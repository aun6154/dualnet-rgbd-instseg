import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.resnet import resnet50
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from collections import OrderedDict
import numpy as np

try:
    from torchvision.models.detection import roi_heads as _roi_heads_module
    from torchvision.models.detection.roi_heads import project_masks_on_boxes as _project_masks_on_boxes
    _DICE_LOSS_AVAILABLE = True
except ImportError:
    _DICE_LOSS_AVAILABLE = False


def _dice_loss(pred_logits, targets, eps=1e-6):
    """마스크 전체(덩어리)의 겹침 비율을 직접 최적화하는 loss.
    BCE는 픽셀 하나하나를 독립적으로 채점해서 애매한 픽셀이 각지게(블록형으로)
    흔들리기 쉬운 반면, Dice는 '이 덩어리 전체가 GT와 얼마나 겹치는가'를 직접
    최적화하므로 더 매끄럽고 뭉친 형태를 학습하는 경향이 있음."""
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.flatten(1)
    targets_flat = targets.flatten(1)
    intersection = (pred_flat * targets_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def _maskrcnn_loss_with_dice(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, dice_weight=1.0):
    """torchvision 기본 maskrcnn_loss와 동일한 타겟 생성 로직을 쓰되,
    최종 loss를 BCE + dice_weight * Dice로 합산."""
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        _project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    selected_logits = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
    bce = F.binary_cross_entropy_with_logits(selected_logits, mask_targets)
    dice = _dice_loss(selected_logits, mask_targets)
    return bce + dice_weight * dice


def _boundary_weight_map(mask_targets, dilate=3, edge_weight=3.0):
    """GT 마스크에서 '경계 근처 band'만 골라 가중치 맵을 만든다.
    Dice처럼 마스크 전체 면적의 밸런스를 바꾸는 게 아니라, 경계에서 먼 안쪽/바깥쪽
    픽셀은 기존 BCE와 동일하게(가중치 1.0) 두고 경계 부근 픽셀만 edge_weight배 더
    벌점을 주므로, 마스크가 전체적으로 넓어지거나 좁아지는 편향이 생기지 않는다.

    mask_targets: (N, H, W) 0/1 float tensor
    dilate       : 경계 band의 두께(픽셀, 정확히는 kernel 반경)
    edge_weight  : 경계 픽셀에 곱할 가중치 배수
    """
    m = mask_targets.unsqueeze(1)  # (N, 1, H, W)
    k = dilate * 2 + 1
    kernel = torch.ones((1, 1, k, k), device=mask_targets.device)

    # dilation: max-pool로 근사 (conv+threshold보다 값 폭주 없이 안전)
    dilated = F.max_pool2d(m, kernel_size=k, stride=1, padding=dilate)
    # erosion: 배경(1-m)을 dilation한 뒤 반전 = mask의 erosion
    eroded = 1.0 - F.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=dilate)

    # "dilated 안쪽이지만 eroded 바깥쪽" = 마스크 경계를 둘러싼 band
    boundary = (dilated - eroded).clamp(0, 1)
    weight = 1.0 + (edge_weight - 1.0) * boundary
    return weight.squeeze(1)  # (N, H, W)


def _maskrcnn_loss_boundary_weighted(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs,
                                      edge_weight=3.0, dilate=3):
    """torchvision 기본 maskrcnn_loss와 동일한 타겟 생성 로직을 쓰되,
    최종 loss를 '경계 근처만 가중치를 높인 BCE'로 계산."""
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        _project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    selected_logits = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
    weight_map = _boundary_weight_map(mask_targets, dilate=dilate, edge_weight=edge_weight)
    bce = F.binary_cross_entropy_with_logits(selected_logits, mask_targets, weight=weight_map)
    return bce


def enable_boundary_weighted_mask_loss(edge_weight=3.0, dilate=3):
    """MaskRCNN의 mask loss 계산 함수를 '경계 가중 BCE'로 교체(monkey-patch)한다.
    모델을 만들기 전에 한 번만 호출. Dice loss(enable_dice_mask_loss)와는
    동시에 켜면 나중에 호출한 쪽이 덮어쓰므로 둘 중 하나만 쓰세요."""
    if not _DICE_LOSS_AVAILABLE:
        raise ImportError(
            "torchvision.models.detection.roi_heads.project_masks_on_boxes를 "
            "찾을 수 없습니다. 설치된 torchvision 버전이 내부 구조를 바꿨을 수 있으니, "
            "'pip show torchvision'으로 버전을 확인해서 알려주세요."
        )

    def _patched(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        return _maskrcnn_loss_boundary_weighted(mask_logits, proposals, gt_masks, gt_labels,
                                                 mask_matched_idxs, edge_weight=edge_weight, dilate=dilate)

    _roi_heads_module.maskrcnn_loss = _patched
    print(f"[Boundary Loss] mask loss를 경계 band(두께 {dilate}px)만 {edge_weight}배 가중한 BCE로 교체했습니다.")


def enable_dice_mask_loss(dice_weight=1.0):
    """MaskRCNN의 mask loss 계산 함수를 BCE+Dice 조합으로 교체(monkey-patch)한다.
    모델을 만들기 전에 한 번만 호출하면 됨. torchvision의 비공식 내부 API
    (project_masks_on_boxes)에 의존하므로, torchvision 버전이 크게 바뀌면
    깨질 수 있음 — 이 경우 ImportError 메시지로 바로 알 수 있게 해둠."""
    if not _DICE_LOSS_AVAILABLE:
        raise ImportError(
            "torchvision.models.detection.roi_heads.project_masks_on_boxes를 "
            "찾을 수 없습니다. 설치된 torchvision 버전이 내부 구조를 바꿨을 수 있으니, "
            "'pip show torchvision'으로 버전을 확인해서 알려주세요."
        )

    def _patched(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        return _maskrcnn_loss_with_dice(mask_logits, proposals, gt_masks, gt_labels,
                                         mask_matched_idxs, dice_weight=dice_weight)

    _roi_heads_module.maskrcnn_loss = _patched
    print(f"[Dice Loss] mask loss를 BCE + {dice_weight} * Dice 조합으로 교체했습니다.")


def adaptive_depth_normalize(depth_mm, low_pct=1, high_pct=99, valid_mask=None):
    """
    고정된 depth_min/depth_max(mm) 대신, 이미지마다 실제로 관측된 depth 분포의
    1~99 percentile을 그때그때 0~1로 매핑한다.

    왜 필요한가: 합성 데이터(카메라~바닥 거리 300~1200mm)와 실물 데이터(268~584mm)처럼
    절대 촬영 거리가 도메인마다 다르면, 고정 mm 범위로는 두 도메인을 같은 스케일에
    놓을 수 없다. "이 장면 안에서 상대적으로 가깝다/멀다"라는 정보만 남기면
    절대 거리 차이에 영향을 덜 받는다.

    depth_mm    : (H, W) depth 값 (mm 단위, 0은 보통 결측치)
    valid_mask  : (H, W) bool, True인 픽셀만 percentile 계산에 사용 (결측치 제외용)
    """
    if valid_mask is None:
        valid_mask = depth_mm > 0
    valid_vals = depth_mm[valid_mask]
    if valid_vals.size == 0:
        return np.zeros_like(depth_mm, dtype=np.float32)

    lo = np.percentile(valid_vals, low_pct)
    hi = np.percentile(valid_vals, high_pct)
    if hi <= lo:
        hi = lo + 1e-6

    depth_norm = np.clip(depth_mm, lo, hi)
    depth_norm = (depth_norm - lo) / (hi - lo)
    return depth_norm.astype(np.float32)


def mask_nms(masks, scores, labels, iou_thresh=0.5):
    """
    박스 대신 실제 마스크 겹침(IoU)으로 중복 검출을 억제.
    길쭉한 물체 두 개는 박스끼리는 잘 안 겹쳐 보여도(box_nms를 통과해도)
    실제 마스크는 크게 겹치는 경우가 흔해서, 박스 기준 NMS만으로는
    "같은 물체를 여러 번 잡는" 문제를 못 거르는 경우가 있음.
    score가 높은 순으로 훑으면서, 같은 클래스이고 마스크 IoU가 임계값 이상이면
    낮은 score 쪽을 제거.
    """
    order = np.argsort(-scores)
    binary_masks = [m > 0.5 for m in masks]
    suppressed = set()
    keep = []
    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for j in order:
            if j == idx or j in suppressed:
                continue
            if labels[j] != labels[idx]:
                continue
            inter = np.logical_and(binary_masks[idx], binary_masks[j]).sum()
            union = np.logical_or(binary_masks[idx], binary_masks[j]).sum()
            iou = inter / union if union > 0 else 0.0
            if iou > iou_thresh:
                suppressed.add(j)
    return sorted(keep)


def clean_mask(binary_mask, close_kernel=7, open_kernel=5):
    """
    예측 마스크의 각진(블록) 노이즈를 정리한다.
    1) Morphological closing으로 작은 구멍/블록형 결손을 메움
    2) Morphological opening으로 튀어나온 자잘한 조각을 제거
    3) 가장 큰 연결 영역만 남겨 파편을 제거 (한 인스턴스는 하나의 덩어리여야 하므로)
    binary_mask: (H, W) uint8, 0/1
    """
    import cv2 as _cv2

    mask_u8 = (binary_mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8

    close_k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    open_k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    cleaned = _cv2.morphologyEx(mask_u8, _cv2.MORPH_CLOSE, close_k)
    cleaned = _cv2.morphologyEx(cleaned, _cv2.MORPH_OPEN, open_k)

    # 가장 큰 연결 영역만 유지
    num_labels, labels_im = _cv2.connectedComponents(cleaned)
    if num_labels <= 1:
        return cleaned  # 전부 배경이면 그대로 반환
    largest_label = 1 + np.argmax(
        [(labels_im == i).sum() for i in range(1, num_labels)]
    )
    return (labels_im == largest_label).astype(np.uint8)


class TwoStreamFusionBackbone(nn.Module):
    """
    RGB 스트림과 Depth 스트림을 각각 layer1~layer4까지 독립적으로 통과시키고,
    매 레벨(layer1/2/3/4 출력)마다 element-wise 덧셈으로 융합한 뒤,
    4개 레벨을 표준 FPN(FeaturePyramidNetwork)에 태워 멀티스케일 피처를 만든다.

    기존(layer1 이후 단일 지점 융합) 대비 개선점:
      - 저수준(layer1, 경계선 등)부터 고수준(layer4, 의미/형태)까지 각 스케일에서
        RGB/Depth가 상호보완적으로 정보를 합침
      - FPN 덕분에 RPN/ROI Head가 여러 해상도의 피처를 동시에 참고 가능
        → 겹쳐 쌓인 객체를 분리하는 능력, 마스크 경계 정교함 모두 개선 기대
    """
    def __init__(self):
        super().__init__()
        self.rgb_stream = resnet50(weights='DEFAULT')
        self.depth_stream = resnet50(weights='DEFAULT')

        # 4채널 입력을 받기 위해 conv1을 4채널로 교체
        self.rgb_stream.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_stream.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 레벨별 채널 수: layer1=256, layer2=512, layer3=1024, layer4=2048 (ResNet50 기준)
        # 각 레벨에서 RGB+Depth를 더한 뒤 conv로 한 번 더 섞어준다.
        self.fuse1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fuse2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.fuse3 = nn.Sequential(nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True))
        self.fuse4 = nn.Sequential(nn.Conv2d(2048, 2048, 3, padding=1), nn.BatchNorm2d(2048), nn.ReLU(inplace=True))

        # 표준 FPN: 4개 레벨(256/512/1024/2048채널)을 받아 모두 256채널로 통일하고,
        # LastLevelMaxPool로 5번째(가장 저해상도) 레벨을 추가 생성 (torchvision 표준 관례)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
            extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = 256

    @staticmethod
    def _stem(stream, x):
        x = stream.conv1(x)
        x = stream.bn1(x)
        x = stream.relu(x)
        x = stream.maxpool(x)
        return x

    def forward(self, x):
        # 각 스트림은 서로 다른 모달리티(RGB/Depth)의 저수준 특징을 독립적으로 학습하도록
        # layer2~4도 "융합된 피처"가 아니라 "자기 스트림의 피처"를 계속 이어받는다.
        # (융합은 FPN에 넘기는 출력 지점에서만 발생 — 스트림 내부 표현은 서로 오염시키지 않음)
        r_stem = self._stem(self.rgb_stream, x)
        d_stem = self._stem(self.depth_stream, x)

        r1 = self.rgb_stream.layer1(r_stem);  d1 = self.depth_stream.layer1(d_stem)
        r2 = self.rgb_stream.layer2(r1);      d2 = self.depth_stream.layer2(d1)
        r3 = self.rgb_stream.layer3(r2);      d3 = self.depth_stream.layer3(d2)
        r4 = self.rgb_stream.layer4(r3);      d4 = self.depth_stream.layer4(d3)

        f1 = self.fuse1(r1 + d1)
        f2 = self.fuse2(r2 + d2)
        f3 = self.fuse3(r3 + d3)
        f4 = self.fuse4(r4 + d4)

        # FPN 입력 키는 관례상 '0'(가장 고해상도) ~ '3'(가장 저해상도) 순서
        feats = OrderedDict([("0", f1), ("1", f2), ("2", f3), ("3", f4)])
        return self.fpn(feats)  # -> OrderedDict('0','1','2','3','pool') 5레벨 반환


def get_two_stream_maskrcnn(num_classes=2, box_nms_thresh=0.3, box_score_thresh=0.05):
    """
    box_nms_thresh   : 같은 클래스 내 박스끼리 이 값 이상 겹치면 낮은 점수쪽을 제거.
                        기본(0.5)보다 낮춰서(0.3), 서로 가깝게 붙어있는 진짜 다른 인스턴스가
                        중복 제안으로 오인되어 뭉개지는 걸 줄이면서도, 진짜 중복(같은 객체를
                        여러 번 예측한 것)은 여전히 잘 걸러지도록 함.
    box_score_thresh : 추론 시 이 점수 미만인 예측은 아예 버림 (기본 낮게 둬서 나중에
                        test 스크립트의 --score_thresh로 유연하게 조정 가능하게 함).
    """
    backbone = TwoStreamFusionBackbone()

    # FPN이 5개 레벨('0'~'3' + LastLevelMaxPool의 'pool')을 반환하므로,
    # anchor_generator도 5레벨에 맞춰 지정 (torchvision 표준 FPN 설정과 동일한 값)
    # 종횡비: 신발 밑창처럼 길쭉한 형태를 고려해 기본(0.5,1,2)보다 훨씬 좁고 긴 비율도 추가.
    # (RPN이 물체 형태와 안 맞는 박스를 제안하면, 인접한 두 인스턴스가 뭉치거나
    #  하나가 둘로 쪼개지는 문제로 이어지기 쉬움)
    # 크기: 레벨당 1개(32,64,128...)만 쓰면 그 사이 크기의 물체를 놓치기 쉬워서,
    # 인접 레벨과 겹치는 중간 크기도 추가해 스케일 커버리지를 촘촘하게 함.
    anchor_generator = AnchorGenerator(
        sizes=((24, 32), (48, 64), (96, 128), (192, 256), (384, 512)),
        aspect_ratios=((0.2, 0.33, 0.5, 1.0, 2.0, 3.0, 5.0),) * 5
    )

    # 마스크 브랜치 RoIAlign 해상도를 기본 14x14 -> 28x28로 2배 상향.
    # (마스크 브랜치는 완전 컨볼루션 구조라 해상도를 바꿔도 head 구조는 그대로 사용 가능.
    #  좁고 긴 밑창의 경계를 더 정교하게 그리기 위한 조치. 연산량/메모리는 다소 늘어남)
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2
    )

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        mask_roi_pool=mask_roi_pool,
        box_nms_thresh=box_nms_thresh,
        box_score_thresh=box_score_thresh,
        image_mean=[0.485, 0.456, 0.406, 0.5],  # 4채널 평균
        image_std=[0.229, 0.224, 0.225, 0.5]    # 4채널 표준편차
    )
    return model


if __name__ == "__main__":
    print("--- 3. 모델 전방향(Forward) 테스트 ---")
    model = get_two_stream_maskrcnn(num_classes=2)
    model.eval()

    dummy_input = torch.rand(2, 4, 800, 800)

    try:
        with torch.no_grad():
            output = model(dummy_input)
        print("✅ 투-스트림 네트워크 모델 통과 성공!")
    except Exception as e:
        print("❌ 실행 중 에러가 발생했습니다:")
        import traceback
        traceback.print_exc()

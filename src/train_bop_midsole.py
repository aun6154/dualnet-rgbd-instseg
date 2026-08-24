"""
train_bop_midsole.py
---------------------
Two-Stream RGB-D Mask R-CNN 학습 스크립트

- mask_rcnn_trainer.py 의 get_two_stream_maskrcnn() 을 그대로 불러와 사용합니다.
- BOPMidsoleInstanceDataset: BOP 포맷(RGB + Depth(.exr) + 개별 Instance Mask)을 읽어
  4채널(RGB+D) 텐서와 Mask R-CNN용 target(dict)을 생성합니다.
- 표준 torchvision Detection 학습 루프(train_one_epoch / evaluate_loss)를 포함합니다.

*** 가정한 폴더 구조 ***
data_root/
├── rgb/            000000.png, 000001.png, ...
├── depth/          000000.exr, 000001.exr, ...   (단위: mm 가정)
└── mask/           000000.png, 000001.png, ...   (1채널 라벨맵: 0=배경, 1~N=인스턴스 ID)

박스(bbox)는 gt.yml을 파싱하지 않고 라벨맵에서 각 인스턴스 ID 영역의 bounding box를 직접 계산합니다.
"""

import os
# OpenCV가 .exr(OpenEXR) 파일을 읽을 수 있도록 cv2 import 전에 반드시 설정해야 합니다.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import glob
import argparse
import time
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from mask_rcnn_trainer import (
    get_two_stream_maskrcnn, adaptive_depth_normalize,
    enable_dice_mask_loss, enable_boundary_weighted_mask_loss,
)


# ----------------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------------
class BOPMidsoleInstanceDataset(Dataset):
    def __init__(self, root_dir, depth_min=300.0, depth_max=1200.0, min_visible_px=300, transforms=None):
        """
        root_dir       : 위 구조를 따르는 데이터셋 루트 경로
        depth_min/max  : [더 이상 사용 안 함] 이제 이미지별 adaptive percentile 정규화를 쓰므로
                          무시됨. 이전 코드/명령어와의 호환을 위해 인자만 남겨둠.
        min_visible_px : 이 픽셀 수 미만으로 보이는 인스턴스는 학습 대상에서 제외.
                         (적당히 가려진 인스턴스는 빈 피킹 태스크의 핵심이라 그대로 두고,
                          거의 안 보이는 파편 수준(라벨 노이즈에 가까운 것)만 걸러내기 위함.
                          이미지 자체를 버리는 게 아니라 해당 인스턴스만 target에서 제외.)
        transforms     : (image_tensor, target) -> (image_tensor, target) 형태의 augmentation 함수 (선택)
        """
        self.root_dir = root_dir
        self.depth_min = depth_min  # 더 이상 사용 안 함 (호환용)
        self.depth_max = depth_max  # 더 이상 사용 안 함 (호환용)
        self.min_visible_px = min_visible_px
        self.transforms = transforms

        rgb_files = sorted(glob.glob(os.path.join(root_dir, "rgb", "*.png")))
        if len(rgb_files) == 0:
            raise FileNotFoundError(
                f"'{root_dir}/rgb' 에서 png 파일을 찾지 못했습니다. 폴더 구조를 확인해주세요."
            )
        self.img_ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in rgb_files]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # ---- RGB 로드 (0~1 정규화, Mask R-CNN 내부 transform이 mean/std 정규화를 다시 수행) ----
        rgb_path = os.path.join(self.root_dir, "rgb", f"{img_id:06d}.png")
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # (H, W, 3)

        # ---- Depth 로드 (.exr, mm 단위 가정) 후 이미지별 adaptive 정규화 ----
        depth_path = os.path.join(self.root_dir, "depth", f"{img_id:06d}.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth 파일을 읽을 수 없습니다: {depth_path}")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        # 고정 mm 범위 대신, 이 이미지에서 관측된 depth 분포 기준으로 0~1 정규화
        # (합성/실물 간 절대 촬영 거리가 달라도 상대적 형태 정보는 동일하게 유지됨)
        depth = adaptive_depth_normalize(depth.astype(np.float32))

        # ---- RGB + Depth 결합 -> (4, H, W) ----
        rgbd = np.dstack([rgb, depth])  # (H, W, 4)
        image_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).float()

        # ---- 인스턴스 마스크 로드 + 박스 계산 ----
        # mask/{img_id:06d}.png 는 1채널 라벨맵: 픽셀값 0=배경, 1~N=인스턴스 ID
        # (파일이 인스턴스별로 나뉘어 있지 않고, 한 이미지에 모든 인스턴스가 색상(값)으로 구분되어 들어있음)
        mask_path = os.path.join(self.root_dir, "mask", f"{img_id:06d}.png")
        label_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if label_map is None:
            raise FileNotFoundError(f"마스크 파일을 읽을 수 없습니다: {mask_path}")

        instance_ids = np.unique(label_map)
        instance_ids = instance_ids[instance_ids != 0]  # 배경(0) 제외

        # 오른쪽/왼쪽 구분 없이 단일 클래스(midsole=1)로 학습.
        # (좌우 대칭 형태 때문에 RPN/classifier가 클래스를 헷갈려 같은 물체를
        #  서로 다른 클래스로 중복 예측하고, NMS가 클래스별로 동작해 이 중복을
        #  걸러내지 못하는 문제가 있었음. 좌/우 구분은 필요하면 후처리 단계에서
        #  마스크 형태 기반으로 별도 판별하는 게 더 안정적)
        masks, boxes, labels = [], [], []
        H, W = rgb.shape[:2]
        for inst_id in instance_ids:
            binary_mask = (label_map == inst_id).astype(np.uint8)
            visible_px = int(binary_mask.sum())
            if visible_px == 0:
                continue
            if visible_px < self.min_visible_px:
                # 거의 안 보이는 파편(라벨 노이즈에 가까움)은 학습 대상에서 제외.
                # 적당히 가려진 인스턴스는 빈 피킹의 핵심 케이스이므로 여기서 걸러지지 않음.
                continue
            ys, xs = np.where(binary_mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            if x_max <= x_min or y_max <= y_min:
                continue

            masks.append(binary_mask)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # 단일 클래스

        if len(masks) == 0:
            masks_np = np.zeros((0, H, W), dtype=np.uint8)
            boxes_np = np.zeros((0, 4), dtype=np.float32)
            labels_np = np.zeros((0,), dtype=np.int64)
        else:
            masks_np = np.stack(masks, axis=0)
            boxes_np = np.array(boxes, dtype=np.float32)
            labels_np = np.array(labels, dtype=np.int64)

        num_objs = masks_np.shape[0]
        boxes_t = torch.as_tensor(boxes_np, dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": torch.as_tensor(labels_np, dtype=torch.int64),  # 1=오른쪽 솔, 2=왼쪽 솔
            "masks": torch.as_tensor(masks_np, dtype=torch.uint8),
            "image_id": torch.tensor([img_id]),
            "area": (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
                    if num_objs > 0 else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target


class _AugmentedSubset(torch.utils.data.Dataset):
    """torch.utils.data.Subset을 감싸서, val_set은 그대로 두고 train_set에만
    augmentation을 적용하기 위한 얇은 wrapper."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image_tensor, target = self.subset[idx]
        image_tensor, target = self.transform(image_tensor, target)
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def random_hflip_transform(image_tensor, target, p=0.5):
    """
    좌우 반전 augmentation. 단일 클래스라 라벨 스왑은 필요 없고,
    이미지·박스·마스크만 뒤집으면 됨.
    """
    if torch.rand(1).item() >= p:
        return image_tensor, target

    W = image_tensor.shape[-1]
    image_tensor = image_tensor.flip(-1)
    target["masks"] = target["masks"].flip(-1)

    boxes = target["boxes"].clone()
    if boxes.numel() > 0:
        x1 = W - boxes[:, 2]
        x2 = W - boxes[:, 0]
        boxes[:, 0], boxes[:, 2] = x1, x2
    target["boxes"] = boxes

    return image_tensor, target


# ----------------------------------------------------------------------------
# 2. Train / Eval Loop
# ----------------------------------------------------------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, mask_loss_weight=1.0,
                     warmup=False, warmup_factor=1.0 / 1000, warmup_iters=None, grad_clip_norm=1.0):
    """
    warmup       : epoch 0(첫 epoch)에서만 True로 넘겨서, 이터레이션마다 lr을
                   warmup_factor -> 1.0으로 선형 증가시킴. 랜덤 초기화된 RPN/head가
                   사전학습 backbone과 섞여 있는 상태에서 처음부터 --lr(0.005)을
                   그대로 쓰면 초반 gradient가 튀어 nan으로 발산하기 쉬움 — 이걸
                   완화하는 torchvision 공식 detection 레퍼런스의 표준 관행.
    grad_clip_norm: 매 스텝 gradient norm을 이 값으로 clip. 특정 배치에서 유독 큰
                   loss(예: aspect ratio가 극단적인 anchor의 box 오차)가 나와도
                   그 한 스텝이 weight 전체를 nan으로 날리는 걸 막아줌.
    """
    model.train()
    total_loss = 0.0

    lr_scheduler_warmup = None
    if warmup:
        warmup_iters = warmup_iters or min(1000, len(data_loader) - 1)
        lr_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # mask 경계 정교함을 더 우선시하도록 loss_mask에 가중치를 곱해서 합산
        weighted = {k: (v * mask_loss_weight if k == "loss_mask" else v) for k, v in loss_dict.items()}
        losses = sum(loss for loss in weighted.values())

        if not torch.isfinite(losses):
            # nan/inf가 뜬 바로 그 순간 즉시 알아채서 멈추게 함.
            # (지금까지는 이걸 못 걸러서 7 epoch 내내 nan으로 헛돈 것)
            print(f"  !! [Epoch {epoch}][{i}/{len(data_loader)}] loss가 nan/inf입니다. "
                  f"loss_dict={ {k: v.item() for k, v in loss_dict.items()} }")
            raise FloatingPointError(
                "Loss가 nan/inf가 되었습니다. lr을 낮추거나(--lr), warmup/grad_clip이 켜져 있는지 확인하세요."
            )

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if lr_scheduler_warmup is not None:
            lr_scheduler_warmup.step()

        total_loss += losses.item()
        if i % print_freq == 0:
            loss_str = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"[Epoch {epoch}][{i}/{len(data_loader)}] total_loss: {losses.item():.4f} ({loss_str})")

    return total_loss / max(1, len(data_loader))


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    # torchvision MaskRCNN은 train() 모드에서만 loss dict를 반환하므로,
    # gradient 계산만 no_grad로 막고 train() 모드를 유지합니다.
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    return total_loss / max(1, len(data_loader))


# ----------------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="BOP 데이터셋 루트 경로")
    parser.add_argument("--epochs", type=int, default=200, help="최대 epoch 상한 (early stopping으로 그 전에 멈출 수 있음)")
    parser.add_argument("--patience", type=int, default=8, help="val_loss가 이만큼 연속으로 개선 안 되면 학습 종료")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="이보다 작은 개선은 '개선 없음'으로 간주")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--log_dir", type=str, default="./logs/twostream_rgbd")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                         help="gradient norm clipping 값. loss가 nan으로 발산하는 것을 방지.")
    parser.add_argument("--keep_last_n", type=int, default=2,
                         help="epoch별 전체 체크포인트(.tar)를 최근 몇 개까지만 남기고 자동 삭제할지. "
                              "best.pth는 별도로 항상 유지되므로 영향 없음. 디스크 용량 부족 방지용.")
    parser.add_argument("--resume", type=str, default=None, help="재개할 체크포인트(.tar) 경로")
    parser.add_argument("--reset_optimizer", action="store_true", default=False,
                         help="--resume 사용 시 모델 가중치는 그대로 불러오되 optimizer/scheduler 상태는 "
                              "리셋하고 --lr 값으로 다시 시작. lr이 조기에 지나치게 감쇠되어 학습이 "
                              "멈춘 경우, 체크포인트를 버리지 않고 이어서 재학습할 때 사용.")
    parser.add_argument("--lr_patience", type=int, default=6,
                         help="ReduceLROnPlateau patience (기존 3 -> 6으로 완화. batch_size가 작아 "
                              "val_loss 노이즈가 큰 상황에서 3은 너무 민감하게 lr을 깎았음)")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                         help="ReduceLROnPlateau factor (기존 0.1 -> 0.5로 완화. 0.1은 한 번 발동시 "
                              "lr을 1/10로 급격히 깎아, 노이즈로 오발동하면 회복이 어려웠음)")
    parser.add_argument("--depth_min", type=float, default=300.0)
    parser.add_argument("--depth_max", type=float, default=1200.0)
    parser.add_argument("--min_visible_px", type=int, default=300,
                         help="이 픽셀 수 미만으로 보이는(거의 안 보이는 파편) 인스턴스는 학습에서 제외")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=2, help="배경 포함 클래스 수 (배경+midsole=2, 좌우 구분 없음)")
    parser.add_argument("--augment", action="store_true", default=True, help="좌우반전 augmentation 사용 (기본 켜짐)")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--box_nms_thresh", type=float, default=0.3, help="같은 클래스 박스끼리 이 값 이상 겹치면 중복 제거")
    parser.add_argument("--mask_loss_weight", type=float, default=2.0, help="loss_mask에 곱할 가중치 (마스크 경계 정교함 우선 학습)")
    parser.add_argument("--use_dice_loss", action="store_true", default=False,
                         help="mask loss를 BCE+Dice 조합으로 계산 (기본 꺼짐 — precision/recall 저하로 인해 롤백. "
                              "필요시 --use_dice_loss로 다시 켤 수 있음)")
    parser.add_argument("--no_dice_loss", dest="use_dice_loss", action="store_false")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="BCE에 더할 Dice loss의 가중치")
    parser.add_argument("--use_boundary_loss", action="store_true", default=False,
                         help="mask loss를 '경계 band만 가중한 BCE'로 계산. Dice와 달리 마스크 전체 "
                              "면적 밸런스는 건드리지 않아 precision/recall 저하 리스크가 낮음. "
                              "--use_dice_loss와 동시에 켜지 마세요 (나중 호출이 덮어씀).")
    parser.add_argument("--boundary_edge_weight", type=float, default=3.0,
                         help="경계 band 픽셀에 곱할 가중치 배수")
    parser.add_argument("--boundary_dilate", type=int, default=3,
                         help="경계 band 두께(픽셀 반경)")
    args = parser.parse_args()

    if args.use_dice_loss and args.use_boundary_loss:
        raise ValueError("--use_dice_loss 와 --use_boundary_loss 는 동시에 켤 수 없습니다. 하나만 선택하세요.")

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = BOPMidsoleInstanceDataset(
        args.data_root, depth_min=args.depth_min, depth_max=args.depth_max,
        min_visible_px=args.min_visible_px
    )
    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"전체 {len(full_dataset)}장 -> train {n_train} / val {n_val}")

    if args.augment:
        # val_set은 그대로 두고, train_set에만 좌우반전 augmentation을 씌움
        # (원본 full_dataset은 공유되므로 Subset을 감싸는 방식으로 train에만 적용)
        train_set = _AugmentedSubset(train_set, random_hflip_transform)
        print("좌우반전 augmentation 적용됨 (train만)")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), collate_fn=collate_fn
    )

    if args.use_dice_loss:
        enable_dice_mask_loss(dice_weight=args.dice_weight)
    elif args.use_boundary_loss:
        enable_boundary_weighted_mask_loss(edge_weight=args.boundary_edge_weight, dilate=args.boundary_dilate)

    model = get_two_stream_maskrcnn(num_classes=args.num_classes, box_nms_thresh=args.box_nms_thresh)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    # val_loss가 정체되면 자동으로 lr을 줄여줌 (고정 step 대신 실제 학습 정체 시점에 반응)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=1e-6
    )

    start_epoch = 0
    prev_elapsed_seconds = 0.0
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        if args.reset_optimizer:
            # 모델 가중치는 이어받되, optimizer/scheduler는 새로 만들어서 --lr 값으로 재시작.
            # lr이 조기에 5e-6 수준까지 감쇠되어 사실상 학습이 멈춘 상태에서, 가중치를 버리지
            # 않고 살아있는 lr로 이어서 학습을 재개하기 위함.
            start_epoch = ckpt["epoch"] + 1
            prev_elapsed_seconds = ckpt.get("elapsed_seconds", 0.0)
            print(f"체크포인트 로드 완료: {args.resume} (epoch {start_epoch}부터 재개, "
                  f"optimizer/scheduler는 리셋 -> lr {args.lr}부터 재시작, "
                  f"기존 누적 학습시간 {prev_elapsed_seconds/3600:.2f}시간)")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            lr_scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            prev_elapsed_seconds = ckpt.get("elapsed_seconds", 0.0)  # 이전 학습 시간 이어받기
            print(f"체크포인트 로드 완료: {args.resume} (epoch {start_epoch}부터 재개, "
                  f"기존 누적 학습시간 {prev_elapsed_seconds/3600:.2f}시간)")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_start_time = time.time() - prev_elapsed_seconds  # resume 시 이전 시간을 이어서 누적
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                      mask_loss_weight=args.mask_loss_weight,
                                      warmup=(epoch == start_epoch),  # 이 학습이 시작되는 첫 epoch에만 warmup
                                      grad_clip_norm=args.grad_clip_norm)
        val_loss = evaluate_loss(model, val_loader, device)
        lr_scheduler.step(val_loss)  # ReduceLROnPlateau는 step()에 지표를 넘겨줘야 함

        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - train_start_time
        print(f"===== Epoch {epoch} 완료 | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"lr: {optimizer.param_groups[0]['lr']:.6f} | "
              f"이번 epoch: {epoch_elapsed/60:.1f}분 | 누적: {total_elapsed/3600:.2f}시간 =====")

        ckpt_path = os.path.join(args.log_dir, f"{epoch}.tar")
        try:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
                "val_loss": val_loss,
                "elapsed_seconds": total_elapsed,  # 여기까지 누적 학습 시간(초) 기록
            }, ckpt_path)
        except (RuntimeError, OSError) as e:
            # 디스크 용량 부족 등으로 저장이 중간에 실패해도 학습 프로세스 자체는 죽지 않게 함.
            # 실패한 파일은 손상되어 있을 수 있으니 바로 지움.
            print(f"  !! 체크포인트 저장 실패 (epoch {epoch}): {e}")
            print(f"  !! 디스크 용량을 확인하세요: df -h  /  du -sh {args.log_dir}/*")
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        else:
            # 저장 성공 시, 최근 --keep_last_n개만 남기고 오래된 epoch 체크포인트는 자동 삭제
            # (best.pth는 아래에서 별도 관리되므로 여기서 지워지지 않음)
            all_ckpts = sorted(
                glob.glob(os.path.join(args.log_dir, "*.tar")),
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
            for old_ckpt in all_ckpts[:-args.keep_last_n]:
                os.remove(old_ckpt)

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.log_dir, "best.pth"))
            print("  -> 최고 성능 갱신, best.pth 저장")
        else:
            epochs_no_improve += 1
            print(f"  -> val_loss 개선 없음 ({epochs_no_improve}/{args.patience})")

        if epochs_no_improve >= args.patience:
            total_elapsed = time.time() - train_start_time
            print(f"\n조기 종료: val_loss가 {args.patience} epoch 연속 개선되지 않았습니다 "
                  f"(최종 best_val_loss: {best_val_loss:.4f}, epoch {epoch - epochs_no_improve}). "
                  f"총 학습 시간: {total_elapsed/3600:.2f}시간")
            break
    else:
        total_elapsed = time.time() - train_start_time
        print(f"\n최대 epoch({args.epochs})에 도달하여 학습을 종료합니다 "
              f"(best_val_loss: {best_val_loss:.4f}). 총 학습 시간: {total_elapsed/3600:.2f}시간")


if __name__ == "__main__":
    main()

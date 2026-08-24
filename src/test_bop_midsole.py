"""
test_bop_midsole.py
---------------------
학습된 Two-Stream RGB-D Mask R-CNN 체크포인트를 로드해서:
  1) 이미지별 예측(box/score/mask) 개수와 confidence 통계를 출력
  2) GT vs 예측 마스크를 나란히 시각화한 PNG를 저장
  3) IoU 0.5 기준 간단한 precision/recall(암기 여부를 가늠하는 용도, 정식 mAP는 아님)

사용법:
    python test_bop_midsole.py --data_root <경로> --checkpoint logs/twostream_rgbd/best.pth
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")  # 파일 저장 전용 백엔드. cv2의 Qt 플러그인과 충돌 방지 (GUI 창 안 띄움)
import matplotlib.pyplot as plt

from mask_rcnn_trainer import get_two_stream_maskrcnn, mask_nms
from train_bop_midsole import BOPMidsoleInstanceDataset


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def visualize(rgb, gt_masks, pred_masks, pred_scores, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb); axes[0].set_title("RGB 원본"); axes[0].axis("off")

    gt_overlay = rgb.copy()
    for m in gt_masks:
        color = np.random.randint(0, 255, 3)
        gt_overlay[m > 0] = gt_overlay[m > 0] * 0.4 + color * 0.6
    axes[1].imshow(gt_overlay.astype(np.uint8))
    axes[1].set_title(f"GT ({len(gt_masks)}개 인스턴스)"); axes[1].axis("off")

    pred_overlay = rgb.copy()
    for m, s in zip(pred_masks, pred_scores):
        color = np.random.randint(0, 255, 3)
        pred_overlay[m > 0.5] = pred_overlay[m > 0.5] * 0.4 + color * 0.6
    axes[2].imshow(pred_overlay.astype(np.uint8))
    axes[2].set_title(f"예측 ({len(pred_masks)}개, score>0.5)"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="best.pth 또는 {epoch}.tar 경로")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="./test_results")
    parser.add_argument("--depth_min", type=float, default=300.0)
    parser.add_argument("--depth_max", type=float, default=1200.0)
    parser.add_argument("--max_images", type=int, default=20, help="너무 많으면 시간이 오래 걸리므로 상한")
    parser.add_argument("--num_classes", type=int, default=2, help="학습 때 사용한 num_classes와 반드시 일치해야 함 (배경+midsole=2)")
    parser.add_argument("--mask_nms_thresh", type=float, default=0.5,
                         help="마스크 IoU가 이 값을 넘으면 중복 검출로 간주해 제거. 0이면 끔")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BOPMidsoleInstanceDataset(args.data_root, depth_min=args.depth_min, depth_max=args.depth_max)
    print(f"총 {len(dataset)}장 로드됨 (최대 {args.max_images}장까지만 테스트)")

    model = get_two_stream_maskrcnn(num_classes=args.num_classes)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # best.pth는 state_dict 그대로, {epoch}.tar는 dict 안에 "model_state" 키로 감싸져 있음
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_gt, total_tp, total_pred = 0, 0, 0
    all_scores = []
    all_mask_ious = []  # 매칭된 쌍들의 픽셀 마스크 IoU 전체 (경계 정교함 지표)

    with torch.no_grad():
        for idx in range(min(len(dataset), args.max_images)):
            image_tensor, target = dataset[idx]
            output = model([image_tensor.to(device)])[0]

            keep = output["scores"] >= args.score_thresh
            pred_boxes = output["boxes"][keep].cpu().numpy()
            pred_masks = output["masks"][keep].squeeze(1).cpu().numpy()  # (N,H,W)
            pred_scores = output["scores"][keep].cpu().numpy()
            pred_labels = output["labels"][keep].cpu().numpy()

            # 박스 NMS로 못 거른 마스크 중복(같은 물체를 여러 번 잡는 것) 추가 제거
            if args.mask_nms_thresh > 0 and len(pred_masks) > 1:
                nms_keep = mask_nms(pred_masks, pred_scores, pred_labels, iou_thresh=args.mask_nms_thresh)
                pred_boxes = pred_boxes[nms_keep]
                pred_masks = pred_masks[nms_keep]
                pred_scores = pred_scores[nms_keep]
                pred_labels = pred_labels[nms_keep]
            all_scores.extend(pred_scores.tolist())

            gt_boxes = target["boxes"].numpy()
            gt_masks = target["masks"].numpy()
            gt_labels = target["labels"].numpy()

            matched = set()
            tp = 0
            matched_mask_ious = []  # 매칭된 쌍의 실제 픽셀 마스크 IoU (박스 IoU와 별개 지표)
            for pi, (pb, pl) in enumerate(zip(pred_boxes, pred_labels)):
                best_iou, best_j = 0, -1
                for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if j in matched or gl != pl:  # IoU뿐 아니라 클래스(오른쪽/왼쪽)도 일치해야 인정
                        continue
                    iou = compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= args.iou_thresh:
                    matched.add(best_j)
                    tp += 1
                    # 박스가 매칭된 쌍에 대해 실제 마스크(픽셀) IoU도 계산
                    pred_bin = pred_masks[pi] > 0.5
                    gt_bin = gt_masks[best_j] > 0
                    inter = np.logical_and(pred_bin, gt_bin).sum()
                    union = np.logical_or(pred_bin, gt_bin).sum()
                    mask_iou = inter / union if union > 0 else 0.0
                    matched_mask_ious.append(mask_iou)

            total_gt += len(gt_boxes)
            total_tp += tp
            total_pred += len(pred_boxes)
            all_mask_ious.extend(matched_mask_ious)

            rgb_vis = (image_tensor[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            save_path = os.path.join(args.out_dir, f"{idx:03d}_compare.png")
            visualize(rgb_vis, gt_masks, pred_masks, pred_scores, save_path)

            mean_mask_iou_this_img = np.mean(matched_mask_ious) if matched_mask_ious else 0.0
            print(f"[{idx}] GT {len(gt_boxes)}개 / "
                  f"예측 {len(pred_boxes)}개(score≥{args.score_thresh}) / "
                  f"매칭(IoU≥{args.iou_thresh}) {tp}개 / 평균 마스크IoU: {mean_mask_iou_this_img:.3f} / score 범위: "
                  f"{pred_scores.min() if len(pred_scores) else 0:.3f}~{pred_scores.max() if len(pred_scores) else 0:.3f}")

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    mean_score = float(np.mean(all_scores)) if all_scores else 0.0
    mean_mask_iou = float(np.mean(all_mask_ious)) if all_mask_ious else 0.0

    print("\n===== 요약 =====")
    print(f"전체 GT 인스턴스: {total_gt}개 / 전체 예측: {total_pred}개 / 매칭(IoU≥{args.iou_thresh}): {total_tp}개")
    print(f"Precision: {precision:.3f} / Recall: {recall:.3f}")
    print(f"평균 confidence score: {mean_score:.3f}")
    print(f"매칭된 쌍의 평균 픽셀 마스크 IoU: {mean_mask_iou:.3f}  <- 마스크 경계 정교함의 핵심 지표")
    print(f"시각화 결과 저장 위치: {args.out_dir}/")

    if mean_mask_iou > 0 and mean_mask_iou < 0.5:
        print("\n참고: 평균 마스크 IoU가 0.5 미만입니다. 박스 위치는 맞아도 마스크 경계 자체가")
        print("      부정확하다는 뜻이라, 시각화에서 '지저분해 보이는' 인상이 착시가 아니라")
        print("      실제 정량 지표로도 확인된 것입니다.")

    if mean_score > 0.99 and precision > 0.99 and recall > 0.99 and len(dataset) < 50:
        print("\n⚠️  경고: 데이터셋 크기가 매우 작고(정밀도/재현율/신뢰도 모두 거의 1.0) 지표가 완벽에 가깝습니다.")
        print("   → 이 결과가 val_loss가 아니라 '학습에 이미 쓰인 이미지'에 대한 것이라면 암기(overfitting) 가능성이 높습니다.")
        print("   → 학습에 전혀 쓰이지 않은(hold-out) 이미지로 다시 테스트해보시는 걸 권장합니다.")


if __name__ == "__main__":
    main()

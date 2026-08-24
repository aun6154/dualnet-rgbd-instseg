# dualnet-rgbd-instseg

Dual-backbone(앙상블) RGB-D Mask R-CNN으로 쌓여있는 신발 중창(midsole)의 인스턴스
마스크를 검출하는 bin-picking perception 프로젝트.

> 같은 조직에 모달리티 분리형 **RGB-D two-stream** 레포가 별도로 있습니다.
> 이 레포는 그 접근을 시도한 뒤 **dual-backbone 앙상블(동일 4채널 입력을 서로
> 다르게 초기화된 백본 두 개가 각자 학습)** 방식으로 pivot한 후속 실험입니다.
> "two-stream"이라는 이름을 쓰지 않는 이유이기도 합니다.

Bin-picking 환경에서 쌓여있는 신발 중창(midsole)의 인스턴스 마스크를 RGB-D 이미지로부터
검출하는 모델. Mask R-CNN(torchvision)을 기반으로, 서로 다르게 초기화된 ResNet-50 백본
두 개가 동일한 RGB+D 4채널 입력을 각자 받아 학습하고, 각 FPN 레벨(layer1~4)마다 두
백본의 출력을 융합하는 **이중 백본 앙상블(dual-backbone ensemble) 구조**를 사용합니다.
(자세한 설계 배경은 아래 "아키텍처 노트" 참고)

## 구조

```
src/
├── mask_rcnn_trainer.py   # 모델 정의(backbone/FPN/anchor), NMS/후처리 유틸, dice/boundary loss
├── train_bop_midsole.py   # 학습 스크립트 (BOP 포맷 데이터셋 로더 포함)
└── test_bop_midsole.py    # 체크포인트 평가 + GT-예측 비교 시각화
scripts/
├── train.sh                # 학습 실행 예시
├── test.sh                 # 평가 실행 예시
└── check_ckpt.py           # 체크포인트 무결성(손상 여부) 검사
```

## 데이터 포맷

BOP 포맷을 가정합니다.

```
data_root/
├── rgb/     000000.png, 000001.png, ...
├── depth/   000000.exr, 000001.exr, ...   (mm 단위)
└── mask/    000000.png, 000001.png, ...   (1채널 라벨맵: 0=배경, 1~N=인스턴스 ID)
```

## 사용법

```bash
pip install -r requirements.txt

# 학습
DATA_ROOT=./data/Dual_Soles LOG_DIR=./logs/run1 bash scripts/train.sh

# 평가 (hold-out 테스트셋)
DATA_ROOT=./data/Dual_Soles_test CHECKPOINT=./logs/run1/best.pth bash scripts/test.sh

# 체크포인트 무결성 확인 (디스크 fail 등으로 손상됐는지)
python scripts/check_ckpt.py ./logs/run1/32.tar
```

주요 학습 옵션 (`train_bop_midsole.py --help` 참고):

| 옵션 | 설명 |
|---|---|
| `--lr_patience` / `--lr_factor` | ReduceLROnPlateau 완화 설정 (기본 patience=6, factor=0.5) |
| `--grad_clip_norm` | Gradient clipping — loss nan 발산 방지 |
| `--keep_last_n` | 최근 N개 체크포인트만 유지, 나머지 자동 삭제 (디스크 용량 관리) |
| `--resume` / `--reset_optimizer` | 체크포인트 재개. `--reset_optimizer`는 가중치만 이어받고 lr/optimizer는 새로 시작 |
| `--use_boundary_loss` | 마스크 경계 band에만 가중치를 준 BCE (Dice보다 안전한 boundary 개선 옵션) |
| `--use_dice_loss` | BCE+Dice 결합. **주의**: 이 프로젝트에서는 precision/recall 저하로 롤백된 이력 있음 |

## 아키텍처 노트

`TwoStreamFusionBackbone`이라는 이름과 달리, 두 ResNet-50 스트림은 RGB 전용/Depth 전용으로
모달리티가 분리되어 입력되지 않고 **동일한 RGB+D 4채널 결합 텐서를 그대로 두 스트림에 입력**합니다.
이는 버그가 아니라 **의도적인 설계 변경 이력**입니다 — 모달리티별로 분리된 진짜 two-stream(RGB만
vs Depth만)을 먼저 시도했으나 성능이 좋지 않아, 서로 다르게 초기화된 두 백본이 동일한 결합 입력을
각자 학습해 앙상블하는 현재 방식으로 방향을 틀었습니다. 클래스명/변수명은 이전 설계의 흔적이 남아
있는 것이므로, 코드를 읽을 때 이름만 보고 모달리티 분리 구조를 가정하지 않도록 주의가 필요합니다.

## Loss 실험 이력

- Dice loss는 precision/recall 저하로 비활성화된 이력 있음 (`--use_dice_loss` 기본 꺼짐).
- Boundary-weighted BCE(`--use_boundary_loss`)는 Dice보다 안전한 대안으로 도입, 마스크 전체
  면적 밸런스는 건드리지 않고 경계 band 픽셀에만 가중치를 줌.
- 현재 실험 방향: 결합 4채널 입력을 받는 이중 백본 앙상블 + adaptive depth normalization
  (percentile 기반) + mask-based NMS + morphological 후처리 조합으로 sim-to-real 갭을 좁히는 중.

## 참고한 선행 연구

- SD-Mask-RCNN (Danielczuk et al., 2019) — depth-only synthetic 학습 기반 bin-picking 세그멘테이션
- UOIS-Net (Xie et al.) — 임베딩 기반 클러스터링으로 겹친 유사 물체 분리

#!/bin/bash
# 학습 실행 스크립트.
# nan 발산 방지 장치(warmup + gradient clipping)가 반영된 안정화 버전입니다.
# lr=0.0025는 batch_size=2 기준으로 튜닝된 값입니다 — 배치 크기를 바꾸면 같이 조정하세요.

DATA_ROOT="${DATA_ROOT:-./data/Dual_Soles}"
LOG_DIR="${LOG_DIR:-./logs/twostream_rgbd}"

python src/train_bop_midsole.py \
  --data_root "$DATA_ROOT" \
  --log_dir "$LOG_DIR" \
  --lr 0.0025 \
  --grad_clip_norm 1.0 \
  --lr_patience 6 \
  --lr_factor 0.5 \
  --batch_size 2 \
  --epochs 200 \
  --patience 8 \
  --keep_last_n 2

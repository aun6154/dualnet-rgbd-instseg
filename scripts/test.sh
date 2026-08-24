#!/bin/bash
# Hold-out 테스트셋 평가 스크립트.

DATA_ROOT="${DATA_ROOT:-./data/Dual_Soles_test}"
CHECKPOINT="${CHECKPOINT:-./logs/twostream_rgbd/best.pth}"

python src/test_bop_midsole.py \
  --data_root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --out_dir ./test_results

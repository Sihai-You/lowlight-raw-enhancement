#!/bin/bash
# 便携版一键训练 + 测试（无硬编码 conda 路径；数据/结果路径可用环境变量覆盖）。
set -e
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-./dataset}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

python train.py \
  --input_dir "$DATA_DIR/0.01" --gt_dir "$DATA_DIR/gt" \
  --gt_raw_dir "$DATA_DIR/gt_raw" --gt_dev_raw_dir "$DATA_DIR/gt_raw" \
  --gt_mode dev_raw --ratio auto --base_ch 32 --ssim_weight 0.15 \
  --epochs "${EPOCHS:-100}" --se --save_dir "$RESULTS_DIR/checkpoints"

python test.py \
  --input_dir "$DATA_DIR/0.01" --gt_dir "$DATA_DIR/gt" \
  --gt_raw_dir "$DATA_DIR/gt_raw" \
  --checkpoint "$RESULTS_DIR/checkpoints/best.pth" \
  --output_dir "$RESULTS_DIR/visuals" --num_images "${NUM_IMAGES:-5}"
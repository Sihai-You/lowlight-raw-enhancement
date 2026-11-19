#!/bin/bash
# SE / no-SE 联合训练（便携版；通过 CUDA_VISIBLE_DEVICES 选 GPU，按顺序执行）。
# 依赖 dataset/multi_train、multi_gt、multi_gt_raw 软链接目录（见 build_multi_train.py）。
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-./dataset}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
EPOCHS="${EPOCHS:-100}"

COMMON="--input_dir $DATA_DIR/multi_train --gt_dir $DATA_DIR/multi_gt \
--gt_mode dev_raw --gt_dev_raw_dir $DATA_DIR/multi_gt_raw \
--ratio auto --gt_raw_dir $DATA_DIR/multi_gt_raw \
--base_ch 32 --ssim_weight 0.15 --epochs $EPOCHS"

echo "== SE 模型 (GPU ${CUDA_VISIBLE_DEVICES:-1}) =="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" $PY train.py $COMMON --se \
  --save_dir "$RESULTS_DIR/checkpoints/abl_devraw_multi_se_real"

echo "== no-SE 模型 (GPU ${CUDA_VISIBLE_DEVICES:-2}) =="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" $PY train.py $COMMON \
  --save_dir "$RESULTS_DIR/checkpoints/abl_devraw_multi_nose_real"
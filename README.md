# Lightweight CNN for Extremely Low-Light RAW Image Enhancement

A lightweight, fully-convolutional neural network for joint **denoising** and
**brightness recovery** of extremely low-light RAW images, in a single forward pass.

> **Note:** This manuscript is currently under review at *Remote Sensing* (MDPI).

## Highlights

- **Single-stage RAW-to-sRGB**: 4-channel packed Bayer RAW input → 3-channel sRGB output.
- **Lightweight**: 0.49M / 1.94M / 7.82M parameters for `base_ch = 8 / 16 / 32`.
- **Strong quality**: 34.37 dB PSNR on the SIED (Sony) test set under the dev-RAW protocol, +13.31 dB over the fixed-ISP gain baseline.
- **Cross-camera generalization**: 33.05 dB on Canon images without fine-tuning.
- **Three illuminance levels**: 0.01–0.1 / 0.001–0.01 / 0.0001–0.001 lx.
- **Real-time edge deployment**: 82.6 FPS on V100 (base_ch=8); 5.98 / 7.61 FPS (fp32 / fp16) on Jetson Orin Nano.

## Repository structure

```
.
├── code/                    # training / testing / benchmarking
│   ├── model.py             # network definition (lightweight U-Net)
│   ├── dataset.py           # dataset / dataloader
│   ├── train.py             # training entry point
│   ├── test.py              # evaluation / inference
│   ├── benchmark.py         # PSNR / SSIM / LPIPS benchmark
│   ├── efficiency_bench.py  # FLOPs / params / speed benchmark
│   ├── run.sh               # one-command train + test
│   └── ...
└── paper/                   # manuscript
    ├── paper.tex            # LaTeX source (MDPI)
    ├── paper.pdf            # compiled PDF
    ├── Definitions/         # MDPI LaTeX class / style files
    └── figures/
```

## Dataset

The [SIED (See-in-the-Extremely-Dark)](https://github.com/JianghaiSCU/SIED) dataset
is used for training and evaluation.

## Requirements

See [`code/requirements.txt`](code/requirements.txt). Core dependencies: PyTorch,
rawpy, numpy, Pillow, lpips, scikit-image, opencv-python-headless.

## Quick start

```bash
cd code
pip install -r requirements.txt

# one-command train + test (set DATA_DIR to the SIED dataset path)
DATA_DIR=/path/to/sied ./run.sh
```

Manual usage:

```bash
python train.py \
  --input_dir $DATA_DIR/0.01 --gt_dir $DATA_DIR/gt \
  --gt_raw_dir $DATA_DIR/gt_raw --gt_dev_raw_dir $DATA_DIR/gt_raw \
  --gt_mode dev_raw --ratio auto --base_ch 32 --ssim_weight 0.15 --se \
  --save_dir results/checkpoints

python test.py \
  --input_dir $DATA_DIR/0.01 --gt_dir $DATA_DIR/gt \
  --gt_raw_dir $DATA_DIR/gt_raw \
  --checkpoint results/checkpoints/best.pth \
  --output_dir results/visuals
```

## Citation

A formal citation will be added once the paper is published.

# -*- coding: utf-8 -*-
"""全分辨率 sRGB 口径公平对比（对齐 SIED 论文全分辨率 2160x3840）。

对 challenge 集全帧推理我们的全分辨率模型（out_ch=12 + pixel_shuffle），
与全分辨率对齐 GT 计算 PSNR/SSIM/LPIPS(alex)，并与论文 Table 5.2 的 21~25dB 对比。

参考（论文 Sony 子集 0.01-0.1 lux，全分辨率 sRGB，PSNR@全帧）：
  SID 21.93, DID 21.64, SGN 22.41, LLPackNet 21.57, RRT 21.11,
  LDC 21.74, MCR 22.87, DNF 24.24, RAWMamba 23.94, SIED(论文) 24.98

用法:
  python benchmark_srgb_full.py --checkpoint checkpoints/srgb_full_l1_100/best.pth \
      --split dataset/challenge_split.json --out results/benchmark_srgb_full.json
"""
import os, json, argparse
import numpy as np
import torch
import cv2
from model import UNet
from dataset_1 import load_raw, compute_ratio_map
from benchmark import get_lpips, eval_one, dev_short_rgb, DEV

# 论文参考值（索尼子集全分辨率 sRGB PSNR，单位 dB）
PAPER_PSNR = {
    'SID': 21.93, 'DID': 21.64, 'SGN': 22.41, 'LLPackNet': 21.57, 'RRT': 21.11,
    'LDC': 21.74, 'MCR': 22.87, 'DNF': 24.24, 'RAWMamba': 23.94, 'SIED(论文)': 24.98,
}

NPY_DIR = './dataset/gt_aligned_4k_npy'
ALIGNED4K_DIR = './dataset/gt_aligned_4k'
GT_DIR = './dataset/gt'
RAW_DIR = './dataset/0.01'

_MODEL = None


def load_gt_full(name):
    """全分辨率对齐 GT (float32 [0,1]) + 有效掩码 (bool)。"""
    npy = os.path.join(NPY_DIR, name + '.npy')
    if os.path.exists(npy):
        gt = np.load(npy).astype(np.float32) / 255.0
        mp = os.path.join(NPY_DIR, name + '.mask.npy')
        mask = np.load(mp) > 127 if os.path.exists(mp) else np.ones(gt.shape[:2], bool)
        return gt, mask
    ap = os.path.join(ALIGNED4K_DIR, name + '.jpg')
    if os.path.exists(ap):
        gt = np.array(cv2.cvtColor(cv2.imread(ap), cv2.COLOR_BGR2RGB),
                      np.float32) / 255.0
        mp = os.path.join(ALIGNED4K_DIR, name + '.mask.png')
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127 if os.path.exists(mp) \
            else np.ones(gt.shape[:2], bool)
        return gt, mask
    gt = np.array(cv2.cvtColor(cv2.imread(os.path.join(GT_DIR, name + '.jpg')),
                               cv2.COLOR_BGR2RGB), np.float32) / 255.0
    return gt, np.ones(gt.shape[:2], bool)


def load_full_model(ckpt, device):
    global _MODEL
    if _MODEL is None:
        raw = torch.load(ckpt, map_location=device, weights_only=False)
        full_res = raw.get('gt_mode') == 'srgb_full' or \
            raw['model']['outc.weight'].shape[0] == 12
        m = UNet(in_ch=4, out_ch=3, base_ch=raw.get('base_ch', 32),
                 bilinear=raw.get('bilinear', False), full_res=full_res).to(device).eval()
        m.load_state_dict(raw['model'])
        _MODEL = m
    return _MODEL


def run_full(ckpt, name, rmap, device):
    m = load_full_model(ckpt, device)
    inp = load_raw(os.path.join(RAW_DIR, name + '.raw'), ratio=rmap[name])
    t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = m(t).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.clip(out, 0, 1).astype(np.float32)


def naive_full(name, rmap):
    """ISP 显影短曝光 + 上采样到全分辨率（无学习锚点）。"""
    dev = dev_short_rgb(name, rmap)                       # [1080,1920,3]
    return cv2.resize(dev, (3840, 2160), interpolation=cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='dataset/challenge_split.json')
    ap.add_argument('--out', default='results/benchmark_srgb_full.json')
    ap.add_argument('--with-naive', action='store_true')
    args = ap.parse_args()

    split = json.load(open(args.split))
    names = split['all'] if isinstance(split, dict) else split
    rmap = compute_ratio_map(RAW_DIR, './dataset/gt_raw')
    device = DEV

    ours = dict(psnr=[], ssim=[], lpips=[])
    naive = dict(psnr=[], ssim=[], lpips=[]) if args.with_naive else None

    for name in names:
        gt, mask = load_gt_full(name)
        pred = run_full(args.checkpoint, name, rmap, device)
        r = eval_one(pred, gt, mask)
        for k in r:
            ours[k].append(r[k])
        if naive is not None:
            rn = eval_one(naive_full(name, rmap), gt, mask)
            for k in rn:
                naive[k].append(rn[k])
        print(f"{name}: PSNR={r['psnr']:.2f}dB SSIM={r['ssim']:.4f} LPIPS={r['lpips']:.4f}")

    table = {}
    for k in ('psnr', 'ssim', 'lpips'):
        v = np.array(ours[k])
        table[f'ours_{k}'] = dict(mean=float(v.mean()), std=float(v.std()))
    if naive is not None:
        for k in ('psnr', 'ssim', 'lpips'):
            v = np.array(naive[k])
            table[f'naive_{k}'] = dict(mean=float(v.mean()), std=float(v.std()))

    print('\n==== 全分辨率 sRGB 结果（challenge 集 %d 帧）====' % len(names))
    print(f"ours  : PSNR={table['ours_psnr']['mean']:.2f}±{table['ours_psnr']['std']:.2f}dB  "
          f"SSIM={table['ours_ssim']['mean']:.4f}  LPIPS={table['ours_lpips']['mean']:.4f}")
    if naive is not None:
        print(f"naive : PSNR={table['naive_psnr']['mean']:.2f}±{table['naive_psnr']['std']:.2f}dB  "
              f"SSIM={table['naive_ssim']['mean']:.4f}  LPIPS={table['naive_lpips']['mean']:.4f}")
    print('\n==== 论文 Table 5.2 索尼子集全分辨率 sRGB PSNR 对比 ====')
    print(f"{'方法':<14}{'PSNR(dB)':>10}")
    for mth, v in PAPER_PSNR.items():
        print(f"{mth:<14}{v:>10.2f}")
    print(f"{'ours(全分辨率)':<12}{table['ours_psnr']['mean']:>10.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(table, open(args.out, 'w'), indent=1)
    print(f'\n结果已保存: {args.out}')


if __name__ == '__main__':
    main()
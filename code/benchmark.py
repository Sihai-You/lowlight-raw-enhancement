# -*- coding: utf-8 -*-
"""统一基准评估框架（对比方法 + 指标）。

口径（两个目标域）：
  srgb   参考 = 配准 GT JPG（dataset/gt_aligned），指标只在有效区(mask)内计算
  devraw 参考 = 固定 ISP 显影的长曝光 RAW（全帧 1080x1920，无 mask）

指标（与 SIED 论文 Table 5 对齐）：
  PSNR ↑ / SSIM ↑ / LPIPS(alex) ↓
  SSIM/LPIPS 输入范围 [0,1]；LPIPS 内部按 2*x-1 转 [-1,1]（与论文 data_transform 一致）。

方法注册：METHODS[name] = callable(name, rmap, device) -> rgb[H,W,3] float32 [0,1]
逐步把 9 个外部基线 + naive + ours 注进来。

用法:
  python benchmark.py --split dataset/challenge_split.json \
      --checkpoint checkpoints/ab_rawgt_100/best.pth --methods naive,ours
"""
import os, json, argparse
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_fn
from model import UNet
from dataset_1 import (load_raw, load_raw_linear, packed_to_rgb_linear,
                       gray_world_wb, develop_rgb, compute_ratio_map, RAW_H, RAW_W)
from test import load_aligned_gt, psnr_masked

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LPIPS_FN = None
RAW_DIR = './dataset/0.01'
GT_RAW_DIR = './dataset/gt_raw'


def get_lpips():
    global LPIPS_FN
    if LPIPS_FN is None:
        LPIPS_FN = lpips.LPIPS(net='alex').to(DEV).eval()
    return LPIPS_FN


def dev_long_rgb(name, rmap, raw_dir=None, gt_raw_dir=None):
    """固定 ISP 显影长曝光 RAW（dev-RAW 口径参考）。"""
    long = load_raw_linear(os.path.join(gt_raw_dir or GT_RAW_DIR, name + '.raw'), 1.0)
    rgb = packed_to_rgb_linear(long)
    wb = gray_world_wb(rgb)
    return develop_rgb(rgb, wb)


def dev_short_rgb(name, rmap, raw_dir=None):
    """固定 ISP 显影短曝光 RAW（naive 放大基线）。"""
    short = load_raw_linear(os.path.join(raw_dir or RAW_DIR, name + '.raw'), rmap[name])
    rgb = packed_to_rgb_linear(short)
    wb = gray_world_wb(rgb)
    return develop_rgb(rgb, wb)


# ---------------- 方法注册 ----------------
METHODS = {}
_MODEL_CACHE = {}
_OURS_CKPT = None
_SID_CKPT = 'checkpoints/best.pth'  # SID 同架构 RAW->sRGB(JPG 目标) 模型


def register(name):
    def dec(fn):
        METHODS[name] = fn
        return fn
    return dec


def _load_devraw_input(name, rmap):
    """dev-RAW 口径输入：白平衡后的短曝光线性 packed Bayer（与训练 _load_dev_raw_pair 一致）。"""
    short = load_raw_linear(os.path.join(RAW_DIR, name + '.raw'), rmap[name])
    long = load_raw_linear(os.path.join(GT_RAW_DIR, name + '.raw'), 1.0)
    wb = gray_world_wb(packed_to_rgb_linear(long))
    wb4 = np.array([wb[0], wb[1], wb[1], wb[2]], np.float32)
    return np.clip(short * wb4[None, None, :], 0, 1).astype(np.float32)


def _run_unet(ckpt, name, rmap, device, tta=False):
    """加载指定 checkpoint 的 4 通道 UNet 并前向推理 -> rgb[H,W,3] [0,1]。"""
    if ckpt not in _MODEL_CACHE:
        raw = torch.load(ckpt, map_location=device, weights_only=False)
        m = UNet(in_ch=4, out_ch=3, base_ch=raw.get('base_ch', 32),
                 bilinear=raw.get('bilinear', False),
                 use_se=raw.get('se', False)).to(device).eval()
        m.load_state_dict(raw['model'])
        _MODEL_CACHE[ckpt] = (m, raw.get('gt_mode', 'aligned'))
    m, gt_mode = _MODEL_CACHE[ckpt]
    if gt_mode == 'dev_raw':
        inp = _load_devraw_input(name, rmap)
    else:
        inp = load_raw(os.path.join(RAW_DIR, name + '.raw'), ratio=rmap[name])
    t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = m(t) if not tta else _tta_forward(m, t)
    return np.clip(out.squeeze(0).cpu().numpy().transpose(1, 2, 0), 0, 1).astype(np.float32)


def _tta_forward(model, x):
    """8 折 D4 测试时自集成（TTA）。x: [1,4,H,W]，变换与训练增强一致
    （fliplr/flipud/rot90，作用于已打包 Bayer 表示）。"""
    preds = [model(x)]
    for k in (1, 2, 3):
        preds.append(torch.rot90(model(torch.rot90(x, k, (2, 3))), 4 - k, (2, 3)))
    preds.append(torch.flip(model(torch.flip(x, (3,))), (3,)))   # 水平翻转
    preds.append(torch.flip(model(torch.flip(x, (2,))), (2,)))   # 垂直翻转
    # 翻转 + 旋转组合（逆变换 = 先反向旋转再反向翻转）
    xh = torch.flip(x, (3,))
    preds.append(torch.flip(torch.rot90(model(torch.rot90(xh, 1, (2, 3))), 3, (2, 3)), (3,)))
    xv = torch.flip(x, (2,))
    preds.append(torch.flip(torch.rot90(model(torch.rot90(xv, 1, (2, 3))), 3, (2, 3)), (2,)))
    return torch.stack(preds).mean(0)


@register('naive')
def method_naive(name, rmap, device):
    return dev_short_rgb(name, rmap).astype(np.float32)


@register('ours')
def method_ours(name, rmap, device):
    return _run_unet(_OURS_CKPT, name, rmap, device)


@register('ours_tta')
def method_ours_tta(name, rmap, device):
    return _run_unet(_OURS_CKPT, name, rmap, device, tta=True)


@register('sid')
def method_sid(name, rmap, device):
    return _run_unet(_SID_CKPT, name, rmap, device)


# ---------------- 传统非学习基线 ----------------
def _lime_illumination(T_hat, alpha=0.15, rho=2.0, iters=10):
    """LIME 光照图细化（加权 L1 TV，ADMM）。T_hat: [H,W] float [0,1]。"""
    H, W = T_hat.shape
    T_hat = T_hat.astype(np.float64)
    fx = np.fft.fftfreq(W).reshape(1, W).astype(np.float64)
    fy = np.fft.fftfreq(H).reshape(H, 1).astype(np.float64)
    otf_dx = np.exp(2j * np.pi * fx) - 1.0
    otf_dy = np.exp(2j * np.pi * fy) - 1.0
    denom = alpha + rho * (np.abs(otf_dx) ** 2 + np.abs(otf_dy) ** 2)
    F_TH = np.fft.fft2(T_hat)

    eps = 1e-6
    dx = np.roll(T_hat, -1, axis=1) - T_hat
    dy = np.roll(T_hat, -1, axis=0) - T_hat
    Wx = 1.0 / (np.abs(dx) + eps)
    Wy = 1.0 / (np.abs(dy) + eps)

    T = T_hat.copy()
    Gx = dx.copy(); Gy = dy.copy()
    Yx = np.zeros_like(T); Yy = np.zeros_like(T)  # 已缩放对偶变量

    for _ in range(iters):
        rhs = alpha * F_TH + rho * (np.conj(otf_dx) * np.fft.fft2(Gx - Yx)
                                    + np.conj(otf_dy) * np.fft.fft2(Gy - Yy))
        T = np.real(np.fft.ifft2(rhs / denom))
        gx = (np.roll(T, -1, axis=1) - T) + Yx
        gy = (np.roll(T, -1, axis=0) - T) + Yy
        Gx = np.sign(gx) * np.maximum(np.abs(gx) - Wx / rho, 0.0)
        Gy = np.sign(gy) * np.maximum(np.abs(gy) - Wy / rho, 0.0)
        Yx = Yx + (np.roll(T, -1, axis=1) - T) - Gx
        Yy = Yy + (np.roll(T, -1, axis=0) - T) - Gy
    return np.clip(T, 0.0, 1.0).astype(np.float32)


@register('lime')
def method_lime(name, rmap, device):
    img = dev_short_rgb(name, rmap).astype(np.float64)  # 低光 RGB [0,1]
    T = _lime_illumination(img.max(axis=2))
    T = np.maximum(T, 1e-4) ** 0.7  # 光照 gamma（LIME 默认 0.7）
    R = img / T[..., None]
    return np.clip(R, 0.0, 1.0).astype(np.float32)


@register('hist_eq')
def method_hist_eq(name, rmap, device):
    import cv2
    img = dev_short_rgb(name, rmap)
    hsv = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return out


# ---------------- 指标 ----------------
def ssim_full(a, b):
    return float(ssim_fn(a, b, channel_axis=2, data_range=1.0))


def ssim_masked(a, b, mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m3 = np.repeat(mask[y0:y1, x0:x1, None], 3, 2).astype(np.float32)
    return float(ssim_fn(a[y0:y1, x0:x1] * m3, b[y0:y1, x0:x1] * m3,
                         channel_axis=2, data_range=1.0))


def lpips_score(a, b):
    """a,b: [H,W,3] float [0,1] -> LPIPS(alex)。"""
    t = lambda x: torch.from_numpy((2.0 * x - 1.0).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(DEV)
    with torch.no_grad():
        return float(get_lpips()(t(a), t(b)).item())


def eval_one(pred, ref, mask):
    """单一参考口径：返回 {psnr, ssim, lpips}。mask 为 None 表示全帧。"""
    if mask is None:
        mse = ((pred - ref) ** 2).mean()
        psnr = 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)
        ss = ssim_full(pred, ref)
        lp = lpips_score(pred, ref)
    else:
        m3 = np.repeat(mask[:, :, None], 3, 2).astype(bool)
        psnr = float(psnr_masked(pred, ref, m3))
        ss = ssim_masked(pred, ref, mask)
        lp = lpips_score(pred * m3, ref * m3)
    return dict(psnr=psnr, ssim=ss, lpips=lp)


def save_visual(name, dev_gt, preds, out_dir):
    """保存 各方法|dev-RAW参考 逐帧横向对比图（论文图1/图4 取材）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    keys = list(preds.keys())
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(3.6 * (len(keys) + 1), 3.4))
    for ax, k in zip(axes[:-1], keys):
        ax.imshow(np.clip(preds[k], 0, 1))
        ax.set_title(k, fontsize=10)
        ax.axis('off')
    axes[-1].imshow(np.clip(dev_gt, 0, 1))
    axes[-1].set_title('dev-RAW reference', fontsize=10)
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name}_compare.png'), dpi=150)
    plt.close(fig)


def evaluate(methods, names, rmap, checkpoint, out_path, save_dir=None):
    global _OURS_CKPT
    _OURS_CKPT = checkpoint

    methods = [m for m in methods if m in METHODS]
    results = {d: {m: dict(psnr=[], ssim=[], lpips=[]) for m in methods}
               for d in ('srgb', 'devraw')}

    for name in names:
        gt_jpg, mask = load_aligned_gt(name, './dataset/gt_aligned', './dataset/gt')
        dev_gt = dev_long_rgb(name, rmap)
        preds = {}
        for m in methods:
            pred = METHODS[m](name, rmap, DEV)
            preds[m] = pred
            for domain, (ref, msk) in dict(srgb=(gt_jpg, mask),
                                           devraw=(dev_gt, None)).items():
                r = eval_one(pred, ref, msk)
                for k in r:
                    results[domain][m][k].append(r[k])
        if save_dir:
            save_visual(name, dev_gt, preds, save_dir)

    # 汇总
    table = {}
    for domain in ('srgb', 'devraw'):
        table[domain] = {}
        for m in methods:
            agg = {}
            for k in ('psnr', 'ssim', 'lpips'):
                v = np.array(results[domain][m][k])
                agg[k] = dict(mean=float(v.mean()), std=float(v.std()))
            table[domain][m] = agg
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(table, open(out_path, 'w'), indent=1)
    print(json.dumps(table, indent=1))
    return table


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dataset/challenge_split.json')
    ap.add_argument('--checkpoint', default='checkpoints/ab_rawgt_100/best.pth')
    ap.add_argument('--sid-checkpoint', default='checkpoints/best.pth')
    ap.add_argument('--methods', default='naive,ours')
    ap.add_argument('--out', default='results/benchmark.json')
    ap.add_argument('--raw-dir', default='./dataset/0.01',
                    help='短曝光输入目录（跨照度评估时切到 0.001/0.0001）')
    ap.add_argument('--ratio-cache', default='.ratio_cache.npz',
                    help='ratio 缓存文件名')
    ap.add_argument('--save-visuals', action='store_true',
                    help='保存逐帧对比图到 results/visuals（论文图1/图4 取材）')
    args = ap.parse_args()
    _SID_CKPT = args.sid_checkpoint
    RAW_DIR = args.raw_dir
    split = json.load(open(args.split))
    names = split['all']
    rmap = compute_ratio_map(args.raw_dir, './dataset/gt_raw',
                             cache_name=args.ratio_cache)
    save_dir = 'results/visuals' if args.save_visuals else None
    evaluate(args.methods.split(','), names, rmap, args.checkpoint, args.out,
             save_dir=save_dir)
    print(f"test set: {len(names)} frames; methods: {args.methods}; raw_dir: {args.raw_dir}")
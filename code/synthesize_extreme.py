# -*- coding: utf-8 -*-
"""从 SID (Sony) 0.01× 子集合成 0.001 lx / 0.0001 lx 极低照度 RAW。

物理模型（信号相关噪声，Poisson-Gaussian，单位 DN = 去黑电平后的数字值）:
  y_s = C_DN / s + sqrt((C_DN / s) * a) * eps1 + b * eps2,   eps~N(0,1)

  C_DN  = 长曝光 RAW 去黑电平后的干净信号（长曝光 SNR 极高，近似无噪）
  s     = 曝光尺度：0.01×=100, 0.001 lx=1000, 0.0001 lx=10000
  a     = 散粒噪声系数（每单位 DN 信号的方差增量，等价于 DN/电子 增益，传感器常量）
  b     = 读出噪声标准差（DN，传感器常量，与照度无关）

a、b 为传感器常量，从真实 0.01× 数据估计后可直接外推到更低照度。
生成文件仍保留黑电平 512、uint16 存储，与 dataset/0.01 格式一致；
后续 compute_ratio_map 会自动对 0.001/0.0001 目录估计 ratio≈1000/10000。

用法:
  python synthesize_extreme.py --estimate-only              # 仅估计 a,b
  python synthesize_extreme.py --levels 0.001,0.0001        # 生成（全 1680 对）
"""
import os
import argparse
import numpy as np
from dataset_1 import RAW_H, RAW_W, RAW_CH, RAW_DTYPE, BLACK_LEVEL

SRC_SHORT_DIR = './dataset/0.01'
SRC_LONG_DIR = './dataset/gt_raw'
OUT_DIRS = {'0.001': './dataset/0.001', '0.0001': './dataset/0.0001'}
SCALES = {'0.001': 1000.0, '0.0001': 10000.0}
NOMINAL_S0 = 100.0  # 0.01× 名义曝光尺度


def _read_dn(path):
    """读 uint16 RAW，减去黑电平，返回 float32 [H,W,4]（去黑电平 DN，可为负）。"""
    d = np.fromfile(path, dtype=RAW_DTYPE).reshape(RAW_H, RAW_W, RAW_CH).astype(np.float32)
    return d - BLACK_LEVEL


def _list_names():
    import glob
    names = sorted(os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(SRC_SHORT_DIR, '*.raw')))
    return names


def estimate_noise_params(sample_pairs=80, seed=42):
    """从真实 0.01×/长曝光对估计 a（散粒噪声系数）、b（读出噪声 std，DN）。

    - b：长曝光近黑像素（纯读出噪声，信号≈0）上短曝光 DN 的稳健标准差。
    - a：平滑区域（局部梯度小，避免短/长曝光对不齐带来的伪信号相关方差）拟合
      var(resid) = a * mu + b^2。
    """
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        uniform_filter = None
    rng = np.random.default_rng(seed)
    names = _list_names()
    idx = rng.choice(len(names), size=min(sample_pairs, len(names)), replace=False)

    dark_vars, mus, resids = [], [], []
    for i in idx:
        name = names[i]
        C = _read_dn(os.path.join(SRC_LONG_DIR, name + '.raw'))
        y = _read_dn(os.path.join(SRC_SHORT_DIR, name + '.raw'))
        # 读出噪声：长曝光近黑像素上方差
        dark = C < 3.0
        if dark.sum() > 50:
            dark_vars.append(y[dark].var())
        # 平滑区：5x5 均值滤波残差小 && 中亮信号
        mu = C / NOMINAL_S0
        resid = y - mu
        if uniform_filter is not None:
            Lm = uniform_filter(C.astype(np.float64), size=5, mode='nearest')
            smooth = np.abs(C - Lm) / (np.abs(Lm) + 1.0) < 0.15
        else:
            smooth = np.ones(C.shape, bool)
        sel = smooth & (C >= 20) & (C <= 200)
        if sel.sum() > 2000:
            mus.append(mu[sel].ravel())
            resids.append(resid[sel].ravel())
    b = float(np.sqrt(np.median(np.array(dark_vars)) if dark_vars else 0.0))

    mus = np.concatenate(mus).astype(np.float64)
    resids = np.concatenate(resids).astype(np.float64)
    edges = np.linspace(0.2, 5.0, 25)
    centers = 0.5 * (edges[:-1] + edges[1:])
    var, cnt = np.zeros_like(centers), np.zeros_like(centers)
    for j in range(len(centers)):
        m = (mus >= edges[j]) & (mus < edges[j + 1])
        cnt[j] = m.sum()
        if m.sum() > 300:
            var[j] = resids[m].var()
    valid = var > 0
    A = np.vstack([centers[valid], np.ones(valid.sum())]).T
    slope, intercept = np.linalg.lstsq(A, var[valid], rcond=None)[0]
    a = float(max(slope, 0.0))
    if intercept > 0:
        b = float(np.sqrt(intercept))
    print(f"[噪声估计] 散粒噪声系数 a={a:.4f} DN, 读出噪声 b={b:.4f} DN "
          f"(用 {sample_pairs} 对，暗区像素 {int(np.sum(cnt))}）")
    return a, b


def synthesize(name, C, s, a, b, rng):
    mu = C / s
    noise = np.sqrt(np.maximum(mu, 0.0) * a) * rng.standard_normal(C.shape) \
        + b * rng.standard_normal(C.shape)
    y = mu + noise
    return np.clip(np.rint(y + BLACK_LEVEL), 0, 65535).astype(RAW_DTYPE)


def write_ratio_cache(out_dir, scale, names):
    """写 .ratio_cache.npz：合成数据曝光比为已知精确值 scale，避免 auto 估计
    在极低照度下被黑电平裁剪偏置污染（实测 0.001 会被低估到 ~271 而非 1000）。"""
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, '.ratio_cache.npz'),
             names=np.array(names, dtype=object),
             ratios=np.full(len(names), scale, dtype=np.float32))
    print(f"[{os.path.basename(out_dir)}] 已写 .ratio_cache.npz (ratio={scale:.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', default='0.001,0.0001')
    ap.add_argument('--sample-pairs', type=int, default=80)
    ap.add_argument('--a', type=float, default=None, help='手动指定散粒噪声系数，否则自动估计')
    ap.add_argument('--b', type=float, default=None, help='手动指定读出噪声 std，否则自动估计')
    ap.add_argument('--estimate-only', action='store_true')
    ap.add_argument('--subset', type=int, default=None, help='仅生成前 N 对（调试用）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    levels = [l.strip() for l in args.levels.split(',')]
    if args.a is None or args.b is None:
        a, b = estimate_noise_params(sample_pairs=args.sample_pairs, seed=args.seed)
        if args.a is not None:
            a = args.a
        if args.b is not None:
            b = args.b
    else:
        a, b = args.a, args.b
    if args.estimate_only:
        return

    rng = np.random.default_rng(args.seed)
    names = _list_names()
    if args.subset:
        names = names[:args.subset]
    for lv in levels:
        assert lv in SCALES, f"未知照度档 {lv}"
        out_dir = OUT_DIRS[lv]
        os.makedirs(out_dir, exist_ok=True)
        s = SCALES[lv]
        for n, name in enumerate(names):
            C = _read_dn(os.path.join(SRC_LONG_DIR, name + '.raw'))
            out = synthesize(name, C, s, a, b, rng)
            out.tofile(os.path.join(out_dir, name + '.raw'))
            if (n + 1) % 200 == 0:
                print(f"[{lv}] {n+1}/{len(names)} 生成完成")
        write_ratio_cache(out_dir, s, names)
        print(f"[{lv}] 全部 {len(names)} 对已写入 {out_dir}")
    print(f"完成：a={a:.6f}, b={b:.4f}，档位 {levels}")


if __name__ == '__main__':
    main()
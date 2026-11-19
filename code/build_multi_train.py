# -*- coding: utf-8 -*-
"""构建三级照度联合训练目录（软链接，不复制大文件）。

dataset/multi_train/{name}__x1.raw   -> dataset/0.01/{name}.raw   (0.01 lx 档)
dataset/multi_train/{name}__x10.raw  -> dataset/0.001/{name}.raw  (0.001 lx 档)
dataset/multi_train/{name}__x100.raw -> dataset/0.0001/{name}.raw (0.0001 lx 档)

同时软链接 gt_raw 与 gt jpg，并写 multi_train/.ratio_cache.npz。
注意：真实数据的曝光比并非 100/1000/10000 十倍递进（实测 0.001≈1.5×、
0.0001≈3× 于 0.01 的暗度），因此每一档的 ratio 都必须按 gt_raw 逐样本实测，
不能再用合成时的固定 1000/10000。
"""
import os
import glob
import numpy as np
from dataset_1 import compute_ratio_map

BASE = os.path.dirname(os.path.abspath(__file__))
MT = os.path.join(BASE, 'dataset', 'multi_train')
MG = os.path.join(BASE, 'dataset', 'multi_gt_raw')
MJ = os.path.join(BASE, 'dataset', 'multi_gt')

LEVELS = [('x1', '0.01'), ('x10', '0.001'), ('x100', '0.0001')]


def main():
    names = sorted(os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(BASE, 'dataset', 'gt_raw', '*.raw')))

    # 每档分别按 gt_raw 实测逐样本曝光比（生成/读取各自的 .ratio_cache.npz）
    maps = {}
    for suffix, srcdir in LEVELS:
        maps[suffix] = compute_ratio_map(
            os.path.join(BASE, 'dataset', srcdir),
            os.path.join(BASE, 'dataset', 'gt_raw'),
            cache_name='.ratio_cache.npz', robust=False)

    for d in (MT, MG, MJ):
        os.makedirs(d, exist_ok=True)

    cache_names, cache_ratios = [], []
    for name in names:
        for suffix, srcdir in LEVELS:
            new = f'{name}__{suffix}'
            os.symlink(os.path.join(BASE, 'dataset', srcdir, name + '.raw'),
                       os.path.join(MT, new + '.raw'))
            os.symlink(os.path.join(BASE, 'dataset', 'gt_raw', name + '.raw'),
                       os.path.join(MG, new + '.raw'))
            os.symlink(os.path.join(BASE, 'dataset', 'gt', name + '.jpg'),
                       os.path.join(MJ, new + '.jpg'))
            cache_names.append(new)
            cache_ratios.append(float(maps[suffix].get(name, 100.0)))

    cache_ratios = np.array(cache_ratios, dtype=np.float32)
    np.savez(os.path.join(MT, '.ratio_cache.npz'),
             names=np.array(cache_names, dtype=object),
             ratios=cache_ratios)
    print(f'完成：{len(cache_names)} 个训练样本（{len(names)} 场景 × 3 档）')
    for suffix, _ in LEVELS:
        sub = cache_ratios[[cache_names[i].endswith(suffix) for i in range(len(cache_names))]]
        print(f'  {suffix}: median={np.median(sub):.1f} 范围 {sub.min():.1f}~{sub.max():.1f}')


if __name__ == '__main__':
    main()
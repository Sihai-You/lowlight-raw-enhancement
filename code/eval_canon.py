# -*- coding: utf-8 -*-
"""Canon 跨相机泛化评估（dev-RAW 口径，仅 devraw 域）。

在 SIED Canon 0.01 官方 180 场景测试集上，用 Sony 训练的 SE 模型评估。
复用 benchmark.py 的 devraw 管线，将 RAW_DIR/GT_RAW_DIR 切到 canon 子集。
输出 PSNR/SSIM/LPIPS (全帧) 的 naive 与 ours 结果。

用法:
  python eval_canon.py \
      --checkpoint ./checkpoints/abl_devraw_multi_se_clean/best.pth \
      --raw-dir ./dataset/canon/0.01 --gt-raw-dir ./dataset/canon/gt_raw \
      --split ./dataset/test_canon_0.01.txt --out ./results/benchmark_canon_cross.json
"""
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import benchmark as B

DEV = 'cuda' if B.torch.cuda.is_available() else 'cpu'


def ids_from_split(path):
    out = []
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        p = l.split()[0]
        out.append(os.path.splitext(os.path.basename(p))[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',
                    default='./checkpoints/abl_devraw_multi_se_clean/best.pth')
    ap.add_argument('--raw-dir', default='./dataset/canon/0.01')
    ap.add_argument('--gt-raw-dir', default='./dataset/canon/gt_raw')
    ap.add_argument('--split', default='./dataset/test_canon_0.01.txt')
    ap.add_argument('--out', default='./results/benchmark_canon_cross.json')
    args = ap.parse_args()

    B.RAW_DIR = args.raw_dir
    B.GT_RAW_DIR = args.gt_raw_dir

    ids = ids_from_split(args.split)
    print('n_ids =', len(ids), 'first =', ids[:3], flush=True)

    rmap = B.compute_ratio_map(args.raw_dir, args.gt_raw_dir,
                               cache_name='.ratio_cache_canon.npz')
    print('ratio map done, n =', len(rmap), flush=True)

    results = {m: dict(psnr=[], ssim=[], lpips=[]) for m in ('naive', 'ours')}
    for i, name in enumerate(ids):
        ref = B.dev_long_rgb(name, rmap)
        preds = {
            'naive': B.dev_short_rgb(name, rmap),
            'ours': B._run_unet(args.checkpoint, name, rmap, DEV),
        }
        for m, pred in preds.items():
            r = B.eval_one(pred, ref, None)
            for k in r:
                results[m][k].append(r[k])
        if (i + 1) % 20 == 0:
            print(f'progress {i+1}/{len(ids)}', flush=True)

    agg = {m: {k: dict(mean=float(np.array(v).mean()), std=float(np.array(v).std()))
               for k, v in d.items()} for m, d in results.items()}
    print(json.dumps(agg, indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(agg, open(args.out, 'w'), indent=1)
    print('saved to', args.out)


if __name__ == '__main__':
    main()
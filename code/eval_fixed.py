# -*- coding: utf-8 -*-
"""固定挑战集全帧评估（v8 调试度量基建）。

指标（全部在配准有效区内）：
  PSNR（掩码内）、SSIM（掩码外接框，框内外双方置零）、
  pred/gt 亮度比、饱和度比、平均 RGB 偏置。
产物：每帧三联图（pred | gt | 3x 残差热图）+ metrics.json + 汇总打印。

用法:
  python eval_fixed.py --checkpoint checkpoints/best.pth --tag v7base
"""
import os, sys, json, argparse
import numpy as np
import torch
import cv2
from PIL import Image
from model import UNet
from dataset_1 import load_raw, compute_ratio_map, RAW_H, RAW_W
from test import load_aligned_gt, psnr_masked
from skimage.metrics import structural_similarity as ssim_fn


def sat_mean(rgb):
    mx = rgb.max(1); mn = rgb.min(1)
    return float(((mx - mn) / np.maximum(mx, 1e-6)).mean())


def evaluate(ckpt_path, split_path, out_dir, use_robust_ratio=False):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = UNet(in_ch=4, out_ch=3, base_ch=ck.get('base_ch', 32),
                 bilinear=ck.get('bilinear', True)).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    cache_name = '.ratio_cache_robust.npz' if use_robust_ratio else '.ratio_cache.npz'
    rmap = compute_ratio_map('./dataset/0.01', './dataset/gt_raw', cache_name=cache_name)

    split = json.load(open(split_path))
    names = split['all']
    os.makedirs(out_dir, exist_ok=True)
    TW, TH = 560, 315
    montage = np.zeros((TH * len(names), TW * 3, 3), np.uint8)
    rows = []
    with torch.no_grad():
        for k, name in enumerate(names):
            inp = load_raw(f'./dataset/0.01/{name}.raw', ratio=rmap[name])
            t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            out = np.clip(model(t).squeeze(0).cpu().numpy().transpose(1, 2, 0), 0, 1)
            gt, mask = load_aligned_gt(name, './dataset/gt_aligned', './dataset/gt')
            m3 = np.repeat(mask[:, :, None], 3, 2).astype(bool)
            psnr = float(psnr_masked(out, gt, m3))
            ys, xs = np.where(mask)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            mc = m3[y0:y1, x0:x1]
            ss = float(ssim_fn(gt[y0:y1, x0:x1] * mc, out[y0:y1, x0:x1] * mc,
                               channel_axis=2, data_range=1.0))
            go, oo = gt[y0:y1, x0:x1][mask[y0:y1, x0:x1]], out[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
            l_gt, l_oo = float(go.mean()), float(oo.mean())
            luma_ratio = l_oo / max(l_gt, 1e-6)
            sat_ratio = sat_mean(oo) / max(sat_mean(go), 1e-6)
            bias = (oo.mean(0) - go.mean(0))
            rows.append(dict(name=name, psnr=psnr, ssim=ss, luma_ratio=luma_ratio,
                             sat_ratio=sat_ratio, bias_r=float(bias[0]),
                             bias_g=float(bias[1]), bias_b=float(bias[2])))

            def tile(x):
                return cv2.resize((np.clip(x, 0, 1) * 255).astype(np.uint8),
                                  (TW, TH), interpolation=cv2.INTER_AREA)
            err = np.clip(np.abs(out - gt) * 3, 0, 1)
            heat = cv2.applyColorMap((tile(err).mean(2)).astype(np.uint8),
                                     cv2.COLORMAP_JET)[..., ::1]
            for c, im in enumerate([np.where(m3, out, 0), np.where(m3, gt, 0)]):
                montage[k*TH:(k+1)*TH, c*TW:(c+1)*TW] = tile(im)
            montage[k*TH:(k+1)*TH, 2*TW:3*TW] = heat
            cv2.rectangle(montage[k*TH:(k+1)*TH, 0:TW], (0, 0), (560, 22), (0, 0, 0), -1)
            cv2.putText(montage[k*TH:(k+1)*TH, 0:TW],
                        f"{name} p={psnr:.1f} sat={sat_ratio:.2f} lum={luma_ratio:.2f}",
                        (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(out_dir, 'montage.png'),
                cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    json.dump(rows, open(os.path.join(out_dir, 'metrics.json'), 'w'), indent=1)

    ps = np.array([r['psnr'] for r in rows])
    sr = np.array([r['sat_ratio'] for r in rows])
    lr = np.array([r['luma_ratio'] for r in rows])
    ss = np.array([r['ssim'] for r in rows])
    summary = dict(n=len(rows), psnr_mean=float(ps.mean()), psnr_p10=float(np.percentile(ps, 10)),
                   psnr_min=float(ps.min()), ssim_mean=float(ss.mean()),
                   sat_ratio_median=float(np.median(sr)), sat_ratio_p10=float(np.percentile(sr, 10)),
                   sat_ratio_p90=float(np.percentile(sr, 90)),
                   luma_p10=float(np.percentile(lr, 10)), luma_p90=float(np.percentile(lr, 90)))
    json.dump(summary, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/best.pth')
    ap.add_argument('--split', default='dataset/challenge_split.json')
    ap.add_argument('--tag', default='eval')
    ap.add_argument('--robust_ratio', action='store_true')
    args = ap.parse_args()
    evaluate(args.checkpoint, args.split, f'results/{args.tag}', args.robust_ratio)

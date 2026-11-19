"""
测试脚本 - 低光图像增强

功能:
  1. 加载训练好的模型
  2. 对全帧输入暗图做增强
  3. 保存结果图 (输入、输出、配准 GT 三列对比)
  4. 在配准 GT 的有效区域内计算 PSNR / SSIM（GT 已与 RAW packed 平面对齐）
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from model import UNet
from dataset_1 import (load_raw, load_jpg, compute_ratio_map, DEFAULT_RATIO,
                       RAW_H, RAW_W, ALIGNED_DIR)

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./dataset/0.01')
    parser.add_argument('--gt_dir', type=str, default='./dataset/gt')
    parser.add_argument('--aligned_dir', type=str, default=ALIGNED_DIR)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pth')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--ratio', type=float, default=None,
                        help='短曝光放大倍数；默认读取 checkpoint（auto=逐样本估计），可强制指定')
    parser.add_argument('--gt_raw_dir', type=str, default='./dataset/gt_raw')
    return parser.parse_args()

def natural_key(name):
    """按数值自然排序: 1, 2, ..., 10, 100（而非字典序 1,10,100,1000）"""
    digits = ''.join(c if c.isdigit() else ' ' for c in name).split()
    return [int(x) if x.isdigit() else x for x in digits] or [name]

def psnr_masked(pred, target, mask):
    d = (pred.astype(np.float64) - target.astype(np.float64)) ** 2
    mse = d[mask].mean()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)

def load_aligned_gt(name, aligned_dir, gt_dir):
    """返回 (gt[H,W,3], mask[H,W] bool)；无配准文件时回退 1080p 全图。"""
    ap = os.path.join(aligned_dir, name + '.jpg')
    if os.path.exists(ap):
        gt = np.array(Image.open(ap).convert('RGB').resize((RAW_W, RAW_H), Image.BILINEAR),
                      dtype=np.float32) / 255.0
        mp = os.path.join(aligned_dir, name + '.mask.png')
        if os.path.exists(mp):
            mask = np.array(Image.open(mp).resize((RAW_W, RAW_H), Image.NEAREST)) > 127
        else:
            mask = np.ones((RAW_H, RAW_W), bool)
        return gt, mask
    gt = np.array(Image.open(os.path.join(gt_dir, name + '.jpg'))
                  .convert('RGB').resize((RAW_W, RAW_H), Image.BILINEAR),
                  dtype=np.float32) / 255.0
    return gt, np.ones((RAW_H, RAW_W), bool)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型（架构参数从 checkpoint 恢复，兼容旧版 checkpoint）
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_ch = ckpt.get('base_ch', 32)
    bilinear = ckpt.get('bilinear', True)
    model = UNet(in_ch=4, out_ch=3, base_ch=base_ch, bilinear=bilinear).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    ckpt_ratio = ckpt.get('ratio', DEFAULT_RATIO)
    if args.ratio is not None:
        auto_ratio, ratio = False, args.ratio
    elif ckpt_ratio == 'auto':
        auto_ratio, ratio = True, None
    else:
        auto_ratio, ratio = False, float(ckpt_ratio)
    ratio_map = compute_ratio_map(args.input_dir, args.gt_raw_dir) if auto_ratio else None
    print(f"加载模型: {args.checkpoint} (epoch={ckpt['epoch']}, "
          f"val_loss={ckpt['best_val_loss']:.5f}, ratio={'auto' if auto_ratio else ratio})")

    # 获取测试图像列表（先过滤 .raw，再按数值自然排序，最后取前 N 张）
    all_files = sorted(os.listdir(args.input_dir))
    names = [os.path.splitext(n)[0] for n in all_files if n.endswith('.raw')]
    names.sort(key=natural_key)
    names = names[:args.num_images]

    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for name in names:
            inp_path = os.path.join(args.input_dir, name + '.raw')
            if not os.path.exists(os.path.join(args.gt_dir, name + '.jpg')):
                continue

            cur_ratio = ratio_map[name] if auto_ratio else ratio
            inp = load_raw(inp_path, ratio=cur_ratio)  # (H, W, 4)
            inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
            out = model(inp_t)
            out = np.clip(out.squeeze(0).cpu().numpy().transpose(1, 2, 0), 0, 1)

            gt, mask = load_aligned_gt(name, args.aligned_dir, args.gt_dir)
            m3 = np.repeat(mask[:, :, None], 3, axis=2)

            p = psnr_masked(out, gt, m3)
            psnr_list.append(p)
            msg = f"{name}: PSNR={p:.2f}dB"
            if HAS_SSIM:
                # 仅在配准有效区内计算：取掩码外接框（mask 已腐蚀 4px，
                # 避开 warp 黑边），避免整帧 75% 黑区把 SSIM 拉低
                ys, xs = np.where(mask)
                y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
                mc = m3[y0:y1, x0:x1]
                s = ssim_fn(gt[y0:y1, x0:x1] * mc, out[y0:y1, x0:x1] * mc,
                            channel_axis=2, data_range=1.0)
                ssim_list.append(s)
                msg += f"  SSIM={s:.4f}"
            msg += f"  有效区={mask.mean()*100:.0f}%"
            print(msg)

            # 输入可视化：R / mean(Gr,Gb) / B
            inp_vis = np.clip(np.stack([
                inp[:, :, 0],
                (inp[:, :, 1] + inp[:, :, 2]) / 2.0,
                inp[:, :, 3]], axis=-1), 0, 1)
            gt_vis = gt * m3
            canvas = np.concatenate([inp_vis, out * m3, gt_vis], axis=1)
            canvas = (canvas * 255).astype(np.uint8)
            save_path = os.path.join(args.output_dir, f'{name}.png')
            Image.fromarray(canvas).save(save_path)
            print(f"保存: {save_path}")

    print(f"\n共评估 {len(psnr_list)} 张，平均 PSNR={np.mean(psnr_list):.2f}dB", end='')
    if ssim_list:
        print(f"，平均 SSIM={np.mean(ssim_list):.4f}")
    else:
        print("\n(未安装 scikit-image，跳过 SSIM；可 pip install scikit-image)")
    print(f"完成！结果保存在 {args.output_dir}/")

if __name__ == '__main__':
    main()

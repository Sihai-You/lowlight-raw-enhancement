"""
训练脚本 - 低光图像增强

功能:
  1. 加载配对数据集 (暗图RAW -> 正常光sRGB)
  2. 训练 U-Net 网络（转置卷积上采样，SID 原版配置）
  3. L1 + SSIM 混合损失，Adam + cosine 学习率退火
  4. 每个 epoch 在验证集上评估，保存最佳模型，支持断点续训

注意: GT 已预配准到 RAW packed 平面（1080x1920），与输入同坐标同尺寸，
      损失/指标仅在配准有效区（mask）内计算
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import UNet
from dataset_1 import LowLightDataset, list_paired_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./dataset/0.01')
    parser.add_argument('--gt_dir', type=str, default='./dataset/gt')
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--ssim_weight', type=float, default=0.15,
                        help='混合损失中 1-SSIM 项的权重，0 表示纯 L1')
    parser.add_argument('--color_weight', type=float, default=0.0,
                        help='Lab a/b 色度 L1 损失权重（针对去饱和，推荐 0.02~0.1），0=关闭')
    parser.add_argument('--luma_weight', type=float, default=0.0,
                        help='逐帧 RGB 均值匹配损失权重（针对曝光散布，推荐 0.3），0=关闭')
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--se', action='store_true',
                        help='在 DoubleConv 后加 SE 通道注意力')
    parser.add_argument('--bilinear', action='store_true',
                        help='上采样使用双线性插值（默认用转置卷积，容量更大）')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--ratio', type=str, default='auto',
                        help="短曝光放大倍数：'auto'=用 gt_raw 逐样本估计(推荐)，或传数值如 100")
    parser.add_argument('--gt_raw_dir', type=str, default='./dataset/gt_raw')
    parser.add_argument('--gt_mode', type=str, default='aligned',
                        choices=['aligned', 'dev_raw', 'srgb_full'],
                        help='GT 来源：aligned=预配准JPG；dev_raw=显影长曝光RAW(纯去噪)；'
                             'srgb_full=4K全分辨率对齐JPG(对齐论文口径)')
    parser.add_argument('--gt_dev_raw_dir', type=str, default='./dataset/gt_raw')
    parser.add_argument('--valid_frac', type=float, default=1.0,
                        help='训练裁剪窗口最小有效像素占比（旋转小有效区可用 0.95）')
    parser.add_argument('--robust_ratio', action='store_true',
                        help='使用 10-80 百分位鲁棒曝光比缓存 .ratio_cache_robust.npz')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='梯度累积步数（batch_size 为微批，等效批=batch_size×accum）')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练（省显存）')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()

def seed_everything(seed):
    """固定随机种子，保证数据划分/裁剪/增强可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_dataset(names, val_ratio, seed):
    """按比例划分训练集和验证集（固定种子，划分可复现）"""
    n = len(names)
    n_val = max(1, int(n * val_ratio))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    val_idx = set(indices[:n_val].tolist())
    train_names = [names[i] for i in range(n) if i not in val_idx]
    val_names = [names[i] for i in val_idx]
    return train_names, val_names

def masked_l1(pred, target, mask):
    """按有效区域掩码归一化的 L1（mask: [B,1,H,W]）。"""
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() * pred.shape[1] + 1e-8)


# sRGB(D65)->Lab 色度转换矩阵（仅在首次使用时按设备/精度缓存）
_SRGB_MAT = None


def lab_ab(img):
    """sRGB [B,3,H,W] in [0,1] -> 归一化 Lab 色度 (a,b) in ~[-0.8,0.8]，可微。"""
    global _SRGB_MAT
    if _SRGB_MAT is None or _SRGB_MAT.device != img.device or _SRGB_MAT.dtype != img.dtype:
        _SRGB_MAT = torch.tensor(
            [[0.4124564, 0.3575761, 0.1804375],
             [0.2126729, 0.7151522, 0.0721750],
             [0.0193339, 0.1191920, 0.9503041]],
            device=img.device, dtype=img.dtype)
    lin = torch.where(img <= 0.04045, img / 12.92,
                      ((img.clamp_min(0) + 0.055) / 1.055).pow(2.4))
    xyz = torch.einsum('ij,bjhw->bihw', _SRGB_MAT, lin)
    xyz = xyz / xyz.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    eps = 216.0 / 24389.0
    f = torch.where(xyz > eps, xyz.clamp_min(0).pow(1.0 / 3.0),
                    ((24389.0 / 27.0) * xyz + 16.0) / 116.0)
    a = (500.0 * (f[:, 0] - f[:, 1]) / 128.0).unsqueeze(1)
    b = (200.0 * (f[:, 1] - f[:, 2]) / 128.0).unsqueeze(1)
    return a, b


def chroma_loss(pred, target, mask):
    """有效区内 Lab a/b 通道的掩码 L1（缓解 L1 回归导致的去饱和）。"""
    pa, pb = lab_ab(pred)
    ta, tb = lab_ab(target)
    d = torch.abs(pa - ta) + torch.abs(pb - tb)
    return (d * mask).sum() / (mask.sum() * 2 + 1e-8)


def exposure_loss(pred, target, mask):
    """逐帧 RGB 均值匹配损失（有效区内）：压缩曝光/色偏的逐帧散布。

    诊断依据：pred/gt 亮度比散布 0.72~1.6，过曝帧 RGB 削顶导致去饱和
    （曝光准确帧饱和度比 1.31，严重过曝帧仅 0.39）。
    pred/target: [B,3,H,W]，mask: [B,1,H,W]
    """
    cnt = mask.sum(dim=(2, 3)).clamp_min(1.0)              # [B,1]
    po = (pred * mask).sum(dim=(2, 3)) / cnt               # [B,3]
    to = (target * mask).sum(dim=(2, 3)) / cnt
    return torch.abs(po - to).mean()

def batch_psnr(pred, target, mask, max_val=1.0):
    """按有效区域掩码计算每个样本 PSNR (dB) 后平均。"""
    diff2 = (pred - target) ** 2
    cnt = mask.sum(dim=(1, 2, 3)) * pred.shape[1]
    mse = (diff2 * mask).flatten(1).sum(dim=1) / cnt.clamp_min(1.0)
    psnr = torch.where(
        mse > 0,
        10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-10)),
        torch.full_like(mse, 99.0)
    )
    return psnr.mean().item()

def _gaussian_window(size=11, sigma=1.5, channels=3, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win = torch.outer(g, g)
    return win.expand(channels, 1, size, size).contiguous()

def ssim_metric(pred, target, win=None, mask=None):
    """标准 11x11 高斯窗 SSIM，输入/输出均为 [B,3,H,W] 且范围 [0,1]；
    可传 [B,1,H,W] 有效区域掩码。"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    if win is None:
        win = _gaussian_window(device=pred.device)
    pad = win.shape[-1] // 2
    if mask is not None:
        pred = pred * mask
        target = target * mask
    mu1 = F.conv2d(pred, win, padding=pad, groups=3)
    mu2 = F.conv2d(target, win, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, win, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(target * target, win, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred * target, win, padding=pad, groups=3) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if mask is None:
        return ssim_map.mean()
    m3 = mask.expand_as(ssim_map)
    return (ssim_map * m3).sum() / m3.sum().clamp_min(1.0)

def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 数据集（只扫描一次磁盘，再构造训练/验证两个数据集）
    all_names = list_paired_names(args.input_dir, args.gt_dir)
    print(f"共 {len(all_names)} 对有效配对图像")
    train_names, val_names = split_dataset(all_names, args.val_ratio, args.seed)

    ratio_kw = {} if args.ratio == 'auto' else {'ratio': float(args.ratio)}
    train_dataset = LowLightDataset(args.input_dir, args.gt_dir, args.patch_size,
                                    augment=True, gt_raw_dir=args.gt_raw_dir,
                                    valid_frac=args.valid_frac,
                                    robust_ratio=args.robust_ratio,
                                    ratio_cache='.ratio_cache_robust.npz'
                                    if args.robust_ratio else None,
                                    gt_mode=args.gt_mode,
                                    gt_dev_raw_dir=args.gt_dev_raw_dir,
                                    **ratio_kw)
    train_dataset.names = train_names
    val_dataset = LowLightDataset(args.input_dir, args.gt_dir, args.patch_size,
                                  augment=False, gt_raw_dir=args.gt_raw_dir,
                                  robust_ratio=args.robust_ratio,
                                  ratio_cache='.ratio_cache_robust.npz'
                                  if args.robust_ratio else None,
                                  gt_mode=args.gt_mode,
                                  gt_dev_raw_dir=args.gt_dev_raw_dir,
                                  **ratio_kw)
    val_dataset.names = val_names
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=max(1, args.num_workers // 2))

    # 模型（默认转置卷积上采样，容量与 SID 原版一致）
    full_res = args.gt_mode == 'srgb_full'
    model = UNet(in_ch=4, out_ch=3, base_ch=args.base_ch,
                 bilinear=args.bilinear, full_res=full_res, use_se=args.se).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"模型参数量: {n_params:.2f}M (base_ch={args.base_ch}, "
          f"上采样={'bilinear' if args.bilinear else 'ConvTranspose2d'}, SE={'开' if args.se else '关'})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')
    use_amp = args.amp and device.type == 'cuda'
    print(f"等效批大小: {args.batch_size * args.accum_steps} "
          f"(微批 {args.batch_size} × 累积 {args.accum_steps})，AMP={'开' if use_amp else '关'}")
    l1_criterion = nn.L1Loss()
    ssim_window = _gaussian_window(device=device)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"从 epoch {start_epoch} 恢复训练")

    def save_ckpt(path, epoch, val_loss):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best_val_loss': val_loss,
                    'ratio': 'auto' if args.ratio == 'auto' else float(args.ratio),
                    'robust_ratio': args.robust_ratio,
                    'base_ch': args.base_ch, 'bilinear': args.bilinear,
                    'se': args.se, 'gt_mode': args.gt_mode}, path)

    def calc_loss(out, gt, mask):
        """返回 (总损失, L1)；色度/曝光项按开关叠加。"""
        l1 = masked_l1(out, gt, mask)
        loss = l1 + args.ssim_weight * (1.0 - ssim_metric(out, gt, ssim_window, mask))
        if args.color_weight > 0:
            loss = loss + args.color_weight * chroma_loss(out, gt, mask)
        if args.luma_weight > 0:
            loss = loss + args.luma_weight * exposure_loss(out, gt, mask)
        return loss, l1

    # 训练循环（支持 AMP 与梯度累积，微批 batch_size × accum_steps = 等效批）
    accum = args.accum_steps
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = 0
        for i, (inp, gt, mask) in enumerate(train_loader):
            inp, gt, mask = inp.to(device), gt.to(device), mask.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(inp)
                # GT 已与输入同坐标（预配准），仅在有效区域计算损失
                loss, l1 = calc_loss(out.float(), gt, mask)
                loss = loss / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item() * accum
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch+1} [{i+1}/{len(train_loader)}] "
                      f"loss={loss.item() * accum:.5f} (L1={l1.item():.5f})")

        # 验证（中心附近有效区裁剪、无增强，结果可复现）
        model.eval()
        val_loss = 0
        val_l1 = 0
        val_psnr = 0
        with torch.no_grad():
            for inp, gt, mask in val_loader:
                inp, gt, mask = inp.to(device), gt.to(device), mask.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(inp).float()
                v_loss, l1 = calc_loss(out, gt, mask)
                val_l1 += l1.item()
                val_loss += v_loss.item()
                val_psnr += batch_psnr(out, gt, mask)
        val_loss /= len(val_loader)
        val_l1 /= len(val_loader)
        val_psnr /= len(val_loader)
        train_loss /= len(train_loader)
        scheduler.step()
        print(f"Epoch {epoch+1}  训练loss={train_loss:.5f}  "
              f"验证loss={val_loss:.5f}(L1={val_l1:.5f})  "
              f"PSNR={val_psnr:.2f}dB  lr={optimizer.param_groups[0]['lr']:.2e}")

        # 保存
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_ckpt(os.path.join(args.save_dir, 'best.pth'), epoch, best_val_loss)
            print(f"  -> 保存最佳模型 (val_loss={best_val_loss:.5f}, PSNR={val_psnr:.2f}dB)")

        save_ckpt(os.path.join(args.save_dir, 'last.pth'), epoch, best_val_loss)

    print(f"训练完成，最佳验证loss: {best_val_loss:.5f}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Jetson 上实测 UNet base_ch=8/16/32 的推理 FPS（输入 4x1080x1920）。
只依赖 torch，无权重/数据集依赖；推理延迟与权重数值无关。"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        hid = max(ch // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.se = SE(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.conv(x))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, use_se)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, use_se=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, use_se)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, base_ch=32, bilinear=True, full_res=False, use_se=False):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch, use_se)
        self.down1 = Down(base_ch, base_ch * 2, use_se)
        self.down2 = Down(base_ch * 2, base_ch * 4, use_se)
        self.down3 = Down(base_ch * 4, base_ch * 8, use_se)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch * 8, base_ch * 16 // factor, use_se)
        self.up1 = Up(base_ch * 16, base_ch * 8 // factor, bilinear, use_se)
        self.up2 = Up(base_ch * 8, base_ch * 4 // factor, bilinear, use_se)
        self.up3 = Up(base_ch * 4, base_ch * 2 // factor, bilinear, use_se)
        self.up4 = Up(base_ch * 2, base_ch, bilinear, use_se)
        self.full_res = full_res
        out_c = out_ch * 4 if full_res else out_ch
        self.outc = nn.Conv2d(base_ch, out_c, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        if self.full_res:
            out = F.pixel_shuffle(out, 2)
        return torch.sigmoid(out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bench(base_ch, bilinear, H=1080, W=1920, warmup=5, iters=20):
    torch.cuda.empty_cache()
    m = UNet(in_ch=4, out_ch=3, base_ch=base_ch, bilinear=bilinear,
             full_res=False, use_se=False).cuda().eval()
    nparam = count_parameters(m) / 1e6
    x = torch.randn(1, 4, H, W, device='cuda')
    with torch.no_grad():
        for _ in range(warmup):
            y = m(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            y = m(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
    fps = iters / dt
    ms = dt / iters * 1000.0
    print(f"base_ch={base_ch} bilinear={bilinear}: params={nparam:.2f}M | "
          f"{iters} iters / {dt:.3f}s => FPS={fps:.2f}  ({ms:.1f} ms/frame)")
    del m, x, y
    torch.cuda.empty_cache()
    return fps


if __name__ == '__main__':
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    print(f"input = 4 x 1080 x 1920, fp32, GPU={torch.cuda.get_device_name(0)}")
    # 与论文 V100 口径一致：默认 bilinear=True
    for bc in (8, 16, 32):
        try:
            bench(bc, bilinear=True)
        except Exception as e:
            print(f"base_ch={bc} FAILED: {type(e).__name__}: {e}")
    print("DONE_SPEED_BENCH")
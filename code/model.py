"""
U-Net 模型用于 RAW 低光图像增强
输入: 4 通道 Bayer 子采样图 (R, Gr, Gb, B)
输出: 3 通道 sRGB 图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    """Squeeze-and-Excitation 通道注意力（轻量，reduction 控制瓶颈）。"""
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
    """双卷积块（可选 SE 通道注意力）"""
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
        # 处理尺寸差异
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
        # 全分辨率模式：12 通道经 pixel_shuffle(2) 上采样为 2x 分辨率 3 通道 sRGB
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

if __name__ == '__main__':
    model = UNet(in_ch=4, out_ch=3, base_ch=32)
    print(f"参数量: {count_parameters(model) / 1e6:.2f}M")
    x = torch.randn(1, 4, 512, 512)
    y = model(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")

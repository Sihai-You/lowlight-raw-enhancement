"""
SID 数据集加载器
读取 Sony ARW RAW 文件，进行黑电平校正、白平衡、Bayer 子采样
输入: 短曝光(低光) RAW -> 4通道 Bayer (R, Gr, Gb, B)
输出: 长曝光(正常光) RAW 处理后的 sRGB 图像
"""

import os
import glob
import numpy as np
import rawpy
import torch
from torch.utils.data import Dataset
from PIL import Image

def pack_raw_bayer(raw):
    """将 RAW 图像打包为 4 通道 Bayer 子采样 (H/2, W/2, 4)"""
    # 获取 Bayer 模式并打包
    raw_array = raw.raw_image_visible.astype(np.float32)
    
    # 黑电平校正
    black_level = raw.black_level_per_channel
    white_level = raw.white_level
    
    # 按通道分别校正
    H, W = raw_array.shape
    H2, W2 = H // 2, W // 2
    
    # 假设 Bayer 模式为 RGGB (Sony)
    r = raw_array[0:H:2, 0:W:2] - black_level[0]
    gr = raw_array[0:H:2, 1:W:2] - black_level[1]
    gb = raw_array[1:H:2, 0:W:2] - black_level[2]
    b = raw_array[1:H:2, 1:W:2] - black_level[3]
    
    out = np.stack([r, gr, gb, b], axis=-1)
    out = np.maximum(out, 0) / (white_level - max(black_level))
    return out

def apply_white_balance(raw, img_4ch):
    """应用相机白平衡到 4 通道图像"""
    wb = raw.camera_whitebalance
    # wb: [R, G, B, G] 或类似
    img_4ch[:, :, 0] *= wb[0] / wb[1]  # R
    img_4ch[:, :, 3] *= wb[2] / wb[1]  # B
    return img_4ch

def raw_to_srgb(raw):
    """使用 rawpy 默认管线将 RAW 转为 sRGB (用于长曝光参考图)"""
    rgb = raw.postprocess(
        use_camera_wb=True,
        output_bps=8,
        no_auto_bright=True,
        gamma=(1, 1),
        exp_shift=1.0
    )
    return rgb.astype(np.float32) / 255.0

def read_list_file(list_path):
    """读取 SID 列表文件，返回 (短曝光路径, 长曝光路径, iso, fnumber) 列表"""
    pairs = []
    with open(list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                short_path = parts[0]
                long_path = parts[1]
                iso = float(parts[2]) if len(parts) > 2 else 0
                fnum = float(parts[3]) if len(parts) > 3 else 0
                pairs.append((short_path, long_path, iso, fnum))
    return pairs

class SIDDataset(Dataset):
    """SID Sony 数据集"""
    
    def __init__(self, data_dir, list_file, patch_size=512, augment=True):
        self.data_dir = data_dir
        self.pairs = read_list_file(list_file)
        self.patch_size = patch_size
        self.augment = augment
        
        # 过滤掉不存在的文件
        valid = []
        for short_p, long_p, iso, fnum in self.pairs:
            sp = os.path.join(data_dir, short_p)
            lp = os.path.join(data_dir, long_p)
            if os.path.exists(sp) and os.path.exists(lp):
                valid.append((short_p, long_p, iso, fnum))
        self.pairs = valid
        print(f"数据集: {len(self.pairs)} 对有效图像 (来自 {len(self.pairs)} 列表条目)")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        short_path, long_path, iso, fnum = self.pairs[idx]
        short_path = os.path.join(self.data_dir, short_path)
        long_path = os.path.join(self.data_dir, long_path)
        
        # 读取短曝光 RAW (输入)
        with rawpy.imread(short_path) as raw:
            short_4ch = pack_raw_bayer(raw)
            short_4ch = apply_white_balance(raw, short_4ch)
        
        # 读取长曝光 RAW (参考) -> sRGB
        with rawpy.imread(long_path) as raw:
            long_rgb = raw_to_srgb(raw)
        
        # 计算曝光比，放大短曝光
        # 从文件名解析曝光时间, e.g. 10003_00_0.1s.ARW
        short_exp = self._parse_exposure(os.path.basename(short_path))
        long_exp = self._parse_exposure(os.path.basename(long_path))
        ratio = min(long_exp / max(short_exp, 1e-6), 300)
        
        short_4ch = short_4ch * ratio
        short_4ch = np.clip(short_4ch, 0, 1)
        
        # 裁剪 patch
        H, W = short_4ch.shape[:2]
        ps = self.patch_size
        
        if self.augment and H > ps and W > ps:
            h = np.random.randint(0, H - ps)
            w = np.random.randint(0, W - ps)
        else:
            h, w = 0, 0
        
        short_patch = short_4ch[h:h+ps, w:w+ps, :]
        long_patch = long_rgb[h*2:(h+ps)*2, w*2:(w+ps)*2, :]  # 长曝光是全分辨率
        
        # 数据增强
        if self.augment:
            short_patch, long_patch = self._augment(short_patch, long_patch)
        
        # 转 tensor: (H, W, C) -> (C, H, W)
        short_tensor = torch.from_numpy(short_patch.transpose(2, 0, 1)).float()
        long_tensor = torch.from_numpy(long_patch.transpose(2, 0, 1)).float()
        
        return short_tensor, long_tensor
    
    def _parse_exposure(self, filename):
        """从文件名解析曝光时间, e.g. 10003_00_0.1s.ARW -> 0.1"""
        try:
            parts = filename.replace('.ARW', '').split('_')
            for p in parts:
                if p.endswith('s'):
                    return float(p[:-1])
        except:
            pass
        return 1.0
    
    def _augment(self, short, long_img):
        """随机翻转和旋转"""
        if np.random.rand() > 0.5:
            short = short[::-1, :, :].copy()
            long_img = long_img[::-1, :, :].copy()
        if np.random.rand() > 0.5:
            short = short[:, ::-1, :].copy()
            long_img = long_img[:, ::-1, :].copy()
        if np.random.rand() > 0.5:
            short = np.rot90(short, 1).copy()
            long_img = np.rot90(long_img, 1).copy()
        return short, long_img

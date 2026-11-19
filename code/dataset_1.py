"""
数据集加载模块 - SID 低光增强数据集

数据格式:
  - 暗图输入: dataset/0.01/*.raw  -> uint16, Bayer 4通道打包 (1080,1920,4)
  - 标签图像: dataset/gt/*.jpg    -> sRGB, (2160,3840,3)

配对规则: 同名文件配对，如 0.01/1.raw 对应 gt/1.jpg

RAW 预处理（SID 标准管线）:
  1. 减黑电平 512（Sony 14-bit RAW 的 black level）
  2. 除以有效白电平 (16383 - 512)
  3. 乘曝光比（0.01 曝光集合 ratio=100，即长/短曝光时间比）
  4. clip 到 [0, 1]
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# RAW 文件参数（根据数据集实际格式确定）
RAW_H = 1080
RAW_W = 1920
RAW_CH = 4
RAW_DTYPE = np.uint16
BLACK_LEVEL = 512      # Sony 14-bit RAW 黑电平
WHITE_LEVEL = 16383    # Sony 14-bit RAW 白电平（不是 uint16 的 65535）
DEFAULT_RATIO = 100.0  # 0.01 短曝光集合相对长曝光的放大倍数

def load_raw(file_path, ratio=DEFAULT_RATIO,
             black_level=BLACK_LEVEL, white_level=WHITE_LEVEL):
    """
    加载 .raw 文件 -> float32 数组 [H, W, 4]，范围 [0, 1]

    处理: 减黑电平 -> 除白电平 -> 乘曝光比 -> clip
    """
    data = np.fromfile(file_path, dtype=RAW_DTYPE).reshape(RAW_H, RAW_W, RAW_CH)
    data = data.astype(np.float32)
    data = np.maximum(data - black_level, 0) / (white_level - black_level)
    data = np.clip(data * ratio, 0.0, 1.0)
    return data

def load_jpg(file_path):
    """
    加载 .jpg -> float32 数组 [H, W, 3]，归一化到 [0, 1]
    """
    img = Image.open(file_path).convert('RGB')
    img = np.array(img, dtype=np.float32) / 255.0
    return img


def load_raw_linear(file_path, ratio=1.0,
                    black_level=BLACK_LEVEL, white_level=WHITE_LEVEL):
    """线性域加载 RAW：去黑电平 -> 除白电平 -> 乘曝光比，不 clip。"""
    data = np.fromfile(file_path, dtype=RAW_DTYPE).reshape(RAW_H, RAW_W, RAW_CH)
    data = np.maximum(data.astype(np.float32) - black_level, 0) / (white_level - black_level)
    return data * ratio


def packed_to_rgb_linear(packed):
    """1080x1920x4 packed Bayer(RGGB) -> 1080x1920x3 线性 RGB (R,(Gr+Gb)/2,B)。"""
    return np.stack([packed[..., 0],
                     (packed[..., 1] + packed[..., 2]) / 2,
                     packed[..., 3]], -1)


def gray_world_wb(lin_rgb):
    """长曝光线性 RGB [H,W,3] -> 灰度世界白平衡系数 [3]（非饱和/非近黑像素）。"""
    ok = (lin_rgb.max(2) < 0.95) & (lin_rgb.mean(2) > 0.01)
    if ok.sum() < 100:
        return np.ones(3, np.float32)
    cm = lin_rgb[ok].mean(0)
    if not np.isfinite(cm).all() or cm.mean() < 1e-6:
        return np.ones(3, np.float32)
    return (cm.mean() / np.maximum(cm, 1e-6)).astype(np.float32)


def develop_rgb(lin_rgb, wb, gamma=2.2):
    """线性 RGB + 白平衡 -> gamma 域 [0,1]。"""
    return np.clip(lin_rgb * wb[None, None, :], 0, 1) ** (1.0 / gamma)

def list_paired_names(input_dir, gt_dir):
    """收集 input/gt 同名配对的文件名（只做一次磁盘扫描）"""
    raw_files = sorted(glob.glob(os.path.join(input_dir, '*.raw')))
    names = [os.path.splitext(os.path.basename(f))[0] for f in raw_files]
    names = [n for n in names
             if os.path.exists(os.path.join(gt_dir, n + '.jpg'))]
    return names

def compute_ratio_map(input_dir, gt_raw_dir, cache_name='.ratio_cache.npz', robust=False):
    """
    逐样本估计真实曝光比 gt_raw / short_raw（最小二乘增益，抗暗部噪声）。
    结果缓存到 input_dir 下的 npz，避免每次训练重算。

    robust=True 时仅用长曝光信号 10%~80% 百分位内的像素（剔除饱和高光与
    近黑噪声），缓解全图均值法在高亮/极暗场景上的增益偏差。
    """
    cache_path = os.path.join(input_dir, cache_name)
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return dict(zip(d['names'].tolist(), d['ratios'].tolist()))

    print(f"未发现 ratio 缓存，根据 {gt_raw_dir} 逐样本估计曝光比"
          f"（{'robust 10-80 百分位' if robust else '未饱和均值'}，仅首次）...")
    ratio_map = {}
    raw_files = sorted(glob.glob(os.path.join(input_dir, '*.raw')))
    sat = (WHITE_LEVEL - BLACK_LEVEL) * 0.9
    for f in raw_files:
        name = os.path.splitext(os.path.basename(f))[0]
        gt_path = os.path.join(gt_raw_dir, name + '.raw')
        if not os.path.exists(gt_path):
            ratio_map[name] = DEFAULT_RATIO
            continue
        short = np.maximum(
            np.fromfile(f, RAW_DTYPE).astype(np.float32).reshape(-1)[::37] - BLACK_LEVEL, 0)
        long = np.maximum(
            np.fromfile(gt_path, RAW_DTYPE).astype(np.float32).reshape(-1)[::37] - BLACK_LEVEL, 0)
        # 用长曝光有效位置的信号均值之比估计增益（百万级像素均值对暗部噪声稳定；
        # 最小二乘会在极暗图上被噪声分母严重低估）
        if robust:
            lo, hi = np.percentile(long, [10, 80])
            mask = (long >= lo) & (long <= hi)
        else:
            mask = long < sat
        s_mean = float(short[mask].mean())
        k = float(long[mask].mean()) / max(s_mean, 1e-6)
        ratio_map[name] = float(np.clip(k, 1.0, 300.0))
    np.savez(cache_path,
             names=np.array(list(ratio_map.keys()), dtype=object),
             ratios=np.array(list(ratio_map.values()), dtype=np.float32))
    vals = np.array(list(ratio_map.values()))
    print(f"曝光比估计完成: median={np.median(vals):.1f}, "
          f"范围 {vals.min():.1f}~{vals.max():.1f}，已缓存到 {cache_path}")
    return ratio_map

ALIGNED_DIR = './dataset/gt_aligned'
ALIGNED4K_DIR = './dataset/gt_aligned_4k'
ALIGNED4K_NPY_DIR = './dataset/gt_aligned_4k_npy'


def _integral(mask):
    """掩码积分图（用于 O(1) 判断某裁剪窗口是否全有效）。"""
    m = (mask > 127).astype(np.float64)
    ii = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(m, 0), 1)
    return ii


def _valid_positions(mask, ps, frac):
    """返回掩码中 ps×ps 窗口有效像素占比 >= frac 的裁剪位置 (top, left) 数组。"""
    ii = _integral(mask)
    h, w = mask.shape
    ys, xs = np.mgrid[0:h - ps + 1, 0:w - ps + 1]
    s = (ii[ys + ps, xs + ps] - ii[ys, xs + ps] - ii[ys + ps, xs] + ii[ys, xs])
    return np.argwhere(s >= ps * ps * frac - 0.5)  # (N,2): [top,left]


def _all_valid_positions(mask, ps):
    """返回掩码中 ps×ps 全部有效的裁剪位置 (top, left) 数组（验证集惰性缓存用）。"""
    return _valid_positions(mask, ps, 1.0)


def full_to_packed_mask(mask_full):
    """全分辨率掩码 [2160,3840] -> packed 平面掩码 [1080,1920]。
    一个 packed 像素对应的 2x2 全分辨率块内全有效才算有效（块内 min）。"""
    m = (mask_full > 127).astype(np.uint8)
    packed = m.reshape(RAW_H, 2, RAW_W, 2).min(axis=(1, 3))
    return (packed > 0).astype(np.uint8) * 255


class LowLightDataset(Dataset):
    """
    低光增强数据集（GT 已预配准到 RAW packed 平面坐标 1080x1920）

    Args:
        input_dir: 暗图目录 (dataset/0.01)
        gt_dir: 原始标签目录 (dataset/gt)，仅用于配对扫描
        patch_size: 裁剪大小（input/gt/mask 同坐标同尺寸）
        augment: True=随机裁剪(仅有效区)+翻转旋转; False=中心裁剪无增强
        ratio: 短曝光放大倍数；'auto' 表示用 gt_raw 逐样本估计（推荐）
        gt_raw_dir: ratio='auto' 时使用的长曝光 RAW 目录
        aligned_dir: 预配准 GT 目录（{name}.jpg + {name}.mask.png）；
                     目录或文件缺失时回退为 1080p 直接缩放（全图有效掩码）
    """

    def __init__(self, input_dir, gt_dir, patch_size=512, augment=True,
                 ratio='auto', gt_raw_dir='./dataset/gt_raw',
                 aligned_dir=ALIGNED_DIR, valid_frac=1.0,
                 ratio_cache=None, robust_ratio=False,
                 gt_mode='aligned', gt_dev_raw_dir='./dataset/gt_raw',
                 aligned4k_dir=ALIGNED4K_DIR):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.aligned_dir = aligned_dir
        self.aligned4k_dir = aligned4k_dir
        self.gt_mode = gt_mode            # 'aligned'=预配准JPG | 'dev_raw'=显影长RAW | 'srgb_full'=4K对齐JPG
        self.gt_dev_raw_dir = gt_dev_raw_dir
        self.patch_size = patch_size
        self.augment = augment
        # 训练裁剪允许的最小有效像素占比（1.0=必须全有效；旋转小有效区时 0.95
        # 可消除"无可用裁剪位→退回固定中心"导致的采样退化）
        self.valid_frac = float(valid_frac)
        self.auto_ratio = (ratio == 'auto')
        self.ratio = DEFAULT_RATIO if self.auto_ratio else float(ratio)
        self.names = list_paired_names(input_dir, gt_dir)
        if self.auto_ratio:
            full_map = compute_ratio_map(
                input_dir, gt_raw_dir,
                cache_name=ratio_cache or '.ratio_cache.npz',
                robust=robust_ratio)
            self.ratio_map = {n: full_map.get(n, DEFAULT_RATIO) for n in self.names}
        else:
            self.ratio_map = None
        n_aligned = sum(
            os.path.exists(os.path.join(aligned_dir, n + '.jpg')) for n in self.names)
        self.use_aligned = n_aligned > 0
        if self.use_aligned and n_aligned < len(self.names):
            print(f"警告: 仅 {n_aligned}/{len(self.names)} 个样本存在配准 GT，缺失者回退直接缩放")
        print(f"加载 {len(self.names)} 对图像 (ratio={'auto' if self.auto_ratio else self.ratio}, "
              f"gt_mode={self.gt_mode}, aligned_gt={self.use_aligned})")

    def _load_dev_raw_pair(self, name, ratio):
        """显影短/长 RAW 对（阶段 G：GT 切换为固定 ISP 显影的长曝光 RAW）。

        输入=白平衡后的短曝光线性 packed Bayer [1080,1920,4]（clip[0,1]，WB 与 GT 一致）；
        GT=白平衡+gamma2.2 的长曝光 RGB [1080,1920,3]（全幅有效，无几何错位）。
        任务退化为纯去噪+demosaic+gamma，天花板从 JPG 的 16dB 提升到 22dB+。
        """
        short = load_raw_linear(os.path.join(self.input_dir, name + '.raw'), ratio=ratio)
        long = load_raw_linear(os.path.join(self.gt_dev_raw_dir, name + '.raw'), ratio=1.0)
        l_rgb = packed_to_rgb_linear(long)
        wb = gray_world_wb(l_rgb)
        wb4 = np.array([wb[0], wb[1], wb[1], wb[2]], np.float32)
        inp = np.clip(short * wb4[None, None, :], 0, 1).astype(np.float32)
        gt = develop_rgb(l_rgb, wb).astype(np.float32)
        return inp, gt

    def _load_gt(self, name):
        """返回 (gt[H,W,3] float32, mask[H,W] uint8)，与 RAW packed 平面同坐标。"""
        ap = os.path.join(self.aligned_dir, name + '.jpg')
        if self.use_aligned and os.path.exists(ap):
            gt = np.array(Image.open(ap).convert('RGB').resize((RAW_W, RAW_H), Image.BILINEAR),
                          dtype=np.float32) / 255.0
            mp = os.path.join(self.aligned_dir, name + '.mask.png')
            if os.path.exists(mp):
                mask = np.array(Image.open(mp).resize((RAW_W, RAW_H), Image.NEAREST))
            else:
                mask = np.full((RAW_H, RAW_W), 255, np.uint8)
            return gt, mask
        # 回退：原始 4K GT 直接缩放到 packed 平面尺寸
        gt = np.array(Image.open(os.path.join(self.gt_dir, name + '.jpg'))
                      .convert('RGB').resize((RAW_W, RAW_H), Image.BILINEAR),
                      dtype=np.float32) / 255.0
        return gt, np.full((RAW_H, RAW_W), 255, np.uint8)

    def _load_gt_full(self, name):
        """返回 (gt, mask) 全分辨率对齐 GT，均为 uint8 [2160,3840]（gt 为 RGB）。
        优先读 uint8 .npy 缓存（mmap）；float 化与 /255 在 __getitem__ 裁剪后进行，
        使整帧 JPEG 解码被部分读取取代。"""
        np_path = os.path.join(ALIGNED4K_NPY_DIR, name + '.npy')
        if os.path.exists(np_path):
            gt = np.load(np_path, mmap_mode='r')          # uint8 [2160,3840,3]
            mp = os.path.join(ALIGNED4K_NPY_DIR, name + '.mask.npy')
            mask = np.load(mp, mmap_mode='r') if os.path.exists(mp) \
                else np.full((RAW_H * 2, RAW_W * 2), 255, np.uint8)
            return gt, mask
        ap = os.path.join(self.aligned4k_dir, name + '.jpg')
        if os.path.exists(ap):
            gt = np.array(Image.open(ap).convert('RGB'))
            mp = os.path.join(self.aligned4k_dir, name + '.mask.png')
            mask = np.array(Image.open(mp)) if os.path.exists(mp) \
                else np.full((RAW_H * 2, RAW_W * 2), 255, np.uint8)
            return gt, mask
        # 回退：原始 4K GT（全图有效掩码）
        gt = np.array(Image.open(os.path.join(self.gt_dir, name + '.jpg')).convert('RGB'))
        return gt, np.full((RAW_H * 2, RAW_W * 2), 255, np.uint8)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        ratio = self.ratio_map[name] if self.auto_ratio else self.ratio
        if self.gt_mode == 'dev_raw':
            inp, gt = self._load_dev_raw_pair(name, ratio)
            mask = np.full((RAW_H, RAW_W), 255, np.uint8)
            scale = 1
        elif self.gt_mode == 'srgb_full':
            inp = load_raw(os.path.join(self.input_dir, name + '.raw'), ratio=ratio)
            gt, mask = self._load_gt_full(name)
            scale = 2
        else:
            inp = load_raw(os.path.join(self.input_dir, name + '.raw'), ratio=ratio)
            gt, mask = self._load_gt(name)
            scale = 1
        # 裁剪位置三模式均在 packed 平面(1080x1920)选取；srgb_full 用块内全有效掩码
        sel_mask = full_to_packed_mask(mask) if scale == 2 else mask

        h, w = RAW_H, RAW_W
        ps = min(self.patch_size, h, w)
        cs = ps  # 实际裁剪边长（小有效区帧可能 < ps，随后 pad 回 ps）
        if self.augment:
            # 只在 GT 有效区内随机裁剪：积分图 O(1) 校验 + 拒绝采样
            ii = _integral(sel_mask)
            top = left = -1
            need = ps * ps * self.valid_frac - 0.5
            for _ in range(60):
                ty = int(np.random.randint(0, h - ps + 1))
                tx = int(np.random.randint(0, w - ps + 1))
                s = (ii[ty + ps, tx + ps] - ii[ty, tx + ps]
                     - ii[ty + ps, tx] + ii[ty, tx])
                if s >= need:
                    top, left = ty, tx
                    break
            if top < 0:  # 拒绝采样失败：枚举所有达标位置
                pos = _valid_positions(sel_mask, ps, self.valid_frac)
                if len(pos) > 0:
                    top, left = pos[np.random.randint(len(pos))]
                else:
                    # 旋转/窄小有效区放不下 ps：逐级缩小裁剪边长（最低 256）
                    top = left = -1
                    for cs in (448, 384, 320, 256):
                        if cs >= ps:
                            continue
                        pos = _valid_positions(sel_mask, cs, self.valid_frac)
                        if len(pos) > 0:
                            top, left = pos[np.random.randint(len(pos))]
                            break
                    if top < 0:  # 理论兜底：中心裁剪
                        cs = ps
                        top, left = (h - ps) // 2, (w - ps) // 2
        else:
            # 验证：优先中心，其次距中心最近的有效位置（位置惰性缓存）
            if not hasattr(self, '_val_pos'):
                self._val_pos = {}
            if name not in self._val_pos:
                self._val_pos[name] = _all_valid_positions(sel_mask, ps)
            pos = self._val_pos[name]
            ct, cl = (h - ps) // 2, (w - ps) // 2
            if len(pos) > 0:
                d = np.abs(pos[:, 0] - ct) + np.abs(pos[:, 1] - cl)
                top, left = pos[int(np.argmin(d))]
            else:
                top, left = ct, cl
        # 输入按 packed 坐标裁剪；GT/掩码按 scale 倍坐标裁剪（srgb_full 为 2x）
        inp = inp[slice(top, top + cs), slice(left, left + cs)]
        g0, g1 = scale * top, scale * left
        gt = gt[g0:g0 + scale * cs, g1:g1 + scale * cs]
        mask = mask[g0:g0 + scale * cs, g1:g1 + scale * cs]
        if self.gt_mode == 'srgb_full':  # uint8 全分辨率 GT -> float32 [0,1]（仅裁剪块）
            gt = gt.astype(np.float32) / 255.0
        if cs < ps:  # 小裁剪补回：输入按 ps 补（packed），GT/掩码按 scale*ps 补
            ph, pw = ps - cs, ps - cs
            t0 = int(np.random.randint(0, ph + 1)) if self.augment else ph // 2
            l0 = int(np.random.randint(0, pw + 1)) if self.augment else pw // 2
            inp = np.pad(inp, ((t0, ph - t0), (l0, pw - l0), (0, 0)), mode='edge')
            gt = np.pad(gt, ((t0 * scale, (ph - t0) * scale),
                             (l0 * scale, (pw - l0) * scale), (0, 0)),
                        mode='constant', constant_values=0)
            mask = np.pad(mask, ((t0 * scale, (ph - t0) * scale),
                                 (l0 * scale, (pw - l0) * scale)),
                          mode='constant', constant_values=0)

        # 数据增强（input/gt/mask 同步；packed Bayer 表示下翻转旋转合法）
        if self.augment:
            if np.random.rand() > 0.5:
                inp = np.fliplr(inp).copy(); gt = np.fliplr(gt).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                inp = np.flipud(inp).copy(); gt = np.flipud(gt).copy()
                mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                inp = np.rot90(inp, k).copy(); gt = np.rot90(gt, k).copy()
                mask = np.rot90(mask, k).copy()

        inp = torch.from_numpy(np.ascontiguousarray(inp.transpose(2, 0, 1)))
        gt = torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1)))
        mask = torch.from_numpy((mask > 127).astype(np.float32)[None])
        return inp, gt, mask

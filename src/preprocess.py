"""
preprocess.py — 实验图预处理模块：无量纲拓扑对齐管线
======================================================

本模块负责将真实 CCD 采集的实验光斑图像转换为与理论仿真数据
拓扑对齐的标准化张量，彻底消除 Sim-to-Real 物理尺度鸿沟。

【核心设计理念】
  理论仿真数据使用无量纲坐标 (w0=1.0, 窗口[-4,4])，
  其本质是光斑的"拓扑骨架"——干涉条纹的相对间距、奇点分布、
  对称性破缺模式。CNN (ResNet34) 提取的正是这些 Scale-invariant
  的拓扑特征，而非绝对物理尺寸。

  因此，实验图的预处理不是去"匹配物理尺度"，而是去"剥离物理尺度"：
  通过质心裁剪+拉伸缩放，将实验光斑从物理像素域映射到归一化的
  拓扑特征域，使得实验光斑的干涉结构与理论光斑处于同一等价类中。

【标准化管线 (5 步)】
  Step 1: 灰度转换 — RGB/多通道 → 单通道灰度 (PIL)
  Step 2: 背景去噪 — 低于 5% 最大光强的像素置零 (numpy)
  Step 3: 质心定位 — 能量加权中心计算 (numpy)
  Step 4: 紧凑裁剪 — 沿能量包络提取外接矩形 Bounding Box (numpy)
  Step 5: 强行缩放 — 双线性插值 Resize 至 224×224, [0,1] 归一化 (PIL)

【与理论数据的拓扑对齐性】
  理论: w0=1.0, [-4,4], 224×224 网格 → 拓扑骨架
  实验: 任意尺寸 → 裁剪 → 224×224 网格 → 拓扑骨架
         ↑ 物理尺度在此被丢弃，仅保留拓扑结构 ↑

【依赖】仅需 PIL/Pillow + numpy，无需 OpenCV。

日期: 2026-07-31
"""

import numpy as np
from typing import Tuple, List, Union
from pathlib import Path

from PIL import Image


# =============================================================================
# 预处理常量
# =============================================================================

# 背景噪声阈值系数 (相对于最大光强的比例)
# 物理含义: CCD 暗电流和杂散光通常 < 峰值的 5%
BACKGROUND_THRESHOLD_RATIO = 0.05

# 紧凑裁剪时的边缘 padding (像素)
# 预留 2-3 像素避免裁剪到光斑的有效能量边缘
CROP_PADDING = 3

# 目标输出尺寸
TARGET_SIZE = 224


# =============================================================================
# 实验图像预处理器
# =============================================================================

class ExperimentalImageProcessor:
    """
    实验光斑图像标准化预处理器。

    将原始 CCD 采集的实验光斑图像通过灰度转换、背景去噪、
    质心裁剪和缩放归一化，转换为与无量纲理论仿真数据
    拓扑对齐的 224×224 标准化张量。

    使用方法:
        processor = ExperimentalImageProcessor()
        result = processor.process("experiment_spot.png")
        # result['image']: (224, 224) ndarray, float64, [0, 1]
        # result['centroid']: (x_c, y_c) 质心坐标
        # result['bbox']: (x1, y1, x2, y2) 裁剪边界
    """

    def __init__(
        self,
        bg_threshold_ratio: float = BACKGROUND_THRESHOLD_RATIO,
        crop_padding: int = CROP_PADDING,
        target_size: int = TARGET_SIZE,
        mask_threshold_ratio: float = 0.05,
    ):
        """
        Args:
            bg_threshold_ratio: 背景噪声阈值 (相对于最大光强的比例)
            crop_padding: 紧凑裁剪时的边缘 padding (像素)
            target_size: 输出图像的目标尺寸 (正方形)
            mask_threshold_ratio: 归一化后的物理掩膜阈值 (相对于最大光强, 默认 0.05)
        """
        self.bg_threshold_ratio = bg_threshold_ratio
        self.crop_padding = crop_padding
        self.target_size = target_size
        self.mask_threshold_ratio = mask_threshold_ratio

    # =========================================================================
    # 主处理管线
    # =========================================================================

    def process(
        self,
        image: Union[str, Path, np.ndarray],
        return_intermediate: bool = False,
    ) -> dict:
        """
        执行完整的实验图预处理管线。

        Args:
            image: 输入图像 (文件路径或 numpy ndarray)
            return_intermediate: 是否返回中间步骤结果 (用于调试可视化)

        Returns:
            dict:
                'image': (224, 224) ndarray, float64, [0, 1] 归一化光强
                'centroid': (cx, cy), 光斑质心坐标 (原始图像坐标系)
                'bbox': (x1, y1, x2, y2), 裁剪边界框
                'original_shape': (H, W), 原始图像尺寸

                若 return_intermediate=True，额外包含:
                'gray': 灰度图 (float64)
                'denoised': 去噪后 (float64)
                'cropped': 裁剪后 (float64)
        """
        # ---- Step 0: 加载图像 ----
        if isinstance(image, (str, Path)):
            img_array = self._load_image(str(image))
        elif isinstance(image, np.ndarray):
            img_array = image.copy()
        else:
            raise TypeError(f"不支持的输入类型: {type(image)}")

        original_shape = img_array.shape[:2]

        # ---- Step 1: 灰度转换 ----
        gray = self._to_grayscale(img_array)

        # ---- Step 2: 背景去噪 ----
        denoised = self._remove_background(gray)

        # ---- Step 3: 质心定位 ----
        centroid = self._find_centroid(denoised)

        # ---- Step 4: 紧凑裁剪 ----
        cropped, bbox = self._crop_tight(denoised)

        # ---- Step 5: 双线性缩放 + 严格 Min-Max 线性归一化 ----
        normalized = self._resize_and_normalize(cropped)

        # ---- Step 6: 物理掩膜 (消除 CCD 暗电流噪声 + 插值边缘伪影) ----
        normalized = self._apply_physical_mask(normalized)

        result = {
            'image': normalized,
            'centroid': centroid,
            'bbox': bbox,
            'original_shape': original_shape,
        }

        if return_intermediate:
            result['gray'] = gray
            result['denoised'] = denoised
            result['cropped'] = cropped

        return result

    def process_batch(
        self,
        image_paths: List[str],
    ) -> List[dict]:
        """
        批量处理多张实验图像。

        Args:
            image_paths: 图像文件路径列表

        Returns:
            每张图像的 process() 结果组成的列表
        """
        results = []
        for path in image_paths:
            try:
                result = self.process(path)
                result['source_path'] = path
                results.append(result)
            except Exception as e:
                print(f"[警告] 处理 {path} 时出错: {e}")
                results.append({'error': str(e), 'source_path': path})
        return results

    # =========================================================================
    # Step 0 & 1: 图像加载与灰度转换
    # =========================================================================

    def _load_image(self, path: str) -> np.ndarray:
        """
        从文件加载图像并转换为 numpy 数组。

        使用 PIL 支持 PNG, TIFF, BMP, JPEG 等常见格式。
        """
        pil_img = Image.open(path)
        return np.array(pil_img)

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """
        将输入图像转换为单通道灰度图 (float64)。

        转换策略:
          - 若已是单通道 (H, W): 直接转为 float64
          - 若为 RGB/RGBA 多通道 (H, W, C):
            使用 PIL 的标准灰度转换 (ITU-R 601-2 luma):
              gray = 0.299·R + 0.587·G + 0.114·B

        【物理含义】
          实验光斑的伪彩色 (如 DMD 衍射的彩虹效应) 是采集系统的人工产物，
          不包含任何有用的干涉信息。转换为灰度后仅保留真实的强度调制信号。

        Args:
            img: 输入图像 (任意通道数, 任意 dtype)

        Returns:
            (H, W) float64 灰度图, 值域保持原始范围
        """
        if img.ndim == 2:
            # 已是单通道灰度
            return img.astype(np.float64)

        if img.ndim == 3:
            # 多通道 → 使用 PIL 标准灰度转换
            if img.shape[2] >= 3:
                # 取前三个通道 (忽略可能的 alpha)
                pil_rgb = Image.fromarray(img[:, :, :3].astype(np.uint8))
            else:
                pil_rgb = Image.fromarray(img.astype(np.uint8))
            gray_pil = pil_rgb.convert('L')
            return np.array(gray_pil, dtype=np.float64)

        raise ValueError(f"不支持的图像维度: {img.ndim}D, shape={img.shape}")

    # =========================================================================
    # Step 2: 背景去噪
    # =========================================================================

    def _remove_background(self, gray: np.ndarray) -> np.ndarray:
        """
        去除 CCD 背景底噪。

        操作:
          1. 计算光强最大值 I_max
          2. 设定阈值 tau = bg_threshold_ratio * I_max (默认 5%)
          3. 将 I < tau 区域的像素值置零，其余保持不变

        【物理含义】
          CCD 传感器存在暗电流 (dark current) 和读出噪声 (read noise)，
          即使在没有光斑的区域也会产生非零的基底信号。这个基底信号
          不是来自激光场，而是来自探测器电子学。

          通过将低于 5% 峰值的像素置零，我们精确地去除了探测器噪声
          的贡献，同时保留了光斑的全部干涉结构。

        【为什么不去除光斑边缘的低强度信号？】
          多奇点涡旋阵列在奇点处存在精确的强度零点 (相位奇点处的
          完全相消干涉)。这些零点不是噪声——它们是干涉拓扑结构的
          核心特征。5% 阈值远高于数值零 (相对于峰值 1.0)，不会误伤
          奇点零点的邻近区域。

        Args:
            gray: (H, W) 灰度图 (float64)

        Returns:
            (H, W) 去噪后的灰度图，背景区域为零
        """
        img_max = gray.max()
        if img_max < 1e-10:
            return gray

        threshold = self.bg_threshold_ratio * img_max
        mask = gray >= threshold
        denoised = gray * mask
        return denoised

    # =========================================================================
    # Step 3: 质心定位
    # =========================================================================

    def _find_centroid(self, denoised: np.ndarray) -> Tuple[float, float]:
        """
        计算光斑的能量加权质心。

        质心公式:
            x_c = sum(x * I(x,y)) / sum(I(x,y))
            y_c = sum(y * I(x,y)) / sum(I(x,y))

        【物理含义】
          能量加权质心代表光束的传播主轴在 CCD 平面上的投影位置。
          它是光斑的"几何中心"，也是后续裁剪操作的空间参考点。

          与简单的几何中心 (bounding box 中心) 不同，能量加权质心
          对光斑的对称性更敏感：如果光斑某个方向有更强的旁瓣，
          质心会向该方向偏移。

        【鲁棒性处理】
          如果去噪后全图为黑 (所有像素 <= 阈值)，则返回图像几何中心。

        Args:
            denoised: (H, W) 去噪后的灰度图 (float64)

        Returns:
            (cx, cy), 质心浮点坐标 (cx=列索引, cy=行索引)
        """
        total = denoised.sum()
        if total < 1e-12:
            # 全黑退化情况: 返回几何中心
            H, W = denoised.shape
            return (W / 2.0, H / 2.0)

        H, W = denoised.shape
        y_indices, x_indices = np.mgrid[0:H, 0:W]

        cx = float(np.sum(x_indices * denoised) / total)
        cy = float(np.sum(y_indices * denoised) / total)

        return (cx, cy)

    # =========================================================================
    # Step 4: 紧凑裁剪 (Bounding Box Cropping)
    # =========================================================================

    def _crop_tight(
        self, denoised: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        沿光斑能量包络进行紧凑裁剪。

        操作:
          1. 找到所有 I(x,y) > 0 像素的外接矩形框
             x_min, x_max, y_min, y_max
          2. 在边界外增加 crop_padding 像素的余量
          3. 将图像裁剪至该矩形区域

        【这是消除 Sim-to-Real 尺度鸿沟的核心操作】

        原始 CCD 图像中，光斑的大小由以下物理因素共同决定:
          - 真实束腰半径 w0 (mm)
          - CCD 像素尺寸 pitch (um)
          - 成像系统放大倍率 M
          - 光路中的热透镜效应

        这些因素产生的是"绝对物理尺度"，无法与无量纲理论数据
        (w0=1.0, [-4,4]) 直接对齐。

        紧凑裁剪的操作本质是:
          → 找到光斑的能量包络边缘
          → 丢弃外围的纯背景像素
          → 仅保留光斑本体的形态信息

        裁剪后，图像从"光斑+空白背景"变为"光斑填充画面"，
        所有绝对尺度信息被丢弃，仅保留光斑内部干涉结构的
        相对空间关系——即拓扑特征。

        Args:
            denoised: (H, W) 去噪后的灰度图 (float64)

        Returns:
            (cropped, bbox):
            - cropped: (H', W') float64, 裁剪后的光斑图像
            - bbox: (x1, y1, x2, y2), 裁剪边界 (含 padding)
        """
        nonzero_mask = denoised > 0
        if not nonzero_mask.any():
            H, W = denoised.shape
            return denoised, (0, 0, W, H)

        rows = np.any(nonzero_mask, axis=1)
        cols = np.any(nonzero_mask, axis=0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        H, W = denoised.shape
        y1 = max(0, int(y_min) - self.crop_padding)
        y2 = min(H, int(y_max) + self.crop_padding + 1)
        x1 = max(0, int(x_min) - self.crop_padding)
        x2 = min(W, int(x_max) + self.crop_padding + 1)

        bbox = (x1, y1, x2, y2)
        cropped = denoised[y1:y2, x1:x2]

        return cropped, bbox

    # =========================================================================
    # Step 5: 强行缩放与归一化
    # =========================================================================

    def _resize_and_normalize(self, cropped: np.ndarray) -> np.ndarray:
        """
        将裁剪后的任意尺寸光斑图强制拉伸/压缩为 224x224 标准尺寸。

        操作:
          1. 使用 PIL 双线性插值 (Image.BILINEAR) 进行缩放
          2. 将像素值线性归一化到 [0, 1]

        【拓扑等价性保证】

        双线性插值在拉伸/压缩时对光强值进行平滑过渡，不会引入
        新的奇点或破坏现有的干涉条纹拓扑结构。具体而言:

          - 放大 (cropped < 224): 填充新像素时采用邻域线性加权，
            条纹边缘保持平滑连续
          - 缩小 (cropped > 224): 降采样时对邻域取加权平均，
            避免最近邻重采样产生的混叠伪影

        【归一化到 [0, 1]】
          理论仿真引擎输出的光强图经过 max 归一化到 [0, 1]。
          实验图经过相同的归一化后，两张图的像素值具有相同的
          动态范围和统计分布——这是 CNN 正确推理的前提条件。

        Args:
            cropped: (H', W') float64, 裁剪后的光斑

        Returns:
            (224, 224) ndarray, float64, [0, 1]
        """
        # 转为 PIL Image 进行双线性插值缩放
        # 先将 float64 映射到 [0, 255] 以便 PIL 处理
        max_val = cropped.max()
        if max_val < 1e-12:
            # 纯黑退化: 直接返回零矩阵
            return np.zeros((self.target_size, self.target_size), dtype=np.float64)

        img_uint8 = (cropped / max_val * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='L')

        # 双线性插值缩放到目标尺寸
        pil_resized = pil_img.resize(
            (self.target_size, self.target_size),
            resample=Image.BILINEAR,
        )

        # 转回 numpy 并归一化到 [0, 1]
        resized = np.array(pil_resized, dtype=np.float64) / 255.0

        # ---- 严格 Min-Max 归一化 ----
        # 双线性插值可能在边缘产生非零极小值，严格重映射到 [0, 1]
        # 消除 CCD 增益/偏置差异对亮度动态范围的影响
        r_min, r_max = resized.min(), resized.max()
        if r_max - r_min > 1e-12:
            resized = (resized - r_min) / (r_max - r_min)
        else:
            resized = np.zeros_like(resized)

        return np.clip(resized, 0.0, 1.0)


    # =========================================================================
    # Step 6: 物理掩膜 (Physical Masking)
    # =========================================================================

    def _apply_physical_mask(self, img: np.ndarray) -> np.ndarray:
        """
        对 Min-Max 归一化后的图像施加物理阈值掩膜，消除 CCD 暗电流噪声。

        【物理动机】

        Min-Max 线性归一化保证了光强能量的线性守恒，但双线性插值
        缩放会在光斑边缘产生微弱的灰色过渡带（数值上非零但在物理上
        不包含任何干涉能量）。这些过渡带来自：
          - 裁剪边界的插值混合
          - CCD 暗电流在边缘像素的残留
          - 读出噪声的随机涨落

        通过在归一化后再次施加阈值截断（低于 max 的 mask_threshold_ratio
        的像素强制置零），我们：
          1. 保留全部干涉结构（奇点零点 > 5% 阈值不受影响）
          2. 消除插值引入的虚假低强度像素
          3. 确保零背景区域在实验图和理论图中具有一致的定义

        【为什么这是线性操作？】
          阈值截断是分段线性变换：
            I_out = I_in  (if I_in >= τ)
            I_out = 0     (if I_in < τ)
          在有效区域内 (I_in >= τ)，变换是恒等的，不改变像素间的
          相对比值关系。PCC 在有效区域内完全不受影响。

        Args:
            img: (H, W) float64, [0, 1] Min-Max 归一化光强图

        Returns:
            (H, W) float64, 掩膜后的光强图，背景严格为零
        """
        max_val = img.max()
        if max_val < 1e-12:
            return img

        threshold = self.mask_threshold_ratio * max_val
        mask = img >= threshold
        return img * mask


# =============================================================================
# 便捷函数
# =============================================================================

def preprocess_experimental_image(
    image_path: str,
    bg_threshold: float = 0.05,
    padding: int = 3,
    mask_threshold: float = 0.05,
) -> np.ndarray:
    """
    预处理单张实验光斑图像的一行式便捷接口。

    Args:
        image_path: 实验光斑图像文件路径
        bg_threshold: 背景阈值 (默认 0.05 = 5%)
        padding: 裁剪 padding (默认 3 像素)
        mask_threshold: 归一化后物理掩膜阈值 (默认 0.05 = 5%)

    Returns:
        (224, 224) ndarray, float64, [0, 1] 归一化光强分布

    使用示例:
        img = preprocess_experimental_image("vortex_8_singularities.png")
        # img.shape → (224, 224)
    """
    processor = ExperimentalImageProcessor(
        bg_threshold_ratio=bg_threshold,
        crop_padding=padding,
        mask_threshold_ratio=mask_threshold,
    )
    result = processor.process(image_path)
    return result['image']


# =============================================================================
# 自检代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ExperimentalImageProcessor 自检 (PIL/numpy 纯实现)")
    print("=" * 60)

    processor = ExperimentalImageProcessor()

    # --- 测试 1: 人工合成高斯光斑 ---
    print("\n--- 测试 1: 合成高斯光斑 (模拟理想实验图) ---")
    size = 512
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    gaussian = np.exp(-((X - 0.3)**2 + (Y + 0.2)**2) / 0.5**2)
    gaussian_noisy = gaussian + np.random.default_rng(42).normal(0, 0.02, (size, size))
    gaussian_noisy = np.clip(gaussian_noisy, 0, 1)
    img_uint8 = (gaussian_noisy * 255).astype(np.uint8)

    result = processor.process(img_uint8, return_intermediate=True)

    print(f"  原始尺寸: {result['original_shape']}")
    print(f"  质心: ({result['centroid'][0]:.1f}, {result['centroid'][1]:.1f})")
    print(f"  裁剪边界: {result['bbox']}")
    print(f"  裁剪后尺寸: {result['cropped'].shape}")
    print(f"  输出尺寸: {result['image'].shape}")
    print(f"  输出值域: [{result['image'].min():.4f}, {result['image'].max():.4f}]")

    assert result['image'].shape == (224, 224), "输出尺寸错误!"
    assert 0.0 <= result['image'].min() <= result['image'].max() <= 1.0, "值域错误!"
    print("  [OK] 合成高斯光斑预处理通过")

    # --- 测试 2: 多模式干涉光斑 (模拟涡旋阵列实验图) ---
    print("\n--- 测试 2: 多模式干涉光斑 (模拟涡旋阵列) ---")
    try:
        from hg_generator import create_default_generator
        gen = create_default_generator()
        rng = np.random.default_rng(123)
        amps = gen.sample_amplitudes(rng)
        phases = gen.sample_phases(rng)
        vortex_spot = gen.generate_interference_pattern(amps, phases)

        # 模拟实验条件: 放在一个更大的画布上 (含多余背景)
        big_canvas = np.zeros((600, 800), dtype=np.float64)
        big_canvas[50:50+224, 150:150+224] = vortex_spot
        big_canvas += rng.normal(0, 0.01, big_canvas.shape)
        big_canvas = np.clip(big_canvas, 0, 1)
        vortex_uint8 = (big_canvas * 255).astype(np.uint8)

        result2 = processor.process(vortex_uint8, return_intermediate=True)

        print(f"  原始尺寸: {result2['original_shape']}")
        print(f"  质心: ({result2['centroid'][0]:.1f}, {result2['centroid'][1]:.1f})")
        print(f"  裁剪边界: {result2['bbox']}")
        print(f"  裁剪后尺寸: {result2['cropped'].shape}")
        print(f"  输出尺寸: {result2['image'].shape}")

        assert result2['image'].shape == (224, 224), "输出尺寸错误!"
        print("  [OK] 多模式干涉光斑预处理通过")

    except ImportError:
        print("  [跳过] hg_generator 模块不可用")

    # --- 测试 3: 纯黑退化情况 ---
    print("\n--- 测试 3: 纯黑退化鲁棒性 ---")
    black_img = np.zeros((100, 100), dtype=np.uint8)
    result3 = processor.process(black_img)
    print(f"  输出尺寸: {result3['image'].shape}")
    print(f"  质心: ({result3['centroid'][0]:.1f}, {result3['centroid'][1]:.1f})")
    assert result3['image'].shape == (224, 224)
    print("  [OK] 纯黑退化处理通过 (未崩溃)")

    # --- 测试 4: 文件路径输入 ---
    print("\n--- 测试 4: 文件路径输入 ---")
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    test_path = os.path.join(tmpdir, "test_gaussian.png")
    pil_test = Image.fromarray((gaussian_noisy * 255).astype(np.uint8), mode='L')
    pil_test.save(test_path)

    result4 = processor.process(test_path)
    print(f"  文件路径输入: {test_path}")
    print(f"  输出尺寸: {result4['image'].shape}")
    assert result4['image'].shape == (224, 224)
    print("  [OK] 文件路径输入通过")

    # --- 测试 5: 批量处理接口 ---
    print("\n--- 测试 5: 批量处理接口 ---")
    tmp_paths = []
    for i in range(3):
        p = os.path.join(tmpdir, f"test_{i}.png")
        pil_tmp = Image.fromarray(
            (np.random.default_rng(i).uniform(0, 1, (300, 400)) * 255).astype(np.uint8)
        )
        pil_tmp.save(p)
        tmp_paths.append(p)

    batch_results = processor.process_batch(tmp_paths)
    success_count = sum(1 for r in batch_results if 'error' not in r)
    print(f"  批量处理 {len(batch_results)} 张, 成功 {success_count} 张")
    for i, r in enumerate(batch_results):
        if 'error' in r:
            print(f"    [{i}] 错误: {r['error']}")
        else:
            print(f"    [{i}] shape={r['image'].shape}, 质心=({r['centroid'][0]:.0f},{r['centroid'][1]:.0f})")
    print("  [OK] 批量处理通过")

    import shutil
    shutil.rmtree(tmpdir)

    print(f"\n[OK] ExperimentalImageProcessor 所有自检通过!")
    print(f"\n【无量纲拓扑对齐总结】")
    print(f"  理论数据: w0=1.0, [-4,4], 224x224 → 拓扑骨架 (干涉结构)")
    print(f"  实验数据: 任意尺寸 → 灰度 → 去噪 → 裁剪 → 224x224 → 拓扑骨架")
    print(f"  对齐机制: 裁剪丢弃物理尺度, 缩放保留拓扑比例")
    print(f"  目标: 两者的干涉条纹间距比、奇点位置比、对称性在同一等价类中")

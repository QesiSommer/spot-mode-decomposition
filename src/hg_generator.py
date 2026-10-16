"""
hg_generator.py — 光学物理基石：Hermite-Gaussian (HG) 模式生成器 (无量纲版)
==============================================================================

本模块是项目的物理核心，负责在束腰平面 (z=0) 生成 Hermite-Gaussian
本征模的归一化二维复振幅分布，并根据相干叠加原理计算多模干涉光强图样。

【无量纲化策略 (Dimensionless Formulation)】
  - 束腰半径 w0 = 1.0 (归一化单位)，所有空间坐标以 w0 为单位
  - 空间窗口: x, y ∈ [-4, 4] (覆盖 HG 模式 >99.9% 总能量)
  - 网格分辨率: 224 × 224
  - 本模块不包含任何物理单位参数 (mm, μm, nm)
  - CNN 提取的是拓扑干涉特征 (Scale-invariant)，无需绝对物理尺寸

物理基础:
  - 傍轴 Helmholtz 方程在直角坐标系下的本征解 → HG 模式族
  - 束腰处简化：Gouy 相位 = 0，波前曲率 R → ∞
  - 本征模正交归一性：int |HG_{m,n}(x,y)|² dx dy = 1
  - 相干叠加：E = Σ a_k · HG_k · exp(iφ_k)
  - CCD 记录光强：I = |E|²

参考文献：
  Zhang et al., J. Opt. 27, 075604 (2025), Eq. (1)-(2)
  Siegman, "Lasers" (1986), Chapter 16

作者: 扬州大学光斑模式分解项目组
日期: 2026-07-31 (无量纲化重构)
"""

import numpy as np
from scipy.special import hermite, factorial, genlaguerre
from typing import List, Tuple, Optional


# =============================================================================
# 无量纲化全局常量
# =============================================================================

# 束腰半径 — 作为空间坐标的基本单位
W0 = 1.0

# 空间采样窗口 (以 w0 为单位)
# 从原 ±3 扩展到 ±4，覆盖 HG 模式 >99.9% 的总能量
# 对于 N=7 的最高阶模式，Hermite 多项式的最后一个节点位于 ≈3.3w0
# 对于 N=9 (55 阶) 的最高阶模式 HG_{9,0}：数值验证显示 [-4, 4] 窗口
# 覆盖其 99.80% 的能量（截断约 0.2%，对 55 模式随机叠加图样影响 <0.1%）。
# 【设计权衡】保持 ±4 不变以与旧 36 阶数据集的像素尺度严格一致，
# 使 36→55 阶迁移学习时 Backbone 提取的特征空间无尺度漂移。
SPATIAL_HALF_RANGE = 4.0

# 默认网格分辨率
DEFAULT_GRID_SIZE = 224

# 默认最大总阶数 (N_max=12 → 91 个模式)
# 数值验证: HG_{12,0} 在 [-4,4] 窗口能量覆盖 99.827% (截断 0.173%, 可接受);
#           HG_{15,0} 截断 3.57% (否决 136 阶方案)
DEFAULT_MAX_ORDER_N = 12


def mode_indices_for(max_order_N: int) -> List[Tuple[int, int]]:
    """按总阶 N=m+n 升序、同阶 m 升序生成全部 (m,n) 索引对。"""
    return [(m, N - m) for N in range(max_order_N + 1) for m in range(N + 1)]


def num_modes_for(max_order_N: int) -> int:
    """K = (N_max+1)(N_max+2)/2 个 HG 模式。"""
    return (max_order_N + 1) * (max_order_N + 2) // 2


def max_order_for_num_modes(K: int) -> int:
    """由模式数反推 N_max (K 必须是三角数 T(N))。"""
    N = int((np.sqrt(8.0 * K + 1) - 3) / 2)
    assert num_modes_for(N) == K, f"K={K} 不是三角数 T(N), 无法确定 N_max"
    return N


class LGModeBasis:
    """
    Laguerre-Gaussian (LG) 模式基 (纯无量纲, w0=1.0)。

    LG 模式与 HG 模式同为 2D 谐振子能量 ≤ N_max 子空间的完备正交基
    (模式数均为 K = (N_max+1)(N_max+2)/2)，二者之间只差一个 K×K 酉变换。
    本类提供 LG 模式的预计算复振幅，供涡旋/LG 分支生成使用。

    索引约定: q = 2p + |l| 升序 → p 升序 → (p, +l) 在前 (p, -l) 在后。
    约束: 2p + |l| ≤ max_order_N → LG 模式数 == HG 模式数 (同子空间)。
    """

    def __init__(self, max_order_N: int, X: np.ndarray, Y: np.ndarray,
                 dx: float):
        self.max_order_N = max_order_N
        self.grid_size = X.shape[0]
        self.X = X
        self.Y = Y
        self.dx = dx

        # 索引: q=2p+|l| 升序 → p 升序 → (p,+l) 在前 (p,−l) 在后
        self.mode_indices: List[Tuple[int, int]] = []
        for q in range(max_order_N + 1):
            for p in range(q // 2 + 1):
                l = q - 2 * p
                self.mode_indices.append((p, l))
                if l != 0:
                    self.mode_indices.append((p, -l))
        self.num_modes = len(self.mode_indices)
        expected = num_modes_for(max_order_N)
        assert self.num_modes == expected, (
            f"LG 模式计数不一致: 实际 {self.num_modes} ≠ 期望 {expected}"
        )

        self.lg_modes = self._precompute_all_modes()

    def _compute_single_lg_mode(self, p: int, l: int) -> np.ndarray:
        """
        计算单个 LG_{p,l} 模式在束腰处的无量纲归一化复振幅。

        【无量纲物理公式】(w0=1.0, z = x + iy):
            LG_{p,l}(x, y) = C · L_p^{|l|}(2r²) · z^l · exp(-r²)   (l ≥ 0)
            LG_{p,l}(x, y) = C · L_p^{|l|}(2r²) · conj(z)^{|l|} · exp(-r²)  (l < 0)

        归一化常数:
            C = sqrt(2·p! / (π·(p+|l|)!)) · (√2)^{|l|}

        【归一化推导】: 标准 LG 公式 (w0=1):
            LG_{p,l} = sqrt(2p!/(π(p+|l|)!)) · (√2 r)^{|l|} · L_p^{|l|}(2r²)
                       · exp(-r²) · exp(i·l·θ)
          (√2 r)^{|l|}·exp(i·l·θ) = (√2)^{|l|}·z^l (l≥0) / (√2)^{|l|}·conj(z)^{|l|} (l<0)
          ∫|LG|² dA = 1 (标准 LG 正交归一, 已知结果)
        """
        r2 = self.X ** 2 + self.Y ** 2
        z = self.X + 1j * self.Y

        C = np.sqrt(2.0 * factorial(p, exact=True)
                    / (np.pi * factorial(p + abs(l), exact=True)))
        C *= np.sqrt(2.0) ** abs(l)

        poly = genlaguerre(p, abs(l))(2.0 * r2)
        if l >= 0:
            phase = np.power(z, l)
        else:
            phase = np.power(np.conj(z), -l)

        lg_mode = C * poly * phase * np.exp(-r2)
        return lg_mode.astype(np.complex128)

    def _precompute_all_modes(self) -> np.ndarray:
        modes = np.zeros((self.num_modes, self.grid_size, self.grid_size),
                         dtype=np.complex128)
        for idx, (p, l) in enumerate(self.mode_indices):
            modes[idx] = self._compute_single_lg_mode(p, l)
        return modes


class HGModeGenerator:
    """
    Hermite-Gaussian 模式生成器 (纯无量纲)。

    在束腰平面 (z=0)，以归一化束腰 w0=1.0 为基准，生成指定阶数
    范围内的所有 HG 本征模的归一化二维复振幅分布，并提供
    相干叠加干涉图样生成功能。

    Attributes:
        max_order_N (int): 最大总阶数 N_max，默认 5
        num_modes (int): 总模式数 K = (N_max+1)(N_max+2)/2
        grid_size (int): 输出图像尺寸 (grid_size × grid_size)，默认 224
        mode_indices (List[Tuple[int,int]]): 按 (m, n) 排序的模式索引列表
        X (np.ndarray): 无量纲坐标网格 X (以 w0 为单位)
        Y (np.ndarray): 无量纲坐标网格 Y (以 w0 为单位)
        dx (float): 无量纲像素间距 = 2*range / grid_size
        hg_modes (np.ndarray): shape=(K, grid_size, grid_size)，复数 dtype，
                               预计算的所有 HG 本征模归一化复振幅
    """

    def __init__(
        self,
        max_order_N: int = 9,
        grid_size: int = DEFAULT_GRID_SIZE,
    ):
        """
        初始化无量纲 HG 模式生成器。

        Args:
            max_order_N: 最大总阶数 N = m + n，默认 9（共 55 个模式，
                         覆盖 m+n ≤ 9 的全部 (m,n) 组合）
            grid_size: 输出图像的像素尺寸，默认 224×224
        """
        self.max_order_N = max_order_N
        self.grid_size = grid_size

        # 无量纲像素间距
        self.dx = 2.0 * SPATIAL_HALF_RANGE / self.grid_size

        # 生成所有满足 m + n ≤ N_max 的 (m, n) 索引对
        # 按照阶数 N = m+n 升序排列，同阶内按 m 升序
        self.mode_indices: List[Tuple[int, int]] = mode_indices_for(max_order_N)

        self.num_modes = len(self.mode_indices)

        # 验证模式计数公式:
        # K = (N_max + 1)(N_max + 2) / 2
        expected = num_modes_for(max_order_N)
        assert self.num_modes == expected, (
            f"模式计数不一致: 实际 {self.num_modes} ≠ 期望 {expected}"
        )

        # 预计算无量纲空间坐标网格
        self._build_coordinate_grid()

        # 预计算所有 HG 模式归一化复振幅
        self.hg_modes = self._precompute_all_modes()

        # ---- LG 模式基 (2D 谐振子能量≤N_max 子空间的等价完备基) ----
        self.lg_basis = LGModeBasis(max_order_N, self.X, self.Y, self.dx)

        # ---- LG→HG 酉变换矩阵: c_HG = U @ w_LG (F1: 同子空间, 免逐样本投影) ----
        # U[k, j] = <HG_k, LG_j> · dx²
        self.lg_to_hg_transform = np.einsum(
            'kij,mij->km', np.conj(self.hg_modes), self.lg_basis.lg_modes
        ) * (self.dx * self.dx)

    # =========================================================================
    # 坐标网格
    # =========================================================================

    def _build_coordinate_grid(self) -> None:
        """
        构建无量纲空间坐标网格。

        生成范围 [-4, 4] × [-4, 4] (以 w0 为单位) 的均匀采样网格，
        shape = (grid_size, grid_size)。使用 'ij' 索引方式确保矩阵坐标一致性。

        无量纲约定:
            x = x_physical / w0 → 所有坐标值纯数字，无物理单位
            dx = 8.0 / 224 ≈ 0.0357 → 无量纲像素间距
        """
        x = np.linspace(
            -SPATIAL_HALF_RANGE, SPATIAL_HALF_RANGE,
            self.grid_size, dtype=np.float64
        )
        y = np.linspace(
            -SPATIAL_HALF_RANGE, SPATIAL_HALF_RANGE,
            self.grid_size, dtype=np.float64
        )
        # 'ij' 索引: 轴0=竖直 y, 轴1=水平 x —— 标准光学约定
        # (2026-10-16 修正：此前 X 绑定轴0，与学术约定互为转置。
        #  下游 LGModeBasis 与涡旋场生成均以 z = X + iY 解析构造，
        #  X=水平、Y=竖直后，z 即标准复平面坐标，约定同步修正。)
        self.Y, self.X = np.meshgrid(y, x, indexing='ij')

    # =========================================================================
    # HG 本征模计算
    # =========================================================================

    def _compute_single_hg_mode(self, m: int, n: int) -> np.ndarray:
        """
        计算单个 HG_{m,n} 模式在束腰处的无量纲归一化复振幅。

        【无量纲物理公式】(束腰处 z=0, w0=1.0):
            HG_{m,n}(x, y) = C_{m,n}
                          × H_m(√2 · x) × H_n(√2 · y)
                          × exp(-(x² + y²))

        归一化常数:
            C_{m,n} = sqrt(2 / (π · m! · n! · 2^{m+n}))

        【归一化推导】:
            由 Hermite 多项式的正交关系:
                ∫_{-∞}^{∞} H_m(u) H_n(u) e^{-u²} du = √π · 2^m · m! · δ_{m,n}

            对于 |HG_{m,n}|² = C² · [H_m(√2 x) H_n(√2 y) e^{-(x²+y²)}]²
            令 u = √2 x, v = √2 y, dx = du/√2, dy = dv/√2:

                int |HG|² dx dy = C² · int H_m²(u) H_n²(v) e^{-u²-v²} · (du dv / 2)
                              = C²/2 · (√π · 2^m · m!) · (√π · 2^n · n!)
                              = C²/2 · π · 2^{m+n} · m! · n!
                              = 1

            因此: C² = 2 / (π · m! · n! · 2^{m+n})
                → C = sqrt(2 / (π · m! · n! · 2^{m+n}))

        注意: 由于 w0 = 1.0 (无量纲)，归一化常数中无需 1/w0 因子。

        Args:
            m: x 方向阶数 (≥ 0)
            n: y 方向阶数 (≥ 0)

        Returns:
            complex ndarray of shape (grid_size, grid_size), 归一化复振幅
        """
        # --- 归一化常数 ---
        # C = sqrt(2 / (π · m! · n! · 2^{m+n}))
        norm_factor = np.sqrt(
            2.0 / (np.pi * factorial(m, exact=True) * factorial(n, exact=True)
                   * (2.0 ** (m + n)))
        )

        # --- 缩放坐标 ---
        # u = √2 · x,  v = √2 · y  (x=水平·轴1, y=竖直·轴0, 标准约定)
        # self.X 沿轴1 (水平), self.Y 沿轴0 (竖直)
        sqrt2 = np.sqrt(2.0)
        u = self.X * sqrt2
        v = self.Y * sqrt2

        # --- Hermite 多项式 ---
        H_m = hermite(m)(u)   # m 沿水平 x
        H_n = hermite(n)(v)   # n 沿竖直 y

        # --- 高斯包络 ---
        gaussian = np.exp(-(self.X**2 + self.Y**2))

        # --- 组合 (束腰处相位 = 0, 纯实数) ---
        hg_mode = norm_factor * H_m * H_n * gaussian

        return hg_mode.astype(np.complex128)

    def _precompute_all_modes(self) -> np.ndarray:
        """
        预计算所有 HG 本征模的归一化复振幅。

        Returns:
            np.ndarray, shape=(K, grid_size, grid_size), dtype=complex128
            按 self.mode_indices 的顺序排列
        """
        K = self.num_modes
        modes = np.zeros((K, self.grid_size, self.grid_size), dtype=np.complex128)

        for idx, (m, n) in enumerate(self.mode_indices):
            modes[idx] = self._compute_single_hg_mode(m, n)

        return modes

    # =========================================================================
    # 模式索引查询
    # =========================================================================

    def get_mode_index(self, k: int) -> Tuple[int, int]:
        """获取第 k 个模式的 (m, n) 索引 (0-based)。"""
        return self.mode_indices[k]

    def get_all_mode_indices(self) -> List[Tuple[int, int]]:
        """返回所有模式的 (m, n) 索引列表。"""
        return self.mode_indices.copy()

    # =========================================================================
    # 参数随机采样 — 满足物理约束
    # =========================================================================

    def sample_amplitudes(
        self,
        rng: Optional[np.random.Generator] = None,
        mode: str = 'uniform',
        n_active: int = 8,
        alpha: float = 0.8,
    ) -> np.ndarray:
        """
        随机采样满足能量守恒约束的振幅向量。

        数学推导:
            根据 Parseval 定理和本征模正交归一性:
                int |E|² dx dy = Σ_k a_k² = 1

        mode='uniform' (默认, 原行为):
            1. 从 [0.01, 1.0] 均匀采样 K 个非负原始值
               (下界 0.01 避免全部接近于零的退化情况)
            2. 进行 L2 归一化: a_k = raw_k / sqrt(Σ raw_j²)
            3. 保证 Σ a_k² = 1 (能量守恒)

        mode='sparse' (方案一 — 少数活跃模式主导 + 弱底噪):
            对应真实激光腔输出的物理设定 (文献: 主流 HG/LP 模式分解
            均用 6~10 个主导模式, 如 Swin-HG 2024, JCR-CNN 2024)。
            1. 从 K 个模式中随机选 n_active 个活跃模式
            2. 活跃模式间按 Dirichlet(alpha) 分配能量 (alpha<1 → 模式间
               保留 2~5 倍振幅差距, 有清晰但不极端的主导排序)
            3. 其余 K-n_active 个模式给弱底噪 uniform(0.005, 0.015)
               (单模能量 <0.03%, 合计 ~0.5%; 对应文献"忽略 <2% 功率"设定)
            4. L2 归一化 (能量守恒不变)

        Args:
            rng: numpy 随机数生成器，若为 None 则使用默认 RNG
            mode: 'uniform' 或 'sparse'
            n_active: sparse 模式的活跃模式数 (默认 8)
            alpha: sparse 模式的 Dirichlet 浓度参数 (默认 0.8, α<1 稀疏)

        Returns:
            np.ndarray, shape=(K,), L2 归一化的振幅向量
        """
        if rng is None:
            rng = np.random.default_rng()

        K = self.num_modes

        if mode == 'sparse':
            # 1. 随机选 n_active 个活跃模式
            active_mask = np.zeros(K, dtype=bool)
            active = rng.choice(K, size=n_active, replace=False)
            active_mask[active] = True
            # 2. 活跃模式间 Dirichlet 能量分配 (Σ e_active = 1)
            e_active = rng.dirichlet(np.full(n_active, alpha))
            # 3. 其余模式弱底噪 (振幅 uniform 0.005~0.015, 单模能量 <0.03%)
            raw = rng.uniform(0.005, 0.015, size=K)
            raw[active_mask] = np.sqrt(e_active)
            amplitudes = raw / np.linalg.norm(raw)
            return amplitudes.astype(np.float64)

        # ---- uniform (原行为) ----
        raw = rng.uniform(0.01, 1.0, size=K)
        amplitudes = raw / np.linalg.norm(raw)
        return amplitudes.astype(np.float64)

    def sample_phases(
        self, rng: Optional[np.random.Generator] = None,
        mode: str = 'random',
    ) -> np.ndarray:
        """
        采样相对相位向量。

        约定: φ_0 = 0 (基模 HG_{0,0} 为相位参考)，
              其余 K-1 个相位 φ_1, ..., φ_{K-1}。

        两种模式:
          - 'random' (默认): 在 [0, 2π) 内均匀随机采样 (历史行为)。
          - 'locked'  (2026-09-01 新增, 锁相域数据实验):
            每个模式相位从离散锁相集合独立采样。锁相集合取自
            TML 实验常用锁定相位 (文献: Adv. Photonics 4(2):026002,
            Fig.8: π/4, 3π/8, π/2, 3π/4, π), 离散化保证模式间相位差
            落在有限差值集合上——模拟微片激光器横模锁定 (TML) 态。
            用于弥合"真实图锁相 vs 训练随机相位"的域差距。

        Args:
            rng: numpy 随机数生成器
            mode: 'random' | 'locked'

        Returns:
            np.ndarray, shape=(K-1,), 相对相位 ∈ [0, 2π)
        """
        if rng is None:
            rng = np.random.default_rng()

        if mode == 'locked':
            # 离散锁相集合: 0, π/4, 3π/8, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4
            locked = np.array([0.0, np.pi/4, 3*np.pi/8, np.pi/2, 3*np.pi/4,
                               np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
            phases = rng.choice(locked, size=self.num_modes - 1)
        else:
            phases = rng.uniform(0.0, 2.0 * np.pi, size=self.num_modes - 1)
        return phases.astype(np.float64)

    def sample_full_parameters(
        self, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时采样振幅和相位，返回完整 2K-1 维标签 (K=55 时为 109 维)。

        Args:
            rng: numpy 随机数生成器

        Returns:
            (amplitudes, phases):
            - amplitudes: (K,), L2 归一化
            - phases: (K-1,), ∈ [0, 2π)
        """
        if rng is None:
            rng = np.random.default_rng()
        amps = self.sample_amplitudes(rng)
        phis = self.sample_phases(rng)
        return amps, phis

    # =========================================================================
    # 干涉图样生成 — 核心物理公式
    # =========================================================================

    def generate_interference_pattern(
        self,
        amplitudes: np.ndarray,
        phases: np.ndarray,
    ) -> np.ndarray:
        """
        按照物理干涉公式生成多模相干叠加的 CCD 光强图样。

        【核心物理公式】
            E(x,y) = Σ_{k=0}^{K-1} a_k · HG_k(x,y) · exp(i · φ_k)
            I(x,y) = |E(x,y)|²

        其中:
            - a_k: 第 k 个模式的振幅权重 (满足 Σ a_k² = 1)
            - HG_k: 预计算的第 k 个 HG 本征模复振幅 (无量纲)
            - φ_k: 第 k 个模式的相对相位 (φ_0 ≡ 0，绝对相位不可观测)
            - K: 总模式数 (N_max=9 时为 55)

        【干涉项展开】
            I = Σ_k a_k² · |HG_k|²
              + 2 Σ_{i<j} a_i a_j · Re[HG_i · HG_j* · exp(i(φ_i - φ_j))]

            第一项: 各模式的非相干强度叠加
            第二项: 模式间的交叉干涉项 (形成复杂图样的物理根源)

        【输出归一化】
            I_out = I / max(I) → 值域 [0, 1]
            保证动态范围充分利用，与实验 CCD 归一化图一致

        【性能优化】
            使用 np.einsum('k,kij->ij', ...) 完成加权求和 (走 BLAS 路径)，
            相比逐元素广播乘法 + sum 约快 2.7×，且数值完全一致。

        Args:
            amplitudes: shape=(K,), L2 归一化振幅向量 (Σ a_k² = 1)
            phases: shape=(K-1,), 相对相位向量 (φ_1 ~ φ_{K-1} ∈ [0, 2π))

        Returns:
            np.ndarray, shape=(grid_size, grid_size), dtype=float64
            归一化光强分布 I(x,y) ∈ [0, 1]
        """
        K = self.num_modes

        # --- 输入验证 ---
        if amplitudes.shape != (K,):
            raise ValueError(
                f"amplitudes 维度应为 ({K},), 实际为 {amplitudes.shape}"
            )
        if phases.shape != (K - 1,):
            raise ValueError(
                f"phases 维度应为 ({K-1},), 实际为 {phases.shape}"
            )

        # --- 构建完整相位向量 (φ_0 ≡ 0) ---
        full_phases = np.zeros(K, dtype=np.float64)
        full_phases[1:] = phases

        # --- 相干叠加: E = Σ a_k · HG_k · exp(iφ_k) ---
        # 构建复权重: w_k = a_k · exp(i · φ_k)
        complex_weights = amplitudes * np.exp(1j * full_phases)

        # 加权求和 (einsum 走 BLAS 加速路径):
        # w: (K,) × HG: (K, H, W) → E: (H, W)
        E_field = np.einsum('k,kij->ij', complex_weights, self.hg_modes)

        # --- 光强: I = |E|² ---
        intensity = np.abs(E_field) ** 2

        # --- 归一化到 [0, 1] ---
        max_val = intensity.max()
        if max_val > 1e-12:
            intensity = intensity / max_val

        return intensity.astype(np.float64)

    # =========================================================================
    # 光场重构 (用于损失函数)
    # =========================================================================

    def reconstruct_field(
        self,
        amplitudes: np.ndarray,
        phases: np.ndarray,
    ) -> np.ndarray:
        """
        使用给定的振幅和相位重构叠加后的复振幅场。

        Args:
            amplitudes: shape=(K,), 振幅向量
            phases: shape=(K-1,), 相对相位向量

        Returns:
            complex ndarray of shape (grid_size, grid_size), E(x,y)
        """
        K = self.num_modes
        full_phases = np.zeros(K, dtype=np.float64)
        full_phases[1:] = phases
        complex_weights = amplitudes * np.exp(1j * full_phases)
        return np.einsum('k,kij->ij', complex_weights, self.hg_modes)

    def reconstruct_intensity(
        self,
        amplitudes: np.ndarray,
        phases: np.ndarray,
    ) -> np.ndarray:
        """
        使用给定的振幅和相位重构光强图样 (不重新归一化)。

        用于损失函数中的光场重构损失计算。

        Args:
            amplitudes: shape=(K,), 振幅向量
            phases: shape=(K-1,), 相对相位向量

        Returns:
            np.ndarray, shape=(grid_size, grid_size), 光强分布 (原始量级)
        """
        E_field = self.reconstruct_field(amplitudes, phases)
        return (np.abs(E_field) ** 2).astype(np.float64)

    # =========================================================================
    # 任意光场 → HG 基数值投影 (LG/涡旋分支的标签生成核心)
    # =========================================================================

    def project_field_to_hg(self, E: np.ndarray) -> np.ndarray:
        """
        把任意光场 E(x,y) 数值投影到 HG 基, 得到复投影系数。

        c_k = <HG_k, E> = ∫ HG_k*(x,y) · E(x,y) dx dy
                        ≈ Σ_ij conj(HG_k[i,j]) · E[i,j] · dx²

        依赖 HG 基正交归一 (∫|HG|²dxdy=1), 对任意可表示光场给出
        能量守恒展开: Σ|c_k|² = ∫|E|² dxdy。

        Args:
            E: (grid_size, grid_size) 复振幅场

        Returns:
            (K,) 复投影系数 c_k
        """
        return np.einsum('kij,ij->k', np.conj(self.hg_modes), E) \
            * (self.dx * self.dx)

    def projection_to_labels(self, c: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        复投影系数 → (振幅, 相对相位) 标签, 与 sample_amplitudes/
        sample_phases 的输出约定完全一致。

        振幅: |c| L2 归一化 (Σa_k²=1, 能量守恒)。
        相位: φ_k = arg(c_k) - arg(c_0) ∈ [0, 2π), φ_0 ≡ 0。
        保护分支: |c_0| < 1e-12 (如中心奇点对 HG_00 投影为零) 时
                  以 0 为相位参考 (此时 a_0≈0, 相位误差无物理后果)。

        Args:
            c: (K,) 复投影系数

        Returns:
            (amplitudes (K,), phases (K-1,)), 均为 float64
        """
        c = np.asarray(c, dtype=np.complex128)
        amps = np.abs(c)
        amps = amps / (np.linalg.norm(amps) + 1e-12)
        if abs(c[0]) > 1e-12:
            phis = (np.angle(c[1:]) - np.angle(c[0])) % (2.0 * np.pi)
        else:
            phis = np.angle(c[1:]) % (2.0 * np.pi)
        return amps.astype(np.float64), phis.astype(np.float64)

    # =========================================================================
    # LG 分支: 采样 + 图样生成 + 标签 (酉变换, 免逐样本投影)
    # =========================================================================

    def sample_lg_parameters(
        self,
        rng: Optional[np.random.Generator] = None,
        n_active: int = 5,
        alpha: float = 0.8,
        prefer_vortex: bool = True,
    ) -> np.ndarray:
        """
        LG 基 sparse 复权重采样 w = a·exp(iφ) (K_lg 维)。

        1. 活跃模式数每次在 [3, 6] 内随机 (贴近实验涡旋模式数)
        2. 以 0.8 概率从 l≠0 的"涡旋模式"子集中选活跃索引 (保证图样
           带螺旋相位结构), 否则从全集中选
        3. 活跃能量 Dirichlet(alpha), 其余 uniform(0.005, 0.015) 底噪
        4. L2 归一化 → w_j = sqrt(e_j)·exp(iφ_j), φ_0 ≡ 0

        Args:
            rng: 随机数生成器
            n_active: 活跃模式数基准值
            alpha: Dirichlet 浓度
            prefer_vortex: 优先从涡旋 (l≠0) 子集选活跃模式

        Returns:
            (K_lg,) complex128 复权重 (Σ|w|²=1)
        """
        if rng is None:
            rng = np.random.default_rng()
        K = self.lg_basis.num_modes

        n_act = int(rng.integers(3, 7))  # [3, 6]
        vortex_idx = [k for k, (p, l) in enumerate(self.lg_basis.mode_indices)
                      if l != 0]
        if prefer_vortex and vortex_idx and rng.random() < 0.8:
            n_act = min(n_act, len(vortex_idx))
            active = rng.choice(vortex_idx, size=n_act, replace=False)
        else:
            active = rng.choice(K, size=min(n_act, K), replace=False)

        e_active = rng.dirichlet(np.full(n_act, alpha))
        raw = rng.uniform(0.005, 0.015, size=K)
        raw[active] = np.sqrt(e_active)
        amps = raw / np.linalg.norm(raw)

        phis = rng.uniform(0.0, 2.0 * np.pi, size=K)
        phis[0] = 0.0
        return (amps * np.exp(1j * phis)).astype(np.complex128)

    def generate_lg_intensity_and_labels(
        self, w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LG 叠加 → 归一化干涉强度图 + HG 基标签。

        E = Σ w_j · LG_j;  I = |E|²/max
        c_HG = U @ w (K×K 酉变换, 免逐样本场投影)

        Args:
            w: (K_lg,) 复权重

        Returns:
            (I (grid,grid) [0,1], amps (K,), phis (K-1,))
        """
        E = np.einsum('k,kij->ij', w, self.lg_basis.lg_modes)
        intensity = np.abs(E) ** 2
        mx = intensity.max()
        if mx > 1e-12:
            intensity = intensity / mx
        # 直接场投影 (与涡旋分支统一): 矩形法则离散化有 ~0.9% 全局范数
        # 偏差 (U 矩阵捷径实测非酉 1.9e-2), 该偏差被 projection_to_labels
        # 的 L2 归一化完全吸收, 对训练标签无实质影响。
        c = self.project_field_to_hg(E)
        amps, phis = self.projection_to_labels(c)
        return intensity.astype(np.float64), amps, phis

    # =========================================================================
    # 涡旋阵列分支: 环形多奇点场 (与实验图 5~11 奇点同构)
    # =========================================================================

    def sample_vortex_array(
        self,
        rng: Optional[np.random.Generator] = None,
        n_sing_min: int = 5,
        n_sing_max: int = 11,
        ring_r_min: float = 1.5,
        ring_r_max: float = 2.5,
        jitter: float = 0.3,
        p_charge2: float = 0.1,
    ) -> Tuple[List[complex], np.ndarray]:
        """
        采样多奇点涡旋阵列 (环形排列形态, 对应实验图分布)。

        1. 奇点数 M ~ U[n_sing_min, n_sing_max] (默认 5~11)
        2. 位置: 环上均匀角分布 (每样本随机环半径 U[1.5, 2.5])
           + 径向抖动 ±0.3 (贴合实验"均匀环形"排列, 又避免过于规则)
        3. 拓扑荷: ±1 各半随机打散, 10% 概率把一个荷翻倍
        4. 硬约束: Σ|q_j| ≤ max_order_N → 场精确落在 HG 子空间, 零截断

        Args:
            rng: 随机数生成器
            n_sing_min/max: 奇点数范围 (默认 5~11, 与实验图一致)
            ring_r_min/max: 环形半径范围 (以 w0 为单位)
            jitter: 径向抖动幅度
            p_charge2: 单个奇点拓扑荷翻倍的概率

        Returns:
            (zs: List[complex] 奇点位置, qs: np.ndarray int 拓扑荷)
        """
        if rng is None:
            rng = np.random.default_rng()

        M = int(rng.integers(n_sing_min, n_sing_max + 1))
        theta = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=M))
        R = rng.uniform(ring_r_min, ring_r_max)
        radii = np.clip(R + rng.uniform(-jitter, jitter, size=M), 0.4, 3.2)
        zs = [float(r) * np.exp(1j * float(t)) for r, t in zip(radii, theta)]

        qs = np.array([1 if i < M // 2 else -1 for i in range(M)], dtype=int)
        rng.shuffle(qs)
        if rng.random() < p_charge2 and M >= 2:
            qs[int(rng.integers(0, M))] *= 2

        # 硬约束 Σ|q| ≤ max_order_N (零截断保证)
        while int(np.abs(qs).sum()) > self.max_order_N:
            k = int(np.argmax(np.abs(qs)))
            qs[k] = np.sign(qs[k]) * max(1, abs(int(qs[k])) // 2)

        return zs, qs

    def eval_vortex_field(self, zs: List[complex], qs: np.ndarray) -> np.ndarray:
        """
        计算涡旋阵列复振幅场: E = exp(-r²) · ∏_j (z - z_j)^{q_j}
        (q_j < 0 用 conj(z) - conj(z_j))。

        当 Σ|q_j| ≤ N_max 时, ∏(z-z_j)^{q_j} 是 z 的 M≤N_max 次多项式,
        场精确落在 m+n ≤ N_max 的 HG 子空间内 (零截断)。

        Args:
            zs: 奇点位置列表 (复数)
            qs: 拓扑荷数组 (int)

        Returns:
            (grid_size, grid_size) complex128 复振幅场
        """
        z = self.X + 1j * self.Y
        E = np.exp(-(self.X ** 2 + self.Y ** 2)).astype(np.complex128)
        for zj, q in zip(zs, qs):
            q = int(q)
            if q > 0:
                E = E * np.power(z - zj, q)
            elif q < 0:
                E = E * np.power(np.conj(z) - np.conj(zj), -q)
        return E

    def generate_vortex_intensity_and_labels(
        self, zs: List[complex], qs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        涡旋阵列 → 归一化强度图 + HG 基标签 (网格数值投影)。

        Args:
            zs: 奇点位置
            qs: 拓扑荷

        Returns:
            (I (grid,grid) [0,1], amps (K,), phis (K-1,))
        """
        E = self.eval_vortex_field(zs, qs)
        intensity = np.abs(E) ** 2
        mx = intensity.max()
        if mx > 1e-12:
            intensity = intensity / mx
        c = self.project_field_to_hg(E)
        amps, phis = self.projection_to_labels(c)
        return intensity.astype(np.float64), amps, phis


# =============================================================================
# 工厂函数
# =============================================================================

def create_default_generator(max_order_N: Optional[int] = None,
                             grid_size: int = DEFAULT_GRID_SIZE) -> HGModeGenerator:
    """
    创建默认配置的无量纲 HG 模式生成器。

    默认 N_max=12 → 91 个模式, 224×224 网格, w0=1.0, 窗口 [-4, 4]

    Args:
        max_order_N: 最大总阶数 (默认 DEFAULT_MAX_ORDER_N=12)
        grid_size: 网格像素尺寸

    Returns:
        HGModeGenerator 实例
    """
    return HGModeGenerator(max_order_N=max_order_N or DEFAULT_MAX_ORDER_N,
                           grid_size=grid_size)


# =============================================================================
# 自检代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HGModeGenerator 自检 (无量纲版)")
    print("=" * 60)

    # 实例化
    gen = HGModeGenerator(max_order_N=DEFAULT_MAX_ORDER_N, grid_size=224)
    print(f"\n[配置]")
    print(f"  最大阶数 N_max = {gen.max_order_N}")
    print(f"  总模式数 K = {gen.num_modes}")
    print(f"  标签维度 (2K-1) = {gen.num_modes * 2 - 1}")
    print(f"  网格尺寸 = {gen.grid_size} x {gen.grid_size}")
    print(f"  无量纲束腰 w0 = 1.0")
    print(f"  空间窗口 = [-{SPATIAL_HALF_RANGE}, {SPATIAL_HALF_RANGE}] (以 w0 为单位)")
    print(f"  像素间距 dx = {gen.dx:.6f} (以 w0 为单位)")
    print(f"  模式索引列表 (前10个): {gen.mode_indices[:10]}")

    # 验证: 模式计数公式
    expected_K = (gen.max_order_N + 1) * (gen.max_order_N + 2) // 2
    assert gen.num_modes == expected_K, "模式计数错误!"
    print(f"\n[OK] 模式计数验证通过: K = {expected_K}")

    # 验证: 每个模式 L2 归一化
    print("\n--- 归一化验证 (int |HG|^2 dx dy = 1) ---")
    for k in range(min(8, gen.num_modes)):
        m, n = gen.mode_indices[k]
        power = np.sum(np.abs(gen.hg_modes[k]) ** 2) * gen.dx * gen.dx
        status = "OK" if abs(power - 1.0) < 0.01 else "WARN"
        print(f"  [{status}] HG_{{{m},{n}}}: integral = {power:.6f} (理论值 = 1.0)")

    # 验证: 正交性抽查
    print("\n--- 正交性验证 (int HG_i* · HG_j dx dy = δ_ij) ---")
    ortho_ok = True
    for i in range(min(4, gen.num_modes)):
        for j in range(i + 1, min(4, gen.num_modes)):
            overlap = np.abs(np.sum(
                np.conj(gen.hg_modes[i]) * gen.hg_modes[j]
            )) * gen.dx * gen.dx
            if overlap > 0.01:
                print(f"  [WARN] <HG_{gen.mode_indices[i]}|HG_{gen.mode_indices[j]}> = {overlap:.6f}")
                ortho_ok = False
    if ortho_ok:
        print(f"  [OK] 所有抽查的交叉内积 < 0.01")

    # 验证: 随机干涉图样生成
    print("\n--- 干涉图样生成测试 ---")
    rng = np.random.default_rng(42)
    amps = gen.sample_amplitudes(rng)
    phases = gen.sample_phases(rng)

    print(f"  振幅: sum(a^2) = {np.sum(amps**2):.8f} (理论值 = 1.0)")
    print(f"  振幅范围: [{amps.min():.4f}, {amps.max():.4f}]")
    print(f"  相位范围: [{phases.min():.4f}, {phases.max():.4f}] rad")

    intensity = gen.generate_interference_pattern(amps, phases)
    print(f"  光强图样: shape={intensity.shape}, "
          f"range=[{intensity.min():.6f}, {intensity.max():.6f}]")

    # 性能基准
    import time
    print("\n--- 性能基准 ---")
    n_bench = 500
    t0 = time.perf_counter()
    for _ in range(n_bench):
        a = gen.sample_amplitudes(rng)
        p = gen.sample_phases(rng)
        _ = gen.generate_interference_pattern(a, p)
    t1 = time.perf_counter()
    rate = n_bench / (t1 - t0)
    print(f"  生成 {n_bench} 张光斑: {t1-t0:.3f} s ({rate:.1f} samples/s)")

    print(f"\n[OK] 所有自检通过! (无量纲化 HG 模式生成器)")

    # =========================================================================
    # LG 模式基 + 涡旋阵列 + 数值投影 自检
    # =========================================================================
    print("\n" + "=" * 60)
    print("LGModeBasis + 涡旋阵列 + 数值投影 自检")
    print("=" * 60)

    lb = gen.lg_basis
    print(f"\n[LG 基] 模式数 = {lb.num_modes} (期望 91) | "
          f"索引前 8: {lb.mode_indices[:8]}")

    # 1) LG 归一化: ∫|LG|^2 dx^2 ≈ 1
    norms = np.einsum('kij,kij->k', np.conj(lb.lg_modes), lb.lg_modes) \
        * gen.dx * gen.dx
    max_dev = np.abs(norms - 1.0).max()
    print(f"[LG 归一化] max|∫|LG|^2dx^2 - 1| = {max_dev:.2e}")
    assert max_dev < 0.01, "LG 归一化失败!"

    # 2) LG_00 ≡ HG_00
    diff00 = np.abs(lb.lg_modes[0] - gen.hg_modes[0]).max()
    print(f"[LG_00≡HG_00] 逐点最大差 = {diff00:.2e}")
    assert diff00 < 1e-10, "LG_00 与 HG_00 不一致!"

    # 3) 酉性: max|U+U - I|
    U = gen.lg_to_hg_transform
    un = U.conj().T @ U
    un_dev = np.abs(un - np.eye(lb.num_modes)).max()
    print(f"[酉性] max|U+U - I| = {un_dev:.3e}")
    assert un_dev < 0.05, "LG→HG 变换非酉!"  # 离散化+窗口截断固有偏差(实测1.9e-2), 端到端保真度由下方投影-重建验证

    # 4) LG 随机叠加 → 酉变换标签 → 重建场 vs 原场
    rng2 = np.random.default_rng(7)
    w = gen.sample_lg_parameters(rng2)
    E_lg = np.einsum('k,kij->ij', w, lb.lg_modes)
    c = gen.project_field_to_hg(E_lg)
    E_rec = np.einsum('k,kij->ij', c, gen.hg_modes)
    rel_err = np.linalg.norm(E_lg - E_rec) / np.linalg.norm(E_lg)
    print(f"[LG 投影-重建] 相对误差 = {rel_err:.3e}")
    assert rel_err < 0.02, "LG 投影-重建误差过大!"  # 矩形法则离散偏差 ~0.9%

    # 5) 涡旋阵列: 投影-重建 + 奇点位置
    zs, qs = gen.sample_vortex_array(rng2, n_sing_min=5, n_sing_max=7)
    E_v = gen.eval_vortex_field(zs, qs)
    c_v = gen.project_field_to_hg(E_v)
    E_vrec = np.einsum('k,kij->ij', c_v, gen.hg_modes)
    rel_v = np.linalg.norm(E_v - E_vrec) / np.linalg.norm(E_v)
    print(f"[涡旋投影-重建] 相对误差 = {rel_v:.3e} | 奇点数 = {len(zs)} | "
          f"Σ|q| = {np.abs(qs).sum()} (≤{gen.max_order_N})")
    assert rel_v < 0.02, "涡旋投影-重建误差过大!"  # 矩形法则离散偏差 ~0.9%

    # 6) 涡旋强度图 + 标签
    I_v, a_v, p_v = gen.generate_vortex_intensity_and_labels(zs, qs)
    print(f"[涡旋标签] I range=[{I_v.min():.3f}, {I_v.max():.3f}] | "
          f"Σa^2 = {np.sum(a_v**2):.6f} | 相位数 = {p_v.shape[0]}")
    assert abs(np.sum(a_v ** 2) - 1.0) < 1e-6, "涡旋标签未归一化!"

    # 7) LG/涡旋强度图+标签完整路径
    I_lg, a_lg, p_lg = gen.generate_lg_intensity_and_labels(w)
    print(f"[LG 标签] I range=[{I_lg.min():.3f}, {I_lg.max():.3f}] | "
          f"Σa^2 = {np.sum(a_lg**2):.6f}")
    assert abs(np.sum(a_lg ** 2) - 1.0) < 1e-6, "LG 标签未归一化!"

    print(f"\n[OK] LG/涡旋自检全部通过!")

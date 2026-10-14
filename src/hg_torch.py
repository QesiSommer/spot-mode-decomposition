"""
hg_torch.py — PyTorch 可微分 HG 模式引擎
==========================================

本模块是 hg_generator.py 的 PyTorch 移植版本，专门用于训练时的
物理信息损失 (Physics-Informed Loss) 计算。

【核心功能】
  将预计算的 HG 本征模作为固定张量，在 PyTorch 自动微分框架下
  执行相干叠加和光强重构。梯度从重构光强 → 光场 → 参数
  (amplitudes, phases) 全程可微。

【物理信息损失原理】
  Model: I_input → (a_k, φ_k)   [参数回归]
  Physics: (a_k, φ_k) → E_field → I_recon   [可微重建]
  Loss: MSE(I_input, I_recon)   [图像域监督]

  梯度流: I_recon → E_field → φ_k → model.backbone
         解决了"单张光强图无法唯一确定相位"的多模态简并问题
         ——损失在图像域计算，天然处理等价相位配置。

用法:
  engine = TorchHGEngine(max_order_N=9, grid_size=224, device='cuda')
  amps = torch.randn(B, 55).abs()  # 将被 L2 归一化
  phases = torch.randn(B, 54)      # 将被约束到 [0, 2π]
  I_recon = engine(amps, phases)   # (B, 224, 224), 可微

日期: 2026-07-31, 2026-08-03 (升级至 55 阶, m+n ≤ 9)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import hermite, factorial
from typing import Tuple


class TorchHGEngine(nn.Module):
    """
    PyTorch 可微分 HG 模式叠加引擎。

    预计算所有 HG 本征模（不可训练），在 forward 中执行
    可微的相干叠加和光强计算。
    """

    def __init__(
        self,
        max_order_N: int = 9,
        grid_size: int = 224,
        spatial_range: float = 4.0,
        device: str = 'cuda',
    ):
        """
        Args:
            max_order_N: 最大总阶数 (默认 9, m+n ≤ 9 → 55 个 HG 模式,
                         与 55 阶数据集/网络输出严格对齐)
            grid_size: 网格尺寸
            spatial_range: 空间范围 (以 w0 为单位)
            device: 计算设备
        """
        super().__init__()
        self.max_order_N = max_order_N
        self.grid_size = grid_size
        self.spatial_range = spatial_range

        # 生成模式索引
        self.mode_indices = []
        for N in range(max_order_N + 1):
            for m in range(N + 1):
                self.mode_indices.append((m, N - m))
        self.num_modes = len(self.mode_indices)

        # 预计算 HG 模式 (固定权重, 不可训练)
        hg_np = self._compute_all_hg_modes_numpy()
        self.register_buffer(
            'hg_modes',
            torch.tensor(hg_np, dtype=torch.complex64, device=device),
        )
        # 实部和虚部分开存储以加速
        self.register_buffer('hg_real', self.hg_modes.real.float())
        self.register_buffer('hg_imag', self.hg_modes.imag.float())

    def _compute_all_hg_modes_numpy(self) -> np.ndarray:
        """使用 numpy/scipy 预计算所有 HG 模式 (与 hg_generator.py 一致)。"""
        # 坐标网格
        x = np.linspace(-self.spatial_range, self.spatial_range,
                        self.grid_size, dtype=np.float64)
        y = np.linspace(-self.spatial_range, self.spatial_range,
                        self.grid_size, dtype=np.float64)
        X, Y = np.meshgrid(x, y, indexing='ij')

        modes = np.zeros(
            (self.num_modes, self.grid_size, self.grid_size),
            dtype=np.complex64
        )

        for idx, (m, n) in enumerate(self.mode_indices):
            norm = np.sqrt(
                2.0 / (np.pi * factorial(m, exact=True)
                       * factorial(n, exact=True) * (2.0 ** (m + n)))
            )
            sqrt2 = np.sqrt(2.0)
            Hm = hermite(m)(X * sqrt2)
            Hn = hermite(n)(Y * sqrt2)
            gaussian = np.exp(-(X**2 + Y**2))
            modes[idx] = (norm * Hm * Hn * gaussian).astype(np.complex64)

        return modes

    def forward(
        self,
        amplitudes: torch.Tensor,
        phases: torch.Tensor,
    ) -> torch.Tensor:
        """
        可微相干叠加 → 光强重构。

        E = Σ a_k * HG_k * exp(i * φ_k)
        I = |E|²

        Args:
            amplitudes: (B, K), 将被 L2 归一化
            phases: (B, K-1), 相对相位 (φ_0 ≡ 0 自动处理)

        Returns:
            (B, grid_size, grid_size), 重构光强, 可微
        """
        B = amplitudes.shape[0]
        K = self.num_modes

        # ---- L2 归一化振幅 ----
        amps = amplitudes.abs()
        amps = amps / (amps.norm(p=2, dim=1, keepdim=True) + 1e-8)  # (B, K)

        # ---- 构建完整相位 (φ_0 = 0) ----
        full_phases = torch.cat([
            torch.zeros(B, 1, device=phases.device, dtype=phases.dtype),
            phases,
        ], dim=1)  # (B, K)

        # ---- 复振幅叠加 ----
        # E = Σ_k a_k * HG_k * exp(i*φ_k)
        # 使用实部/虚部分离以兼容非复数张量运算

        cos_phi = torch.cos(full_phases)  # (B, K)
        sin_phi = torch.sin(full_phases)  # (B, K)

        # 复权重: w_k = a_k * exp(i*φ_k) = a_k*(cos_φ_k + i*sin_φ_k)
        w_real = amps * cos_phi  # (B, K)
        w_imag = amps * sin_phi  # (B, K)

        # 加权求和 (利用 einsum 进行批量矩阵乘法)
        # HG: (K, H, W), w: (B, K) → E: (B, H, W)
        E_real = torch.einsum('bk,khw->bhw', w_real, self.hg_real)  # (B, H, W)
        E_imag = torch.einsum('bk,khw->bhw', w_imag, self.hg_real)
        # 注意: HG 模式在束腰处为纯实数, 所以虚部贡献也乘 hg_real

        # 光强: I = |E|² = E_real² + E_imag²
        intensity = E_real**2 + E_imag**2  # (B, H, W)

        # ---- 归一化到 [0, 1] (每样本独立) ----
        max_vals = intensity.amax(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        intensity = intensity / (max_vals + 1e-8)

        return intensity


def create_torch_engine(device: str = 'cuda') -> TorchHGEngine:
    """创建默认配置的 PyTorch HG 引擎 (55 阶, 与网络输出对齐)。"""
    return TorchHGEngine(max_order_N=9, grid_size=224, device=device)

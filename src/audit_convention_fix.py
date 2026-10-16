# -*- coding: utf-8 -*-
"""
audit_convention_fix.py — 约定修正独立复核（2026-10-16，审核用）
=====================================================================
从解析标准式从零构造（不 import conv_check_std 等此前自检脚本），
直接核对三处被修改的基生成代码：
  A) hg_torch.TorchHGEngine.hg_real/hg_modes
  B) search_two_mode_lg.build_lg_modes
  C) hg_generator（HGModeBasis + LGModeBasis + 涡旋场）

标准式（Siegman / Saleh & Teich / MATLAB）：
  HG(m,n) = sqrt(2/(π m! n! 2^{m+n})) · H_m(√2 x) · H_n(√2 y) · e^{-(x²+y²)}
            m 沿水平 x（轴1），n 沿竖直 y（轴0）
  LG(p,l) = sqrt(2p!/(π(p+|l|)!)) · (√2 r)^|l| · L_p^|l|(2r²) · e^{-r²}
            · e^{i·l·θ},  θ = atan2(y, x)

判据：复场 z-ratio = field/analytic 应为 |z|=1.000000 且 arg≈0
（或强度 pcc=1 + 相位差恒定）。附加：imshow 视觉判据（光斑行列数）。
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import hermite, genlaguerre, factorial

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SIZE, EXTENT = 224, 4.0
x = np.linspace(-EXTENT, EXTENT, SIZE, dtype=np.float64)
y = np.linspace(-EXTENT, EXTENT, SIZE, dtype=np.float64)
# 数组 (i, j): i 沿轴0=行=竖直 y，j 沿轴1=列=水平 x
Xg, Yg = np.meshgrid(x, y, indexing='xy')   # Xg(i,j)=x_j 水平, Yg(i,j)=y_i 竖直


def hg_std(m, n):
    norm = np.sqrt(2.0 / (np.pi * factorial(m, exact=True)
                          * factorial(n, exact=True) * 2.0 ** (m + n)))
    return norm * hermite(m)(Xg * np.sqrt(2)) \
        * hermite(n)(Yg * np.sqrt(2)) * np.exp(-(Xg ** 2 + Yg ** 2))


def lg_std(p, l):
    R = np.sqrt(Xg ** 2 + Yg ** 2)
    TH = np.arctan2(Yg, Xg)          # y 竖直、x 水平
    C = np.sqrt(2.0 * factorial(p, exact=True)
                / (np.pi * factorial(p + abs(l), exact=True)))
    poly = genlaguerre(p, abs(l))(2 * R ** 2)
    return (C * (np.sqrt(2) * R) ** abs(l) * poly
            * np.exp(-R ** 2) * np.exp(1j * l * TH)).astype(np.complex128)


def zratio(field, ref):
    """z = field/ref，返回 |z| 均值 ± 标准差 与 arg 众数（度）。"""
    mask = np.abs(ref) > 0.05 * np.abs(ref).max()
    z = field[mask] / ref[mask]
    az = np.abs(z)
    arg = np.degrees(np.angle(z))
    arg0 = arg[np.abs(arg) < 170]      # 排除符号翻转混入
    return float(az.mean()), float(az.std()), float(np.median(arg0))


print('=' * 72)
print('A) hg_torch.TorchHGEngine —— 修正后应与标准式复场级一致')
print('=' * 72)
from hg_torch import TorchHGEngine
eng = TorchHGEngine(max_order_N=10, grid_size=SIZE,
                    spatial_range=EXTENT, device='cpu')
fails = 0
for (m, n) in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
               (3, 1), (4, 2), (6, 4), (0, 5), (5, 0)]:
    k = eng.mode_indices.index((m, n))
    F = eng.hg_modes.numpy()[k].astype(np.complex128)
    ref = hg_std(m, n)
    am, asd, amed = zratio(F, ref)
    ok = abs(am - 1) < 1e-5 and asd < 1e-5 and abs(amed) < 0.01
    fails += not ok
    print(f'  HG({m},{n}): |z|={am:.6f}±{asd:.1e}  arg={amed:+.4f}°  '
          f'{"PASS" if ok else "FAIL"}')
# 视觉判据：HG(m,n) 光斑应 (m+1)列 × (n+1)行
def blob_axes(I):
    """返回 (行数, 列数)：亮块沿轴0/轴1的投影峰数。"""
    th = 0.35 * I.max()
    proj_r = (I > th).sum(axis=1)
    proj_c = (I > th).sum(axis=0)
    def peaks(p):
        c = 0; inside = False
        for v in p:
            if v > 0 and not inside: c += 1; inside = True
            elif v == 0: inside = False
        return c
    return peaks(proj_r), peaks(proj_c)

print('  视觉判据 HG(m,n) = (m+1)列 × (n+1)行:')
for (m, n) in [(2, 0), (1, 1), (6, 4), (0, 5)]:
    k = eng.mode_indices.index((m, n))
    I = np.abs(eng.hg_modes.numpy()[k]) ** 2
    nr, nc = blob_axes(I)
    ok = (nr == n + 1) and (nc == m + 1)
    fails += not ok
    print(f'  HG({m},{n}): 行×列 = {nr}×{nc}  期望 {n+1}×{m+1}  '
          f'{"PASS" if ok else "FAIL"}')

print()
print('=' * 72)
print('B) search_two_mode_lg.build_lg_modes —— 修正后应与标准式一致')
print('   (该基无归一化常数——预存在情况，非本次改动；渲染端由')
print('    max|I| 归一吸收。判据：|z| 恒定=形状一致 + arg=0=相位一致)')
print('=' * 72)
from search_two_mode_lg import build_lg_modes
modes_lg, idx_lg = build_lg_modes(max_order=10, size=SIZE,
                                  extent=EXTENT, device='cpu')
modes_lg = modes_lg.numpy()
for (p, l) in [(0, 1), (0, -1), (0, 3), (1, 2), (2, 0), (1, -3),
               (0, 5), (3, 1), (0, -3), (2, 1)]:
    k = idx_lg.index((p, l))
    F = modes_lg[k].astype(np.complex128)
    ref = lg_std(p, l)
    am, asd, amed = zratio(F, ref)
    ok = asd < 1e-5 and abs(amed) < 0.01
    fails += not ok
    print(f'  LG({p},{l}): |z|={am:.4f}±{asd:.1e}(恒定)  '
          f'arg={amed:+.4f}°  {"PASS" if ok else "FAIL"}')
# 螺旋手性判据：LG(0,+1) 相位沿逆时针递增（数学正方向）
k1 = idx_lg.index((0, 1))
F = modes_lg[k1].astype(np.complex128)
r0 = 80
ph = []
for ang_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
    a = np.radians(ang_deg)
    px, py = int(112 + r0 * np.cos(a)), int(112 + r0 * np.sin(a))
    ph.append(np.angle(F[py, px]))
dphi = np.diff(np.unwrap(np.array(ph)))
handedness = np.degrees(dphi.mean())   # 正=逆时针递增=标准 e^{+ilθ}
ok = handedness > 40    # 每 45° 步进应约 +45°
fails += not ok
print(f'  手性判据 LG(0,+1): 相位绕行 {handedness:+.1f}°/45°步 '
      f'{"PASS (逆时针, e^{+ilθ})" if ok else "FAIL"}')

print()
print('=' * 72)
print('C) hg_generator（HGModeBasis/LGModeBasis/涡旋场）')
print('=' * 72)
from hg_generator import HGModeGenerator, LGModeBasis
gen = HGModeGenerator(grid_size=SIZE)
# HG 基（hg_generator 无公开单模 API，走内部 _compute_single_hg_mode）
for (m, n) in [(1, 0), (0, 1), (3, 2), (0, 4), (4, 0), (2, 2)]:
    F = gen._compute_single_hg_mode(m, n)
    ref = hg_std(m, n)
    am, asd, amed = zratio(F.astype(np.complex128), ref)
    ok = abs(am - 1) < 1e-5 and asd < 1e-5 and abs(amed) < 0.01
    fails += not ok
    print(f'  HG({m},{n}): |z|={am:.6f}±{asd:.1e}  arg={amed:+.4f}°  '
          f'{"PASS" if ok else "FAIL"}')
# LG 基（LGModeBasis 用 gen.X/gen.Y）
lg_basis = LGModeBasis(10, gen.X, gen.Y, gen.dx)
for (p, l) in [(0, 1), (0, -1), (0, 3), (1, 2), (2, 1)]:
    k = lg_basis.mode_indices.index((p, l))
    F = lg_basis.lg_modes[k].astype(np.complex128)
    ref = lg_std(p, l)
    am, asd, amed = zratio(F, ref)
    ok = abs(am - 1) < 1e-5 and asd < 1e-5 and abs(amed) < 0.01
    fails += not ok
    print(f'  LG({p},{l}): |z|={am:.6f}±{asd:.1e}  arg={amed:+.4f}°  '
          f'{"PASS" if ok else "FAIL"}')
# 涡旋场：q=+1 奇点放 z0=+1+0i（物理坐标 1.0 → 栅格约 x=140px），
# 必须绕【奇点】采样相位（绕栅格中心是错的），r=10px 小圆逆时针一圈
zs = [1.0 + 0.0j]
qs = np.array([1])
Ev = gen.eval_vortex_field(zs, qs)
# 奇点栅格位置：物理 1.0 单位 = 1.0/(2*4/224) px?  EXTENT=4 → dx = 8/224
dxx = (2 * 4.0) / SIZE
cx = int(112 + 1.0 / dxx)          # z0 实部 → 列
cy = 112                            # z0 虚部 0 → 行
r1 = 10
phv = []
for ang_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
    a = np.radians(ang_deg)
    px = int(round(cx + r1 * np.cos(a)))
    py = int(round(cy + r1 * np.sin(a)))
    phv.append(np.angle(Ev[py, px]))
phv.append(phv[0])                  # 闭合
w = np.degrees(np.sum(np.diff(np.unwrap(np.array(phv)))))
ok = 300 < w < 420    # 逆时针一圈相位应累积 +360°
fails += not ok
print(f'  涡旋场 q=+1 绕奇点({cx},{cy}) 相位累积: {w:.0f}°/360°  '
      f'{"PASS" if ok else "FAIL"}')

print()
print('=' * 72)
print(f'总判据: {"ALL PASS ✓" if fails == 0 else f"{fails} FAIL ✗"}')
print('=' * 72)

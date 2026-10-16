# -*- coding: utf-8 -*-
"""
conv_check_std.py — 约定自检（2026-10-16 轴序修正后）

验证三件事：
  1. 新代码的 HG(m,n) 与解析标准式（m 沿水平 x）复场级一致；
  2. 新代码的 LG(p,l) 与解析标准式（θ=atan2(y,x)）复场级一致；
  3. 新基与旧基的映射关系 = 换算规则的回归基础：
     B_new(HG(m,n)) = B_old(HG(n,m))        (数组同一位置)
     B_new(LG(p,l)) = e^{-i·l·π/2}·B_old(LG(p,-l))
"""
import sys
import numpy as np
from pathlib import Path
from scipy.special import hermite, genlaguerre, factorial

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from search_two_mode_lg import build_lg_modes          # 新代码（已修）
from hg_torch import TorchHGEngine                     # 新代码（已修）
import torch

SIZE, EXTENT = 224, 4.0
ok_all = True


def pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    den = np.sqrt(((a - a.mean()) ** 2).sum() * ((b - b.mean()) ** 2).sum())
    return float(((a - a.mean()) * (b - b.mean())).sum() / den)


def zstat(a, b):
    """a,b 复场：b = e^{iarg}·a 的程度，|z|=1 即只差恒定相位。"""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    z = np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    return z


def std_hg_field(m, n):
    """解析标准式：m 沿水平 x（轴1），n 沿竖直 y（轴0）。"""
    x = np.linspace(-EXTENT, EXTENT, SIZE)
    y = np.linspace(-EXTENT, EXTENT, SIZE)
    Y, X = np.meshgrid(y, x, indexing='ij')
    norm = np.sqrt(2.0 / (np.pi * factorial(m, exact=True)
                          * factorial(n, exact=True)
                          * (2.0 ** (m + n))))
    return (norm * hermite(m)(X * np.sqrt(2)) * hermite(n)(Y * np.sqrt(2))
            * np.exp(-(X ** 2 + Y ** 2))).astype(np.complex128)


def std_lg_field(p, l):
    """解析标准式：θ = atan2(y, x)（y 竖直·轴0，x 水平·轴1）。"""
    x = np.linspace(-EXTENT, EXTENT, SIZE)
    y = np.linspace(-EXTENT, EXTENT, SIZE)
    Y, X = np.meshgrid(y, x, indexing='ij')
    R = np.sqrt(X ** 2 + Y ** 2)
    TH = np.arctan2(Y, X)
    return (genlaguerre(p, abs(l))(2 * R ** 2)
            * (np.sqrt(2) * R) ** abs(l) * np.exp(-R ** 2)
            * np.exp(1j * l * TH)).astype(np.complex128)


print('==== 1. 新代码 HG vs 解析标准式（复场级） ====')
# max_order_N=10：覆盖 HG(6,4)/(3,4)/(4,3) 等 m+n=10 的交付组合
eng = TorchHGEngine(max_order_N=10, grid_size=SIZE, device='cpu')
lab_hg = {f'HG({m},{n})': k for k, (m, n) in enumerate(eng.mode_indices)}
for (m, n) in [(0, 0), (0, 3), (3, 0), (1, 1), (6, 4), (4, 6), (3, 3),
               (7, 1), (1, 7), (2, 0), (0, 2)]:
    F = eng.hg_modes[lab_hg[f'HG({m},{n})']].numpy().astype(np.complex128)
    S = std_hg_field(m, n)
    z = zstat(F, S)
    ok = abs(abs(z) - 1) < 1e-3 and abs(z - 1) < 1e-3
    ok_all &= ok
    print(f'  HG({m},{n}): z={z:+.6f} |z|={abs(z):.6f} '
          f'{"PASS" if ok else "FAIL"}')

print('\n==== 2. 新代码 LG vs 解析标准式（复场级） ====')
lg_modes, lg_idx = build_lg_modes(10, SIZE, EXTENT, 'cpu')
lg_lab = {f'LG({p},{l})': k for k, (p, l) in enumerate(lg_idx)}
for (p, l) in [(0, 1), (0, -1), (0, 3), (0, -3), (0, -4), (1, -1),
               (2, -1), (1, -3), (2, 0), (0, 4), (0, 5)]:
    F = lg_modes[lg_lab[f'LG({p},{l})']].numpy().astype(np.complex128)
    S = std_lg_field(p, l)
    z = zstat(F, S)
    ok = abs(z - 1) < 1e-3
    ok_all &= ok
    print(f'  LG({p},{l}): z={z:+.6f} {"PASS" if ok else "FAIL"}')

print('\n==== 3. 新基 vs 旧基映射（换算规则回归基础） ====')
# 旧基 = 修正前的实现（这里直接内联旧代码，避免再维护一份文件）


def old_hg_field(m, n):
    """修正前实现：X 绑定 x 且 ij 索引 → m 沿轴0。"""
    x = np.linspace(-EXTENT, EXTENT, SIZE)
    y = np.linspace(-EXTENT, EXTENT, SIZE)
    X, Y = np.meshgrid(x, y, indexing='ij')
    norm = np.sqrt(2.0 / (np.pi * factorial(m, exact=True)
                          * factorial(n, exact=True)
                          * (2.0 ** (m + n))))
    return (norm * hermite(m)(X * np.sqrt(2)) * hermite(n)(Y * np.sqrt(2))
            * np.exp(-(X ** 2 + Y ** 2))).astype(np.complex128)


def old_lg_field(p, l):
    """修正前实现：X 沿轴0 → TH 与标准方位角转置。"""
    xs = np.linspace(-EXTENT, EXTENT, SIZE)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    R = np.sqrt(X ** 2 + Y ** 2)
    TH = np.arctan2(Y, X)
    return (genlaguerre(p, abs(l))(2 * R ** 2)
            * (np.sqrt(2) * R) ** abs(l) * np.exp(-R ** 2)
            * np.exp(1j * l * TH)).astype(np.complex128)


print('  [HG] B_new(m,n) ?= B_old(n,m)')
for (m, n) in [(0, 3), (3, 1), (1, 3), (3, 0), (6, 4), (4, 6), (7, 1),
               (1, 7), (5, 1), (1, 5), (2, 3), (3, 2), (2, 0), (0, 2),
               (4, 0), (0, 4), (3, 3), (1, 1)]:
    Fn = eng.hg_modes[lab_hg[f'HG({m},{n})']].numpy().astype(np.complex128)
    Fo = old_hg_field(n, m)
    z = zstat(Fn, Fo)
    ok = abs(z - 1) < 1e-3
    ok_all &= ok
    print(f'  HG({m},{n})~old HG({n},{m}): z={z:+.6f} '
          f'{"PASS" if ok else "FAIL"}')

print('  [LG] B_new(p,l) ?= e^{+i·l·π/2}·B_old(p,-l)')
for (p, l) in [(0, -3), (0, -4), (1, -1), (2, -1), (1, -3), (0, 3),
               (0, 1), (2, 0)]:
    Fn = lg_modes[lg_lab[f'LG({p},{l})']].numpy().astype(np.complex128)
    Fo = old_lg_field(p, -l)
    z = zstat(Fn, Fo * np.exp(1j * l * np.pi / 2))
    ok = abs(z - 1) < 1e-3
    ok_all &= ok
    print(f'  LG({p},{l})~old LG({p},{-l}): z={z:+.6f} '
          f'{"PASS" if ok else "FAIL"}')

print('\n==== 4. 额外：旋转不变性（l→-l 强度相同）抽查 ====')
for (p, l) in [(0, 3), (1, 1), (2, 1)]:
    a = lg_modes[lg_lab[f'LG({p},{l})']].numpy()
    b = lg_modes[lg_lab[f'LG({p},{-l})']].numpy()
    c = pcc(np.abs(a) ** 2, np.abs(b) ** 2)
    print(f'  |LG({p},{l})|² vs |LG({p},{-l})|²: pcc={c:+.4f}')

print(f'\n{"=" * 60}\n约定自检总判定: {"ALL PASS" if ok_all else "FAIL"}')
sys.exit(0 if ok_all else 1)

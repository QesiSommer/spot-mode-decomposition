# -*- coding: utf-8 -*-
r"""
shijie_style_matlab_plot.py — 师姐式 MATLAB 复现（2026-10-17）
====================================================================
像师姐那样，**只提取八图分解结果的发布参数**（模式标签、能量占比、
相对相位 —— 标准记号，源自 10.16汇报/overview_data.json），用
MATLAB 标准解析式独立构造每个模式并纯叠加，与实验图对比：

  HG(m,n) = sqrt(2/(π·m!·n!·2^{m+n}))·H_m(√2x)·H_n(√2y)·e^{-(x²+y²)}
            m = 水平 x 阶数（标准约定）
  LG(p,l) = (√2r)^|l|·L_p^|l|(2r²)·e^{-r²}·e^{i·l·θ}, θ=atan2(y,x)
            （交付文档所载公式，与参数拟合口径一致）

拟合方式与师姐相同：纯叠加后仅拟合**全局尺度 s + 整体平移**，
不含逐模 sx/sy/rot（那些不在发布参数表里）。

不修改任何既有文件；输出写入新文件夹：
  归档_八图分解\师姐式MATLAB复现_2026-10-17\
"""
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import hermite, genlaguerre, factorial
from scipy.ndimage import shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from window_metrics import make_windows, _support_mask

OUT = ROOT / '归档_八图分解' / '师姐式MATLAB复现_2026-10-17'
(OUT / 'exp_processed').mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
SIZE, EXTENT = 224, 4.0

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---- MATLAB 标准坐标：x 水平(轴1)，y 竖直(轴0) ----
x = np.linspace(-EXTENT, EXTENT, SIZE)
y = np.linspace(-EXTENT, EXTENT, SIZE)
Y, X = np.meshgrid(y, x, indexing='ij')
R = np.sqrt(X ** 2 + Y ** 2)
TH = np.arctan2(Y, X)


def hg(m, n):
    norm = np.sqrt(2.0 / (np.pi * factorial(m, exact=True)
                          * factorial(n, exact=True) * 2.0 ** (m + n)))
    return (norm * hermite(m)(X * np.sqrt(2)) * hermite(n)(Y * np.sqrt(2))
            * np.exp(-(X ** 2 + Y ** 2))).astype(np.complex128)


def lg(p, l):
    # 交付文档所载公式（未含归一化常数，与参数拟合口径一致）
    return ((np.sqrt(2) * R) ** abs(l) * genlaguerre(p, abs(l))(2 * R ** 2)
            * np.exp(-R ** 2) * np.exp(1j * l * TH)).astype(np.complex128)


def field_of(label):
    fam = label[:2]
    a, b = (int(v) for v in label[3:-1].split(','))
    return hg(a, b) if fam == 'HG' else lg(a, b)


def pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel() - np.mean(a)
    b = np.asarray(b, dtype=np.float64).ravel() - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else -1.0


def blur2(I):
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy()


def C_s02(Ie, Ie_b, I):
    return 0.5 * (pcc(Ie, I) + pcc(Ie_b, blur2(I)))


def zoom_center(I, s):
    """强度域整体缩放（等价于对复场做 rescale 后取模方，中心对齐）。"""
    if abs(s - 1.0) < 1e-9:
        return I
    H = I.shape[0]
    n = int(round(H * s))
    t = torch.tensor(I, dtype=torch.float32)[None, None]
    z = torch.nn.functional.interpolate(t, size=(n, n), mode='bilinear',
                                        align_corners=False)[0, 0].numpy()
    out = np.zeros((H, H))
    if n >= H:
        c = (n - H) // 2
        out = z[c:c + H, c:c + H]
    else:
        c = (H - n) // 2
        out[c:c + n, c:c + n] = z
    return out


def fit_scale_shift(I0, Ie, Ie_b):
    """师姐式拟合：全局尺度 s + 整体平移 (dy,dx)，C_s02 最大化。"""
    best = None
    for s in np.arange(0.70, 1.451, 0.05):
        I1 = zoom_center(I0, float(s))
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                I2 = ndshift(I1, (dy, dx), order=1, mode='nearest')
                c = C_s02(Ie, Ie_b, I2)
                if best is None or c > best[0]:
                    best = (c, float(s), float(dy), float(dx))
    _, s0, dy0, dx0 = best
    for s in np.arange(s0 - 0.04, s0 + 0.041, 0.01):
        I1 = zoom_center(I0, float(s))
        for dy in np.arange(dy0 - 1, dy0 + 1.01, 0.5):
            for dx in np.arange(dx0 - 1, dx0 + 1.01, 0.5):
                I2 = ndshift(I1, (dy, dx), order=1, mode='nearest')
                c = C_s02(Ie, Ie_b, I2)
                if c > best[0]:
                    best = (c, float(s), float(dy), float(dx))
    return best


data = json.load(open(ROOT / '10.16汇报' / 'overview_data.json',
                      encoding='utf-8'))

rows = []
for d in data:
    cn, name = d['cn'], d['en']
    # 实验图（用 cn → 文件名映射）
    NAME = {'4奇点': '4Singularities', '5奇点': '5Singularities',
            '6奇点': '6Singularities', '8奇点1': '8Singularities(1)',
            '8奇点2': '8Singularities(2)', '9奇点': '9Singularities',
            '10奇点': '10Singularities', '11奇点': '11Singularities'}[cn]
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    Ie_b = blur2(Ie)

    # ---- 纯参数叠加（MATLAB 标准式） ----
    E = np.zeros((SIZE, SIZE), dtype=np.complex128)
    for m in d['modes']:
        a = np.sqrt(m['energy'] / 100.0)
        ph = np.deg2rad(m['phase'])
        E = E + a * np.exp(1j * ph) * field_of(m['name'])
    I0 = np.abs(E) ** 2
    I0 = I0 / max(I0.max(), 1e-30)
    C_nofit = C_s02(Ie, Ie_b, I0)

    # ---- 师姐式 全局尺度+平移 拟合 ----
    c_fit, s, dy, dx = fit_scale_shift(I0, Ie, Ie_b)
    I_fit = np.clip(ndshift(zoom_center(I0, s), (dy, dx), order=1,
                            mode='nearest'), 0, None)

    # ---- 56px 窗热图 ----
    I_fit_b = blur2(I_fit)
    mask = _support_mask(Ie)
    windows, _ = make_windows(224, 56)
    valid = [sl for sl in windows if mask[sl].mean() >= 0.5]
    hm = np.full((4, 4), np.nan)
    for k, sl in enumerate(valid):
        hm[k // 4, k % 4] = pcc(Ie_b[sl], I_fit_b[sl])

    # 导出实验处理图（供 MATLAB 脚本对照）
    plt.imsave(OUT / 'exp_processed' / f'exp_{name}.png',
               Ie, cmap='gray', vmin=0, vmax=1)

    rows.append({'cn': cn, 'name': name, 'N': d['N'],
                 'combo': ' + '.join(m['name'] for m in d['modes']),
                 'C_nofit': C_nofit, 'C_fit': c_fit,
                 's': s, 'dy': dy, 'dx': dx,
                 'C_delivered': d['C_s02'], 'W_avg_delivered': d['W_avg'],
                 'hm': hm, 'Ie': Ie, 'I_fit': I_fit})
    print(f'[{cn}] 纯参数(无拟合) C={C_nofit:.4f} | 拟合后 C={c_fit:.4f} '
          f'(s={s:.2f}, shift=({dy:+.1f},{dx:+.1f})) | '
          f'交付(含逐模自由度) C={d["C_s02"]:.4f}')

# ---- 汇总图：8 行 × 3 列 ----
fig = plt.figure(figsize=(12.5, 20))
gs = fig.add_gridspec(8, 3, width_ratios=[1, 1, 1], hspace=0.32,
                      wspace=0.08)
for i, r in enumerate(rows):
    ax = fig.add_subplot(gs[i, 0])
    ax.imshow(r['Ie'], cmap='inferno', vmin=0, vmax=1)
    if i == 0:
        ax.set_title('实验图（预处理后）', fontsize=12)
    ax.axis('off')
    ax = fig.add_subplot(gs[i, 1])
    ax.imshow(r['I_fit'], cmap='inferno', vmin=0, vmax=1)
    if i == 0:
        ax.set_title('纯参数 MATLAB 式拟合', fontsize=12)
    ax.set_title(f"{r['name']}  C_s02={r['C_fit']:.4f}\n"
                 f"s={r['s']:.2f}  shift=({r['dy']:+.1f},{r['dx']:+.1f})",
                 fontsize=9.5)
    ax.axis('off')
    ax = fig.add_subplot(gs[i, 2])
    cmap = matplotlib.colormaps['RdYlGn'].copy()
    cmap.set_bad('0.75')
    ax.imshow(r['hm'], cmap=cmap, vmin=0.4, vmax=1.0)
    for a in range(4):
        for b in range(4):
            if not np.isnan(r['hm'][a, b]):
                ax.text(b, a, f'{r["hm"][a, b]:.2f}', ha='center',
                        va='center', fontsize=8)
    if i == 0:
        ax.set_title('区域相关系数（56px）', fontsize=12)
    ax.axis('off')
fig.suptitle('师姐式纯参数复现（能量占比+相对相位 → 标准公式叠加，'
             '仅拟合全局尺度/平移；不含逐模 sx/sy/rot 自由度）\n'
             '记号：HG(m,n) m=水平；LG 按交付文档公式',
             fontsize=13, y=0.995)
fig.savefig(OUT / '八图_师姐式纯参数拟合.png', dpi=115,
            bbox_inches='tight')
plt.close(fig)

# ---- 汇总 JSON ----
out_rows = [{k: v for k, v in r.items() if k not in ('Ie', 'I_fit', 'hm')}
            for r in rows]
json.dump(out_rows, open(OUT / '复现汇总.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

print('\n==== 汇总（纯参数复现 vs 交付） ====')
print(f'{"图":8s} {"纯参数C":>8s} {"拟合后C":>8s} {"交付C":>8s} {"差":>7s}')
for r in rows:
    print(f'{r["cn"]:8s} {r["C_nofit"]:8.4f} {r["C_fit"]:8.4f} '
          f'{r["C_delivered"]:8.4f} {r["C_fit"]-r["C_delivered"]:+7.4f}')
print(f'\nDONE -> {OUT}')

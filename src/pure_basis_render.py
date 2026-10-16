# -*- coding: utf-8 -*-
r"""
pure_basis_render.py — 纯基候选渲染出图（第3层前置）
======================================================
对搜索 JSON 中的冠军解做五联图 + 参数明细图（交付口径渲染）。
用法: python pure_basis_render.py <cn> <track>
例:   python pure_basis_render.py 4奇点 HG
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
from scipy.ndimage import shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'closed_routes'))  # morph_search_lib 所在目录
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
import morph_search_lib as L

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

CN = sys.argv[1]
TRACK = sys.argv[2].upper()
NAME = {'4奇点': '4Singularities', '8奇点1': '8Singularities(1)',
       '10奇点': '10Singularities'}[CN]
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
SIZE = 224

d = json.load(open(ROOT / '归档_八图分解' / '纯基调整_2026-10-16'
                   / '第2层_搜索' / CN / '搜索结果.json',
                   encoding='utf-8'))
pool = d[TRACK]['best_struct'] or d[TRACK]['best_any']
b = pool[0]
combo = b['combo']
theta = b['theta']
sx, sy, rot, shiftv = b['sx'], b['sy'], b['rot'], b['shift']

modes, labels = build_joint_modes(10, SIZE, 4.0, 'cpu')
modes_np = modes.numpy()
lab2k = {l: k for k, l in enumerate(labels)}
fields = [modes_np[lab2k[m]] for m in combo]

# 交付口径渲染（zoom_axis_c 同径）
def render_dof(fields, theta, sx, sy, shiftv, rot):
    E = np.zeros((SIZE, SIZE), dtype=np.complex128)
    n = len(fields)
    for i, Fm in enumerate(fields):
        if abs(sx[i] - 1.0) > 1e-9:
            Fm = L.zoom_axis_c(Fm, sx[i], 1)
        if abs(sy[i] - 1.0) > 1e-9:
            Fm = L.zoom_axis_c(Fm, sy[i], 0)
        if abs(rot[i]) > 1e-9:
            Fm = L.ndrotate(Fm.real, rot[i], reshape=False, order=1) \
                + 1j * L.ndrotate(Fm.imag, rot[i], reshape=False,
                                  order=1)
        phi = 0.0 if i == 0 else theta[n + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-30)
    I = ndshift(I, shiftv, order=1, mode='nearest')
    return np.clip(I, 0, None)

I_model = render_dof(fields, theta, sx, sy, shiftv, rot)
Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
Ie_b = gaussian_blur(torch.tensor(Ie, dtype=torch.float32)
                     .unsqueeze(0), 2.0)[0].numpy()


def pcc(a, b):
    a = np.asarray(a).ravel() - np.asarray(a).mean()
    b = np.asarray(b).ravel() - np.asarray(b).mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else -1.0


I_b = gaussian_blur(torch.tensor(I_model, dtype=torch.float32)
                    .unsqueeze(0), 2.0)[0].numpy()
C = 0.5 * (pcc(Ie, I_model) + pcc(Ie_b, I_b))
pts_m = find_dark_points(I_model)
pts_e = find_dark_points(Ie)
mt = match_points(pts_e, pts_m)
print(f'{CN} 纯{TRACK} 冠军 {"+".join(combo)}')
print(f'  C_s02 = {C:.4f} (搜索记录 {b["C_s02"]:.4f})')
print(f'  cores {len(pts_m)} vs {len(pts_e)}; mean_nn '
      f'{mt["mean"]:.2f}; max {mt["max"]:.2f}')

# ---- 五联图 ----
OUT = ROOT / '归档_八图分解' / '纯基调整_2026-10-16' / '第3层_裁决'
OUT.mkdir(parents=True, exist_ok=True)
raw = plt.imread(f'{IMG}/{NAME}.png')
if raw.ndim == 3 and raw.shape[2] == 4:
    raw = raw[:, :, :3]


def window_pcc_map(I, W=56):
    from scipy.ndimage import binary_dilation
    sup = binary_dilation(Ie > 0.05 * Ie.max(), iterations=8)
    side = SIZE // W
    hm = np.full((side, side), np.nan)
    for i in range(side):
        for j in range(side):
            sl = np.s_[i * W:(i + 1) * W, j * W:(j + 1) * W]
            if sup[sl].any():
                hm[i, j] = pcc(Ie_b[sl], I_b[sl])
    return hm


hm = window_pcc_map(I_model)
fig, axs = plt.subplots(1, 5, figsize=(21.5, 4.4))
axs[0].imshow(raw)
axs[0].set_title('实验原图')
axs[1].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
axs[1].set_title('实验（处理口径）')
axs[2].imshow(I_model, cmap='inferno', vmin=0, vmax=1)
axs[2].set_title(f'纯{TRACK} 拟合\n{"+".join(combo)}\nC_s02={C:.4f}')
axs[3].imshow(np.abs(Ie - I_model), cmap='viridis', vmin=0, vmax=0.6)
axs[3].set_title('|diff|（固定标尺 0~0.6）')
cmap = matplotlib.colormaps['RdYlGn'].copy()
cmap.set_bad('0.7')
axs[4].imshow(hm, cmap=cmap, vmin=0.3, vmax=1.0)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        if not np.isnan(hm[i, j]):
            axs[4].text(j, i, f'{hm[i, j]:.2f}', ha='center',
                        va='center', fontsize=9)
axs[4].set_title('56px 窗 PCC（σ=2）')
for a in axs:
    a.axis('off')
fig.suptitle(f'{NAME}（{CN}）纯{TRACK}基候选 — 核数 '
             f'{len(pts_m)} vs {len(pts_e)}', fontsize=13)
fig.tight_layout()
fig.savefig(OUT / f'{CN}_纯{TRACK}_五联.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)

# ---- 参数明细 ----
N = len(combo)
fig2, axs2 = plt.subplots(1, 2, figsize=(12.5, 4.4))
ener = [u * 100 for u in theta[:N]]
amps = [np.sqrt(max(u, 0)) for u in theta[:N]]
bars = axs2[0].bar(range(N), ener, color='#4C72B0')
axs2[0].set_xticks(range(N))
axs2[0].set_xticklabels(combo, rotation=12)
axs2[0].set_ylabel('能量占比 %')
axs2[0].set_title(f'能量占比（Σ={sum(ener):.1f}%）')
for bb, e, a in zip(bars, ener, amps):
    axs2[0].text(bb.get_x() + bb.get_width() / 2,
                 bb.get_height() + 0.6, f'{e:.2f}%\n(a={a:.4f})',
                 ha='center', fontsize=9)
phs = [0.0] + list(theta[N:])
axs2[1].bar(range(N), [np.degrees(p) for p in phs], color='#DD8452')
axs2[1].set_xticks(range(N))
axs2[1].set_xticklabels(combo, rotation=12)
axs2[1].set_ylabel('相对相位 °')
axs2[1].set_title('相对相位（模式 1 = 0）')
for i, p in enumerate(phs):
    axs2[1].text(i, np.degrees(p) + (3 if p >= 0 else -10),
                 f'{np.degrees(p):+.1f}°', ha='center', fontsize=9)
fig2.suptitle(f'{CN} 纯{TRACK}候选参数明细 '
              f'（sx={sx} sy={sy} rot={rot} shift={shiftv}）',
              fontsize=11)
fig2.tight_layout()
fig2.savefig(OUT / f'{CN}_纯{TRACK}_参数明细.png', dpi=140,
             bbox_inches='tight')
plt.close(fig2)
print(f'[fig] {OUT}\\{CN}_纯{TRACK}_五联.png / 参数明细.png')

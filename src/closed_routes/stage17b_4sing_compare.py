# -*- coding: utf-8 -*-
"""
stage17b_4sing_compare.py -- 4奇点：各方案核数与视觉对比图
============================================================
对比四种解：
  Sep26 原解（4vs4, C_s02 0.8880）
  v3 交付解（5vs4, 0.9163）← 用户指出问题的那个
  v4 修正解（4vs4, 0.8872，LG 冻结）
  v17 冠军（4vs4, 0.8964，新组合）
每图标注核数与 C_s02，直观呈现"核数-分数"的取舍。
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points
from stage0_diagnosis import parse_md
import morph_search_lib as L

OUT = ROOT / '归档_八图分解' / '4奇点_核数修正'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '4Singularities'

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
mnp = modes.numpy()

Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
N_EXP = len(find_dark_points(Ie))

# ---- 四个解 ----
blur = L.blur2
sols = []

# 1) Sep-26
c26, t26, _ = parse_md(ROOT / '归档_八图分解' / '最新结果'
                       / '4奇点_参数与指标.md')
I26 = L.render_dof(mnp, lab2k, c26, t26, [1.0] * 3, [1.0] * 3,
                   [0.0, 0.0], [0.0] * 3)
sols.append(('Sep-26 原解 (4vs4, C=0.8880)', I26, c26))

# 2) v3 交付解（有问题的那个）
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))
a3 = V3['4奇点']['accepted']
Iv3 = L.render_dof(mnp, lab2k, V3['4奇点']['combo'],
                   [float(x) for x in a3['theta']],
                   [float(x) for x in a3['sx']],
                   [float(x) for x in a3['sy']],
                   [float(x) for x in a3['shift']],
                   [float(x) for x in a3['rot']])
sols.append(('v3 交付解 (5vs4, C=0.9163) ← 问题解', Iv3, V3['4奇点']['combo']))

# 3) v4 修正解
V4 = json.load(open(ROOT / '归档_八图分解' / '终检精修v4_2026-10-14'
                    / 'terminal_inspection_v4.json', encoding='utf-8'))
a4 = V4['4奇点']['refined']
Iv4 = L.render_dof(mnp, lab2k, V4['4奇点']['combo'],
                   [float(x) for x in a4['theta']],
                   [float(x) for x in a4['sx']],
                   [float(x) for x in a4['sy']],
                   [float(x) for x in a4['shift']],
                   [float(x) for x in a4['rot']])
sols.append(('v4 修正 (4vs4, LG冻结, C=0.8872)', Iv4, V4['4奇点']['combo']))

# 4) stage17 冠军
S17 = json.load(open(OUT / 'stage17_corelock.json', encoding='utf-8'))
ch = S17['champion']
Ich = L.render_dof(mnp, lab2k, ch['combo'],
                   [float(x) for x in ch['theta']],
                   [float(x) for x in ch['sx']],
                   [float(x) for x in ch['sy']],
                   [float(x) for x in ch['shift']],
                   [float(x) for x in ch['rot']])
sols.append((f'v17 冠军 (4vs4, C={ch["C_s02"]:.4f})', Ich, ch['combo']))

# ---- 图 ----
fig, axs = plt.subplots(1, len(sols) + 1, figsize=(4.2 * (len(sols) + 1),
                                                   4.8))
axs[0].imshow(blur(Ie), cmap='inferno', vmin=0, vmax=1)
axs[0].set_title(f'EXPERIMENT\n{N_EXP} dark cores', fontsize=10)
for y, x, _ in find_dark_points(Ie):
    axs[0].plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.6)
axs[0].axis('off')

for k, (tag, I, combo) in enumerate(sols):
    ax = axs[k + 1]
    ax.imshow(blur(I), cmap='inferno', vmin=0, vmax=1)
    pts = find_dark_points(I)
    n_ok = len(pts) == N_EXP
    col = 'lime' if n_ok else 'red'
    for y, x, _ in pts:
        ax.plot(x, y, 'o', mfc='none', mec=col, ms=11, mew=1.6)
    ax.set_title(f'{tag}\nmodel cores = {len(pts)} '
                 f'{"(MATCH)" if n_ok else "(MISMATCH)"}\n'
                 f'{"+".join(combo)}', fontsize=9,
                 color=('black' if n_ok else 'darkred'))
    ax.axis('off')

fig.suptitle('4Singularities: core-count vs score trade-off '
             '(cyan/lime = matched, red = mismatch)', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / '4奇点_核数对比.png', dpi=135, bbox_inches='tight')
plt.close(fig)
print(f'[done] {OUT / "4奇点_核数对比.png"}')

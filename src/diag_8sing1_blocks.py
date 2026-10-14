# -*- coding: utf-8 -*-
"""
diag_8sing1_blocks.py -- 8(1)"奇点连成一片"的针对性诊断
========================================================

Quantify the "connected vs isolated" phenomenon via dark-region
connected-component analysis (10% threshold), in the gamma domain:
  experiment vs current model.
Also produce a zoomed side-by-side + mask overlay figure.
"""
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points
from calibrate_metrics import render_model
from stage0_diagnosis import parse_md

SRC = ROOT / '归档_八图分解' / '最新结果_新指标'
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()

for name, cn, gamma in [('8Singularities(1)', '8奇点(1)', 0.58),
                        ('5Singularities', '5奇点', 0.70)]:
    combo, params, scale = parse_md(SRC / f'{cn}_参数与指标.md')
    Ie = load_experiment(f'{IMG_DIR}/{name}.png', 'nipy_spectral',
                         'isot')
    I0 = render_model(modes_np, lab2k,
                      {'combo': combo, 'params': params, 'scale': scale})
    Ig = np.clip(gaussian_blur(
        torch.tensor(I0, dtype=torch.float32).unsqueeze(0),
        2.5)[0].numpy(), 0, 1) ** gamma
    Ig = Ig / Ig.max()

    print(f'\n===== {cn} =====')

    def blocks(I, thr=0.10, min_size=8):
        m = I < thr * I.max()
        lab, n = label(m)
        sizes = []
        pos = []
        for k in range(1, n + 1):
            sz = int((lab == k).sum())
            if sz >= min_size:
                ys, xs = np.where(lab == k)
                sizes.append(sz)
                pos.append((int(ys.mean()), int(xs.mean())))
        order = np.argsort(sizes)[::-1]
        return [sizes[i] for i in order], [pos[i] for i in order]

    se, pe = blocks(Ie)
    sm, pm = blocks(Ig)
    print(f'  experiment: {len(se)} dark blocks (>=8px), '
          f'sizes={se[:12]}')
    print(f'  model     : {len(sm)} dark blocks (>=8px), '
          f'sizes={sm[:12]}')

    # core depth & saddle for reference
    from stage0_diagnosis import core_depth, saddle_isolation
    pts_e = find_dark_points(Ie)
    pts_m = find_dark_points(Ig)
    print(f'  cores: exp={len(pts_e)} model={len(pts_m)}')

    # figure: zoom + masks
    if cn.startswith('8'):
        c0, c1 = 40, 190
    else:
        c0, c1 = 60, 164
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
    axs[0].imshow(Ie[c0:c1, c0:c1], cmap='inferno', vmin=0, vmax=1)
    axs[0].set_title('experiment (zoom)')
    axs[1].imshow(Ig[c0:c1, c0:c1], cmap='inferno', vmin=0, vmax=1)
    axs[1].set_title('model gamma (zoom)')
    # mask overlay: exp dark = red, model dark = blue, overlap = purple
    from matplotlib.colors import ListedColormap
    me = Ie < 0.10 * Ie.max()
    mm = Ig < 0.10 * Ig.max()
    ov = np.zeros((*Ie.shape, 3))
    ov[..., 0] = me
    ov[..., 2] = mm
    axs[2].imshow(ov[c0:c1, c0:c1])
    axs[2].set_title('dark masks: red=exp, blue=model')
    # radial depth profile: along the ring
    axs[3].plot(sorted([s for _, s in
                        saddle_isolation(Ie, pts_e)]), 'o-',
                label='experiment saddles')
    axs[3].plot(sorted([s for _, s in
                        saddle_isolation(Ig, pts_m)]), 's-',
                label='model saddles')
    axs[3].set_ylabel('saddle / local peak')
    axs[3].legend(fontsize=8)
    axs[3].grid(alpha=0.3)
    for a in axs[:3]:
        a.axis('off')
    fig.suptitle(f'{name} dark-region connectivity analysis',
                 fontsize=12)
    fig.tight_layout()
    out = ROOT / '归档_八图分解' / '8奇点1_奇点强化'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{cn}_连通块诊断.png', dpi=140,
                bbox_inches='tight')
    plt.close(fig)

print('\n[done]')

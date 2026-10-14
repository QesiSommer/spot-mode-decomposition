# -*- coding: utf-8 -*-
"""
stage2c_sigma_gamma_sweep.py -- (σ, γ) 二维渲染扫描
==================================================

For the incumbent 8(1) solution, sweep the display configuration
(sigma, gamma) and find the one that best reproduces the experiment's
singularity visibility (dark-block count, core depth, saddle levels),
while keeping the appearance PCC as high as possible.

sigma in {1.5, 2.0, 2.5, 3.0}; gamma in {0.58, 0.67, 0.75, 0.85}.
Outputs: 8奇点1_奇点强化/
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
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points
from calibrate_metrics import render_model
from stage0_diagnosis import parse_md, core_depth, saddle_isolation
from stage1_gamma_fix import dark_iou

SRC = ROOT / '归档_八图分解' / '最新结果_新指标'
OUT = ROOT / '归档_八图分解' / '8奇点1_奇点强化'
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

SIGMAS = (1.5, 2.0, 2.5, 3.0)
GAMMAS = (0.58, 0.67, 0.75, 0.85)


def gpcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def blocks(I, thr=0.10, min_size=20, max_size=3000):
    m = I < thr * I.max()
    lab, n = label(m)
    out = []
    for k in range(1, n + 1):
        sz = int((lab == k).sum())
        if min_size <= sz <= max_size:
            out.append(sz)
    return sorted(out, reverse=True)


def main():
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()
    combo0, params0_md, _ = parse_md(SRC / '8奇点(1)_参数与指标.md')
    # use the core-weighted REFINED incumbent params (from stage2b)
    with open(OUT / 'scan_results.json', encoding='utf-8') as f:
        sr = json.load(f)
    combo0 = sr['best']['combo'].split('+')
    params0 = [float(x) for x in sr['best']['params']]
    print(f'refined incumbent params: {[round(x, 4) for x in params0]}')
    Ie = load_experiment(f'{IMG_DIR}/8Singularities(1).png',
                         'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    exp = {'nblk': len(blocks(Ie)),
           'depth': float(np.nanmean(core_depth(Ie, pts_e))),
           'sad_mean': float(np.mean(
               [s for _, s in saddle_isolation(Ie, pts_e)])),
           'sad_min': float(np.min(
               [s for _, s in saddle_isolation(Ie, pts_e)]))}
    print('experiment:', {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in exp.items()})

    I0 = render_model(modes_np, lab2k,
                      {'combo': combo0, 'params': params0, 'scale': None})

    rows = []
    print(f'{"sigma":>6} {"gamma":>6} {"nblk":>5} {"depth":>6} '
          f'{"sad_mean":>8} {"sad_min":>8} {"gpcc":>7} {"iou":>6} '
          f'{"score":>6}')
    for sg in SIGMAS:
        Id = gaussian_blur(torch.tensor(I0, dtype=torch.float32)
                           .unsqueeze(0), float(sg))[0].numpy()
        for g in GAMMAS:
            Ig = np.clip(Id, 0, 1) ** g
            Ig = Ig / max(Ig.max(), 1e-12)
            blk = blocks(Ig)
            pts_m = find_dark_points(Ig)
            dep = float(np.nanmean(core_depth(Ig, pts_m)))
            sad = [s for _, s in saddle_isolation(Ig, pts_m)]
            sc = (2.5 * abs(len(blk) - exp['nblk'])
                  + 2.0 * abs(dep - exp['depth'])
                  + 1.0 * abs(np.mean(sad) - exp['sad_mean']))
            row = {'sigma': sg, 'gamma': g, 'nblk': len(blk),
                   'depth': dep, 'sad_mean': float(np.mean(sad)),
                   'sad_min': float(np.min(sad)),
                   'gpcc': gpcc(Ie, Ig), 'iou': dark_iou(Ie, Ig),
                   'score': float(sc),
                   'blk_sizes': blk[:10]}
            rows.append(row)
            print(f'{sg:>6} {g:>6} {row["nblk"]:>5} {dep:>6.3f} '
                  f'{row["sad_mean"]:>8.3f} {row["sad_min"]:>8.3f} '
                  f'{row["gpcc"]:>7.4f} {row["iou"]:>6.3f} '
                  f'{sc:>6.2f}')

    rows.sort(key=lambda r: r['score'])
    print('\ntop-3 configs by composite score:')
    for r in rows[:3]:
        print(f'  sigma={r["sigma"]}, gamma={r["gamma"]}: '
              f'nblk={r["nblk"]} depth={r["depth"]:.3f} '
              f'sad_mean={r["sad_mean"]:.3f} gpcc={r["gpcc"]:.4f}')

    (OUT / 'sigma_gamma_sweep.json').write_text(json.dumps(
        {'experiment': exp, 'rows': rows}, indent=2,
        ensure_ascii=False, default=str), encoding='utf-8')

    # figure: experiment | top-3 configs (zoom)
    best = rows[:3]
    crop = (slice(56, 168), slice(56, 168))
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
    axs[0].imshow(Ie[crop], cmap='inferno', vmin=0, vmax=1)
    axs[0].set_title(f'experiment (nblk={exp["nblk"]})')
    for k, r in enumerate(best):
        Id = gaussian_blur(torch.tensor(I0, dtype=torch.float32)
                           .unsqueeze(0), float(r['sigma']))[0].numpy()
        Ig = np.clip(Id, 0, 1) ** r['gamma']
        Ig = Ig / Ig.max()
        axs[k + 1].imshow(Ig[crop], cmap='inferno', vmin=0, vmax=1)
        axs[k + 1].set_title(
            f'#{k+1} sigma={r["sigma"]}, gamma={r["gamma"]}\n'
            f'nblk={r["nblk"]} depth={r["depth"]:.3f} '
            f'sad={r["sad_mean"]:.3f} gpcc={r["gpcc"]:.3f}',
            fontsize=9)
    for a in axs:
        a.axis('off')
    fig.suptitle('8Singularities(1) - sigma x gamma render sweep '
                 '(crop)', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'sigma_gamma_扫描.png', dpi=140,
                bbox_inches='tight')
    plt.close(fig)
    print('\n[done]')


if __name__ == '__main__':
    main()

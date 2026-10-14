# -*- coding: utf-8 -*-
"""
stage1_gamma_fix.py -- 阶段1：gamma 校正渲染 + 量化验证
========================================================

Applies the diagnosed display-pipeline gamma to the model rendering:
    model_display = model^gamma   (gamma < 1 lifts dark regions)

  5Singularities : gamma = 0.70  (estimators 0.668 / 0.723)
  8Singularities(1): gamma = 0.58 (estimators 0.586 / 0.570)

Display sigma raised from 2.0 to 2.5 (D3: sigma* ~ 3).

Quantitative verification after the fix:
  - core depth (experiment vs gamma-corrected model)
  - saddle isolation (the "connected vs isolated" metric)
  - dark-region IoU (binarize at 10% threshold, intersection-over-union)

Outputs: 阶段1_gamma修复/
  <cn>_gamma对比.png   experiment | model(no gamma) | model(gamma) | diff
  <cn>_量化验证.json   depth/saddle/IoU before vs after
  阶段1报告.md
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
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points
from calibrate_metrics import render_model
from stage0_diagnosis import parse_md, core_depth, saddle_isolation

SRC = ROOT / '归档_八图分解' / '最新结果_新指标'
BASE = ROOT / '归档_八图分解' / '局部优化_5及8奇点'
OUT = BASE / '阶段1_gamma修复'
OUT.mkdir(exist_ok=True)
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

GAMMAS = {'5Singularities': 0.70, '8Singularities(1)': 0.58}
SIG_DISPLAY = 2.5
CASES = [{'name': '5Singularities', 'cn': '5奇点'},
         {'name': '8Singularities(1)', 'cn': '8奇点(1)'}]


def dark_iou(A, B, thr=0.10):
    a = A < thr * A.max()
    b = B < thr * B.max()
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else np.nan


def main():
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()

    results = {}
    for case in CASES:
        name, cn = case['name'], case['cn']
        g = GAMMAS[name]
        print(f'\n########## {cn} gamma 修复 (gamma={g}, '
              f'sigma={SIG_DISPLAY}) ##########')
        combo, params, scale = parse_md(SRC / f'{cn}_参数与指标.md')
        Ie = load_experiment(f'{IMG_DIR}/{name}.png', 'nipy_spectral',
                             'isot')
        I0 = render_model(modes_np, lab2k,
                          {'combo': combo, 'params': params,
                           'scale': scale})
        I_disp = gaussian_blur(
            torch.tensor(I0, dtype=torch.float32).unsqueeze(0),
            SIG_DISPLAY)[0].numpy()
        I_gamma = np.clip(I_disp, 0, 1) ** g
        I_gamma = I_gamma / max(I_gamma.max(), 1e-12)

        pts_e = find_dark_points(Ie)

        # ---- quantitative verification ----
        dep_e = core_depth(Ie, pts_e)
        sad_e = saddle_isolation(Ie, pts_e)

        pts_m0 = find_dark_points(I_disp)
        dep_m0 = core_depth(I_disp, pts_m0)
        sad_m0 = saddle_isolation(I_disp, pts_m0)

        pts_m1 = find_dark_points(I_gamma)
        dep_m1 = core_depth(I_gamma, pts_m1)
        sad_m1 = saddle_isolation(I_gamma, pts_m1)

        iou_before = dark_iou(Ie, I_disp)
        iou_after = dark_iou(Ie, I_gamma)

        print(f'  core depth : exp={np.nanmean(dep_e):.3f} | '
              f'before={np.nanmean(dep_m0):.3f} -> '
              f'after={np.nanmean(dep_m1):.3f}')
        s_e = [s for _, s in sad_e]
        s_0 = [s for _, s in sad_m0]
        s_1 = [s for _, s in sad_m1]
        print(f'  saddle mean: exp={np.mean(s_e):.3f} | '
              f'before={np.mean(s_0):.3f} -> after={np.mean(s_1):.3f}')
        print(f'  dark IoU   : {iou_before:.3f} -> {iou_after:.3f}')
        print(f'  cores      : exp={len(pts_e)} | before={len(pts_m0)} '
              f'-> after={len(pts_m1)}')

        # ---- comparison figure ----
        raw = plt.imread(f'{IMG_DIR}/{name}.png')
        if raw.ndim == 3 and raw.shape[2] == 4:
            raw = raw[:, :, :3]
        fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
        axs[0].imshow(raw)
        axs[0].set_title('Experiment (original)')
        axs[1].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
        axs[1].set_title('Experiment (processed)')
        axs[2].imshow(I_disp, cmap='inferno', vmin=0, vmax=1)
        axs[2].set_title(f'Model WITHOUT gamma (sigma={SIG_DISPLAY})\n'
                         f'saddle={np.mean(s_0):.2f} vs '
                         f'exp={np.mean(s_e):.2f}')
        axs[3].imshow(I_gamma, cmap='inferno', vmin=0, vmax=1)
        axs[3].set_title(f'Model WITH gamma={g} (sigma={SIG_DISPLAY})\n'
                         f'saddle={np.mean(s_1):.2f} vs '
                         f'exp={np.mean(s_e):.2f}')
        for a in axs:
            a.axis('off')
        fig.suptitle(f'{name} - gamma pipeline fix', fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f'{cn}_gamma对比.png', dpi=140,
                    bbox_inches='tight')
        plt.close(fig)

        # center zoom figure (the region of interest)
        c0, c1 = 52, 172
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.5))
        axs[0].imshow(Ie[c0:c1, c0:c1], cmap='inferno', vmin=0, vmax=1)
        axs[0].set_title('experiment (center)')
        axs[1].imshow(I_disp[c0:c1, c0:c1], cmap='inferno', vmin=0,
                      vmax=1)
        axs[1].set_title('model, no gamma (center)')
        axs[2].imshow(I_gamma[c0:c1, c0:c1], cmap='inferno', vmin=0,
                      vmax=1)
        axs[2].set_title(f'model, gamma={g} (center)')
        for a in axs:
            a.axis('off')
        fig.suptitle(f'{name} - center zoom comparison', fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / f'{cn}_中心放大对比.png', dpi=140,
                    bbox_inches='tight')
        plt.close(fig)

        results[cn] = {
            'gamma': g, 'sigma_display': SIG_DISPLAY,
            'depth_exp': round(float(np.nanmean(dep_e)), 3),
            'depth_before': round(float(np.nanmean(dep_m0)), 3),
            'depth_after': round(float(np.nanmean(dep_m1)), 3),
            'saddle_exp_mean': round(float(np.mean(s_e)), 3),
            'saddle_exp_all': [round(float(x), 3) for x in s_e],
            'saddle_before_mean': round(float(np.mean(s_0)), 3),
            'saddle_before_all': [round(float(x), 3) for x in s_0],
            'saddle_after_mean': round(float(np.mean(s_1)), 3),
            'saddle_after_all': [round(float(x), 3) for x in s_1],
            'iou_before': round(float(iou_before), 3),
            'iou_after': round(float(iou_after), 3),
            'cores_exp': len(pts_e), 'cores_before': len(pts_m0),
            'cores_after': len(pts_m1),
        }

    (OUT / '量化验证.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

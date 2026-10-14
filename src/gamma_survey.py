# -*- coding: utf-8 -*-
"""Gamma survey across all 8 images: is the dark-lifting (gamma) a
global pipeline property?

Robust gamma estimators:
  G1  log-log fit restricted to model intensities in [0.05, 0.5]
      (mid-range: avoids core-center misalignment contamination)
  G2  core-based: gamma = ln(exp_core_min)/ln(model_core_min) using
      mean core depths (broad features)
"""
import sys
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points
from calibrate_metrics import render_model
from stage0_diagnosis import parse_md, core_depth

SRC = ROOT / '归档_八图分解' / '最新结果_新指标'
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
CNS = ['4奇点', '5奇点', '6奇点', '8奇点(1)', '8奇点(2)', '9奇点',
       '10奇点', '11奇点']

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()

print(f'{"img":10s} {"G1(mid-range)":>13s} {"G2(core)":>9s} '
      f'{"exp_dep":>8s} {"mod_dep":>8s}')
rows = []
for cn in CNS:
    md = SRC / f'{cn}_参数与指标.md'
    combo, params, scale = parse_md(md)
    name = md.read_text(encoding='utf-8').splitlines()[0].split()[1]
    Ie = load_experiment(f'{IMG_DIR}/{name}.png', 'nipy_spectral',
                         'isot')
    I0 = render_model(modes_np, lab2k,
                      {'combo': combo, 'params': params, 'scale': scale})
    I2 = gaussian_blur(torch.tensor(I0, dtype=torch.float32)
                       .unsqueeze(0), 2.0)[0].numpy()
    mask = binary_dilation(Ie > 0.05 * Ie.max(), iterations=8)
    e = Ie[mask]
    m = I2[mask]
    # G1: mid-range log-log
    sel = (m >= 0.05) & (m <= 0.5) & (e > 0.005)
    if sel.sum() > 200:
        A = np.vstack([np.log(m[sel]), np.ones(int(sel.sum()))]).T
        g1 = float(np.linalg.lstsq(A, np.log(e[sel]), rcond=None)[0][0])
    else:
        g1 = np.nan
    # G2: core-based
    pts_e = find_dark_points(Ie)
    pts_m = find_dark_points(I2)
    dep_e = np.nanmean(core_depth(Ie, pts_e))
    dep_m = np.nanmean(core_depth(I2, pts_m))
    ce, cm = 1 - dep_e, 1 - dep_m
    g2 = float(np.log(ce) / np.log(cm)) if cm > 1e-4 and ce > 1e-4 \
        else np.nan
    rows.append((cn, g1, g2, dep_e, dep_m))
    print(f'{cn:10s} {g1:13.3f} {g2:9.3f} {dep_e:8.3f} {dep_m:8.3f}')

import json
out = ROOT / '归档_八图分解' / '局部优化_5及8奇点' / '阶段0_诊断'
(out / 'gamma_survey.json').write_text(
    json.dumps([{'cn': cn, 'gamma_midrange': round(g1, 3),
                 'gamma_core': round(g2, 3),
                 'depth_exp': round(float(de), 3),
                 'depth_model': round(float(dm), 3)}
                for cn, g1, g2, de, dm in rows], indent=2),
    encoding='utf-8')
g1s = [r[1] for r in rows if r[1] == r[1]]
print(f'\nG1 mean = {np.nanmean(g1s):.3f} (std {np.nanstd(g1s):.3f})')

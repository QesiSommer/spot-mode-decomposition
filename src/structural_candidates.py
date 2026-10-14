# -*- coding: utf-8 -*-
"""Structural check across candidates: does the dark-core structure
discriminate the champion from the runner-ups?"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch

from search_two_mode import load_experiment
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, render_complex, match_points

OUT = Path(__file__).resolve().parent / 'structural_check'
modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {lab: k for k, lab in enumerate(labels)}

# experiment side
Ie5 = load_experiment(
    os.path.join(os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images')), '5Singularities.png'),
    'nipy_spectral', 'isot')
Ie6 = load_experiment(
    os.path.join(os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images')), '6Singularities.png'),
    'nipy_spectral', 'isot')
pts5 = find_dark_points(Ie5)
pts6 = find_dark_points(Ie6)
print(f'experiment: 5Sing {len(pts5)} cores, 6Sing {len(pts6)} cores')

results = {}

# ---- 5Sing: top candidates from the joint search ----
df5 = pd.read_csv(str(Path(__file__).resolve().parent /
                      'two_mode_search_joint/5Singularities_top.csv'))
print('\n=== 5Singularities candidates ===')
rows5 = []
for _, row in df5.head(6).iterrows():
    m1, m2 = row['mode1'], row['mode2']
    r, phi, s = float(row['r_energy1']), float(row['phi_rad']), \
        float(row['scale'])
    ms = rescale_lg(modes, s).numpy()
    E = render_complex(ms, lab2k[m1], lab2k[m2], r, phi)
    Im = np.abs(E) ** 2
    Im = Im / Im.max()
    pts = find_dark_points(Im)
    match = match_points(pts5, pts)
    d = match['mean']
    flag = 'PASS' if (len(pts) == len(pts5) and d is not None
                      and d < 12) else 'fail'
    rows5.append((f'{m1}+{m2}', len(pts), f'{d:.1f}' if d else '-',
                  float(row['corr_avg']), flag))
    print(f'  {m1:>10s}+{m2:<10s} cores={len(pts):2d} '
          f'(exp {len(pts5)}) meanNN={d if d is None else round(d, 1)}px '
          f'corr={row["corr_avg"]:.4f}  {flag}')
results['5Sing'] = rows5

# ---- 6Sing: candidates from the refinement run ----
ic = json.load(open(str(Path(__file__).resolve().parent / 'refine_6sing' /
                        'refine_6sing_results.json'), encoding='utf-8'))
print('\n=== 6Singularities candidates ===')
rows6 = []
for name, info in ic['candidates'].items():
    m1, m2 = info['pair']
    ms = rescale_lg(modes, info['scale']).numpy()
    E = render_complex(ms, lab2k[m1], lab2k[m2], info['r'],
                       info['phi_rad'])
    Im = np.abs(E) ** 2
    Im = Im / Im.max()
    pts = find_dark_points(Im)
    match = match_points(pts6, pts)
    d = match['mean']
    flag = 'PASS' if (abs(len(pts) - len(pts6)) <= 1 and d is not None
                      and d < 12) else 'fail'
    rows6.append((name, len(pts), f'{d:.1f}' if d else '-',
                  info['corr_avg'], flag))
    print(f'  {name:>22s} cores={len(pts):2d} (exp {len(pts6)}) '
          f'meanNN={d if d is None else round(d, 1)}px '
          f'corr={info["corr_avg"]:.4f}  {flag}')
results['6Sing'] = rows6

(OUT / 'candidate_structural.json').write_text(
    json.dumps(results, indent=2, default=str), encoding='utf-8')
print('\n[done]')

# -*- coding: utf-8 -*-
"""
probeA_odd_harmonics.py -- 探针A：奇阶谐波诊断
=================================================
背景：多图验收单显示模型在奇数 m 阶（1,3,5,7,9）角向幅值≈0，实验
非零（相对差 1.00）。angular_distance 的极坐标采样以图像几何中心
(111.5,111.5) 为中心——若实验图光强质心偏移数像素，即可"制造"
奇阶分量；模型渲染居中故为零。

检验（10奇点 + 8奇点(2)）：
  A1  实验图质心偏移量；
  A2  实验图按质心亚像素重对中后，奇阶幅值下降多少；
  A3  模型施加同样的偏移后，奇阶幅值是否被"制造"出来、与实验相当；
  A4  重对中实验 vs 模型的 D_ang 对比（原始 vs 重对中）。

判定：若重对中后实验奇阶大幅下降、且移位模型奇阶与原始实验相当
     → 伪影（中心化问题）；若实验奇阶保持 → 真实基/模型缺陷。

Outputs: 归档_八图分解/扩展基评估/probeA_odd_harmonics.json + png
"""
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from calibrate_metrics import render_model, angular_distance
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats  # noqa

OUT = ROOT / '归档_八图分解' / '扩展基评估'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

CASES = {
    '10Singularities': {
        'combo': ['LG(0,-3)', 'LG(1,-3)', 'LG(2,-1)', 'HG(3,4)'],
        'theta': json.load(open(
            ROOT / '归档_八图分解' / '10奇点' / '5_新指标修复与验收'
            / 'stage3_delivery.json', encoding='utf-8'))['theta']},
    '8Singularities(2)': {
        'combo': ['HG(0,4)', 'HG(1,3)', 'HG(3,1)', 'HG(5,1)'],
        'theta': json.load(open(
            ROOT / '归档_八图分解' / '8奇点2_三模四模重分解'
            / 'stage7d_delivery.json', encoding='utf-8'))['chosen']['theta']},
}

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()


def centroid(I):
    ys, xs = np.mgrid[0:I.shape[0], 0:I.shape[1]]
    w = np.clip(I, 0, None)
    tot = w.sum()
    return float((w * ys).sum() / tot), float((w * xs).sum() / tot)


def odd_amp(a):
    return float(np.sum(a[1::2]))


results = {}
for name, c in CASES.items():
    Ie = load_experiment(f'{IMG}/{name}.png', 'nipy_spectral', 'isot')
    Im = render_model(modes_np, lab2k,
                      {'combo': c['combo'], 'params': c['theta'],
                       'scale': None})
    Im = Im / max(Im.max(), 1e-12)
    cy, cx = centroid(Ie)
    H, W = Ie.shape
    dy, dx = (H - 1) / 2.0 - cy, (W - 1) / 2.0 - cx
    _, (ae_orig, am_orig) = angular_distance(Ie, Im)
    # A2: recenter experiment (subpixel), keep model
    Ie_rc = ndshift(Ie, (dy, dx), order=1, mode='nearest')
    _, (ae_rc, _) = angular_distance(Ie_rc, Im)
    # A3: shift model by the SAME offset (model moved to where the
    # experiment pattern sits): shift model by (-dy, -dx)
    Im_sh = ndshift(Im, (-dy, -dx), order=1, mode='nearest')
    _, (_, am_sh) = angular_distance(Ie, Im_sh)
    # A4: D_ang before/after recentering
    d_before, _ = angular_distance(Ie, Im)
    d_after, _ = angular_distance(Ie_rc, Im)

    # A2b: optimal shift that MINIMIZES the experiment's odd-amp
    # (centroid != odd-optimal center; if the true pattern is
    #  even-symmetric, some shift should nearly kill the odd content)
    best = None
    for dy_ in np.arange(-3, 3.01, 0.25):
        for dx_ in np.arange(-3, 3.01, 0.25):
            Is = ndshift(Ie, (dy_, dx_), order=1, mode='nearest')
            _, (a_s, _) = angular_distance(Is, Im)
            oa = odd_amp(a_s)
            if best is None or oa < best[0]:
                best = (oa, float(dy_), float(dx_))
    oa_opt, dy_opt, dx_opt = best
    Ie_opt = ndshift(Ie, (dy_opt, dx_opt), order=1, mode='nearest')
    d_ang_opt, _ = angular_distance(Ie_opt, Im)

    r = {'centroid_offset_px': [round(dy, 2), round(dx, 2)],
         'odd_amp_exp_orig': odd_amp(ae_orig),
         'odd_amp_exp_recentred': odd_amp(ae_rc),
         'odd_amp_exp_optshift': oa_opt,
         'opt_shift_px': [dy_opt, dx_opt],
         'odd_amp_model_orig': odd_amp(am_orig),
         'odd_amp_model_shifted': odd_amp(am_sh),
         'D_ang_before': d_before, 'D_ang_after_recentre': d_after,
         'D_ang_after_optshift': d_ang_opt,
         'spec_exp_orig': [round(float(x), 4) for x in ae_orig],
         'spec_exp_recentred': [round(float(x), 4) for x in ae_rc],
         'spec_exp_optshift': None,
         'spec_model_orig': [round(float(x), 4) for x in am_orig],
         'spec_model_shifted': [round(float(x), 4) for x in am_sh]}
    results[name] = r
    _, (a_opt, _) = angular_distance(Ie_opt, Im)
    results[name]['spec_exp_optshift'] = [round(float(x), 4)
                                          for x in a_opt]
    print(f'=== {name} ===')
    print(f'  centroid offset (dy,dx) = ({dy:+.2f}, {dx:+.2f}) px')
    print(f'  odd-amp  exp: {r["odd_amp_exp_orig"]:.3f} -> recentred '
          f'{r["odd_amp_exp_recentred"]:.3f} -> opt-shift({dy_opt:+.2f},'
          f'{dx_opt:+.2f}) {oa_opt:.3f}')
    print(f'  odd-amp  model: {r["odd_amp_model_orig"]:.3f} -> shifted '
          f'{r["odd_amp_model_shifted"]:.3f}')
    print(f'  D_ang: {d_before:.4f} -> recentred {d_after:.4f} -> '
          f'opt-shift {d_ang_opt:.4f}')

# ---- verdict ----
print('\n=== verdict inputs ===')
for name, r in results.items():
    drop = (r['odd_amp_exp_orig'] - r['odd_amp_exp_optshift']) \
        / max(r['odd_amp_exp_orig'], 1e-9)
    results[name]['odd_drop_frac'] = drop
    print(f'{name}: exp odd drops {drop*100:.0f}% at optimal shift; '
          f'floor {r["odd_amp_exp_optshift"]:.3f} vs model 0; '
          f'model shifted odd = {r["odd_amp_model_shifted"]:.3f}')

(OUT / 'probeA_odd_harmonics.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

# ---- figure ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
for row, (name, r) in enumerate(results.items()):
    m = np.arange(11)
    ax = axs[row, 0]
    ax.bar(m - 0.22, r['spec_exp_orig'], width=0.44,
           label='experiment (orig)', color='#c66')
    ax.bar(m + 0.22, r['spec_exp_recentred'], width=0.44,
           label='experiment (recentred)', color='#6a6')
    ax.set_title(f'{name}: experiment spectrum\n'
                 f'centroid offset ({r["centroid_offset_px"][0]},'
                 f'{r["centroid_offset_px"][1]}) px, odd-amp drop '
                 f'{r["odd_drop_frac"]*100:.0f}%', fontsize=10)
    ax.set_xlabel('azimuthal order m')
    ax.legend(fontsize=9)
    ax = axs[row, 1]
    ax.bar(m - 0.22, r['spec_model_orig'], width=0.44,
           label='model (centered)', color='#88a')
    ax.bar(m + 0.22, r['spec_model_shifted'], width=0.44,
           label='model (same shift)', color='#ca8')
    ax.set_title(f'model spectrum: does a shift create odd m?\n'
                 f'model odd {r["odd_amp_model_orig"]:.3f} -> '
                 f'{r["odd_amp_model_shifted"]:.3f} vs exp orig '
                 f'{r["odd_amp_exp_orig"]:.3f}', fontsize=10)
    ax.set_xlabel('azimuthal order m')
    ax.legend(fontsize=9)
fig.suptitle('Probe A: are the odd-m angular harmonics a centering '
             'artifact?', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / 'probeA_奇阶谐波诊断.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)
print(f'\n[done] {OUT}')

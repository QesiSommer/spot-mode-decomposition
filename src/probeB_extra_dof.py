# -*- coding: utf-8 -*-
"""
probeB_extra_dof.py -- 探针B：额外自由度收益测试（逐模束腰/全局平移）
=====================================================================
探针A发现：模型渲染强制居中，而实验图质心可偏 ~2.5px——全局平移是
当前模型缺失的自由度。本探针对三张图（10奇点/8(1)/8(2)）的交付解
做三臂对照（同等评估预算，坐标下降）：

  arm0 对照   ：仅 theta 微扰重搜（同预算）——隔离"额外搜索量"伪增益；
  arm1 平移   ：theta + 全局平移 (dy,dx)；
  arm2 逐模腰 ：theta + 逐模独立束腰 s_i（共享 w₀ 假设放开）；
  arm3 组合   ：theta + 平移 + 逐模腰（收益上界）。

每臂结束做结构验收；报告 C_s02 增益与结构变化。
Outputs: 归档_八图分解/扩展基评估/probeB_extra_dof.json + png
"""
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model

OUT = ROOT / '归档_八图分解' / '扩展基评估'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

CASES = {
    '10Singularities': {
        'combo': ['LG(0,-3)', 'LG(1,-3)', 'LG(2,-1)', 'HG(3,4)'],
        'theta': json.load(open(
            ROOT / '归档_八图分解' / '10奇点' / '5_新指标修复与验收'
            / 'stage3_delivery.json', encoding='utf-8'))['theta']},
    '8Singularities(1)': {
        'combo': ['LG(2,0)', 'HG(1,1)', 'HG(2,0)', 'HG(3,3)'],
        'theta': [0.351744, 0.14786880, 0.47169120, 0.028696,
                  -0.71255, -0.59675, 0.71255]},
    '8Singularities(2)': {
        'combo': ['HG(0,4)', 'HG(1,3)', 'HG(3,1)', 'HG(5,1)'],
        'theta': json.load(open(
            ROOT / '归档_八图分解' / '8奇点2_三模四模重分解'
            / 'stage7d_delivery.json', encoding='utf-8'))['chosen']['theta']},
}

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()
Ie_blur_cache = {}


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def blur2(I):
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy()


def render_extra(combo, theta, scales=None, dxdy=None):
    """Render |E|^2 with optional per-mode scales + global shift."""
    if scales is None and dxdy is None:
        I = render_model(modes_np, lab2k,
                         {'combo': combo, 'params': theta, 'scale': None})
        return I / max(I.max(), 1e-12)
    E = np.zeros_like(modes_np[0])
    for i, m in enumerate(combo):
        F = modes_np[lab2k[m]]
        if scales is not None:
            F = rescale_lg(torch.from_numpy(F).unsqueeze(0),
                           float(scales[i]))[0].numpy()
        E = E + np.sqrt(max(theta[i], 0)) * F \
            * np.exp(1j * theta[len(combo) + i - 1]) if i > 0 else \
            np.sqrt(max(theta[i], 0)) * F
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    if dxdy is not None:
        I = ndshift(I, dxdy, order=1, mode='nearest')
    return I


def eval_full(name, Ie, Ie2, pts_e, combo, theta, scales=None, dxdy=None):
    I = render_extra(combo, theta, scales, dxdy)
    C2 = 0.5 * (pcc(Ie, I) + pcc(Ie2, blur2(I)))
    pts_m = find_dark_points(I)
    mt = match_points(pts_e, pts_m)
    dists = np.array(mt['dists']) if mt['dists'] else np.array([99.0])
    struct = (abs(len(pts_m) - len(pts_e)) <= 1
              and mt['mean'] is not None and mt['mean'] < 12)
    max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.0
    return C2, struct, len(pts_m), float(mt['mean']), max10


def jitter_theta(theta, rng, N, s_en=0.012, s_ph=0.04):
    u = np.array(theta[:N]) + rng.normal(0, s_en, N)
    u = np.clip(u, 0.005, None)
    u = u / u.sum()
    ph = np.array(theta[N:]) + rng.normal(0, s_ph, N - 1)
    return list(u) + list(ph)


def refine_arm(name, Ie, Ie2, pts_e, combo, theta0, arm, budget=250):
    """Coordinate descent for the given arm; returns best record."""
    rng = np.random.default_rng(42)
    N = len(combo)
    cur_th = list(theta0)
    cur_sc = [1.0] * N if arm in ('permode', 'combo') else None
    cur_sh = (0.0, 0.0) if arm in ('shift', 'combo') else None
    best = None
    t0 = time.time()
    n_eval = 0

    def ev(th, sc, sh):
        nonlocal best, n_eval
        n_eval += 1
        C2, struct, nc, mn, m10 = eval_full(name, Ie, Ie2, pts_e, combo,
                                            th, sc, sh)
        if best is None or C2 > best['C_s02']:
            best = {'C_s02': C2, 'struct': struct, 'cores': nc,
                    'mean_nn': mn, 'max10': m10,
                    'theta': list(th), 'scales': (list(sc) if sc
                                                 else None),
                    'shift': (list(sh) if sh else None)}
        return C2

    ev(cur_th, cur_sc, cur_sh)
    # rounds: coordinate descent on enabled extra DOFs + theta jitter
    for r in range(3):
        if cur_sh is not None:
            for ax in range(2):
                for d in np.arange(-2.5, 2.51, 0.5):
                    sh = list(cur_sh)
                    sh[ax] = float(d)
                    ev(cur_th, cur_sc, tuple(sh))
                cur_sh = tuple(best['shift'])
        if cur_sc is not None:
            for i in range(N):
                for s in np.arange(0.92, 1.081, 0.02):
                    sc = list(cur_sc)
                    sc[i] = float(s)
                    ev(cur_th, sc, cur_sh)
                cur_sc = list(best['scales'])
        # theta jitter block (all arms get this, incl. control)
        for k in range(budget // 3):
            th = jitter_theta(cur_th, rng, N)
            ev(th, cur_sc, cur_sh)
        cur_th = list(best['theta'])
    best['n_eval'] = n_eval
    best['time_s'] = round(time.time() - t0, 1)
    return best


results = {}
for name, c in CASES.items():
    Ie = load_experiment(f'{IMG}/{name}.png', 'nipy_spectral', 'isot')
    Ie2 = blur2(Ie)
    pts_e = find_dark_points(Ie)
    combo, theta0 = c['combo'], c['theta']
    print(f'\n=== {name} ({"+".join(combo)}) ===')
    base = eval_full(name, Ie, Ie2, pts_e, combo, theta0)
    print(f'  baseline: C_s02={base[0]:.4f} struct={base[1]} '
          f'cores={base[2]} mean={base[3]:.2f} max10={base[4]:.2f}')
    rec = {'baseline': {'C_s02': base[0], 'struct': base[1],
                        'cores': base[2], 'mean_nn': base[3],
                        'max10': base[4]}}
    for arm in ['control', 'shift', 'permode', 'combo']:
        b = refine_arm(name, Ie, Ie2, pts_e, combo, theta0, arm)
        rec[arm] = b
        print(f'  [{arm:8s}] C_s02 {base[0]:.4f} -> {b["C_s02"]:.4f} '
              f'({b["C_s02"]-base[0]:+.4f}) struct={b["struct"]} '
              f'cores={b["cores"]} max10={b["max10"]:.2f} '
              f'({b["n_eval"]} evals, {b["time_s"]}s)')
        if b['scales']:
            print(f'             scales={np.round(b["scales"],3).tolist()}')
        if b['shift']:
            print(f'             shift={np.round(b["shift"],2).tolist()}')
    results[name] = rec

(OUT / 'probeB_extra_dof.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# ---- summary figure ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
arms = ['control', 'shift', 'permode', 'combo']
names = list(results.keys())
fig, ax = plt.subplots(figsize=(10, 4.6))
w = 0.2
x = np.arange(len(names))
for ai, arm in enumerate(arms):
    vals = [results[n][arm]['C_s02'] - results[n]['baseline']['C_s02']
            for n in names]
    ax.bar(x + (ai - 1.5) * w, vals, width=w, label=arm)
base_vals = [results[n]['baseline']['C_s02'] for n in names]
for i, n in enumerate(names):
    ax.text(i, 0.002, f'base {base_vals[i]:.4f}', ha='center',
            fontsize=8, color='0.3')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel('C_s02 gain over baseline')
ax.axhline(0, color='k', lw=0.8)
ax.legend()
ax.set_title('Probe B: gain from extra degrees of freedom '
             '(equal evaluation budget per arm)')
fig.tight_layout()
fig.savefig(OUT / 'probeB_额外自由度收益.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)
print(f'\n[done] {OUT}')

# -*- coding: utf-8 -*-
"""
calibrate_metrics.py -- 指标阈值与窗口尺寸标定实验
====================================================

Uses 8 known-good solutions (structure-verified) vs 8 known-bad
solutions (high corr but structure failed) to:

  1. calibrate thresholds for:
       - C_s02 = [PCC(sigma=0) + PCC(sigma=2)] / 2
       - separate floors PCC(0), PCC(2)
       - local-window worst-case C_min and failure fraction rho
       - angular-spectrum shape distance D_ang
  2. determine the best window size W* by separation power over
     W in {16, 28, 32, 56, 112} (sigma in {0, 2}).

Outputs: metrics_calibration/calibration_results.json + box plots.
Read-only on archived data.
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
from scipy.ndimage import binary_dilation, map_coordinates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from search_two_mode_lg import rescale_lg

ARCH = ROOT / '归档_八图分解'
OUT = ROOT / 'metrics_calibration'
OUT.mkdir(exist_ok=True)
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
WINDOWS = [16, 28, 32, 56, 112]
SIGMAS_W = [0, 2]

# ---------------------------------------------------------------------------
# build the true/false solution lists from archived data
# ---------------------------------------------------------------------------

def load_solutions():
    with open(ROOT / 'refine_methods_compare' / 'methods_compare.json',
              encoding='utf-8') as f:
        mc = json.load(f)
    with open(ROOT / 'refine_all_new_method' / 'all_results.json',
              encoding='utf-8') as f:
        an = json.load(f)

    truth = []
    for tag, name in [('4Sing', '4Singularities'),
                      ('8Sing1', '8Singularities(1)')]:
        b = mc[tag]['method_B']
        truth.append({'name': name, 'combo': list(mc[tag]['combo']),
                      'params': [float(x) for x in b['theta']],
                      'scale': None, 'src': f'methods_compare/{tag}'})
    for tag, r in an.items():
        truth.append({'name': r['name'], 'combo': list(r['combo']),
                      'params': [float(x) for x in r['new_params']],
                      'scale': r['new_scale'], 'src': f'all_results/{tag}'})

    fake = []
    # 8Sing(1) two-mode fake champion (2 cores)
    d = json.load(open(ARCH / '8奇点(1)' / '2_两模结构验收'
                       / '8Singularities(1)_refine.json',
                       encoding='utf-8'))
    c = d['candidates'][0]
    fake.append({'name': '8Singularities(1)', 'combo': list(c['pair']),
                 'params': [c['r'], 1 - c['r'], -c['phi_rad']],
                 'scale': c['scale'], 'src': '8Sing1 two-mode champion'})
    # 5Sing runner-up (1 core)
    import pandas as pd
    df = pd.read_csv(ARCH / '5奇点' / '1_两模搜索'
                     / '5Singularities_top.csv')
    row = df.iloc[1]
    fake.append({'name': '5Singularities',
                 'combo': [row['mode1'], row['mode2']],
                 'params': [float(row['r_energy1']),
                            1 - float(row['r_energy1']),
                            -float(row['phi_rad'])],
                 'scale': float(row['scale']),
                 'src': '5Sing runner-up (1 core)'})
    # 10/11 two-mode champions (1 core each)
    for tag, folder, name in [('10Sing', '10奇点', '10Singularities'),
                              ('11Sing', '11奇点', '11Singularities')]:
        d = json.load(open(ARCH / folder / '2_两模结构验收'
                           / f'{name}_refine.json', encoding='utf-8'))
        c = d['candidates'][0]
        fake.append({'name': name, 'combo': list(c['pair']),
                     'params': [c['r'], 1 - c['r'], -c['phi_rad']],
                     'scale': c['scale'], 'src': f'{tag} two-mode'})
    # 9Sing three-mode fail (12 cores vs 9)
    d = json.load(open(ARCH / '9奇点' / '3_三模搜索与验收'
                       / '9Singularities_three_mode.json',
                       encoding='utf-8'))
    e = next(x for x in d['entries']
             if set(x['modes']) == {'HG(4,0)', 'HG(3,3)', 'HG(4,2)'})
    fake.append({'name': '9Singularities', 'combo': list(e['modes']),
                 'params': list(e['energy']) + [e['phi2'], e['phi3']],
                 'scale': None, 'src': '9Sing three-mode top (12 cores)'})
    # 10Sing three-mode close solution (12 cores vs 10)
    d = json.load(open(ARCH / '10奇点' / '3_三模搜索与验收'
                       / '10Singularities_three_mode.json',
                       encoding='utf-8'))
    e = next(x for x in d['entries']
             if set(x['modes']) == {'LG(0,-4)', 'HG(2,4)', 'HG(4,2)'})
    fake.append({'name': '10Singularities', 'combo': list(e['modes']),
                 'params': list(e['energy']) + [e['phi2'], e['phi3']],
                 'scale': None, 'src': '10Sing three-mode close (12 cores)'})
    # 10Sing four-mode highest-corr but structure-failed
    d = json.load(open(ARCH / '10奇点' / '4_四模搜索与验收'
                       / '10Singularities_four_mode_guided.json',
                       encoding='utf-8'))
    e = next(x for x in d['entries']
             if set(x['modes']) == {'LG(0,4)', 'LG(2,2)', 'LG(1,4)',
                                    'HG(2,4)'})
    fake.append({'name': '10Singularities', 'combo': list(e['modes']),
                 'params': list(e['energy']) + list(e['phis']),
                 'scale': None,
                 'src': '10Sing four-mode highest-corr (8 cores)'})
    return truth, fake


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def render_model(modes_np, lab2k, sol):
    combo, p, s = sol['combo'], sol['params'], sol['scale']
    N = len(combo)
    mnp = modes_np if s is None else rescale_lg(
        torch.tensor(modes_np), float(s)).numpy()
    tr = [lab2k[m] for m in combo]
    Fs = [np.sqrt(max(p[i], 0)) * mnp[tr[i]] for i in range(N)]
    ph = [0.0] + list(p[N:])
    E = sum(Fs[i] * np.exp(1j * ph[i]) for i in range(N))
    I = np.abs(E) ** 2
    return I / max(I.max(), 1e-12)


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    if d < 1e-12:
        return float('nan')
    return float((a * b).sum() / d)


def window_stats(Ie, Im, W, mask, thr=0.5):
    H, Wd = Ie.shape
    ny, nx = H // W, Wd // W
    corrs = []
    for i in range(ny):
        for j in range(nx):
            sl = (slice(i * W, (i + 1) * W), slice(j * W, (j + 1) * W))
            if mask[sl].mean() < 0.5:
                continue
            a, b = Ie[sl], Im[sl]
            if a.std() < 1e-6 or b.std() < 1e-6:
                continue
            corrs.append(pcc(a, b))
    corrs = np.array([c for c in corrs if np.isfinite(c)])
    if corrs.size == 0:
        return np.nan, np.nan
    return float(corrs.min()), float((corrs < thr).mean())


def angular_distance(Ie, Im, nr=64, nt=96, m_max=10):
    H, W = Ie.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    thetas = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    rs = np.linspace(4, 100, nr)
    Rt, Tt = np.meshgrid(rs, thetas, indexing='ij')
    ys = cy + Rt * np.sin(Tt)
    xs = cx + Rt * np.cos(Tt)

    def spec(I):
        P = map_coordinates(I, [ys.ravel(), xs.ravel()], order=1,
                            mode='nearest').reshape(nr, nt)
        P = P - P.mean(axis=1, keepdims=True)
        S = np.fft.fft(P, axis=1)
        return np.abs(S[:, :m_max + 1]).mean(axis=0)  # amp per order

    ae, am = spec(Ie), spec(Im)
    num = np.sum(np.abs(ae[1:] - am[1:]))
    den = np.sum(ae[1:]) + 1e-12
    return float(num / den), (ae, am)


def evaluate_solution(Ie, sol, modes_np, lab2k):
    I = render_model(modes_np, lab2k, sol)
    I0 = I
    I2 = gaussian_blur(torch.tensor(I, dtype=torch.float32)
                       .unsqueeze(0), 2.0)[0].numpy()
    res = {'name': sol['name'], 'src': sol['src'],
           'pcc0': pcc(Ie, I0), 'pcc2': pcc(Ie, I2)}
    res['C_s02'] = 0.5 * (res['pcc0'] + res['pcc2'])
    mask = binary_dilation(Ie > 0.05 * Ie.max(), iterations=8)
    for W in WINDOWS:
        c0, r0 = window_stats(Ie, I0, W, mask)
        c2, r2 = window_stats(Ie, I2, W, mask)
        res[f'cmin_{W}_s0'] = c0
        res[f'rho_{W}_s0'] = r0
        res[f'cmin_{W}_s2'] = c2
        res[f'rho_{W}_s2'] = r2
    d_ang, _ = angular_distance(Ie, I2)
    res['D_ang'] = d_ang
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    dev = torch.device('cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {lab: k for k, lab in enumerate(labels)}
    modes_np = modes.numpy()

    truth, fake = load_solutions()
    Ie_cache = {}
    for sol in truth + fake:
        if sol['name'] not in Ie_cache:
            Ie_cache[sol['name']] = load_experiment(
                f'{IMG_DIR}/{sol["name"]}.png', 'nipy_spectral', 'isot')

    print('=== TRUE solutions (structure-verified) ===')
    rows_t = []
    for sol in truth:
        r = evaluate_solution(Ie_cache[sol['name']], sol, modes_np, lab2k)
        rows_t.append(r)
        print(f'  {r["name"]:22s} C_s02={r["C_s02"]:.4f} '
              f'(s0={r["pcc0"]:.4f} s2={r["pcc2"]:.4f}) '
              f'D_ang={r["D_ang"]:.4f}')

    print('\n=== FALSE solutions (structure-failed) ===')
    rows_f = []
    for sol in fake:
        r = evaluate_solution(Ie_cache[sol['name']], sol, modes_np, lab2k)
        rows_f.append(r)
        print(f'  {r["name"]:22s} C_s02={r["C_s02"]:.4f} '
              f'(s0={r["pcc0"]:.4f} s2={r["pcc2"]:.4f}) '
              f'D_ang={r["D_ang"]:.4f}   [{r["src"]}]')

    # ---- separation analysis ----
    def sep(metric, higher_is_better=True):
        t = np.array([r[metric] for r in rows_t], dtype=float)
        f = np.array([r[metric] for r in rows_f], dtype=float)
        t, f = t[np.isfinite(t)], f[np.isfinite(f)]
        if t.size == 0 or f.size == 0:
            return None
        if higher_is_better:
            margin = t.min() - f.max()
        else:
            margin = f.min() - t.max()
        z = abs(t.mean() - f.mean()) / np.sqrt(
            t.var() / len(t) + f.var() / len(f) + 1e-12)
        return {'metric': metric, 't_min': float(t.min()),
                't_max': float(t.max()), 't_mean': float(t.mean()),
                'f_min': float(f.min()), 'f_max': float(f.max()),
                'f_mean': float(f.mean()), 'margin': float(margin),
                'z': float(z),
                'threshold': float((t.min() + f.max()) / 2)
                if margin > 0 else None}

    print('\n=== separation analysis ===')
    metrics = ['C_s02', 'pcc0', 'pcc2', 'D_ang']
    for W in WINDOWS:
        metrics += [f'cmin_{W}_s0', f'cmin_{W}_s2',
                    f'rho_{W}_s0', f'rho_{W}_s2']
    sep_rows = []
    for m in metrics:
        hib = not m.startswith('rho')
        s = sep(m, higher_is_better=hib)
        if s is None:
            continue
        sep_rows.append(s)
        if hib:
            line = (f'  {m:14s} true_min={s["t_min"]:.4f} '
                    f'false_max={s["f_max"]:.4f}')
        else:
            line = (f'  {m:14s} true_max={s["t_max"]:.4f} '
                    f'false_min={s["f_min"]:.4f}')
        if s['margin'] > 0 or s['z'] > 1.5:
            print(line + f' margin={s["margin"]:+.4f} z={s["z"]:.2f}'
                  + (f'  -> T={s["threshold"]:.4f}'
                     if s['threshold'] is not None else ''))
        else:
            print(line + f' (weak, z={s["z"]:.2f})')

    best_w = max((s for s in sep_rows if s['metric'].startswith('cmin')),
                 key=lambda s: s['z'])
    print(f'\nbest window (by z-score of C_min): '
          f'{best_w["metric"]}  (z={best_w["z"]:.2f})')

    json.dump({'true': rows_t, 'false': rows_f, 'separation': sep_rows},
              open(OUT / 'calibration_results.json', 'w',
                   encoding='utf-8'), indent=2, ensure_ascii=False)

    # ---- box plots ----
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    panels = [('C_s02', 'C_s02 main metric'),
              ('pcc0', 'PCC sigma=0'), ('pcc2', 'PCC sigma=2'),
              ('D_ang', 'D_ang (lower better)')]
    for ax, (m, title) in zip(axs, panels):
        tv = [r[m] for r in rows_t]
        fv = [r[m] for r in rows_f]
        ax.boxplot([tv, fv], tick_labels=['true', 'false'])
        ax.scatter(np.ones(len(tv)) + np.random.uniform(-.06, .06, len(tv)),
                   tv, c='tab:blue', zorder=3, s=22)
        ax.scatter(np.full(len(fv), 2) + np.random.uniform(-.06, .06,
                                                            len(fv)),
                   fv, c='tab:red', zorder=3, s=22, marker='x')
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'separation_scores.png', dpi=140)

    fig, axs = plt.subplots(1, len(WINDOWS), figsize=(18, 3.6))
    for ax, W in zip(axs, WINDOWS):
        for tag, rows, col in [('true', rows_t, 'tab:blue'),
                               ('false', rows_f, 'tab:red')]:
            vals = [r[f'cmin_{W}_s0'] for r in rows]
            ax.scatter(np.full(len(vals), 1 if tag == 'true' else 2)
                       + np.random.uniform(-.08, .08, len(vals)),
                       vals, c=col, s=20,
                       marker='o' if tag == 'true' else 'x')
        ax.set_title(f'W={W}px ({(224//W)}^2 grid)')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'window_scan_cmin_s0.png', dpi=140)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
probeF_aniso_tilt.py -- 探针F：逐模像散 + 逐模倾斜
=====================================================
在终检解（含全局平移+逐模束腰）基础上，放开：
  - 逐模各向异性缩放 (sx_i, sy_i)（即逐模像散）；
  - 逐模平移 (dx_i, dy_i)（等效逐模倾斜）。
四臂对照（同评估预算，结构+稳健约束）：
  control 纯 theta 重搜 / aniso / tilt / combo。
对象：10奇点、8奇点(1)。诊断性质，不改动交付解。
Outputs: 归档_八图分解/物理模型评估/
"""
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from window_metrics import make_windows, _support_mask
from structural_check import find_dark_points, match_points
from scipy.ndimage import shift as ndshift

OUT = ROOT / '归档_八图分解' / '物理模型评估'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
TI = json.load(open(ROOT / '归档_八图分解' / '终检精修_2026-10-10'
                    / 'terminal_inspection.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def blur_sig(I, sg):
    return I if sg == 0 else gaussian_blur(
        torch.tensor(I, dtype=torch.float32).unsqueeze(0),
        float(sg))[0].numpy()


def _fit_center(a, H, W):
    out = np.zeros((a.shape[0], H, W), np.float64)
    h, w = a.shape[-2], a.shape[-1]
    if h >= H:
        sy = (h - H) // 2
        sy_s, sy_d = slice(sy, sy + H), slice(0, H)
    else:
        sy = (H - h) // 2
        sy_s, sy_d = slice(0, h), slice(sy, sy + h)
    if w >= W:
        sx = (w - W) // 2
        sx_s, sx_d = slice(sx, sx + W), slice(0, W)
    else:
        sx = (W - w) // 2
        sx_s, sx_d = slice(0, w), slice(sx, sx + w)
    out[:, sy_d, sx_d] = a[:, sy_s, sx_s]
    return out


def zoom_axis_c(F, s, axis):
    if abs(s - 1.0) < 1e-6:
        return F
    H, W = F.shape
    t = np.stack([F.real, F.imag], 0)[None]
    t = torch.tensor(t, dtype=torch.float32)
    Hn = int(round(H * s)) if axis == 0 else H
    Wn = int(round(W * s)) if axis == 1 else W
    t = torch.nn.functional.interpolate(
        t, size=(Hn, Wn), mode='bilinear',
        align_corners=False)[0].numpy().astype(np.float64)
    b = _fit_center(t, H, W)
    return b[0] + 1j * b[1]


def render_F(combo, theta, base_scale, sx, sy, pg_shift, pmshift):
    """sx/sy: per-mode aniso factors; pg_shift global; pmshift per-mode."""
    N = len(combo)
    E = np.zeros(modes_np[0].shape, dtype=np.complex128)
    for i, m in enumerate(combo):
        Fm = modes_np[lab2k[m]]
        s = base_scale or 1.0
        if abs(s - 1.0) > 1e-6:
            Fm = rescale_lg(torch.from_numpy(Fm).unsqueeze(0),
                            float(s))[0].numpy()
        if sx is not None and (abs(sx[i] - 1) > 1e-6
                               or abs(sy[i] - 1) > 1e-6):
            Fm = zoom_axis_c(Fm, sx[i], 1)
            Fm = zoom_axis_c(Fm, sy[i], 0)
        if pmshift is not None and (abs(pmshift[0][i]) > 1e-6
                                    or abs(pmshift[1][i]) > 1e-6):
            Fm = ndshift(Fm.real, (pmshift[0][i], pmshift[1][i]),
                         order=1, mode='nearest') + \
                1j * ndshift(Fm.imag, (pmshift[0][i], pmshift[1][i]),
                             order=1, mode='nearest')
        phi = 0.0 if i == 0 else theta[N + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    if pg_shift is not None and (abs(pg_shift[0]) > 1e-6
                                 or abs(pg_shift[1]) > 1e-6):
        I = ndshift(I, pg_shift, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


results = {}
for CN, NAME in [('10奇点', '10Singularities'),
                 ('8奇点1', '8Singularities(1)')]:
    rec = TI[CN]
    combo, N = rec['combo'], rec['N']
    acc = rec['accepted']
    theta0 = [float(x) for x in acc['theta']]
    base_scale = rec['base_scale']
    iso = acc['scales'] if acc['scales'] else [1.0] * N
    gshift = acc['shift'] if acc['shift'] else [0.0, 0.0]
    # fold isotropic scales into base zoom (base_scale * iso_i) — keep as
    # separable: render with base=base_scale and sx=sy=iso as start
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    N_EXP = len(pts_e)
    Ie_b2 = blur_sig(Ie, 2)
    print(f'\n=== {CN} ({NAME}) N={N} ===')

    def ev(theta, sx, sy, pms, pg=None, with_struct=True):
        I = render_F(combo, theta, base_scale, sx, sy,
                     pg if pg is not None else gshift,
                     pms)
        C2 = 0.5 * (pcc(Ie, I) + pcc(Ie_b2, blur_sig(I, 2)))
        pts_m = find_dark_points(I)
        mt = match_points(pts_e, pts_m)
        dists = np.array(mt['dists']) if mt['dists'] else np.array([99.])
        struct = (abs(len(pts_m) - N_EXP) <= 1
                  and mt['mean'] is not None and mt['mean'] < 12)
        max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
        return {'C_s02': C2, 'struct': struct, 'cores': len(pts_m),
                'max10': max10, 'mean_nn': float(mt['mean'])
                if mt['mean'] is not None else 99.,
                'dists': [float(x) for x in dists]}

    base = ev(theta0, list(iso), list(iso), ([0.] * N, [0.] * N))
    margin = max(14.0, base['max10'])
    print(f'  baseline(terminal): C_s02={base["C_s02"]:.4f} '
          f'cores={base["cores"]}/{N_EXP} max10={base["max10"]:.2f}')

    rng = np.random.default_rng(4242)

    def theta_jit(th, rnd):
        u = np.array(th[:N]) + rng.normal(0, 0.010, N)
        u = np.clip(u, 0.005, None)
        u = u / u.sum()
        ph = np.array(th[N:]) + rng.normal(0, 0.035, N - 1)
        return list(u) + list(ph)

    def run_arm(arm, budget_theta=250):
        t0 = time.time()
        cur_th = list(theta0)
        sx = list(iso)
        sy = list(iso)
        pms = ([0.0] * N, [0.0] * N)
        best = dict(base)
        best.update({'theta': cur_th, 'sx': sx, 'sy': sy,
                     'pms': [list(pms[0]), list(pms[1])]})

        def offer(r, th, sxv, syv, pmsv):
            nonlocal best
            if r['struct'] and r['max10'] <= margin \
                    and r['C_s02'] > best['C_s02']:
                best = dict(r)
                best.update({'theta': list(th), 'sx': list(sxv),
                             'sy': list(syv),
                             'pms': [list(pmsv[0]), list(pmsv[1])]})

        for rnd in range(2):
            for k in range(budget_theta):
                th = theta_jit(cur_th if k % 2 else best['theta'], rnd)
                offer(ev(th, sx, sy, pms), th, sx, sy, pms)
            cur_th = best['theta']
            if arm in ('aniso', 'combo'):
                for i in range(N):
                    for ax, arr in [(0, 'sx'), (1, 'sy')]:
                        for step, rad in [(0.02, 0.08), (0.01, 0.03)]:
                            ref = best[arr]
                            for d in np.arange(-rad, rad + 1e-9, step):
                                sxv = list(best['sx'])
                                syv = list(best['sy'])
                                (sxv if ax == 0 else syv)[i] = float(
                                    np.clip(ref[i] + d, 0.75, 1.3))
                                offer(ev(cur_th, sxv, syv, pms),
                                      cur_th, sxv, syv, pms)
            if arm in ('tilt', 'combo'):
                for i in range(N):
                    for ax in (0, 1):
                        for step, rad in [(0.5, 2.0), (0.25, 0.75)]:
                            ref = best['pms'][ax]
                            for d in np.arange(-rad, rad + 1e-9, step):
                                pmsv = [list(best['pms'][0]),
                                        list(best['pms'][1])]
                                pmsv[ax][i] = float(
                                    np.clip(ref[i] + d, -4, 4))
                                offer(ev(cur_th, sx, sy, pmsv),
                                      cur_th, sx, sy, pmsv)
        best['n_time'] = round(time.time() - t0, 1)
        return best

    res = {'baseline': base}
    for arm in ['control', 'aniso', 'tilt', 'combo']:
        b = run_arm(arm)
        res[arm] = b
        print(f'  [{arm:8s}] C_s02 {base["C_s02"]:.4f} -> '
              f'{b["C_s02"]:.4f} ({b["C_s02"]-base["C_s02"]:+.4f}) '
              f'struct={b["struct"]} max10={b["max10"]:.2f} '
              f'({b["n_time"]}s)')
        if arm in ('aniso', 'combo'):
            print(f'             sx={np.round(b["sx"],3).tolist()} '
                  f'sy={np.round(b["sy"],3).tolist()}')
        if arm in ('tilt', 'combo'):
            print(f'             pmshift={np.round(np.array(b["pms"]),2).tolist()}')
    results[CN] = res

(OUT / 'probeF_aniso_tilt.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
arms = ['control', 'aniso', 'tilt', 'combo']
fig, ax = plt.subplots(figsize=(9, 4.4))
x = np.arange(len(results))
w = 0.2
for ai, arm in enumerate(arms):
    vals = [results[c][arm]['C_s02'] - results[c]['baseline']['C_s02']
            for c in results]
    ax.bar(x + (ai - 1.5) * w, vals, width=w, label=arm)
for i, c in enumerate(results):
    ax.text(i, 0.001, f'base {results[c]["baseline"]["C_s02"]:.4f}',
            ha='center', fontsize=8, color='0.3')
ax.set_xticks(x)
ax.set_xticklabels(['10Singularities', '8Singularities(1)'])
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('C_s02 gain over terminal baseline')
ax.legend(fontsize=9)
ax.set_title('Probe F: per-mode astigmatism / tilt gains '
             '(equal budget, structure-constrained)')
fig.tight_layout()
fig.savefig(OUT / 'probeF_像散倾斜收益.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)
print(f'\n[done] {OUT}')

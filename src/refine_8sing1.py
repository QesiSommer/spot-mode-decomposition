# -*- coding: utf-8 -*-
"""
refine_8sing1.py -- 8Singularities(1) 三模解精密二分微调
=========================================================

Champion triple from the three-mode search:
    LG(2,0) + HG(1,1) + HG(2,0),  cavg = 0.9607

Precise refinement of the 5 continuous parameters
(u, v, w = energy shares; phi2, phi3 = relative phases) by
coordinate descent with interval bisection:

  for each parameter: scan 41 points over [x0-h, x0+h],
  take the best, shrink h *= 0.25, repeat until h < 1e-6.

Objective: multi-sigma averaged Pearson corr (same protocol as the
two/three-mode searches, sigmas 0..6), evaluated from precomputed
per-sigma statistics (fast point evaluation).
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from search_three_mode import build_base_fields, eval_c_grid, \
    render_three_mode_np
from structural_check import find_dark_points, match_points

OUT = Path(__file__).resolve().parent / 'refine_8sing1'
OUT.mkdir(exist_ok=True)
IMG = os.path.join(os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images')), '8Singularities(1).png')
SIGMAS = [0, 1, 2, 3, 4, 5, 6]
TRIPLE = ['LG(2,0)', 'HG(1,1)', 'HG(2,0)']


def make_c(u, v, w, p2, p3):
    a1, a2, a3 = np.sqrt(max(u, 0)), np.sqrt(max(v, 0)), \
        np.sqrt(max(w, 0))
    return torch.tensor([[
        u, v, w,
        2 * a1 * a2 * np.cos(p2), 2 * a1 * a2 * np.sin(p2),
        2 * a1 * a3 * np.cos(p3), 2 * a1 * a3 * np.sin(p3),
        2 * a2 * a3 * np.cos(p2 - p3),
        -2 * a2 * a3 * np.sin(p2 - p3)]], dtype=torch.float32)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes = modes.to(dev)
    modes_np = modes.detach().cpu().numpy()
    lab2k = {lab: k for k, lab in enumerate(labels)}

    Ie_np = load_experiment(IMG, 'nipy_spectral', 'isot')
    Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
    pts_e = find_dark_points(Ie_np)

    tr = [lab2k[t] for t in TRIPLE]
    F1, F2, F3 = (modes[tr[0]].unsqueeze(0), modes[tr[1]].unsqueeze(0),
                  modes[tr[2]].unsqueeze(0))
    G9 = build_base_fields(F1, F2, F3)[0]  # (9,224,224)

    # precompute per-sigma statistics (parameter independent)
    stats = []
    for sg in SIGMAS:
        Gb = gaussian_blur(G9, float(sg))
        Ie_b = gaussian_blur(Ie.unsqueeze(0), float(sg))[0]
        me = Ie_b.mean()
        ve = Ie_b.var(unbiased=False)
        Gf = Gb.reshape(9, -1)
        m = Gf.mean(dim=1).unsqueeze(0)
        E = torch.einsum('ki,i->k', Gf, Ie_b.reshape(-1)
                         ).unsqueeze(0) / Gf.shape[1]
        Gram = torch.einsum('ki,li->kl', Gf, Gf
                            ).unsqueeze(0) / Gf.shape[1]
        stats.append((m, E, Gram, me, ve))

    def ev(params):
        c = make_c(*params).to(dev)
        vals = [float(eval_c_grid(c, *s)[0, 0]) for s in stats]
        return float(np.mean(vals)), vals

    # starting point from the three-mode search result
    src = json.load(open(
        str(Path(__file__).resolve().parent / 'three_mode_search' /
            '8Singularities(1)_three_mode.json'), encoding='utf-8'))
    entry = None
    for e in src['entries']:
        if e['modes'] == TRIPLE:
            entry = e
            break
    if entry is None:
        entry = src['entries'][0]
    u, v, w = entry['energy']
    p2, p3 = entry['phi2'], entry['phi3']
    print(f'start: u={u:.4f} v={v:.4f} w={w:.4f} '
          f'phi2={p2:.4f} phi3={p3:.4f}')
    c0, _ = ev((u, v, w, p2, p3))
    print(f'start corr_avg = {c0:.6f}')

    # coordinate descent with bisection shrink
    h = {'u': 0.02, 'v': 0.02, 'w': 0.02, 'p2': 0.10, 'p3': 0.10}
    names = ['u', 'v', 'w', 'p2', 'p3']
    cur = [u, v, w, p2, p3]
    for it in range(12):
        improved = False
        for k, nm in enumerate(names):
            lo = cur[k] - h[nm]
            hi = cur[k] + h[nm]
            if nm in ('u', 'v', 'w'):
                lo, hi = max(0.0, lo), min(1.0, hi)
            pts = np.linspace(lo, hi, 41)
            best_v, best_x = -2.0, cur[k]
            for x in pts:
                trial = list(cur)
                trial[k] = float(x)
                if trial[0] + trial[1] + trial[2] > 1.0:
                    continue
                cv, _ = ev(trial)
                if cv > best_v:
                    best_v, best_x = cv, float(x)
            if abs(best_x - cur[k]) > 1e-9:
                improved = True
            cur[k] = best_x
        h = {nm: max(h[nm] * 0.25, 1e-7) for nm in names}
        cavg, cvals = ev(cur)
        if it % 2 == 0 or not improved:
            print(f'  iter {it + 1:2d}: corr_avg={cavg:.6f} '
                  f'u={cur[0]:.6f} v={cur[1]:.6f} w={cur[2]:.6f} '
                  f'p2={cur[3]:.6f} p3={cur[4]:.6f}')
        if not improved and all(h[nm] <= 1e-6 for nm in names):
            break

    # normalize u,v,w exactly
    s = cur[0] + cur[1] + cur[2]
    cur[0], cur[1], cur[2] = cur[0] / s, cur[1] / s, cur[2] / s
    cavg, cvals = ev(cur)
    u, v, w, p2, p3 = cur
    a1, a2, a3 = np.sqrt(u), np.sqrt(v), np.sqrt(w)
    print(f'\nFINAL: corr_avg={cavg:.6f}')
    print(f'  energies  : {u:.6f} / {v:.6f} / {w:.6f} '
          f'(sum={u + v + w:.6f})')
    print(f'  amplitudes: {a1:.6f} / {a2:.6f} / {a3:.6f} '
          f'(a1^2+a2^2+a3^2={a1**2+a2**2+a3**2:.6f})')
    print(f'  phi2={p2:.6f} rad ({np.degrees(p2):.3f} deg)')
    print(f'  phi3={p3:.6f} rad ({np.degrees(p3):.3f} deg)')
    print(f'  corr by sigma: '
          f'{ {s: round(v_, 4) for s, v_ in zip(SIGMAS, cvals)} }')

    # structural verification at the refined parameters
    I = render_three_mode_np(modes_np, tr[0], tr[1], tr[2],
                             (u, v, w, p2, p3))
    pts_m = find_dark_points(I)
    match = match_points(pts_e, pts_m)
    print(f'  cores: model {len(pts_m)} vs experiment {len(pts_e)} | '
          f'meanNN={match["mean"]:.2f}px max={match["max"]:.2f}px')

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6))
    axs[0].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
    for (x, y, _v) in pts_e:
        axs[0].plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.4)
    axs[0].set_title(f'experiment: {len(pts_e)} cores', fontsize=10)
    axs[1].imshow(I, cmap='inferno', vmin=0, vmax=1)
    for (x, y, _v) in pts_m:
        axs[1].plot(x, y, 'o', mfc='none', mec='red', ms=11, mew=1.4)
    axs[1].set_title(f'{"+".join(TRIPLE)}\ncorr_avg={cavg:.4f} '
                     f'({len(pts_m)} cores)', fontsize=9)
    axs[2].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
    for (x, y, _v) in pts_e:
        axs[2].plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.4)
    for (x, y, _v) in pts_m:
        axs[2].plot(x, y, 'x', color='red', ms=9, mew=1.8)
    axs[2].set_title(f'overlay  meanNN={match["mean"]:.1f}px', fontsize=10)
    for a in axs:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / '8Sing1_refined_structural.png', dpi=140)
    plt.close(fig)

    payload = {
        'image': '8Singularities(1)',
        'triple': TRIPLE,
        'start': {'u': entry['energy'][0], 'v': entry['energy'][1],
                  'w': entry['energy'][2], 'phi2': entry['phi2'],
                  'phi3': entry['phi3'], 'corr_avg': entry['corr_avg']},
        'refined': {'u': u, 'v': v, 'w': w, 'phi2': p2, 'phi3': p3,
                    'amps': [a1, a2, a3], 'corr_avg': cavg,
                    'corr_by_sigma': [round(x, 4) for x in cvals]},
        'structure': {'cores_model': len(pts_m),
                      'cores_experiment': len(pts_e),
                      'mean_nn_px': match['mean'],
                      'max_nn_px': match['max']},
    }
    (OUT / '8Sing1_refined.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

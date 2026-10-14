# -*- coding: utf-8 -*-
"""
structural_check.py -- 结构级验收 (Structural verification)
============================================================

Independent check of two-mode solutions:
  1. detect dark cores (singularity candidates) on the EXPERIMENT
     image with a robust local-minimum detector;
  2. render the candidate theoretical field (complex), detect dark
     cores with the SAME detector;
  3. compare counts and nearest-neighbour positions.

Calibration targets: 6Singularities -> 6 cores, 5Singularities -> 5.
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from scipy.ndimage import minimum_filter, maximum_filter, gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from search_two_mode import load_experiment
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes

OUT = Path(__file__).resolve().parent / 'structural_check'
OUT.mkdir(exist_ok=True)


def find_dark_points(I, blur_sigma=1.5, size=9, dark_ratio=0.20,
                     local_ratio=0.55, local_window=31,
                     floor_ratio=0.004, merge_dist=9):
    """Robust dark-core detection: local minima that are (a) absolutely
    dark w.r.t. the image peak, (b) significantly darker than the local
    maximum within a neighborhood (embedded in brighter structure),
    (c) not part of the flat zero background.

    Calibrated on the experiment images: 5Singularities -> 5 cores
    (center + four around), 6Singularities -> 8 cores (3 per lobe
    + 2 central)."""
    Ib = gaussian_filter(I.astype(np.float64), blur_sigma)
    mx = Ib.max()
    mn = minimum_filter(Ib, size=size)
    lmax = maximum_filter(Ib, size=local_window)
    cand = (Ib <= mn + 1e-12) \
        & (Ib < dark_ratio * mx) \
        & (Ib > floor_ratio * mx) \
        & (Ib < local_ratio * lmax)
    ys, xs = np.where(cand)
    vals = Ib[ys, xs]
    order = np.argsort(vals)
    pts = []
    for k in order:
        y, x = int(ys[k]), int(xs[k])
        for p in pts:
            if (x - p[1]) ** 2 + (y - p[0]) ** 2 < merge_dist ** 2:
                break
        else:
            pts.append((y, x, float(vals[k])))
    return pts


def render_complex(modes_np, i, j, r, phi):
    L1 = modes_np[i]
    L2 = modes_np[j]
    E = np.sqrt(r) * L1 + np.sqrt(1 - r) * L2 * np.exp(1j * phi)
    return E


def match_points(pts_e, pts_m):
    """greedy nearest-neighbour distances from each experimental point
    to the model point set."""
    if not pts_m:
        return {'dists': [], 'mean': None, 'max': None}
    md = [(x, y) for (x, y, _v) in pts_m]
    dists = []
    for (x, y, _v) in pts_e:
        d = min(np.hypot(x - mx_, y - my_) for (mx_, my_) in md)
        dists.append(float(d))
    return {'dists': dists, 'mean': float(np.mean(dists)),
            'max': float(np.max(dists))}


def main():
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {lab: k for k, lab in enumerate(labels)}
    ms_np = modes.detach().cpu().numpy()

    CASES = [
        {'name': '6Singularities',
         'img': os.path.join(os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images')), '6Singularities.png'),
         'pair': ('HG(0,3)', 'HG(3,1)'), 'r': 0.4733, 'phi': 1.5444,
         's': 1.195, 'target': 6},
        {'name': '5Singularities',
         'img': os.path.join(os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images')), '5Singularities.png'),
         'pair': ('LG(1,-1)', 'LG(0,3)'), 'r': 0.326, 'phi': 0.9285,
         's': 1.25, 'target': 5},
    ]

    report = {}
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 9))

    # ---- step 0: calibrate the detector on the experiment images ----
    for ci, case in enumerate(CASES):
        Ie = load_experiment(case['img'], 'nipy_spectral', 'isot')
        pts_e = find_dark_points(Ie)
        # detector calibration sweep for transparency
        sweeps = {}
        for dr in [0.08, 0.12, 0.15, 0.20, 0.25]:
            sweeps[dr] = len(find_dark_points(Ie, dark_ratio=dr))
        print(f"[{case['name']}] experiment dark cores = {len(pts_e)} "
              f"(target {case['target']}) | dark_ratio sweep: {sweeps}")

        # ---- step 1: theoretical candidate ----
        ms = rescale_lg(modes, case['s']).numpy()
        E = render_complex(ms, lab2k[case['pair'][0]],
                           lab2k[case['pair'][1]], case['r'], case['phi'])
        Im = np.abs(E) ** 2
        Im = Im / Im.max()
        pts_m = find_dark_points(Im)

        match = match_points(pts_e, pts_m)
        print(f"[{case['name']}] model ({case['pair'][0]}+{case['pair'][1]}) "
              f"dark cores = {len(pts_m)} | mean NN dist = "
              f"{match['mean'] if match['mean'] is None else round(match['mean'], 2)} px, "
              f"max = {match['max'] if match['max'] is None else round(match['max'], 2)} px")

        report[case['name']] = {
            'pair': list(case['pair']), 'r': case['r'], 'phi': case['phi'],
            's': case['s'],
            'n_core_experiment': len(pts_e),
            'n_core_model': len(pts_m),
            'core_positions_experiment': [(int(x), int(y)) for x, y, _ in pts_e],
            'core_positions_model': [(int(x), int(y)) for x, y, _ in pts_m],
            'match': match,
        }

        # ---- visualization ----
        ax = axs[ci, 0]
        ax.imshow(Ie, cmap='inferno', vmin=0, vmax=1)
        for (x, y, _v) in pts_e:
            ax.plot(x, y, 'o', mfc='none', mec='cyan', ms=12, mew=1.5)
        ax.set_title(f"{case['name']} experiment\n"
                     f"{len(pts_e)} dark cores (target {case['target']})",
                     fontsize=10)
        ax = axs[ci, 1]
        ax.imshow(Im, cmap='inferno', vmin=0, vmax=1)
        for (x, y, _v) in pts_m:
            ax.plot(x, y, 'o', mfc='none', mec='red', ms=12, mew=1.5)
        ax.set_title(f"model {case['pair'][0]}+{case['pair'][1]}\n"
                     f"{len(pts_m)} dark cores", fontsize=10)
        ax = axs[ci, 2]
        ax.imshow(Ie, cmap='inferno', vmin=0, vmax=1)
        for (x, y, _v) in pts_e:
            ax.plot(x, y, 'o', mfc='none', mec='cyan', ms=12, mew=1.5)
        for (x, y, _v) in pts_m:
            ax.plot(x, y, 'x', color='red', ms=9, mew=1.8)
        ax.set_title('overlay (cyan=exp, red=model)', fontsize=10)
        for a in axs[ci]:
            a.axis('off')

    fig.tight_layout()
    fig.savefig(OUT / 'structural_check_6and5.png', dpi=130)
    plt.close(fig)

    (OUT / 'structural_check.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

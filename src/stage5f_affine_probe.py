# -*- coding: utf-8 -*-
"""
stage5f_affine_probe.py -- (b) 像散/坐标变换先导实验
====================================================
Hypothesis: the experiment's -19 deg rotated mesh and outline deformation
come from a COORDINATE transformation of the real optical system
(astigmatism / rotation), not from the mode content itself.

Equivalent formulation (coordinate-change covariance): warping the
experiment image with T^{-1} and fitting with standard modes == fitting
the original image with T-transformed modes.

Probe:
  1. Warp grid: T = (sx, sy, theta) over a coarse grid
     (anisotropic scaling in a rotated frame).
  2. For each T: detect cores on the warped experiment, core-weighted
     refine (short) for 4 candidate combos, record quality
     (core-weighted PCC, mesh cv, core RMS, gpcc).
  3. Anti-overfit control: a KNOWN-BAD combo must not improve.
  4. Refine around the best T, report the gain and the best (T, combo).

Outputs: 归档_八图分解/8奇点1_HG主导搜索/
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage0_diagnosis import core_depth, saddle_isolation
from rerun_new_metrics import block_refine_wavg, GRIDS
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats
from stage5a_grid_diag import grender, gpcc, blocks, squareness_metrics
from stage5b_hg_search import OUT, IMGDIR, NAME, mesh2x4


def warp_image(I, sx, sy, th_deg):
    """Sample I at T(x) where T = R(th) diag(sx,sy) R(th)^T (ellipse
    scaling in a frame rotated by th).  sx>1 <-> image compressed along
    that axis."""
    H, W = I.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    y = yy - cy
    x = xx - cx
    t = np.radians(th_deg)
    ct, st = np.cos(t), np.sin(t)
    xr = ct * x + st * y
    yr = -st * x + ct * y
    xs = xr * sx
    ys = yr * sy
    xi = ct * xs - st * ys + cx
    yi = st * xs + ct * ys + cy
    out = map_coordinates(I, [yi, xi], order=1, mode='constant')
    out = np.clip(out, 0, None)
    m = out.max()
    return out / m if m > 1e-9 else out


def eval_combo_T(dev, modes_g, modes_np, lab2k, Ie_w_np, combo, seed,
                 rounds=3):
    """Warped-experiment evaluation of one combo: detect cores on the
    warped image, core-weighted refine from seed, full metrics."""
    pts_we = find_dark_points(Ie_w_np)
    if len(pts_we) < 5:
        return {'bad': True, 'n_cores_exp': len(pts_we)}
    pts_we_np = np.array([(y, x) for y, x, _ in pts_we], float)
    crop_f = crop_box(pts_we, 224)
    w_f = weight_map(Ie_w_np.shape, pts_we)
    Ie_w_t = torch.tensor(Ie_w_np, dtype=torch.float32, device=dev)
    tr = [lab2k[m] for m in combo]
    ws = weighted_crop_stats(modes_g, Ie_w_t, tr, None, crop_f, w_f,
                             sigmas=(0, 2))
    if len(combo) == 4:
        n_ph, n_en = 15, 11
    else:
        n_ph, n_en = 21, 15
    th, val = block_refine_wavg(ws, seed, len(combo), n_ph, n_en,
                                rounds=rounds)
    I0 = render_model(modes_np, lab2k,
                      {'combo': combo, 'params': th, 'scale': None})
    pts_m0 = find_dark_points(I0)
    if len(pts_m0) >= 2:
        A = np.array([(y, x) for y, x, _ in pts_m0], float)
        C = np.sqrt(((A[:, None, :] - pts_we_np[None, :, :]) ** 2)
                    .sum(-1))
        ri, ci = linear_sum_assignment(C)
        dr = C[ri, ci]
        core_rms = float(np.sqrt((dr ** 2).mean()))
    else:
        core_rms = 99.0
    mesh = mesh2x4(pts_m0)
    Ig = grender(I0, g=0.85)
    blk = blocks(Ig)
    sad = [s for _, s in saddle_isolation(Ig, find_dark_points(Ig))]
    sm = squareness_metrics(Ig, 0.3)
    return {'bad': False, 'n_cores_exp': len(pts_we),
            'wavg': float(val),
            'core_rms': core_rms, 'n_cores_m': len(pts_m0),
            'mesh_ok': bool(mesh.get('ok')),
            'mesh_cv': (mesh.get('cvA', 9) + mesh.get('cvB', 9)) / 2
            if mesh.get('ok') else 9.0,
            'mesh_step': mesh.get('in_row_step', 0) if mesh.get('ok') else 0,
            'mesh_gap': mesh.get('row_gap', 0) if mesh.get('ok') else 0,
            'nblk': len(blk), 'sad_mean': float(np.mean(sad)) if sad else None,
            'gpcc': gpcc(Ie_w_np, Ig),
            'A4_A0': sm['A4_A0'] if sm else None,
            'params': [float(x) for x in th]}


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes_g = modes.to(dev)
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()
    Ie = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')

    # ---- candidate combos + seeds ----
    scan = json.load(open(ROOT / '归档_八图分解' / '8奇点1_四模空间'
                          / 'scan_4mode.json', encoding='utf-8'))
    cur4 = scan['top3'][0]
    phg = json.load(open(OUT / 'pure_hg_deep.json', encoding='utf-8'))
    phg1 = phg['pure_hg_top'][0]
    fin = json.load(open(ROOT / '归档_八图分解' / '8奇点1_奇点强化'
                         / 'final_config.json', encoding='utf-8'))
    hgs = json.load(open(OUT / 'hg_search_results.json',
                         encoding='utf-8'))
    ctrl = None
    for e in hgs['all']:
        if e['combo'] == 'HG(1,0)+HG(3,0)+HG(6,1)':
            ctrl = e
            break
    if ctrl is None:
        ctrl = {'combo': 'HG(1,0)+HG(3,0)+HG(6,1)',
                'params': [0.33, 0.33, 0.34, 0.0, 0.0]}
    combos = [
        {'tag': 'INC4 (LG2,0+HG1,1+HG2,0+HG3,3)', 'combo': cur4['combo'].split('+'),
         'seed': [float(x) for x in cur4['params']]},
        {'tag': 'PHG1 pure-HG best mesh', 'combo': phg1['combo'].split('+'),
         'seed': [float(x) for x in phg1['params']]},
        {'tag': 'INC3 (3-mode incumbent)', 'combo': fin['combo'],
         'seed': [float(x) for x in fin['params']]},
        {'tag': 'CTRL known-bad (overfit check)', 'combo': ctrl['combo'].split('+'),
         'seed': [float(x) for x in ctrl['params']]},
    ]
    print('[setup] combos:', [c['tag'] for c in combos])

    # ---- probe grid ----
    sxs = [0.90, 0.95, 1.00, 1.05, 1.10]
    sys_ = [0.90, 0.95, 1.00, 1.05, 1.10]
    ths = [-24, -16, -8, 0, 8, 16, 24]
    t0 = time.time()
    rows = []
    for th in ths:
        for sx in sxs:
            for sy in sys_:
                Ie_w = warp_image(Ie, sx, sy, th)
                for c in combos:
                    r = eval_combo_T(dev, modes_g, modes_np, lab2k, Ie_w,
                                     c['combo'], c['seed'], rounds=3)
                    r.update({'sx': sx, 'sy': sy, 'th': th, 'tag': c['tag']})
                    rows.append(r)
        print(f'  theta={th:+3d} done  ({time.time()-t0:.0f}s)')

    # ---- analysis ----
    print('\n=== baseline (T = identity) ===')
    for c in combos:
        r = next(x for x in rows if x['tag'] == c['tag'] and x['sx'] == 1.0
                 and x['sy'] == 1.0 and x['th'] == 0)
        print(f'  {c["tag"]:38s} wavg={r.get("wavg", 0):.4f} '
              f'coreRMS={r.get("core_rms", 99):.2f} '
              f'meshcv={r.get("mesh_cv", 9):.2f} gpcc={r.get("gpcc", 0):.4f}')

    print('\n=== best T per combo (by wavg over the grid) ===')
    summary = {}
    for c in combos:
        sub = [x for x in rows if x['tag'] == c['tag'] and not x['bad']]
        base = next(x for x in sub if x['sx'] == 1.0 and x['sy'] == 1.0
                    and x['th'] == 0)
        best = max(sub, key=lambda x: x['wavg'])
        summary[c['tag']] = {'baseline': base, 'best': best}
        print(f'  {c["tag"]}')
        print(f'    base: wavg={base["wavg"]:.4f} coreRMS='
              f'{base["core_rms"]:.2f} meshcv={base["mesh_cv"]:.2f} '
              f'step={base["mesh_step"]:.1f} gap={base["mesh_gap"]:.1f} '
              f'nblk={base["nblk"]} gpcc={base["gpcc"]:.4f}')
        print(f'    best: wavg={best["wavg"]:.4f} @ (sx={best["sx"]}, '
              f'sy={best["sy"]}, th={best["th"]:+d})  '
              f'coreRMS={best["core_rms"]:.2f} meshcv={best["mesh_cv"]:.2f} '
              f'step={best["mesh_step"]:.1f} gap={best["mesh_gap"]:.1f} '
              f'nblk={best["nblk"]} gpcc={best["gpcc"]:.4f}')
        print(f'    gain: dwavg={best["wavg"]-base["wavg"]:+.4f} '
              f'dcoreRMS={best["core_rms"]-base["core_rms"]:+.2f} '
              f'dmeshcv={best["mesh_cv"]-base["mesh_cv"]:+.2f}')

    # ---- anti-overfit check ----
    print('\n=== anti-overfit control ===')
    ctrl_tag = 'CTRL known-bad (overfit check)'
    inc_tag = 'INC4 (LG2,0+HG1,1+HG2,0+HG3,3)'
    s_ctrl = summary[ctrl_tag]
    s_inc = summary[inc_tag]
    print(f'  if T-warp were pure overfitting, the known-bad combo would '
          f'gain as much as the incumbent.')
    print(f'  CTRL gain  = {s_ctrl["best"]["wavg"]-s_ctrl["baseline"]["wavg"]:+.4f}')
    print(f'  INC4 gain  = {s_inc["best"]["wavg"]-s_inc["baseline"]["wavg"]:+.4f}')

    # ---- figures ----
    fig, axs = plt.subplots(2, 3, figsize=(19, 10.4))
    # heatmap wavg at theta=0 for INC4
    ax = axs[0, 0]
    hm = np.full((len(sxs), len(sys_)), np.nan)
    for x in rows:
        if x['tag'] == inc_tag and x['th'] == 0 and not x['bad']:
            hm[sxs.index(x['sx']), sys_.index(x['sy'])] = x['wavg']
    im = ax.imshow(hm, cmap='viridis', origin='lower')
    ax.set_xticks(range(len(sys_)))
    ax.set_xticklabels(sys_)
    ax.set_yticks(range(len(sxs)))
    ax.set_yticklabels(sxs)
    ax.set_xlabel('sy')
    ax.set_ylabel('sx')
    ax.set_title(f'INC4 wavg, theta=0 (base={s_inc["baseline"]["wavg"]:.4f})',
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    # theta profile (best over sx,sy) per combo
    ax = axs[0, 1]
    for c in combos:
        vals = []
        for th in ths:
            sub = [x for x in rows if x['tag'] == c['tag']
                   and x['th'] == th and not x['bad']]
            vals.append(max(x['wavg'] for x in sub) if sub else np.nan)
        ax.plot(ths, vals, 'o-', label=c['tag'][:24], lw=1.5)
    ax.set_xlabel('theta (deg)')
    ax.set_ylabel('max-over-(sx,sy) wavg')
    ax.set_title('rotation response', fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    # core RMS profile
    ax = axs[0, 2]
    for c in combos:
        vals = []
        for th in ths:
            sub = [x for x in rows if x['tag'] == c['tag']
                   and x['th'] == th and not x['bad']]
            vals.append(min(x['core_rms'] for x in sub) if sub else np.nan)
        ax.plot(ths, vals, 'o-', label=c['tag'][:24], lw=1.5)
    ax.set_xlabel('theta (deg)')
    ax.set_ylabel('min core RMS (px)')
    ax.set_title('core-position response', fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    # render best-T warp for INC4 and PHG1
    phg_tag = 'PHG1 pure-HG best mesh'
    for k, tag in enumerate([inc_tag, phg_tag]):
        s = summary[tag]
        b = s['best']
        Ie_w = warp_image(Ie, b['sx'], b['sy'], b['th'])
        ax = axs[1, k]
        ax.imshow(Ie_w, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f'warped experiment (sx={b["sx"]}, sy={b["sy"]}, '
                     f'th={b["th"]:+d})\nfor {tag[:30]}', fontsize=9)
        ax.axis('off')
    ax = axs[1, 2]
    s = summary[inc_tag]
    b = s['best']
    I0 = render_model(modes_np, lab2k,
                      {'combo': combos[0]['combo'], 'params': b['params'],
                       'scale': None})
    ax.imshow(grender(I0, g=0.85), cmap='inferno', vmin=0, vmax=1)
    ax.set_title(f'INC4 refit at best T\nwavg={b["wavg"]:.4f} '
                 f'coreRMS={b["core_rms"]:.2f} nblk={b["nblk"]} '
                 f'gpcc={b["gpcc"]:.4f}', fontsize=9)
    ax.axis('off')
    fig.suptitle('8Singularities(1): affine (astigmatism) probe — '
                 'warped-experiment refits', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / '像散先导实验.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    json.dump({'rows': rows,
               'summary': {k: {'baseline': v['baseline'],
                               'best': v['best']}
                           for k, v in summary.items()}},
              open(OUT / 'affine_probe.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False, default=str)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

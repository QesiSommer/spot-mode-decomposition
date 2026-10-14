# -*- coding: utf-8 -*-
"""
stage5c_mesh_tune.py -- 第3步: 核几何(网格规整度)专项对齐优化
==============================================================
Target: minimize the RMS one-to-one distance between the model's 8
dark cores and the experiment's 8 cores (Hungarian assignment), with a
strong penalty when the core count leaves 8 +/- 1.

This directly attacks the diagnosed gap: the incumbent's mesh is
rotated correctly (-20.6 deg vs -19 deg) and in-row step matches
(22.5 vs 23.5 px), but the row gap is too small (29.8 vs 36.9) and the
in-row spacing is uneven (cv 0.25 vs 0.07).

Method: Nelder-Mead over 7 params (u1..u4, phi2..phi4) of the 4-mode
TOP-1 combo, multi-start from the incumbent + small perturbations.
After convergence: full metrics (mesh params, gamma-domain nblk /
depth / saddle / gpcc) + geometry figure vs experiment.

Output: 归档_八图分解/8奇点1_HG主导搜索/
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment, minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage0_diagnosis import core_depth, saddle_isolation
from stage1_gamma_fix import dark_iou
from stage5a_grid_diag import grender, gpcc, blocks
from stage5b_hg_search import OUT, IMGDIR, NAME, mesh2x4

PENALTY = 50.0


def core_dist_objective(params, modes_np, lab2k, combo, pts_e_np):
    params = np.asarray(params, dtype=np.float64)
    u = np.clip(params[:4], 1e-6, None)
    u = u / u.sum() if u.sum() > 0 else u
    p = list(u) + list(params[4:])
    I0 = render_model(modes_np, lab2k,
                      {'combo': combo, 'params': p, 'scale': None})
    pts_m = find_dark_points(I0)
    n_pen = PENALTY * abs(len(pts_m) - 8)
    if len(pts_m) < 2:
        return 100.0 + n_pen
    A = np.array([(y, x) for y, x, _ in pts_m], float)
    C = np.sqrt(((A[:, None, :] - pts_e_np[None, :, :]) ** 2).sum(-1))
    ri, ci = linear_sum_assignment(C)
    d = C[ri, ci]
    return float(np.sqrt((d ** 2).mean())) + n_pen


def main():
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()
    Ie = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    pts_e_np = np.array([(y, x) for y, x, _ in pts_e], float)

    scan = json.load(open(ROOT / '归档_八图分解' / '8奇点1_四模空间'
                          / 'scan_4mode.json', encoding='utf-8'))
    cur = scan['top3'][0]
    combo = cur['combo'].split('+')
    p0 = np.array([float(x) for x in cur['params']], dtype=np.float64)
    print(f'[setup] combo={"+".join(combo)}')
    print(f'[setup] incumbent params={np.round(p0, 4).tolist()}')
    J0 = core_dist_objective(p0, modes_np, lab2k, combo, pts_e_np)
    print(f'[setup] incumbent J (core RMS px) = {J0:.3f}')

    # ---- multi-start Nelder-Mead ----
    rng = np.random.default_rng(7)
    starts = [p0]
    for k in range(4):
        q = p0.copy()
        q[:4] = np.abs(q[:4] + rng.normal(0, 0.03, 4))
        q[:4] /= q[:4].sum()
        q[4:] = q[4:] + rng.normal(0, 0.12, 3)
        starts.append(q)
    results = []
    for si, s0 in enumerate(starts):
        t0 = time.time()
        res = minimize(core_dist_objective, s0,
                       args=(modes_np, lab2k, combo, pts_e_np),
                       method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-3,
                                'fatol': 1e-3,
                                'initial_simplex': None})
        results.append((float(res.fun), res.x, res.nit))
        print(f'  start {si}: J={res.fun:.3f} px  nit={res.nit}  '
              f'({time.time()-t0:.0f}s)')
    results.sort(key=lambda r: r[0])
    Jb, pb, nitb = results[0]
    print(f'[tune] best J = {Jb:.3f} px (incumbent {J0:.3f})')

    # normalized params of the tuned solution
    u = np.clip(pb[:4], 1e-6, None)
    u = u / u.sum()
    pbest = list(u) + list(pb[4:])
    print(f'[tune] params: u={np.round(u,4).tolist()} '
          f'phi={np.round(pb[4:],4).tolist()}')

    # ---- full metrics: incumbent vs tuned ----
    def full_eval(tag, p):
        I0 = render_model(modes_np, lab2k,
                          {'combo': combo, 'params': p, 'scale': None})
        pts_m = find_dark_points(I0)
        A = np.array([(y, x) for y, x, _ in pts_m], float)
        C = np.sqrt(((A[:, None, :] - pts_e_np[None, :, :]) ** 2).sum(-1))
        ri, ci = linear_sum_assignment(C)
        dr = C[ri, ci]
        mesh = mesh2x4(pts_m)
        per_g = {}
        for gg in (0.75, 0.85):
            Ig = grender(I0, g=gg)
            blk = blocks(Ig)
            pm = find_dark_points(Ig)
            dep = core_depth(Ig, pm)
            sad = [s for _, s in saddle_isolation(Ig, pm)]
            per_g[str(gg)] = {
                'nblk': len(blk), 'blk_sizes': blk[:10],
                'depth': float(np.nanmean(dep)) if len(dep) else None,
                'sad_mean': float(np.mean(sad)) if sad else None,
                'sad_min': float(np.min(sad)) if sad else None,
                'gpcc': gpcc(Ie, Ig), 'iou': dark_iou(Ie, Ig),
                'cores': len(pm)}
        d = {'tag': tag, 'params': [float(x) for x in p], 'combo': combo,
             'n_cores': len(pts_m), 'core_rms': float(np.sqrt(
                 (dr ** 2).mean())), 'core_mean': float(dr.mean()),
             'core_max': float(dr.max()), 'mesh': mesh,
             'per_gamma': per_g}
        print(f'  [{tag}] cores={len(pts_m)} rms={d["core_rms"]:.2f} '
              f'mean={d["core_mean"]:.2f} max={d["core_max"]:.2f} '
              f'meshOK={mesh.get("ok")} '
              f'mesh_rms={(mesh.get("rowA_rms",9)+mesh.get("rowB_rms",9)) if mesh.get("ok") else 9:.2f} '
              f'step={mesh.get("in_row_step",0):.1f} '
              f'gap={mesh.get("row_gap",0):.1f} '
              f'cv={(mesh.get("cvA",0)+mesh.get("cvB",0))/2:.2f}')
        for gg in (0.75, 0.85):
            x = per_g[str(gg)]
            print(f'      gamma={gg}: nblk={x["nblk"]} '
                  f'depth={x["depth"]:.3f} '
                  f'sad={x["sad_mean"]:.3f}/{x["sad_min"]:.3f} '
                  f'gpcc={x["gpcc"]:.4f}')
        return d

    print('\n[compare] incumbent vs tuned:')
    inc = full_eval('incumbent', list(p0[:4] / p0[:4].sum()) + list(p0[4:]))
    tun = full_eval('tuned', pbest)
    # experiment reference
    mesh_e = mesh2x4(pts_e)
    print(f'  [experiment] mesh rms='
          f'{mesh_e["rowA_rms"]+mesh_e["rowB_rms"]:.2f} '
          f'step={mesh_e["in_row_step"]:.1f} '
          f'gap={mesh_e["row_gap"]:.1f} '
          f'cv={(mesh_e["cvA"]+mesh_e["cvB"])/2:.2f} '
          f'sad={np.mean([s for _, s in saddle_isolation(Ie, pts_e)]):.3f}')

    # ---- figure ----
    mesh_tags = []
    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5))
    for k, (tag, d) in enumerate([('incumbent', inc), ('tuned', tun)]):
        p = d['params']
        I0 = render_model(modes_np, lab2k,
                          {'combo': combo, 'params': p, 'scale': None})
        Ig = grender(I0, g=0.85)
        pts_m = find_dark_points(I0)
        A = np.array([(y, x) for y, x, _ in pts_m], float)
        ax = axs[0, k]
        ax.imshow(Ig, cmap='inferno', vmin=0, vmax=1)
        C = np.sqrt(((A[:, None, :] - pts_e_np[None, :, :]) ** 2).sum(-1))
        ri, ci = linear_sum_assignment(C)
        for a, b in zip(ri, ci):
            ax.plot([A[a, 1], pts_e_np[b, 1]], [A[a, 0], pts_e_np[b, 0]],
                    '-', color='cyan', lw=0.8, alpha=0.8)
        for y, x, _v in pts_m:
            ax.plot(x, y, 'x', color='red', ms=8, mew=1.8)
        for y, x, _v in pts_e:
            ax.plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.3)
        ax.set_title(f'{tag}: core RMS={d["core_rms"]:.2f}px '
                     f'max={d["core_max"]:.2f}px\n'
                     f'nblk={d["per_gamma"]["0.85"]["nblk"]} '
                     f'gpcc={d["per_gamma"]["0.85"]["gpcc"]:.4f}', fontsize=10)
        ax.axis('off')
    ax = axs[0, 2]
    ax.imshow(Ie, cmap='inferno', vmin=0, vmax=1)
    for y, x, _v in pts_e:
        ax.plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.3)
    ax.set_title('experiment (cyan = cores)', fontsize=10)
    ax.axis('off')
    # metric bars
    ax = axs[1, 0]
    keys = ['core_rms', 'core_max']
    inc_v = [inc[k] for k in keys]
    tun_v = [tun[k] for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs - 0.18, inc_v, 0.36, label='incumbent')
    ax.bar(xs + 0.18, tun_v, 0.36, label='tuned')
    ax.set_xticks(xs)
    ax.set_xticklabels(['core RMS (px)', 'core max (px)'])
    ax.set_title('position error to experiment cores')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax = axs[1, 1]
    labels3 = ['mesh rms sum', 'cvA+cvB', 'step (px)', 'gap (px)']
    def mvals(d):
        m = d['mesh']
        if not m.get('ok'):
            return [np.nan] * 4
        return [m['rowA_rms'] + m['rowB_rms'], m['cvA'] + m['cvB'],
                m['in_row_step'], m['row_gap']]
    mv_i = mvals(inc)
    mv_t = mvals(tun)
    xs = np.arange(4)
    ax.bar(xs - 0.18, mv_i, 0.36, label='incumbent')
    ax.bar(xs + 0.18, mv_t, 0.36, label='tuned')
    ax.bar(np.arange(4)[2] + 0, [0], 0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels3, fontsize=8)
    ax.set_title('mesh quality (exp: step 23.5, gap 36.9)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax = axs[1, 2]
    for tag, d, col in [('incumbent', inc, 'tab:blue'),
                        ('tuned', tun, 'tab:orange')]:
        per = d['per_gamma']['0.85']
        ax.bar([0 + (0.18 if tag == 'tuned' else -0.18)],
               [per['gpcc']], 0.36, color=col, label=f'{tag} gpcc')
    ax.axhline(gpcc(Ie, Ie), color='gray', lw=0)  # keep ylim sane
    ax.set_ylim(0.9, 1.0)
    ax.set_xticks([0])
    ax.set_xticklabels(['gamma=0.85 appearance PCC'])
    ax.set_title('appearance check')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.suptitle('8Singularities(1): core-geometry targeted tuning of '
                 'the 4-mode TOP-1', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / '核几何对齐优化.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    json.dump({'incumbent': inc, 'tuned': tun, 'J0': J0, 'Jbest': Jb,
               'experiment_mesh': mesh_e},
              open(OUT / 'mesh_tune.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False, default=str)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

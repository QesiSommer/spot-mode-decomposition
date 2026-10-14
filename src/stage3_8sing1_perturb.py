# -*- coding: utf-8 -*-
"""
stage3_8sing1_perturb.py -- 8(1) 微扰第四模扫描
==============================================

Per the confirmed plan:
 1. incumbent triple + each of the other 128 modes (4th-mode energy
    seeded at 7%; refinement may shrink it to 0 = pure incumbent)
 2. refine each quad with the new method (W_avg objective,
    N=4 block refinement)
 3. gamma-domain (0.58, sigma 2.5) evaluation of ALL refined quads:
    global appearance PCC, saddle mean/min (merged-pair detector),
    core depth, dark IoU + structural verification (linear sigma=0)
 4. rank: structural PASS first, then gamma-domain global PCC;
    report saddle metrics for the "obvious cores" goal
    (experiment: saddle mean 0.457, min 0.28)

Outputs: 阶段3_8奇点1微扰/
"""

import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage0_diagnosis import parse_md, core_depth, saddle_isolation
from stage1_gamma_fix import dark_iou
from window_metrics import build_window_stats
from rerun_new_metrics import block_refine_wavg, global_stats_02, \
    c_s02, GRIDS

SRC = ROOT / '归档_八图分解' / '最新结果_新指标'
OUT = ROOT / '归档_八图分解' / '局部优化_5及8奇点' / '阶段3_8奇点1微扰'
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '8Singularities(1)'
CN = '8奇点(1)'
GAMMA = 0.58
SIG_DISP = 2.5
EPS4 = 0.07


def gamma_render(I0, gamma=GAMMA, sg=SIG_DISP):
    Id = gaussian_blur(torch.tensor(I0, dtype=torch.float32)
                       .unsqueeze(0), float(sg))[0].numpy()
    Ig = np.clip(Id, 0, 1) ** gamma
    return Ig / max(Ig.max(), 1e-12)


def gamma_metrics(Ie, Ig, pts_e):
    """appearance PCC + saddle/depth/IoU in the gamma domain."""
    a = Ie.ravel() - Ie.mean()
    b = Ig.ravel() - Ig.mean()
    gpcc = float((a * b).sum()
                 / np.sqrt((a * a).sum() * (b * b).sum()))
    pts_m = find_dark_points(Ig)
    dep = core_depth(Ig, pts_m)
    sad = saddle_isolation(Ig, pts_m)
    s_all = [s for _, s in sad]
    return {
        'gpcc': gpcc,
        'saddle_mean': float(np.mean(s_all)) if s_all else None,
        'saddle_min': float(np.min(s_all)) if s_all else None,
        'depth_mean': float(np.nanmean(dep)) if len(dep) else None,
        'iou': dark_iou(Ie, Ig),
        'cores_gamma': len(pts_m),
    }


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes = modes.to(dev)
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.detach().cpu().numpy()

    combo0, params0, s0 = parse_md(SRC / f'{CN}_参数与指标.md')
    print(f'incumbent: {combo0} scale={s0}')
    Ie_np = load_experiment(f'{IMG_DIR}/{NAME}.png', 'nipy_spectral',
                            'isot')
    Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
    pts_e = find_dark_points(Ie_np)

    # incumbent baseline (gamma domain)
    I0_inc = render_model(modes_np, lab2k,
                          {'combo': combo0, 'params': params0,
                           'scale': s0})
    Ig_inc = gamma_render(I0_inc)
    base = gamma_metrics(Ie_np, Ig_inc, pts_e)
    pts_m0 = find_dark_points(I0_inc)
    mt0 = match_points(pts_e, pts_m0)
    print('incumbent gamma metrics:', base)

    tr0 = [lab2k[m] for m in combo0]
    N = 4
    g = GRIDS[N]
    others = [x for x in range(len(labels)) if x not in tr0]
    print(f'[scan] {len(others)} perturbation modes')

    # ---- 1+2: perturb + refine each quad ----
    t_start = time.time()
    refined = []
    for n, x in enumerate(others):
        quad = sorted(tr0 + [x])
        # seed: incumbent energies * (1-eps), 4th = eps
        u = [p * (1 - EPS4) for p in params0[:3]] + [EPS4]
        seed = list(u) + list(params0[3:]) + [0.0]
        try:
            ws, meta, mask = build_window_stats(modes, Ie, quad, None)
            th, wavg = block_refine_wavg(ws, seed, N, g['n_ph'],
                                         g['n_en'], rounds=5)
        except Exception as e:
            print(f'  skip mode {x}: {e}')
            continue
        refined.append({'quad': quad, 'params': th, 'wavg': wavg})
        if (n + 1) % 32 == 0:
            print(f'  {n+1}/{len(others)} refined '
                  f'({time.time()-t_start:.0f}s)')
    print(f'  refined {len(refined)} quads in '
          f'{time.time()-t_start:.0f}s')

    # ---- 3: gamma-domain evaluation of all ----
    print('[gamma evaluation]')
    evals = []
    for r in refined:
        names = [labels[q] for q in r['quad']]
        I0 = render_model(modes_np, lab2k,
                         {'combo': names, 'params': r['params'],
                          'scale': None})
        Ig = gamma_render(I0)
        gm = gamma_metrics(Ie_np, Ig, pts_e)
        pts_m = find_dark_points(I0)
        mt = match_points(pts_e, pts_m)
        n_ok = abs(len(pts_m) - len(pts_e)) <= 1
        d_ok = mt['mean'] is not None and mt['mean'] < 12
        e4 = r['params'][r['quad'].index(
            [q for q in r['quad'] if q not in tr0][0])]
        evals.append({
            'combo': '+'.join(names), 'quad': r['quad'],
            'params': r['params'], 'e4': e4,
            'wavg': r['wavg'],
            'struct': 'PASS' if (n_ok and d_ok) else 'fail',
            'cores': len(pts_m), 'mean_nn': mt['mean'],
            **gm})
    evals.sort(key=lambda e: -e['gpcc'])

    inc_eval = {'combo': '+'.join(combo0), 'quad': tr0,
                'params': list(params0), 'e4': 0.0, 'wavg': None,
                'struct': 'PASS', 'cores': len(pts_m0),
                'mean_nn': mt0['mean'], 'is_incumbent': True, **base}

    # print top-10 by gpcc
    print(f'{"combo":46s} {"gpcc":>7s} {"sad_mean":>8s} '
          f'{"sad_min":>7s} {"e4":>5s} {"cores":>5s} struct')
    print(f'{"[INCUMBENT] " + inc_eval["combo"]:46s} '
          f'{inc_eval["gpcc"]:7.4f} '
          f'{inc_eval["saddle_mean"]:8.3f} '
          f'{inc_eval["saddle_min"]:7.3f} {"-":>5s} '
          f'{inc_eval["cores"]:5d} {inc_eval["struct"]}')
    for e in evals[:10]:
        sm = '-' if e['saddle_mean'] is None else f'{e["saddle_mean"]:.3f}'
        smin = '-' if e['saddle_min'] is None else f'{e["saddle_min"]:.3f}'
        print(f'{e["combo"]:46s} {e["gpcc"]:7.4f} {sm:>8s} '
              f'{smin:>7s} {e["e4"]:5.3f} {e["cores"]:5d} '
              f'{e["struct"]}')

    # ---- 4: rank ----
    passing = [e for e in evals if e['struct'] == 'PASS']
    pool = passing if passing else evals
    champion = max(pool, key=lambda e: e['gpcc'])
    print(f'\nCHAMPION: {champion["combo"]} gpcc='
          f'{champion["gpcc"]:.4f} struct={champion["struct"]}')

    (OUT / 'perturbation_scan.json').write_text(
        json.dumps({'incumbent': inc_eval, 'evals': evals,
                    'champion': champion}, indent=2,
                   ensure_ascii=False, default=str),
        encoding='utf-8')

    # ---- figure ----
    show = [e for e in ([champion] + pool[:4])
            if not e.get('is_incumbent')][:3]
    fig, axs = plt.subplots(2, 5, figsize=(24, 9.4))
    axs[0, 0].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
    axs[0, 0].set_title('experiment')
    axs[1, 0].imshow(Ie_np[40:190, 40:190], cmap='inferno', vmin=0,
                     vmax=1)
    axs[1, 0].set_title('experiment (zoom)')
    axs[0, 1].imshow(Ig_inc, cmap='inferno', vmin=0, vmax=1)
    axs[0, 1].set_title(f'INCUMBENT {inc_eval["combo"]}\n'
                        f'gpcc={inc_eval["gpcc"]:.3f} sad='
                        f'{inc_eval["saddle_mean"]:.3f}/'
                        f'{inc_eval["saddle_min"]:.3f}', fontsize=9)
    axs[1, 1].imshow(Ig_inc[40:190, 40:190], cmap='inferno', vmin=0,
                     vmax=1)
    for k, e in enumerate(show):
        I0 = render_model(modes_np, lab2k,
                         {'combo': e['combo'].split('+'),
                          'params': e['params'], 'scale': None})
        Ig = gamma_render(I0)
        axs[0, k + 2].imshow(Ig, cmap='inferno', vmin=0, vmax=1)
        tag = 'C' if e is champion else f'#{k+1}'
        sm = '-' if e['saddle_mean'] is None else f'{e["saddle_mean"]:.3f}'
        axs[0, k + 2].set_title(
            f'{tag} {e["combo"]}\n'
            f'gpcc={e["gpcc"]:.3f} sad={sm} '
            f'e4={e["e4"]:.2f}', fontsize=8)
        axs[1, k + 2].imshow(Ig[40:190, 40:190], cmap='inferno',
                            vmin=0, vmax=1)
    for k in range(len(show), 3):
        axs[0, k + 2].axis('off')
        axs[1, k + 2].axis('off')
    for a in axs.ravel():
        a.axis('off')
    fig.suptitle(f'{NAME} stage-3 4th-mode perturbation '
                 f'(gamma={GAMMA}, sigma={SIG_DISP}); '
                 f'C = champion', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / '微扰对比.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

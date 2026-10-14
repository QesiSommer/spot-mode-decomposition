# -*- coding: utf-8 -*-
"""
stage5b2_pure_hg_deep.py -- 纯 HG 深挖 + LG 含量-质量曲线
=========================================================
1. From the cached full-space scores (scan3A.pt): bin ALL 366,145
   combos by total LG energy at their optimum -> per-bin best/median
   PCC -> the "how much LG is needed" curve.  (grid-A scale)
2. Pure-HG deep dive: top-50 pure-HG combos by grid-A score ->
   core-weighted refinement + full-res evaluation (same pipeline as
   stage5b stage C) -> hard evidence about the best pure-HG solutions.
3. Figure: experiment vs best pure-HG renders + bin curve.

Output: 归档_八图分解/8奇点1_HG主导搜索/
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage0_diagnosis import core_depth, saddle_isolation
from stage1_gamma_fix import dark_iou
from rerun_new_metrics import block_refine_wavg, energy_simplex_N, GRIDS
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats
from stage5a_grid_diag import grender, gpcc, blocks
from stage5b_hg_search import (OUT, IMGDIR, NAME, GAMMAS, CORE_W, CORE_R,
                               scan_batch, canon, mesh2x4)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes_g = modes.to(dev)
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()

    Ie = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    crop_f = crop_box(pts_e, 224)
    w_f = weight_map(Ie.shape, pts_e)
    Ie_t = torch.tensor(Ie, dtype=torch.float32, device=dev)
    exp = {'nblk': len(blocks(Ie)),
           'depth': float(np.nanmean(core_depth(Ie, pts_e))),
           'sad_mean': float(np.mean(
               [s for _, s in saddle_isolation(Ie, pts_e)]))}
    mesh_exp = mesh2x4(pts_e)
    print(f'[exp] nblk={exp["nblk"]} mesh ok={mesh_exp["ok"]} '
          f'rms sum={mesh_exp.get("rowA_rms",0)+mesh_exp.get("rowB_rms",0):.2f} '
          f'step={mesh_exp.get("in_row_step"):.1f} '
          f'gap={mesh_exp.get("row_gap"):.1f}')

    # ---- 1. LG-energy binning over ALL combos ----
    ck = torch.load(OUT / 'scan3A.pt', weights_only=False)
    scores = ck['scores'].numpy()
    bestidx = ck['bestidx'].numpy()
    triples = torch.combinations(torch.arange(131), r=3)
    tri_np = triples.numpy()
    lg_mask = np.array([l.startswith('LG') for l in labels])
    selL = lg_mask[tri_np]
    e_pts = energy_simplex_N(3, 6)
    axA = np.linspace(0, 2 * np.pi, 9, endpoint=False)[:8]
    P2, P3 = np.meshgrid(axA, axA, indexing='ij')
    phA = np.stack([P2.ravel(), P3.ravel()], axis=1)
    pmatA = np.concatenate([np.repeat(e_pts, len(phA), 0),
                            np.tile(phA, (len(e_pts), 1))], 1)
    pmat_opt = pmatA[bestidx]
    lg_e = (pmat_opt[:, :3] * selL).sum(axis=1)

    edges = [-1e-9, 1e-9, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.01]
    bin_rows = []
    print('\n[LG bin]  range            n       best     p99      median'
          '   best combo')
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        sel = (lg_e > lo) & (lg_e <= hi)
        n = int(sel.sum())
        if n == 0:
            continue
        sc = scores[sel]
        order = np.argsort(-sc)
        idx_all = np.where(sel)[0]
        bi = idx_all[order[0]]
        combo = '+'.join(labels[x] for x in tri_np[bi])
        row = {'lo': lo, 'hi': hi, 'n': n,
               'best': float(sc.max()),
               'p99': float(np.quantile(sc, 0.99)),
               'median': float(np.median(sc)),
               'best_combo': combo,
               'best_lg_e': float(lg_e[bi]),
               'best_params': [float(x) for x in pmat_opt[bi]]}
        bin_rows.append(row)
        lo_s = '0' if lo < 0 else f'{lo*100:.0f}%'
        hi_s = f'{hi*100:.0f}%'
        print(f'  [{lo_s:>4s},{hi_s:>4s}]  {n:7d}  {sc.max():.4f}  '
              f'{np.quantile(sc,0.99):.4f}  {np.median(sc):.4f}   {combo}')

    # ---- 2. pure-HG deep dive: top-50 by grid-A score ----
    iu_pure = np.where(lg_e <= 1e-9)[0]
    order = iu_pure[np.argsort(-scores[iu_pure])][:50]
    print(f'\n[pure-HG deep] top-50 pure combos, best score='
          f'{scores[order[0]]:.4f}')
    # refinement seeds from grid A optimum params
    g3 = GRIDS[3]
    evals = []
    for rank, ti in enumerate(order):
        tr = tri_np[ti].tolist()
        seed = pmat_opt[ti].tolist()
        ws = weighted_crop_stats(modes_g, Ie_t, tr, None, crop_f, w_f,
                                 sigmas=(0, 2))
        th, val = block_refine_wavg(ws, seed, 3, g3['n_ph'], g3['n_en'],
                                    rounds=4)
        names = [labels[x] for x in tr]
        I0 = render_model(modes_np, lab2k,
                          {'combo': names, 'params': th, 'scale': None})
        pts_m0 = find_dark_points(I0)
        mt = match_points(pts_e, pts_m0)
        struct = ('PASS' if (abs(len(pts_m0) - len(pts_e)) <= 1
                             and mt['mean'] is not None
                             and mt['mean'] < 12) else 'fail')
        mesh = mesh2x4(pts_m0)
        best_nb = 99
        best_gp = 0.0
        per_g = {}
        for gg in GAMMAS:
            Ig = grender(I0, g=gg)
            blk = blocks(Ig)
            pts_m = find_dark_points(Ig)
            dep = core_depth(Ig, pts_m)
            sad = [s for _, s in saddle_isolation(Ig, pts_m)]
            per_g[str(gg)] = {
                'nblk': len(blk),
                'depth': float(np.nanmean(dep)) if len(dep) else None,
                'sad_mean': float(np.mean(sad)) if sad else None,
                'gpcc': gpcc(Ie, Ig), 'iou': dark_iou(Ie, Ig),
                'cores': len(pts_m)}
            best_nb = min(best_nb, abs(len(blk) - exp['nblk']))
            best_gp = max(best_gp, gpcc(Ie, Ig))
        evals.append({'combo': '+'.join(names), 'tr': tr, 'params': th,
                      'scanA': float(scores[ti]), 'wavg_crop': val,
                      'struct': struct, 'cores0': len(pts_m0),
                      'mean_nn': mt['mean'], 'mesh': mesh,
                      'per_gamma': per_g,
                      'best_nblk_gap': best_nb, 'best_gpcc': best_gp})
        if (rank + 1) % 10 == 0:
            print(f'  {rank+1}/50 done')
    evals.sort(key=lambda e: (0 if e['struct'] == 'PASS' else 1,
                              e['best_nblk_gap'], -e['best_gpcc']))
    print('\n[pure-HG deep] ranked top-8:')
    for i, e in enumerate(evals[:8]):
        m = e['mesh']
        mm = (f'mesh[rms={m.get("rowA_rms",0)+m.get("rowB_rms",0):.1f}]'
              if m.get('ok') else 'mesh[FAIL]')
        print(f'  #{i+1} {e["combo"]:38s} scanA={e["scanA"]:.3f} '
              f'struct={e["struct"]:4s} cores={e["cores0"]} '
              f'{mm} nblkGap={e["best_nblk_gap"]} '
              f'gpcc={e["best_gpcc"]:.3f}')

    # ---- 3. figure ----
    fig = plt.figure(figsize=(21, 9.5))
    gs = fig.add_gridspec(2, 4)
    # (top row) LG bin curve
    ax = fig.add_subplot(gs[0, :2])
    xs = [(r['lo'] + r['hi']) / 2 for r in bin_rows]
    xs[0] = 0.0
    ax.plot(xs, [r['best'] for r in bin_rows], 'o-', label='best')
    ax.plot(xs, [r['p99'] for r in bin_rows], 's--', label='p99')
    ax.plot(xs, [r['median'] for r in bin_rows], '^-', label='median')
    ax.axvline(0.3627, color='r', ls=':', lw=1.2,
               label='incumbent LG=36.3%')
    ax.set_xlabel('LG energy (total, at optimum)')
    ax.set_ylabel('core-weighted local PCC (grid A)')
    ax.set_title('How much LG does the experiment need? '
                 '(all 366,145 combos binned)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    for x, r in zip(xs, bin_rows):
        ax.annotate(f'n={r["n"]}', (x, r['best']), fontsize=7,
                    xytext=(0, 6), textcoords='offset points',
                    ha='center')
    # best-combo renders: pure-HG champion vs incumbent
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(Ie, cmap='inferno', vmin=0, vmax=1)
    ax.set_title('experiment')
    ax.axis('off')
    e0 = evals[0]
    gb0 = min(GAMMAS, key=lambda g: abs(
        e0['per_gamma'][str(g)]['nblk'] - exp['nblk']))
    I0 = render_model(modes_np, lab2k,
                      {'combo': e0['combo'].split('+'),
                       'params': e0['params'], 'scale': None})
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(grender(I0, g=gb0), cmap='inferno', vmin=0, vmax=1)
    ax.set_title(f'best pure-HG\n{e0["combo"]}\n'
                 f'cores={e0["cores0"]} nblk='
                 f'{e0["per_gamma"][str(gb0)]["nblk"]} '
                 f'gpcc={e0["best_gpcc"]:.3f}', fontsize=8)
    ax.axis('off')
    # (bottom) top-3 pure-HG full + crop
    for k in range(3):
        e = evals[k]
        gb = min(GAMMAS, key=lambda g: abs(
            e['per_gamma'][str(g)]['nblk'] - exp['nblk']))
        I0 = render_model(modes_np, lab2k,
                          {'combo': e['combo'].split('+'),
                           'params': e['params'], 'scale': None})
        Ig = grender(I0, g=gb)
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(Ig, cmap='inferno', vmin=0, vmax=1)
        pts_m = [(y, x) for y, x, _ in find_dark_points(I0)]
        ax.set_title(f'pure-HG #{k+1}\n{e["combo"]}\n'
                     f'cores={e["cores0"]} (exp 8) nblk='
                     f'{e["per_gamma"][str(gb)]["nblk"]} '
                     f'(exp {exp["nblk"]})', fontsize=7.5)
        ax.axis('off')
    # per-bin evidence table as text panel
    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')
    txt = ('pure-HG deep dive (top-50 by scan score):\n'
           + '\n'.join(
               f'{i+1}. {e["combo"][:34]:34s}\n'
               f'    struct={e["struct"]}  cores={e["cores0"]}/8  '
               f'gpcc={e["best_gpcc"]:.3f}'
               for i, e in enumerate(evals[:4])))
    ax.text(0.0, 1.0, txt, va='top', ha='left', fontsize=8.6,
            family='monospace')
    fig.suptitle('8Singularities(1): LG-requirement curve & pure-HG '
                 'deep dive', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / '纯HG深挖与LG曲线.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    json.dump({'bins': bin_rows,
               'mesh_exp': mesh_exp,
               'pure_hg_top': [{k: (v if k != 'per_gamma' else
                                    {kk: {k3: v3 for k3, v3 in vv.items()}
                                     for kk, vv in v.items()})
                                for k, v in e.items()} for e in evals[:20]]},
              open(OUT / 'pure_hg_deep.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False, default=str)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

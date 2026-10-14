# -*- coding: utf-8 -*-
"""
stage5d_lg2_plus_3hg.py -- LG(2,0)+3HG 四模子空间搜索（网格规整度导向）
======================================================================
Subspace: LG(2,0) fixed + C(66,3) = 45,760 HG triples (4 modes total).
LG(2,0) supplies the ring-shaped bright structure (proven necessary:
full-space optimum sits at LG 36%); the search replaces the HG part to
find a MORE REGULAR 2x4 core mesh.

  A. quarter-res scan over all 45,760 (core-weighted, 2240 pts/combo)
     -> top-400
  B. half-res rescan -> top-60 canon-dedup
  C. refinement (block_refine_wavg N=4) + full-res evaluation:
       core RMS to experiment cores (Hungarian), mesh geometry
       (row rms / cv / step / gap), gamma-domain nblk/depth/saddle/gpcc
     rank: struct PASS > |nblk-8|<=1 > core RMS > gpcc

Output: 归档_八图分解/8奇点1_HG主导搜索/
"""
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from refine_all_new_method import c_table_N
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage0_diagnosis import core_depth, saddle_isolation
from stage1_gamma_fix import dark_iou
from rerun_new_metrics import block_refine_wavg, energy_simplex_N, GRIDS
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats
from stage5a_grid_diag import grender, gpcc, blocks
from stage5b_hg_search import (OUT, IMGDIR, NAME, GAMMAS,
                               scan_batch, canon, mesh2x4, prep_domain)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={dev}')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes_g = modes.to(dev)
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()

    Ie = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    pts_e_np = np.array([(y, x) for y, x, _ in pts_e], float)
    crop_f = crop_box(pts_e, 224)
    w_f = weight_map(Ie.shape, pts_e)
    Ie_t = torch.tensor(Ie, dtype=torch.float32, device=dev)
    exp = {'nblk': len(blocks(Ie)),
           'depth': float(np.nanmean(core_depth(Ie, pts_e))),
           'sad_mean': float(np.mean(
               [s for _, s in saddle_isolation(Ie, pts_e)]))}
    mesh_exp = mesh2x4(pts_e)

    hg_idx = torch.arange(65, 131)
    tri = torch.combinations(hg_idx, r=3)
    lg2k = lab2k['LG(2,0)']
    quads = torch.cat([torch.full((len(tri), 1), lg2k,
                                  dtype=torch.long), tri], 1)
    total = quads.shape[0]
    print(f'[setup] subspace = LG(2,0) + C(66,3) = {total} quads')

    D4 = prep_domain(modes_g, Ie, crop_f, pts_e, 56, 4, (0.0, 0.5), dev)
    D2 = prep_domain(modes_g, Ie, crop_f, pts_e, 112, 2, (0.0, 1.0), dev)

    eA = energy_simplex_N(4, 4)                       # 35 pts
    axA = np.linspace(0, 2 * np.pi, 5, endpoint=False)[:4]
    Q2, Q3, Q4 = np.meshgrid(axA, axA, axA, indexing='ij')
    phA = np.stack([Q2.ravel(), Q3.ravel(), Q4.ravel()], axis=1)
    pmatA = np.concatenate([np.repeat(eA, len(phA), 0),
                            np.tile(phA, (len(eA), 1))], 1)
    CA = c_table_N(pmatA, 4).to(dev)
    eB = energy_simplex_N(4, 5)                       # 56 pts
    axB = np.linspace(0, 2 * np.pi, 6, endpoint=False)[:5]
    R2, R3, R4 = np.meshgrid(axB, axB, axB, indexing='ij')
    phB = np.stack([R2.ravel(), R3.ravel(), R4.ravel()], axis=1)
    pmatB = np.concatenate([np.repeat(eB, len(phB), 0),
                            np.tile(phB, (len(eB), 1))], 1)
    CB = c_table_N(pmatB, 4).to(dev)
    print(f'[setup] gridA={CA.shape[0]} gridB={CB.shape[0]} pts/combo')

    # ---- STAGE A ----
    cache = OUT / 'scan_lg3hg_A.pt'
    if cache.exists():
        ck = torch.load(cache, weights_only=False)
        topA = ck['topA']
        print(f'[A] cached, skip scan')
    else:
        print(f'[A] scan {total} quads (quarter-res)...')
        t0 = time.time()
        heap = []
        BATCH = 64
        for b0 in range(0, total, BATCH):
            qb = quads[b0:b0 + BATCH].to(dev)
            corr = scan_batch(dev, CA, D4, qb)
            mx, mi = corr.max(dim=1)
            for v, ii in zip(mx.tolist(), mi.tolist()):
                heapq.heappush(heap, (float(v), int(b0), int(ii)))
                if len(heap) > 400:
                    heapq.heappop(heap)
            if (b0 // BATCH) % 300 == 299 or b0 + BATCH >= total:
                el = time.time() - t0
                print(f'  {min(b0+BATCH,total)}/{total} ({el:.0f}s)')
        topA = sorted(heap, reverse=True)
        torch.save({'topA': topA}, cache)
        print(f'[A] done {time.time()-t0:.0f}s')
    # decode: (score, b0, local_i) -> global index b0+local_i
    giA = np.array([a[1] + a[2] for a in topA])

    # ---- STAGE B ----
    print('[B] half-res rescan of top-400...')
    t0 = time.time()
    sB = torch.zeros(len(giA))
    bB = torch.zeros(len(giA), dtype=torch.long)
    qiA = quads[torch.tensor(giA)]
    for b0 in range(0, len(giA), 50):
        qb = qiA[b0:b0 + 50].to(dev)
        corr = scan_batch(dev, CB, D2, qb)
        mx, mi = corr.max(dim=1)
        sB[b0:b0 + qb.shape[0]] = mx.cpu()
        bB[b0:b0 + qb.shape[0]] = mi.cpu()
    vB, iB = sB.topk(min(200, len(sB)))
    seen, cands = set(), []
    for v, k in zip(vB.tolist(), iB.tolist()):
        tr = qiA[k].tolist()
        c = canon(tr, labels)
        if c in seen:
            continue
        seen.add(c)
        cands.append({'tr': tr, 'seed': pmatB[bB[k]].tolist(),
                      'scanB': v})
        if len(cands) >= 60:
            break
    print(f'[B] done {time.time()-t0:.0f}s; candidates={len(cands)}')

    # ---- STAGE C ----
    print('[C] refinement + full-res evaluation...')
    g4 = GRIDS[4]
    evals = []
    t0 = time.time()
    for ci, c in enumerate(cands):
        tr = c['tr']
        ws = weighted_crop_stats(modes_g, Ie_t, tr, None, crop_f, w_f,
                                 sigmas=(0, 2))
        th, val = block_refine_wavg(ws, c['seed'], 4, g4['n_ph'],
                                    g4['n_en'], rounds=g4['rounds'])
        names = [labels[x] for x in tr]
        I0 = render_model(modes_np, lab2k,
                          {'combo': names, 'params': th, 'scale': None})
        pts_m0 = find_dark_points(I0)
        C = np.sqrt(((np.array([(y, x) for y, x, _ in pts_m0], float)
                      [:, None, :] - pts_e_np[None, :, :]) ** 2).sum(-1)) \
            if pts_m0 else np.zeros((1, 1))
        if len(pts_m0) >= 2 and pts_m0:
            ri, ci2 = linear_sum_assignment(C)
            dr = C[ri, ci2]
            core_rms = float(np.sqrt((dr ** 2).mean()))
            core_max = float(dr.max())
        else:
            core_rms, core_max = 99.0, 99.0
        mt = match_points(pts_e, pts_m0)
        struct = ('PASS' if (abs(len(pts_m0) - len(pts_e)) <= 1
                             and mt['mean'] is not None
                             and mt['mean'] < 12) else 'fail')
        mesh = mesh2x4(pts_m0)
        per_g = {}
        for gg in GAMMAS:
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
        evals.append({'combo': '+'.join(names), 'tr': tr, 'params': th,
                      'scanB': c['scanB'], 'wavg_crop': val,
                      'struct': struct, 'cores0': len(pts_m0),
                      'core_rms': core_rms, 'core_max': core_max,
                      'mean_nn': mt['mean'], 'mesh': mesh,
                      'per_gamma': per_g})
        if (ci + 1) % 15 == 0:
            print(f'  {ci+1}/{len(cands)} ({time.time()-t0:.0f}s)')
    print(f'[C] done {time.time()-t0:.0f}s')

    def key(e):
        d = e['per_gamma']['0.85']
        return (0 if e['struct'] == 'PASS' else 1,
                0 if abs(d['nblk'] - 8) <= 1 else 1,
                e['core_rms'], -d['gpcc'])
    evals.sort(key=key)

    print('\n=== TOP-15 (struct > nblk > core-RMS > gpcc) ===')
    for i, e in enumerate(evals[:15]):
        m = e['mesh']
        d = e['per_gamma']['0.85']
        mm = (f'mesh[rms={m.get("rowA_rms",0)+m.get("rowB_rms",0):.2f} '
              f'cv={(m.get("cvA",0)+m.get("cvB",0)):.2f} '
              f'step={m.get("in_row_step",0):.1f} '
              f'gap={m.get("row_gap",0):.1f}]'
              if m.get('ok') else 'mesh[FAIL]')
        print(f'  #{i+1:2d} {e["combo"]:50s} struct={e["struct"]:4s} '
              f'coreRMS={e["core_rms"]:5.2f} max={e["core_max"]:5.2f} '
              f'{mm} nblk={d["nblk"]} '
              f'sad={d["sad_mean"] if d["sad_mean"] is None else round(d["sad_mean"],3)} '
              f'gpcc={d["gpcc"]:.4f}')
    print(f'\n[ref] experiment mesh: rms='
          f'{mesh_exp["rowA_rms"]+mesh_exp["rowB_rms"]:.2f} '
          f'cv={(mesh_exp["cvA"]+mesh_exp["cvB"]):.2f} '
          f'step={mesh_exp["in_row_step"]:.1f} '
          f'gap={mesh_exp["row_gap"]:.1f} | nblk={exp["nblk"]} '
          f'sad={exp["sad_mean"]:.3f}')
    print(f'[ref] incumbent 4-mode TOP-1: coreRMS=6.62 max=9.22 '
          f'mesh[rms=2.55 cv=0.50 step=22.5 gap=29.8] nblk=8 '
          f'sad=0.403 gpcc=0.9591')

    json.dump({'experiment': exp, 'mesh_exp': mesh_exp,
               'top15': [{'combo': e['combo'], 'params': e['params'],
                          'struct': e['struct'], 'core_rms': e['core_rms'],
                          'core_max': e['core_max'], 'mesh': e['mesh'],
                          'per_gamma': e['per_gamma']}
                         for e in evals[:15]],
               'all': [{'combo': e['combo'], 'params': e['params'],
                        'struct': e['struct'], 'core_rms': e['core_rms'],
                        'core_max': e['core_max'], 'mesh': e['mesh'],
                        'per_gamma': {k: {kk: vv for kk, vv in v.items()
                                          if kk != 'blk_sizes'}
                                      for k, v in e['per_gamma'].items()}}
                       for e in evals]},
              open(OUT / 'lg3hg_results.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False, default=str)

    # ---- quick figure: exp vs top-2 (full + crop, with core links) ----
    crop = (slice(40, 190), slice(40, 190))
    show = [e for e in evals[:4]]
    fig, axs = plt.subplots(2, 3, figsize=(17, 10.4))
    axs[0, 0].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
    axs[0, 0].set_title(f'EXPERIMENT ({len(pts_e)} cores)', fontsize=10)
    axs[1, 0].imshow(Ie[crop], cmap='inferno', vmin=0, vmax=1)
    axs[1, 0].set_title('experiment (crop)')
    for k in range(2):
        e = show[k]
        I0 = render_model(modes_np, lab2k,
                          {'combo': e['combo'].split('+'),
                           'params': e['params'], 'scale': None})
        Ig = grender(I0, g=0.85)
        d = e['per_gamma']['0.85']
        pts_m0 = find_dark_points(I0)
        axs[0, k + 1].imshow(Ig, cmap='inferno', vmin=0, vmax=1)
        A = np.array([(y, x) for y, x, _ in pts_m0], float)
        if len(A):
            C = np.sqrt(((A[:, None, :] - pts_e_np[None, :, :]) ** 2)
                        .sum(-1))
            ri, ci2 = linear_sum_assignment(C)
            for a, b in zip(ri, ci2):
                axs[0, k + 1].plot([A[a, 1], pts_e_np[b, 1]],
                                   [A[a, 0], pts_e_np[b, 0]], '-',
                                   color='cyan', lw=0.7, alpha=0.75)
        axs[0, k + 1].set_title(
            f'#{k+1} {e["combo"]}\nstruct={e["struct"]} '
            f'coreRMS={e["core_rms"]:.2f} nblk={d["nblk"]} '
            f'gpcc={d["gpcc"]:.4f}', fontsize=8)
        axs[1, k + 1].imshow(Ig[crop], cmap='inferno', vmin=0, vmax=1)
        axs[1, k + 1].set_title(f'sad='
                                f'{d["sad_mean"] if d["sad_mean"] is None else round(d["sad_mean"],3)} '
                                f'depth={round(d["depth"],3) if d["depth"] else None}',
                                fontsize=9)
    for a in axs.ravel():
        a.axis('off')
    fig.suptitle('8Singularities(1): LG(2,0)+3HG subspace — top-2 '
                 '(cyan links = core correspondence)', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'LG+3HG_前三.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

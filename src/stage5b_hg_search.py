# -*- coding: utf-8 -*-
"""
stage5b_hg_search.py -- HG 主导三模全空间搜索（第2步核心）
==========================================================
Full 3-mode space C(131,3) = 366,145 (includes the pure-HG subset
C(66,3) = 45,760).  Funnel:

  A. quarter-res (56^2) core-weighted local scan over ALL triples
     (sigma {0, 0.5}, energy simplex steps=6 x phase 8^2) -> top-400
  B. half-res (112^2) rescan of top-400 (sigma {0, 1}, steps=8 x
     phase 16^2) -> top-40 (canon dedup)
  C. core-weighted refinement (block_refine_wavg, N=3) + full-res
     evaluation:
       - linear domain: 8-core detection + position match (struct)
       - core-geometry: 2x4 mesh criterion (2 parallel rows of 4
         collinear equidistant cores; basis-free) + grid_fit RMS
       - gamma domain (0.75, 0.85): nblk / depth / saddle / gpcc
     rank: struct PASS > mesh ok > mesh RMS > |nblk-8| > gpcc

Every candidate is annotated with its total LG energy so the results
can be split into: pure-HG / LG<=10% / all.

Caches: scan3A.pt (top-400), scan3B.json (top-40 with seeds).
Output: 归档_八图分解/8奇点1_HG主导搜索/
"""
import heapq
import json
import sys
import time
from itertools import combinations
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from search_three_mode import eval_c_grid
from refine_all_new_method import build_base_fields_N, c_table_N
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage0_diagnosis import core_depth, saddle_isolation
from stage1_gamma_fix import dark_iou
from rerun_new_metrics import block_refine_wavg, energy_simplex_N, GRIDS
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats
from stage5a_grid_diag import grid_fit, grender, gpcc, blocks

OUT = ROOT / '归档_八图分解' / '8奇点1_HG主导搜索'
OUT.mkdir(parents=True, exist_ok=True)
IMGDIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '8Singularities(1)'
GAMMAS = (0.75, 0.85)
CORE_W, CORE_R = 4.0, 10


# ---------------------------------------------------------------------------
# generic N-mode core-weighted scan
# ---------------------------------------------------------------------------

def scan_batch(dev, C, stats_fn, qb):
    """qb: (B,N) long on dev. Returns (B,) corr = mean over sigmas of
    best c-grid correlation on the weighted crop."""
    N = qb.shape[1]
    Fs = [stats_fn['modes'][qb[:, i]] for i in range(N)]
    G = build_base_fields_N(Fs)
    B, K = G.shape[0], G.shape[1]
    acc = None
    for sg in stats_fn['sigmas']:
        Gb = gaussian_blur(G.reshape(B * K, *G.shape[2:]),
                           float(sg)).reshape(B, K, *G.shape[2:])
        Ie_b = gaussian_blur(stats_fn['Ie'].unsqueeze(0), float(sg))[0]
        ch = stats_fn['crop']
        Gc = Gb[:, :, ch[0], ch[1]].reshape(B, K, -1)
        Ic = Ie_b[ch[0], ch[1]].reshape(-1)
        wt = stats_fn['wt']
        Wsum = stats_fn['Wsum']
        m = torch.einsum('bki,i->bk', Gc, wt) / Wsum
        E = torch.einsum('bki,i->bk', Gc, Ic * wt) / Wsum
        Gram = torch.einsum('bki,bli->bkl', Gc, Gc * wt) / Wsum
        me = float((Ic * wt).sum() / Wsum)
        ve = float(((Ic - me) ** 2 * wt).sum() / Wsum)
        corr = eval_c_grid(C, m, E, Gram, me, ve)
        acc = corr if acc is None else acc + corr
    return acc / len(stats_fn['sigmas'])


def canon(tr, labels):
    out = []
    for x in tr:
        nm = labels[int(x)]
        kind, rest = nm.split('(')
        pp, ll = rest[:-1].split(',')
        out.append(f'{kind}({pp},{abs(int(ll))})')
    return tuple(sorted(out))


def prep_domain(modes, Ie, crop_f, pts_e, size, factor, sigmas, dev):
    mnp = torch.stack([down_complex(m, size) for m in modes]).to(dev)
    Ied = F.interpolate(
        torch.tensor(Ie, dtype=torch.float32)[None, None],
        size=(size, size), mode='bilinear',
        align_corners=False)[0, 0].to(dev)
    ch = (slice(crop_f[0].start // factor,
                (crop_f[0].stop + factor - 1) // factor),
          slice(crop_f[1].start // factor,
                (crop_f[1].stop + factor - 1) // factor))
    wh = np.ones((size, size), dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    r = CORE_R / factor
    for (y, x, _v) in pts_e:
        d2 = (yy - y / factor) ** 2 + (xx - x / factor) ** 2
        wh[d2 <= r * r] = CORE_W
    wt = torch.tensor(wh, dtype=torch.float32, device=dev)[ch[0],
                                                           ch[1]].reshape(-1)
    return {'modes': mnp, 'Ie': Ied, 'crop': ch, 'wt': wt,
            'Wsum': float(wt.sum()), 'sigmas': sigmas}


def down_complex(x, size):
    t = torch.view_as_real(x).permute(2, 0, 1)[None]
    s = F.interpolate(t, size=(size, size), mode='bilinear',
                      align_corners=False)
    return torch.view_as_complex(s[0].permute(1, 2, 0).contiguous())


# ---------------------------------------------------------------------------
# 2x4 mesh criterion (basis-free)
# ---------------------------------------------------------------------------

def _line_fit(P):
    """PCA line fit: returns (rms, unit_dir, t_sorted_idx)."""
    c = P.mean(0)
    Q = P - c
    U, S, Vt = np.linalg.svd(Q, full_matrices=False)
    d = Vt[0]
    t = Q @ d
    resid = Q - np.outer(t, d)
    rms = float(np.sqrt((resid ** 2).sum(1).mean()))
    return rms, d, t


def mesh2x4(pts):
    """Try to explain 8 cores as 2 parallel rows x 4 collinear,
    equidistant points (basis-free, exhaustive over 35 splits)."""
    P0 = np.array([(y, x) for y, x, _ in pts], dtype=np.float64)
    if len(P0) != 8:
        return {'ok': False, 'n': len(P0)}
    idx = list(range(8))
    best = None
    for A in combinations(idx, 4):
        B4 = [i for i in idx if i not in A]
        PA, PB = P0[list(A)], P0[B4]
        rA, dA, tA = _line_fit(PA)
        rB, dB, tB = _line_fit(PB)
        if rA > 3.5 or rB > 3.5:
            continue
        par = abs(float(dA @ dB))
        if par < np.cos(np.radians(7)):
            continue
        # spacing uniformity within each row
        tA_s = np.sort(tA)
        tB_s = np.sort(tB)
        # orient B to A
        if dA @ dB < 0:
            dA = -dA
        # project both onto dA
        tA_o = PA @ dA
        tB_o = PB @ dA
        oA = np.argsort(tA_o)
        oB = np.argsort(tB_o)
        gA = np.diff(np.sort(tA_o))
        gB = np.diff(np.sort(tB_o))
        cvA = float(gA.std() / (gA.mean() + 1e-9))
        cvB = float(gB.std() / (gB.mean() + 1e-9))
        # pairing displacement consistency
        disp = PB[oB] - PA[oA]
        disp_rms = float(np.sqrt(disp.var(0).sum()))
        row_gap = float(np.linalg.norm(disp.mean(0)))
        score = rA + rB + 8 * (cvA + cvB) + disp_rms
        if best is None or score < best['score']:
            best = {'ok': True, 'score': float(score),
                    'rowA_rms': rA, 'rowB_rms': rB,
                    'par_cos': par, 'cvA': cvA, 'cvB': cvB,
                    'disp_rms': disp_rms, 'row_gap': row_gap,
                    'in_row_step': float((gA.mean() + gB.mean()) / 2),
                    'row_dir_deg': float(np.degrees(
                        np.arctan2(dA[0], dA[1]))),
                    'rows': [sorted(A), sorted(B4)]}
    if best is None:
        return {'ok': False, 'n': 8}
    return best


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={dev}')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes_g = modes.to(dev)
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()
    lg_set = {k for k, l in enumerate(labels) if l.startswith('LG')}

    Ie = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    crop_f = crop_box(pts_e, 224)
    exp = {'nblk': len(blocks(Ie)),
           'depth': float(np.nanmean(core_depth(Ie, pts_e))),
           'sad_mean': float(np.mean(
               [s for _, s in saddle_isolation(Ie, pts_e)]))}
    # experiment geometry reference
    mesh_exp = mesh2x4(pts_e)
    print(f'[setup] experiment: cores={len(pts_e)} '
          f'mesh2x4={mesh_exp["ok"]} '
          f'in_row={mesh_exp.get("in_row_step"):.1f}px '
          f'gap={mesh_exp.get("row_gap"):.1f}px')

    # ---- domains ----
    print('[setup] building downsampled domains...')
    D4 = prep_domain(modes_g, Ie, crop_f, pts_e, 56, 4, (0.0, 0.5), dev)
    D2 = prep_domain(modes_g, Ie, crop_f, pts_e, 112, 2, (0.0, 1.0), dev)

    eA = energy_simplex_N(3, 6)                       # 28 pts
    axA = np.linspace(0, 2 * np.pi, 9, endpoint=False)[:8]
    P2, P3 = np.meshgrid(axA, axA, indexing='ij')
    phA = np.stack([P2.ravel(), P3.ravel()], axis=1)
    pmatA = np.concatenate([np.repeat(eA, len(phA), 0),
                            np.tile(phA, (len(eA), 1))], 1)
    CA = c_table_N(pmatA, 3).to(dev)
    eB = energy_simplex_N(3, 8)                       # 45 pts
    axB = np.linspace(0, 2 * np.pi, 17, endpoint=False)[:16]
    Q2, Q3 = np.meshgrid(axB, axB, indexing='ij')
    phB = np.stack([Q2.ravel(), Q3.ravel()], axis=1)
    pmatB = np.concatenate([np.repeat(eB, len(phB), 0),
                            np.tile(phB, (len(eB), 1))], 1)
    CB = c_table_N(pmatB, 3).to(dev)
    print(f'[setup] grid A: {CA.shape[0]} pts/combo | '
          f'grid B: {CB.shape[0]} pts/combo')

    # ---- STAGE A ----
    total = 131 * 130 * 129 // 6
    triples = torch.combinations(torch.arange(131), r=3)
    tri_np = triples.numpy()
    lg_mask = np.array([labels[i].startswith('LG') for i in range(131)])
    selL = lg_mask[tri_np]                      # (T,3) which slots are LG
    cache_q = OUT / 'scan3A.pt'
    if cache_q.exists():
        ck = torch.load(cache_q, weights_only=False)
        if 'scores' in ck:
            topA = ck['topA']
            scores = ck['scores'].numpy()
            bestidx = ck['bestidx'].numpy()
            print(f'[A] loaded cached full scores '
                  f'({len(scores)} combos), skip scan')
        else:
            cache_q.unlink()
            ck = None
    else:
        ck = None
    if not cache_q.exists():
        print(f'[A] full-space scan over {total} triples '
              f'(quarter-res, core-weighted)...')
        t0 = time.time()
        heap = []
        scores = np.zeros(total, dtype=np.float32)
        bestidx = np.zeros(total, dtype=np.int64)
        BATCH = 96
        done = 0
        for b0 in range(0, total, BATCH):
            qb = triples[b0:b0 + BATCH].to(dev)
            corr = scan_batch(dev, CA, D4, qb)
            mx, mi = corr.max(dim=1)
            mx_np = mx.cpu().numpy()
            mi_np = mi.cpu().numpy()
            scores[b0:b0 + qb.shape[0]] = mx_np
            bestidx[b0:b0 + qb.shape[0]] = mi_np
            for k in range(len(mx_np)):
                heapq.heappush(heap, (float(mx_np[k]), int(b0 + k)))
                if len(heap) > 400:
                    heapq.heappop(heap)
            done += qb.shape[0]
            if (b0 // BATCH) % 600 == 599 or done >= total:
                el = time.time() - t0
                print(f'  {done}/{total}  ({el:.0f}s, eta '
                      f'{el*(total-done)/max(done,1):.0f}s)')
        print(f'[A] done in {time.time()-t0:.0f}s; heap={len(heap)}')
        topA = sorted(heap, reverse=True)
        torch.save({'topA': topA, 'scores': torch.tensor(scores),
                    'bestidx': torch.tensor(bestidx)}, cache_q)
        print(f'[A] top-400 + full scores cached')

    # LG energy of each combo at its grid-A optimum
    pmat_opt = pmatA[bestidx]                    # (T,5)
    lg_e = (pmat_opt[:, :3] * selL).sum(axis=1)

    # ---- STAGE B: per-group top selection + half-res rescan ----
    print('[B] grouped top selection + half-res rescan...')
    t0 = time.time()
    cands = []
    if (OUT / 'scan3B.json').exists():
        cands = json.load(open(OUT / 'scan3B.json', encoding='utf-8'))
        print(f'[B] loaded {len(cands)} candidates from cache')
    else:
        def group_top(mask, k):
            idx = np.where(mask)[0]
            order = idx[np.argsort(-scores[idx])][:k]
            return order.tolist()
        g_all = group_top(np.ones(total, bool), 30)
        g_pure = group_top(lg_e <= 1e-9, 20)
        g_le10 = group_top(lg_e <= 0.10, 20)
        print(f'  group tops: all={len(g_all)} pure={len(g_pure)} '
              f'le10={len(g_le10)}; '
              f'pure best score={scores[g_pure[0]]:.4f} '
              f'(rank in all: last {len(lg_e)} )')
        order_all = np.argsort(-scores)
        rank_of = np.empty(total, dtype=np.int64)
        rank_of[order_all] = np.arange(total)
        print(f'  pure-HG best is global rank #{rank_of[g_pure[0]]+1}; '
              f'le10 best #{rank_of[g_le10[0]]+1} '
              f'(lg_e={lg_e[g_le10[0]]:.3f})')
        union = sorted(set(g_all) | set(g_pure) | set(g_le10))
        print(f'  union = {len(union)} combos to rescan')
        qA = triples[union]
        scoresB = torch.zeros(len(union))
        bidxB = torch.zeros(len(union), dtype=torch.long)
        for b0 in range(0, len(union), 50):
            qb = qA[b0:b0 + 50].to(dev)
            corr = scan_batch(dev, CB, D2, qb)
            mx, mi = corr.max(dim=1)
            scoresB[b0:b0 + qb.shape[0]] = mx.cpu()
            bidxB[b0:b0 + qb.shape[0]] = mi.cpu()
        print(f'[B] done in {time.time()-t0:.0f}s')
        # regroup after half-res rescan
        sB = scoresB.numpy()
        selL_u = lg_mask[tri_np[union]]
        lgE_u = (pmatB[bidxB.numpy()][:, :3] * selL_u).sum(axis=1)

        def topk_from(local_idx, k):
            local_idx = np.asarray(local_idx)
            o = local_idx[np.argsort(-sB[local_idx])][:k]
            return o.tolist()

        local_all = list(range(len(union)))
        pick = set(topk_from(local_all, 22))
        iu_pure = (lgE_u <= 1e-9).nonzero()[0]
        iu_le10 = (lgE_u <= 0.10).nonzero()[0]
        if len(iu_pure):
            pick |= set(topk_from(iu_pure, 14))
        if len(iu_le10):
            pick |= set(topk_from(iu_le10, 14))
        seen = set()
        for li in sorted(pick):
            tr = qA[li].tolist()
            c = canon(tr, labels)
            if c in seen:
                continue
            seen.add(c)
            cands.append({'tr': tr,
                          'seed': pmatB[bidxB[li]].tolist(),
                          'scanB': float(sB[li]),
                          'gA': float(scores[union[li]]),
                          'grp': ('pure' if lgE_u[li] <= 1e-9 else
                                  ('le10' if lgE_u[li] <= 0.10 else 'all'))})
        json.dump(cands, open(OUT / 'scan3B.json', 'w', encoding='utf-8'),
                  ensure_ascii=False)
    print(f'[B] candidates: {len(cands)}')

    # ---- STAGE C: refinement + full-res evaluation ----
    print('[C] core-weighted refinement (N=3) + evaluation...')
    Ie_t = torch.tensor(Ie, dtype=torch.float32, device=dev)
    w_f = weight_map(Ie.shape, pts_e)
    g3 = GRIDS[3]
    evals = []
    t0 = time.time()
    for ci, c in enumerate(cands):
        tr = c['tr']
        ws = weighted_crop_stats(modes_g, Ie_t, tr, None, crop_f, w_f,
                                 sigmas=(0, 2))
        th, val = block_refine_wavg(ws, c['seed'], 3, g3['n_ph'],
                                    g3['n_en'], rounds=4)
        names = [labels[x] for x in tr]
        lg_e = float(sum(th[i] for i, nm in enumerate(names)
                         if nm.startswith('LG')))
        I0 = render_model(modes_np, lab2k,
                          {'combo': names, 'params': th, 'scale': None})
        pts_m0 = find_dark_points(I0)
        mt = match_points(pts_e, pts_m0)
        struct = ('PASS' if (abs(len(pts_m0) - len(pts_e)) <= 1
                             and mt['mean'] is not None
                             and mt['mean'] < 12) else 'fail')
        mesh = mesh2x4(pts_m0)
        per_g = {}
        for gg in GAMMAS:
            Ig = grender(I0, g=gg)
            blk = blocks(Ig)
            pts_m = find_dark_points(Ig)
            dep = core_depth(Ig, pts_m)
            sad = [s for _, s in saddle_isolation(Ig, pts_m)]
            per_g[str(gg)] = {
                'nblk': len(blk), 'blk_sizes': blk[:10],
                'depth': float(np.nanmean(dep)) if len(dep) else None,
                'sad_mean': float(np.mean(sad)) if sad else None,
                'sad_min': float(np.min(sad)) if sad else None,
                'gpcc': gpcc(Ie, Ig), 'iou': dark_iou(Ie, Ig),
                'cores': len(pts_m)}
        evals.append({'combo': '+'.join(names), 'tr': tr, 'params': th,
                      'lg_energy': lg_e, 'wavg_crop': val,
                      'struct': struct, 'cores0': len(pts_m0),
                      'mean_nn': mt['mean'], 'mesh': mesh,
                      'per_gamma': per_g})
        if (ci + 1) % 10 == 0:
            print(f'  {ci+1}/{len(cands)}  ({time.time()-t0:.0f}s)')
    print(f'[C] done in {time.time()-t0:.0f}s')

    def key(e):
        m = e['mesh']
        m_ok = 0 if (m.get('ok') and m.get('rowA_rms', 9) < 2.5
                     and m.get('rowB_rms', 9) < 2.5) else 1
        m_rms = m.get('score', 99) if m.get('ok') else 99
        best_nb = min(abs(e['per_gamma'][str(g)]['nblk'] - exp['nblk'])
                      for g in GAMMAS)
        best_gp = max(e['per_gamma'][str(g)]['gpcc'] for g in GAMMAS)
        return (0 if e['struct'] == 'PASS' else 1, m_ok, m_rms,
                best_nb, -best_gp)

    evals.sort(key=key)

    def show(e, rank):
        m = e['mesh']
        gb = min(GAMMAS, key=lambda g: abs(
            e['per_gamma'][str(g)]['nblk'] - exp['nblk']))
        d = e['per_gamma'][str(gb)]
        mm = (f'mesh[rms={m.get("rowA_rms",0)+m.get("rowB_rms",0):.1f} '
              f'step={m.get("in_row_step",0):.0f} gap={m.get("row_gap",0):.0f}]'
              if m.get('ok') else 'mesh[FAIL]')
        print(f'  #{rank} {e["combo"]:44s} LG={e["lg_energy"]*100:4.1f}% '
              f'struct={e["struct"]:4s} {mm} nblk={d["nblk"]} '
              f'gpcc={d["gpcc"]:.3f}')

    print('\n=== TOP-12 overall ===')
    for i, e in enumerate(evals[:12]):
        show(e, i + 1)

    # group views
    pure = [e for e in evals if e['lg_energy'] <= 1e-9]
    lgle10 = [e for e in evals if e['lg_energy'] <= 0.10]
    print(f'\n=== group sizes: pure-HG={len(pure)} / LG<=10%={len(lgle10)} '
          f'/ all={len(evals)} ===')
    if pure:
        print('--- TOP pure-HG ---')
        for i, e in enumerate(pure[:6]):
            show(e, i + 1)
    if lgle10 and len(lgle10) != len(pure):
        print('--- TOP LG<=10% (incl. pure) ---')
        for i, e in enumerate(lgle10[:6]):
            show(e, i + 1)

    def jsonable(e):
        o = {k: v for k, v in e.items() if k != 'per_gamma'}
        o['per_gamma'] = {k: {kk: vv for kk, vv in v.items()
                              if kk != 'blk_sizes'}
                          for k, v in e['per_gamma'].items()}
        return o

    json.dump({'experiment': exp, 'mesh_exp': mesh_exp,
               'top12': [jsonable(e) for e in evals[:12]],
               'all': [jsonable(e) for e in evals]},
              open(OUT / 'hg_search_results.json', 'w',
                   encoding='utf-8'),
              indent=2, ensure_ascii=False, default=str)

    # ---- quick comparison figure: exp vs top-3 ----
    crop = (slice(40, 190), slice(40, 190))
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.2))
    axs[0, 0].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
    axs[0, 0].set_title(f'EXPERIMENT ({exp["nblk"]} dark cores)',
                        fontsize=11)
    axs[1, 0].imshow(Ie[crop], cmap='inferno', vmin=0, vmax=1)
    axs[1, 0].set_title('experiment (crop)')
    for k, e in enumerate(evals[:3]):
        gb = min(GAMMAS, key=lambda g: abs(
            e['per_gamma'][str(g)]['nblk'] - exp['nblk']))
        I0 = render_model(modes_np, lab2k,
                          {'combo': e['combo'].split('+'),
                           'params': e['params'], 'scale': None})
        Ig = grender(I0, g=gb)
        d = e['per_gamma'][str(gb)]
        m = e['mesh']
        mtag = (f'mesh rms={(m.get("rowA_rms",0)+m.get("rowB_rms",0)):.1f} '
                f'step={m.get("in_row_step",0):.0f}'
                f' gap={m.get("row_gap",0):.0f}'
                if m.get('ok') else 'mesh FAIL')
        axs[0, k + 1].imshow(Ig, cmap='inferno', vmin=0, vmax=1)
        axs[0, k + 1].set_title(
            f'TOP-{k+1} LG={e["lg_energy"]*100:.0f}% '
            f'(g*={gb:.2f})\n{e["combo"]}\n'
            f'{mtag} nblk={d["nblk"]} gpcc={d["gpcc"]:.3f}',
            fontsize=7.5)
        axs[1, k + 1].imshow(Ig[crop], cmap='inferno', vmin=0, vmax=1)
        # overlay model cores + mesh info
        pts_m = [(y, x) for y, x, _ in find_dark_points(I0)]
        axs[1, k + 1].set_title(f'{len(pts_m)} cores (linear)',
                                fontsize=9)
    for a in axs.ravel():
        a.axis('off')
    fig.suptitle('8Singularities(1) HG-led 3-mode search: top-3 '
                 '(quick view)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'HG搜索_快速对比.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

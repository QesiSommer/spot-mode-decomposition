# -*- coding: utf-8 -*-
"""
stage3b_8sing1_4mode.py -- 8(1) 四模全空间搜索（前三视觉对比）
===============================================================

Full 4-mode space: C(131,4) = 11,459,390 quadruples.

Two-stage funnel:
  A. quarter-res (56^2) weighted local scan over ALL quads
     (sharded by first index; core-weighted metric, sigma {0, 0.5},
     grid 35 energies x 125 phases) -> top-800
  B. half-res (112^2) rescan of top-800 (sigma {0,1},
     grid 56 energies x 216 phases) -> top-30 (mirror-deduped)
  C. core-weighted refinement (block_refine_wavg, N=4) + evaluation:
     structural check + gamma sweep {0.58, 0.67, 0.75, 0.85}
     (dark-block count, depth, saddle, gpcc, IoU)
     -> rank -> top-3 comparison figure vs experiment

Outputs: 归档_八图分解/8奇点1_四模空间/
"""

import heapq
import json
import sys
import time
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
from rerun_new_metrics import (
    block_refine_wavg, energy_simplex_N, GRIDS)
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats

OUT = ROOT / '归档_八图分解' / '8奇点1_四模空间'
OUT.mkdir(parents=True, exist_ok=True)
IMGDIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '8Singularities(1)'
GAMMAS = (0.58, 0.67, 0.75, 0.85)


def gpcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def blocks(I, thr=0.10, min_size=20, max_size=3000):
    m = I < thr * I.max()
    lab, n = label(m)
    return sorted((int((lab == k).sum()) for k in range(1, n + 1)
                   if min_size <= int((lab == k).sum()) <= max_size),
                  reverse=True)


def grender(I0, g, sg=2.5):
    Id = gaussian_blur(torch.tensor(I0, dtype=torch.float32)
                       .unsqueeze(0), float(sg))[0].numpy()
    Ig = np.clip(Id, 0, 1) ** g
    return Ig / max(Ig.max(), 1e-12)


def down_complex(x, size):
    t = torch.view_as_real(x).permute(2, 0, 1)[None]
    s = F.interpolate(t, size=(size, size), mode='bilinear',
                      align_corners=False)
    return torch.view_as_complex(
        s[0].permute(1, 2, 0).contiguous())


def canon(p, labels):
    out = []
    for x in p:
        nm = labels[int(x)]
        kind, rest = nm.split('(')
        pp, ll = rest[:-1].split(',')
        out.append(f'{kind}({pp},{abs(int(ll))})')
    return tuple(sorted(out))


def scan_batch(dev, C, stats_fn, qb, B, K):
    """qb: (B,4) long on dev; stats_fn(G) -> list of (m,E,Gram,me,ve)."""
    Fs = [stats_fn['modes'][qb[:, i]] for i in range(4)]
    G = build_base_fields_N(Fs)
    acc = None
    for sg in stats_fn['sigmas']:
        Gb = gaussian_blur(
            G.reshape(B * K, G.shape[2], G.shape[3]),
            float(sg)).reshape(B, K, G.shape[2], G.shape[3])
        Ie_b = gaussian_blur(stats_fn['Ie'].unsqueeze(0),
                             float(sg))[0]
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


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes = modes.to('cpu')
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()
    print('[setup] modes loaded')

    Ie = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    crop_f = crop_box(pts_e, 224)
    w_f = weight_map(Ie.shape, pts_e)
    exp_blk = blocks(Ie)
    exp = {'nblk': len(exp_blk),
           'depth': float(np.nanmean(core_depth(Ie, pts_e))),
           'sad_mean': float(np.mean(
               [s for _, s in saddle_isolation(Ie, pts_e)])),
           'sad_min': float(np.min(
               [s for _, s in saddle_isolation(Ie, pts_e)]))}
    print(f'[setup] experiment: nblk={exp["nblk"]} '
          f'depth={exp["depth"]:.3f} sad={exp["sad_mean"]:.3f}/'
          f'{exp["sad_min"]:.3f}')
    print(f'[setup] crop(full): {crop_f}')

    # ---- prepare q-res and half-res domains ----
    def prep_domain(size, factor, sigmas):
        mnp = torch.stack([down_complex(m, size) for m in modes
                           ]).to(dev)
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
        r = 10.0 / factor
        for (y, x, _v) in pts_e:
            d2 = (yy - y / factor) ** 2 + (xx - x / factor) ** 2
            wh[d2 <= r * r] = 4.0
        wt = torch.tensor(wh, dtype=torch.float32,
                          device=dev)[ch[0], ch[1]].reshape(-1)
        return {'modes': mnp, 'Ie': Ied, 'crop': ch, 'wt': wt,
                'Wsum': float(wt.sum()), 'sigmas': sigmas}

    print('[setup] building downsampled domains...')
    D4 = prep_domain(56, 4, (0.0, 0.5))
    D2 = prep_domain(112, 2, (0.0, 1.0))
    print(f'[setup] q-res crop {D4["crop"]}, half-res crop {D2["crop"]}')

    # ---- grids ----
    eA = energy_simplex_N(4, 4)                      # 35 pts
    axA = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    P2, P3, P4 = np.meshgrid(axA, axA, axA, indexing='ij')
    phA = np.stack([P2.ravel(), P3.ravel(), P4.ravel()], axis=1)
    pmatA = np.concatenate([np.repeat(eA, len(phA), 0),
                            np.tile(phA, (len(eA), 1))], 1)
    CA = c_table_N(pmatA, 4).to(dev)
    print(f'[setup] grid A: {CA.shape[0]} rows')

    eB = energy_simplex_N(4, 5)                      # 56 pts
    axB = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    Q2, Q3, Q4 = np.meshgrid(axB, axB, axB, indexing='ij')
    phB = np.stack([Q2.ravel(), Q3.ravel(), Q4.ravel()], axis=1)
    pmatB = np.concatenate([np.repeat(eB, len(phB), 0),
                            np.tile(phB, (len(eB), 1))], 1)
    CB = c_table_N(pmatB, 4).to(dev)
    print(f'[setup] grid B: {CB.shape[0]} rows')

    # ---- STAGE A: quarter-res full-space scan ----
    total = 131 * 130 * 129 * 128 // 24
    cache_q = OUT / 'quadsA.pt'
    if cache_q.exists():
        quadsA = torch.load(cache_q, weights_only=True)
        print(f'[A] loaded cached {quadsA.shape[0]} quads, skip scan')
    else:
        print(f'[A] full-space scan over {total} quadruples '
              f'(quarter-res)...')
        t0 = time.time()
        heap = []          # (score, i, local_idx)
        BATCH = 128
        done = 0
        for i in range(128):
            rest = torch.combinations(torch.arange(i + 1, 131), r=3)
            L = rest.shape[0]
            q = torch.cat([torch.full((L, 1), i, dtype=torch.long),
                           rest], 1)
            for b0 in range(0, L, BATCH):
                qb = q[b0:b0 + BATCH].to(dev)
                corr = scan_batch(dev, CA, D4, qb, qb.shape[0], 16)
                mx, mi = corr.max(dim=1)
                k = min(3, mx.numel())
                tv, ti = mx.topk(k)
                for v, ii in zip(tv.tolist(), ti.tolist()):
                    heapq.heappush(heap, (float(v), i, b0 + int(ii)))
                    if len(heap) > 800:
                        heapq.heappop(heap)
                done += qb.shape[0]
            if i % 10 == 9 or i == 127:
                el = time.time() - t0
                print(f'  shard {i+1}/128  {done}/{total}  '
                      f'({el:.0f}s, eta '
                      f'{el*(total-done)/max(done,1):.0f}s)')
        print(f'[A] done in {time.time()-t0:.0f}s; heap={len(heap)}')

        # ---- decode top-800 quads ----
        topA = sorted(heap, reverse=True)
        quadsA = []
        cache_rest = {}
        for v, i, li in topA[:800]:
            if i not in cache_rest:
                cache_rest[i] = torch.combinations(
                    torch.arange(i + 1, 131), r=3)
            quad = [i] + cache_rest[i][li].tolist()
            quadsA.append(quad)
        quadsA = torch.tensor(quadsA, dtype=torch.long)
        torch.save(quadsA, cache_q)
        print(f'[A] decoded {quadsA.shape[0]} quads (cached)')

    # ---- STAGE B: half-res rescan ----
    print('[B] half-res rescan of top-800...')
    t0 = time.time()
    scoresB = torch.zeros(quadsA.shape[0])
    bidxB = torch.zeros(quadsA.shape[0], dtype=torch.long)
    for b0 in range(0, quadsA.shape[0], 64):
        qb = quadsA[b0:b0 + 64].to(dev)
        corr = scan_batch(dev, CB, D2, qb, qb.shape[0], 16)
        mx, mi = corr.max(dim=1)
        scoresB[b0:b0 + qb.shape[0]] = mx.cpu()
        bidxB[b0:b0 + qb.shape[0]] = mi.cpu()
    print(f'[B] done in {time.time()-t0:.0f}s')

    vB, iB = scoresB.topk(min(300, len(scoresB)))
    seen, cands = set(), []
    for v, k in zip(vB.tolist(), iB.tolist()):
        tr = quadsA[k].tolist()
        c = canon(tr, labels)
        if c in seen:
            continue
        seen.add(c)
        cands.append({'tr': tr, 'seed': pmatB[bidxB[k]].tolist(),
                      'scanB': v})
        if len(cands) >= 30:
            break
    print('[B] top-5:')
    for c in cands[:5]:
        print(f'  {"+".join(labels[x] for x in c["tr"])}  '
              f'B={c["scanB"]:.4f}')

    # safety injections: current 3-mode best + 4th-mode perturbations
    inj_tr = sorted(lab2k[m] for m in
                    ['LG(2,0)', 'HG(1,1)', 'HG(2,0)'])
    for m4 in ['HG(3,3)', 'HG(3,1)', 'HG(8,2)']:
        q = sorted(inj_tr + [lab2k[m4]])
        if canon(q, labels) not in seen:
            seen.add(canon(q, labels))
            u = [0.34, 0.15, 0.44, 0.07]
            cands.insert(0, {'tr': q, 'seed': list(u) + [0.0, 0.0, 0.0],
                             'scanB': None, 'injected': True})
    print(f'[B] candidates: {len(cands)}')

    # ---- STAGE C: refine + eval ----
    print('[C] core-weighted refinement + gamma evaluation...')
    Ie_t = torch.tensor(Ie, dtype=torch.float32)
    t0 = time.time()
    evals = []
    for c in cands:
        tr = c['tr']
        ws = weighted_crop_stats(modes, Ie_t, tr, None, crop_f, w_f)
        g = GRIDS[4]
        th, val = block_refine_wavg(ws, c['seed'], 4, g['n_ph'],
                                    g['n_en'], rounds=5)
        names = [labels[x] for x in tr]
        I0 = render_model(modes_np, lab2k,
                          {'combo': names, 'params': th, 'scale': None})
        pts_m0 = find_dark_points(I0)
        mt = match_points(pts_e, pts_m0)
        struct = ('PASS' if (abs(len(pts_m0) - len(pts_e)) <= 1
                             and mt['mean'] is not None
                             and mt['mean'] < 12) else 'fail')
        per_g = {}
        for gg in GAMMAS:
            Ig = grender(I0, gg)
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
        evals.append({'combo': '+'.join(names), 'tr': tr,
                      'params': th, 'struct': struct,
                      'cores0': len(pts_m0), 'mean_nn': mt['mean'],
                      'per_gamma': per_g})
    print(f'[C] done in {time.time()-t0:.0f}s')

    def score(e):
        s = None
        for gg in GAMMAS:
            d = e['per_gamma'][str(gg)]
            if d['depth'] is None or d['sad_mean'] is None:
                continue
            sc = (2.5 * abs(d['nblk'] - exp['nblk'])
                  + 2.0 * abs(d['depth'] - exp['depth'])
                  + 1.0 * abs(d['sad_mean'] - exp['sad_mean']))
            if s is None or sc < s[0]:
                s = (sc, gg)
        return s or (99, GAMMAS[0])

    evals.sort(key=lambda e: (
        0 if e['struct'] == 'PASS' else 1, score(e)[0],
        -max(e['per_gamma'][str(g)]['gpcc'] for g in GAMMAS)))

    print('\nranked top-10:')
    for e in evals[:10]:
        sc, gb = score(e)
        d = e['per_gamma'][str(gb)]
        print(f'  {e["combo"]:52s} struct={e["struct"]:4s} '
              f'g*={gb:.2f} nblk={d["nblk"]:2d} depth={d["depth"]:.3f} '
              f'sad={d["sad_mean"]:.3f} gpcc={d["gpcc"]:.4f} '
              f'sc={sc:.2f}')

    # top-3 for the visual review: dedup by EFFECTIVE modes
    # (modes with energy < 0.025 are dropped before canonizing, so
    # a collapsed 4th mode counts as the same solution)
    def redcanon(e):
        tr, params = e['tr'], e['params']
        keep = [i for i in range(len(tr)) if params[i] >= 0.025]
        if len(keep) < 2:
            keep = sorted(range(len(tr)),
                          key=lambda i: -params[i])[:2]
        return tuple(sorted(labels[tr[i]] for i in keep))

    top3 = []
    seen3 = set()
    for e in evals:
        if e['struct'] != 'PASS':
            continue
        c = redcanon(e)
        if c in seen3:
            continue
        seen3.add(c)
        top3.append(e)
        if len(top3) >= 3:
            break
    if len(top3) < 3:
        for e in evals:
            c = redcanon(e)
            if c in seen3:
                continue
            seen3.add(c)
            top3.append(e)
            if len(top3) >= 3:
                break
    print('\ntop-3 (effective-mode dedup):')
    for k, e in enumerate(top3):
        sc, gb = score(e)
        print(f'  TOP-{k+1}: {e["combo"]}  '
              f'e4={e["params"][3]:.3f} g*={gb:.2f} '
              f'(struct={e["struct"]})')

    with open(OUT / 'scan_4mode.json', 'w', encoding='utf-8') as f:
        json.dump({'experiment': exp,
                   'top3': [{'combo': e['combo'],
                             'params': e['params'],
                             'struct': e['struct'],
                             'gamma': score(e)[1],
                             'metrics': e['per_gamma'][
                                 str(score(e)[1])]}
                            for e in top3],
                   'ranked': [{'combo': e['combo'],
                               'struct': e['struct'],
                               'score': score(e)[0],
                               'gamma': score(e)[1],
                               'metrics': e['per_gamma'][
                                   str(score(e)[1])]}
                              for e in evals[:30]]},
                  f, indent=2, ensure_ascii=False, default=str)

    # ---- figure: experiment vs top-3 ----
    crop = (slice(40, 190), slice(40, 190))
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.2))
    axs[0, 0].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
    axs[0, 0].set_title(f'EXPERIMENT\n({exp["nblk"]} dark cores)',
                        fontsize=11)
    axs[1, 0].imshow(Ie[crop], cmap='inferno', vmin=0, vmax=1)
    axs[1, 0].set_title('experiment (crop)')
    for k, e in enumerate(top3):
        gb = score(e)[1]
        d = e['per_gamma'][str(gb)]
        I0 = render_model(modes_np, lab2k,
                          {'combo': e['combo'].split('+'),
                           'params': e['params'], 'scale': None})
        Ig = grender(I0, gb)
        axs[0, k + 1].imshow(Ig, cmap='inferno', vmin=0, vmax=1)
        axs[0, k + 1].set_title(
            f'TOP-{k+1} (struct={e["struct"]})\n{e["combo"]}\n'
            f'g*={gb:.2f} nblk={d["nblk"]} depth={d["depth"]:.3f} '
            f'sad={d["sad_mean"]:.3f}', fontsize=8)
        axs[1, k + 1].imshow(Ig[crop], cmap='inferno', vmin=0, vmax=1)
        axs[1, k + 1].set_title(
            f'nblk={d["nblk"]}  gpcc={d["gpcc"]:.3f}', fontsize=9)
    for a in axs.ravel():
        a.axis('off')
    fig.suptitle('8Singularities(1) - 4-mode space: top-3 vs experiment '
                 '(for visual review)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / '四模前三对比.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

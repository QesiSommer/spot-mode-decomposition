# -*- coding: utf-8 -*-
"""
stage2b_8sing1_local.py -- 8(1) 奇点强化：全空间三模局部扫描
=============================================================

Follow the full 5Sing treatment for 8Sing(1):
 1. dark-core-weighted local scan over ALL 366,145 triples
    (half-res domain for speed, weight x4 on r=10 core disks)
 2. top-30 (mirror-deduped) + incumbent -> core-weighted local
    refinement (block refinement on the weighted crop objective)
 3. evaluation: structural check (linear sigma=0) + gamma-domain
    metrics at gamma in {0.58 0.67 0.75 0.85 0.95}:
      - dark-block count at 10% threshold  (THE connectivity metric;
        experiment = 8 isolated blocks)
      - core depth, saddle mean/min, gpcc, IoU
 4. champion = struct PASS, then block-count match, then depth,
    then saddle, then gpcc

Outputs: 归档_八图分解/8奇点1_奇点强化/
"""

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
from stage0_diagnosis import parse_md, core_depth, saddle_isolation
from stage1_gamma_fix import dark_iou
from rerun_new_metrics import (
    block_refine_wavg, energy_simplex_N, GRIDS)
from stage2_5sing_local import (
    crop_box, weight_map, weighted_crop_stats)

SRC = ROOT / '归档_八图分解' / '最新结果_新指标'
OUT = ROOT / '归档_八图分解' / '8奇点1_奇点强化'
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '8Singularities(1)'
CN = '8奇点(1)'
GAMMAS = (0.58, 0.67, 0.75, 0.85, 0.95)
SIG_DISP = 2.5


def gamma_render(I0, g, sg=SIG_DISP):
    Id = gaussian_blur(torch.tensor(I0, dtype=torch.float32)
                       .unsqueeze(0), float(sg))[0].numpy()
    Ig = np.clip(Id, 0, 1) ** g
    return Ig / max(Ig.max(), 1e-12)


def gpcc(Ie, Ig):
    a = Ie.ravel() - Ie.mean()
    b = Ig.ravel() - Ig.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def blocks(I, thr=0.10, min_size=20, max_size=3000):
    """isolated dark blocks (excluding the huge background block)."""
    m = I < thr * I.max()
    lab, n = label(m)
    out = []
    for k in range(1, n + 1):
        sz = int((lab == k).sum())
        if min_size <= sz <= max_size:
            ys, xs = np.where(lab == k)
            out.append((sz, int(ys.mean()), int(xs.mean())))
    out.sort(reverse=True)
    return out


def half_down(x):
    t = torch.view_as_real(x).permute(2, 0, 1)[None]
    s = F.interpolate(t, size=(x.shape[0] // 2, x.shape[1] // 2),
                      mode='bilinear', align_corners=False)
    return torch.view_as_complex(
        s[0].permute(1, 2, 0).contiguous())


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes = modes.to(dev)
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.detach().cpu().numpy()

    combo0, params0, _s0 = parse_md(SRC / f'{CN}_参数与指标.md')
    print(f'incumbent: {combo0}')
    Ie_np = load_experiment(f'{IMG_DIR}/{NAME}.png', 'nipy_spectral',
                            'isot')
    Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
    pts_e = find_dark_points(Ie_np)

    crop = crop_box(pts_e, 224)
    w = weight_map(Ie_np.shape, pts_e)
    blk_e = blocks(Ie_np)
    exp_ref = {
        'nblk': len(blk_e),
        'blk_sizes': [b[0] for b in blk_e],
        'depth': float(np.nanmean(core_depth(Ie_np, pts_e))),
        'sad_mean': float(np.mean(
            [s for _, s in saddle_isolation(Ie_np, pts_e)])),
        'sad_min': float(np.min(
            [s for _, s in saddle_isolation(Ie_np, pts_e)])),
    }
    print(f'experiment: {exp_ref["nblk"]} blocks '
          f'{exp_ref["blk_sizes"]}, depth={exp_ref["depth"]:.3f}, '
          f'sad={exp_ref["sad_mean"]:.3f}/{exp_ref["sad_min"]:.3f}')

    # ---- half-res scan setup ----
    modes_h = torch.stack([half_down(m) for m in modes]).to(dev)
    Ie_h = F.interpolate(Ie[None, None], size=(112, 112),
                         mode='bilinear', align_corners=False)[0, 0]
    ch = (slice(crop[0].start // 2, (crop[0].stop + 1) // 2),
          slice(crop[1].start // 2, (crop[1].stop + 1) // 2))
    wh = np.ones((112, 112), dtype=np.float64)
    yy, xx = np.mgrid[0:112, 0:112]
    for (y, x, _v) in pts_e:
        d2 = (yy - y / 2.0) ** 2 + (xx - x / 2.0) ** 2
        wh[d2 <= 25.0] = 4.0
    wt = torch.tensor(wh, dtype=torch.float32,
                      device=dev)[ch].reshape(-1)
    Wsum = float(wt.sum())

    e_pts = energy_simplex_N(3, 8)
    ax = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    P2, P3 = np.meshgrid(ax, ax, indexing='ij')
    ph = np.stack([P2.ravel(), P3.ravel()], axis=1)
    pmat = np.concatenate([np.repeat(e_pts, len(ph), axis=0),
                           np.tile(ph, (len(e_pts), 1))], axis=1)
    C = c_table_N(pmat, 3).to(dev)
    print(f'grid: {C.shape[0]} c-rows/triple, crop {ch}')

    triples = torch.combinations(torch.arange(len(labels)), r=3)
    Q = triples.shape[0]
    print(f'[scan] all {Q} triples at half-res, weighted')
    best = torch.empty(Q, device=dev)
    bidx = torch.empty(Q, dtype=torch.long, device=dev)
    batch = 64
    t0 = time.time()
    for b0 in range(0, Q, batch):
        t = triples[b0:b0 + batch].to(dev)
        G = build_base_fields_N([modes_h[t[:, 0]], modes_h[t[:, 1]],
                                 modes_h[t[:, 2]]])
        B, K = G.shape[0], G.shape[1]
        acc = None
        for sg in (0, 1):
            Gb = gaussian_blur(G.reshape(B * K, 112, 112),
                               float(sg)).reshape(B, K, 112, 112)
            Ie_b = gaussian_blur(Ie_h.unsqueeze(0), float(sg))[0]
            Gc = Gb[:, :, ch[0], ch[1]].reshape(B, K, -1)
            Ic = Ie_b[ch].reshape(-1)
            m = torch.einsum('bki,i->bk', Gc, wt) / Wsum
            E = torch.einsum('bki,i->bk', Gc, Ic * wt) / Wsum
            Gram = torch.einsum('bki,bli->bkl', Gc, Gc * wt) / Wsum
            me = float((Ic * wt).sum() / Wsum)
            ve = float(((Ic - me) ** 2 * wt).sum() / Wsum)
            corr = eval_c_grid(C, m, E, Gram, me, ve)
            acc = corr if acc is None else acc + corr
        mx, mi = (acc / 2).max(dim=1)
        best[b0:b0 + B] = mx
        bidx[b0:b0 + B] = mi
        if (b0 // batch) % 1200 == 0:
            print(f'  {b0}/{Q}  ({time.time()-t0:.0f}s)')
    print(f'  scan done in {time.time()-t0:.0f}s')

    # ---- candidates ----
    vals, idxs = best.topk(80)

    def canon(p):
        out = []
        for x in p:
            nm = labels[x]
            kind, rest = nm.split('(')
            pp, ll = rest[:-1].split(',')
            out.append(f'{kind}({pp},{abs(int(ll))})')
        return tuple(sorted(out))

    seen, cands = set(), []
    for k in range(80):
        tr = triples[idxs[k]].tolist()
        c = canon(tr)
        if c in seen:
            continue
        seen.add(c)
        cands.append({'tr': tr, 'seed': pmat[bidx[idxs[k]]].tolist(),
                      'scan': float(vals[k])})
        if len(cands) >= 30:
            break
    inc_tr = sorted(lab2k[m] for m in combo0)
    if canon(inc_tr) not in seen:
        cands.append({'tr': inc_tr, 'seed': list(params0),
                      'scan': None})
    print(f'candidates: {len(cands)}; top-5 scan:')
    for c in cands[:5]:
        print(f'  {"+".join(labels[x] for x in c["tr"])}  '
              f'scan={c["scan"]:.4f}')

    # ---- refinement (core-weighted local objective) ----
    print('[refine] core-weighted local objective')
    t0 = time.time()
    refined = []
    for c in cands:
        tr = c['tr']
        ws = weighted_crop_stats(modes, Ie, tr, None, crop, w)
        g = GRIDS[3]
        th, val = block_refine_wavg(ws, c['seed'], 3, g['n_ph'],
                                    g['n_en'], rounds=6)
        refined.append({'tr': tr, 'params': th, 'wval': val,
                        'scan': c['scan']})
    print(f'  refined {len(refined)} in {time.time()-t0:.0f}s')

    # ---- evaluation ----
    print('[eval] gamma-domain metrics over gamma sweep')
    evals = []
    for r in refined:
        names = [labels[x] for x in r['tr']]
        I0 = render_model(modes_np, lab2k,
                          {'combo': names, 'params': r['params'],
                           'scale': None})
        pts_m0 = find_dark_points(I0)
        mt = match_points(pts_e, pts_m0)
        struct = ('PASS' if (abs(len(pts_m0) - len(pts_e)) <= 1
                             and mt['mean'] is not None
                             and mt['mean'] < 12) else 'fail')
        per_g = {}
        for g in GAMMAS:
            Ig = gamma_render(I0, g)
            blk = blocks(Ig)
            pts_m = find_dark_points(Ig)
            dep = core_depth(Ig, pts_m)
            sad = [s for _, s in saddle_isolation(Ig, pts_m)]
            per_g[str(g)] = {
                'nblk': len(blk),
                'blk_sizes': [b[0] for b in blk[:10]],
                'depth': float(np.nanmean(dep)) if len(dep) else None,
                'sad_mean': float(np.mean(sad)) if sad else None,
                'sad_min': float(np.min(sad)) if sad else None,
                'gpcc': gpcc(Ie_np, Ig),
                'iou': dark_iou(Ie_np, Ig),
                'cores': len(pts_m),
            }
        evals.append({'combo': '+'.join(names), 'tr': r['tr'],
                      'params': r['params'], 'struct': struct,
                      'cores0': len(pts_m0),
                      'mean_nn': mt['mean'], 'per_gamma': per_g,
                      'is_incumbent': r['tr'] == inc_tr})

    def score(e):
        s = None
        for g in GAMMAS:
            d = e['per_gamma'][str(g)]
            if d['depth'] is None or d['sad_mean'] is None:
                continue
            sc = (2.5 * abs(d['nblk'] - exp_ref['nblk'])
                  + 2.0 * abs(d['depth'] - exp_ref['depth'])
                  + 1.0 * abs(d['sad_mean'] - exp_ref['sad_mean']))
            if s is None or sc < s[0]:
                s = (sc, g)
        return s or (99, GAMMAS[0])

    evals.sort(key=lambda e: (
        0 if e['struct'] == 'PASS' else 1,
        score(e)[0],
        -max(e['per_gamma'][str(g)]['gpcc'] for g in GAMMAS)))

    print('\nranked (top-8):')
    for e in evals[:8]:
        sc, gbest = score(e)
        d = e['per_gamma'][str(gbest)]
        flag = ' [INC]' if e['is_incumbent'] else ''
        print(f'  {e["combo"]:44s} struct={e["struct"]:4s} '
              f'gamma*={gbest:.2f} nblk={d["nblk"]:2d} '
              f'depth={d["depth"]:.3f} sad={d["sad_mean"]:.3f} '
              f'gpcc={d["gpcc"]:.4f} score={sc:.3f}{flag}')

    champion = evals[0]
    challenger = next((e for e in evals
                       if not e['is_incumbent']), None)
    print(f'\nBEST CONFIG: {champion["combo"]} '
          f'(incumbent={champion["is_incumbent"]})')

    (OUT / 'scan_results.json').write_text(json.dumps(
        {'experiment': exp_ref, 'best': champion,
         'challenger': challenger, 'evals': evals},
        indent=2, ensure_ascii=False, default=str), encoding='utf-8')

    # ---- figure ----
    panels = []
    I0_inc = render_model(modes_np, lab2k,
                          {'combo': combo0, 'params': params0,
                           'scale': None})
    panels.append((f'experiment', Ie_np))
    Ig58 = gamma_render(I0_inc, 0.58)
    d58 = next(e for e in evals
               if e['is_incumbent'])['per_gamma']['0.58']
    panels.append((f'INCUMBENT gamma=0.58 (delivered)\n'
                   f'nblk={d58["nblk"]} depth={d58["depth"]:.3f}',
                   Ig58))
    gbest = score(champion)[1]
    I0c = render_model(modes_np, lab2k,
                       {'combo': champion['combo'].split('+'),
                        'params': champion['params'], 'scale': None})
    Igc = gamma_render(I0c, gbest)
    dc = champion['per_gamma'][str(gbest)]
    panels.append((f'BEST {champion["combo"]} gamma={gbest:.2f}\n'
                   f'nblk={dc["nblk"]} depth={dc["depth"]:.3f} '
                   f'gpcc={dc["gpcc"]:.3f}', Igc))
    if champion['is_incumbent'] and challenger is not None:
        gc = score(challenger)[1]
        I0x = render_model(modes_np, lab2k,
                           {'combo': challenger['combo'].split('+'),
                            'params': challenger['params'], 'scale': None})
        Igx = gamma_render(I0x, gc)
        dx = challenger['per_gamma'][str(gc)]
        panels.append((f'challenger {challenger["combo"]} '
                       f'gamma={gc:.2f}\nnblk={dx["nblk"]} '
                       f'depth={dx["depth"]:.3f}', Igx))
    else:
        panels.append((f'BEST at gamma=0.58\n'
                       f'nblk={champion["per_gamma"]["0.58"]["nblk"]} '
                       f'depth='
                       f'{champion["per_gamma"]["0.58"]["depth"]:.3f}',
                       gamma_render(I0c, 0.58)))
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.4))
    for k, (title, img) in enumerate(panels[:4]):
        axs[0, k].imshow(img, cmap='inferno', vmin=0, vmax=1)
        axs[0, k].set_title(title, fontsize=9)
        axs[1, k].imshow(img[crop], cmap='inferno', vmin=0, vmax=1)
        axs[1, k].set_title(f'crop', fontsize=9)
    for k in range(len(panels), 4):
        axs[0, k].axis('off')
        axs[1, k].axis('off')
    for a in axs.ravel():
        a.axis('off')
    fig.suptitle(f'{NAME} singularity enhancement '
                 f'(full-space weighted scan, gamma sweep)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / '奇点强化对比.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

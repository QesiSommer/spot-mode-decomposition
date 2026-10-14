# -*- coding: utf-8 -*-
"""
stage2_5sing_local.py -- 5奇点局部穷举搜索
==========================================

Per the confirmed plan:
 1. adaptive crop: bbox of experiment cores + 20px margin
 2. dark-core-weighted local PCC (4x weight in r=10 disks around
    experiment cores), sigma {0,2}, analytic scan over ALL 8515
    two-mode combos at the incumbent scale
 3. top-30 -> scale re-scan (s in 1.10..1.32) -> top-15 + incumbent
 4. global W_avg refinement (new method) for each candidate
 5. gamma-domain (0.70, sigma 2.5) visual metrics:
    local weighted PCC, saddle mean/min, core depth, dark IoU
    + structural verification (linear sigma=0)
 6. rank: structural PASS first, then gamma-local-PCC

Outputs: 阶段2_5奇点局部搜索/
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
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from search_three_mode import eval_c_grid
from refine_all_new_method import build_base_fields_N, c_table_N
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage0_diagnosis import parse_md, core_depth, saddle_isolation
from stage1_gamma_fix import dark_iou
from rerun_new_metrics import refine_candidate, global_stats_02, \
    c_s02, GRIDS

SRC = ROOT / '归档_八图分解' / '最新结果_新指标'
OUT = ROOT / '归档_八图分解' / '局部优化_5及8奇点' / '阶段2_5奇点局部搜索'
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '5Singularities'
CN = '5奇点'
GAMMA = 0.70
SIG_DISP = 2.5
CORE_W, CORE_R = 4.0, 10


# ---------------------------------------------------------------------------
# weighted-crop analytic machinery
# ---------------------------------------------------------------------------

def crop_box(pts, H, margin=20):
    ys = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    y0 = max(0, min(ys) - margin)
    y1 = min(H, max(ys) + margin + 1)
    x0 = max(0, min(xs) - margin)
    x1 = min(H, max(xs) + margin + 1)
    return slice(y0, y1), slice(x0, x1)


def weight_map(shape, pts):
    w = np.ones(shape, dtype=np.float64)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for (y, x, _v) in pts:
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        w[d2 <= CORE_R ** 2] = CORE_W
    return w


def weighted_crop_stats(modes, Ie_full, tr, scale, crop, w, sigmas=(0, 2)):
    """Per-sigma weighted stats on the crop. Returns list of
    (m,E,Gram,me,ve) with shapes (1,K),(1,K),(1,K,K),scalar,scalar."""
    dev = modes.device
    mnp = modes if scale is None else rescale_lg(modes, float(scale))
    Fs = [mnp[k].unsqueeze(0) for k in tr]
    G = build_base_fields_N(Fs)[0]           # (K,H,W)
    K = G.shape[0]
    wt = torch.tensor(w, dtype=torch.float32,
                      device=dev)[crop[0], crop[1]]
    w_flat = wt.reshape(-1)
    Wsum = float(w_flat.sum())
    stats = []
    for sg in sigmas:
        Gb = gaussian_blur(G, float(sg))
        Ie_b = gaussian_blur(Ie_full.unsqueeze(0), float(sg))[0]
        Gc = Gb[:, crop[0], crop[1]].reshape(K, -1)   # (K,npx)
        Ic = Ie_b[crop[0], crop[1]].reshape(-1)
        m = (Gc * w_flat).sum(1) / Wsum
        E = ((Gc * Ic) * w_flat).sum(1) / Wsum
        Gram = (Gc * w_flat) @ Gc.T / Wsum
        me = float((Ic * w_flat).sum() / Wsum)
        ve = float(((Ic - me) ** 2 * w_flat).sum() / Wsum)
        stats.append((m.unsqueeze(0), E.unsqueeze(0),
                     Gram.unsqueeze(0), me, ve))
    return stats


def local_scan(modes, Ie_full, pairs, crop, w, scale, N=2,
               n_r=81, n_phi=120, batch=256):
    """Analytic weighted-crop scan over (pairs x r x phi)."""
    dev = modes.device
    rg = np.linspace(0.02, 0.98, n_r)
    pg = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    U, P = np.meshgrid(rg, pg, indexing='ij')
    # params rows: [u1,u2,phi]
    pmat = np.stack([U.ravel(), 1 - U.ravel(), P.ravel()], axis=1)
    C = c_table_N(pmat, N).to(dev)
    Q = pairs.shape[0]
    best = torch.empty(Q, device=dev)
    bidx = torch.empty(Q, dtype=torch.long, device=dev)
    t0 = time.time()
    for b0 in range(0, Q, batch):
        prs = pairs[b0:b0 + batch].to(dev)
        # batched stats: build fields for all pairs in the batch
        F1 = modes[prs[:, 0]]
        F2 = modes[prs[:, 1]]
        G = build_base_fields_N([F1, F2])    # (B,K,H,W)
        B, K = G.shape[0], G.shape[1]
        wt = torch.tensor(w, dtype=torch.float32, device=dev)[crop]
        w_flat = wt.reshape(-1)
        Wsum = float(w_flat.sum())
        acc = None
        for sg in (0, 2):
            Gb = gaussian_blur(G.reshape(B * K, *G.shape[2:]),
                               float(sg)).reshape(B, K, *G.shape[2:])
            Ie_b = gaussian_blur(Ie_full.unsqueeze(0), float(sg))[0]
            Gc = Gb[:, :, crop[0], crop[1]].reshape(B, K, -1)
            Ic = Ie_b[crop].reshape(-1)
            m = torch.einsum('bki,i->bk', Gc, w_flat) / Wsum
            E = torch.einsum('bki,i->bk', Gc,
                             Ic * w_flat) / Wsum
            Gw = Gc * w_flat
            Gram = torch.einsum('bki,bli->bkl', Gc, Gw) / Wsum
            me = float((Ic * w_flat).sum() / Wsum)
            ve = float(((Ic - me) ** 2 * w_flat).sum() / Wsum)
            corr = eval_c_grid(C, m, E, Gram, me, ve)
            acc = corr if acc is None else acc + corr
        acc = acc / 2
        mx, mi = acc.max(dim=1)
        best[b0:b0 + B] = mx
        bidx[b0:b0 + B] = mi
    print(f'    scan {Q} pairs: {time.time()-t0:.0f}s')
    return best, bidx, pmat


def gamma_render(I0, gamma=GAMMA, sg=SIG_DISP):
    Id = gaussian_blur(torch.tensor(I0, dtype=torch.float32)
                       .unsqueeze(0), float(sg))[0].numpy()
    Ig = np.clip(Id, 0, 1) ** gamma
    return Ig / max(Ig.max(), 1e-12)


def f3(v):
    return '-' if v is None else f'{v:.3f}'


def visual_metrics(Ie, Ig, pts_e, crop, w):
    """gamma-domain local weighted PCC + saddle/depth/IoU."""
    a = Ie[crop].ravel()
    b = Ig[crop].ravel()
    wf = w[crop].ravel()
    am = (a * wf).mean() / wf.mean()
    bm = (b * wf).mean() / wf.mean()
    cov = ((a - am) * (b - bm) * wf).sum()
    va = ((a - am) ** 2 * wf).sum()
    vb = ((b - bm) ** 2 * wf).sum()
    lpcc = float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan
    pts_m = find_dark_points(Ig)
    dep = core_depth(Ig, pts_m)
    sad = saddle_isolation(Ig, pts_m)
    s_all = [s for _, s in sad]
    return {
        'local_pcc_gamma': lpcc,
        'saddle_mean': float(np.mean(s_all)) if s_all else None,
        'saddle_min': float(np.min(s_all)) if s_all else None,
        'depth_mean': float(np.nanmean(dep)) if len(dep) else None,
        'iou': dark_iou(Ie, Ig),
        'cores': len(pts_m),
    }


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes = modes.to(dev)
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.detach().cpu().numpy()

    combo0, params0, s0 = parse_md(SRC / f'{CN}_参数与指标.md')
    print(f'incumbent: {combo0}  s={s0}')
    Ie_np = load_experiment(f'{IMG_DIR}/{NAME}.png', 'nipy_spectral',
                            'isot')
    Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
    pts_e = find_dark_points(Ie_np)
    crop = crop_box(pts_e, 224)
    w = weight_map(Ie_np.shape, pts_e)
    print(f'crop rows {crop[0].start}-{crop[0].stop}, '
          f'cols {crop[1].start}-{crop[1].stop}; '
          f'cores weighted x{CORE_W} in r={CORE_R}')

    # incumbent baseline in gamma domain
    I0_inc = render_model(modes_np, lab2k,
                          {'combo': combo0, 'params': params0,
                           'scale': s0})
    Ig_inc = gamma_render(I0_inc)
    base = visual_metrics(Ie_np, Ig_inc, pts_e, crop, w)
    print('incumbent gamma-domain metrics:', base)

    # ---- step 2: full local scan at incumbent scale ----
    pairs = torch.combinations(torch.arange(len(labels)), r=2)
    print(f'[scan] all {pairs.shape[0]} pairs at s={s0}')
    best, bidx, pmat = local_scan(modes, Ie, pairs, crop, w, s0)
    K_TOP = 30
    vals, idxs = best.topk(K_TOP)
    top_pairs = pairs[idxs.cpu()]
    top_params = pmat[bidx[idxs].cpu().numpy()]
    print(f'  top-5 local PCC: '
          f'{[round(float(v), 4) for v in vals[:5]]}')
    for k in range(5):
        pr = top_pairs[k].tolist()
        print(f'    #{k+1} {labels[pr[0]]}+{labels[pr[1]]} '
              f'lpcc={float(vals[k]):.4f}')

    # ---- step 3: scale re-scan for top-30 ----
    print('[scale re-scan]')
    scale_grid = np.linspace(1.10, 1.32, 8)
    scale_scores = {}
    for s in scale_grid:
        b, _, _ = local_scan(modes, Ie, top_pairs, crop, w, float(s))
        scale_scores[float(s)] = b
    # per-pair best scale
    rows = []
    for k in range(K_TOP):
        scores = torch.stack([scale_scores[float(s)][k]
                              for s in scale_grid])
        si = int(torch.argmax(scores))
        rows.append({'pair': top_pairs[k].tolist(),
                     's': float(scale_grid[si]),
                     'lpcc': float(scores[si]),
                     'p0': top_params[k].tolist()})
    rows.sort(key=lambda r: -r['lpcc'])
    # de-duplicate conjugate mirrors among top-15
    def canon(p):
        names = [labels[x] for x in p]
        out = []
        for nm in names:
            kind, rest = nm.split('(')
            pp, ll = rest[:-1].split(',')
            out.append(f'{kind}({pp},{abs(int(ll))})')
        return tuple(sorted(out))
    seen = set()
    uniq = []
    for r in rows:
        c = canon(r['pair'])
        if c in seen:
            continue
        seen.add(c)
        uniq.append(r)
        if len(uniq) >= 15:
            break
    # incumbent injection
    inc_key = canon([lab2k[m] for m in combo0])
    if inc_key not in seen:
        uniq.append({'pair': sorted(lab2k[m] for m in combo0),
                     's': s0, 'lpcc': float('nan'),
                     'p0': list(params0)})
    print(f'  candidates after dedup + incumbent: {len(uniq)}')

    # ---- step 4: global W_avg refinement ----
    print('[global refinement (new method, W_avg objective)]')
    refined = []
    for r in uniq:
        tr = list(r['pair'])
        seed = r['p0'] if r['p0'][0] + r['p0'][1] <= 1.001 \
            else list(params0)
        gs = global_stats_02(modes, Ie, tr, r['s'])
        try:
            th, wavg, s_best, ws, meta, mask = refine_candidate(
                modes, Ie, tr, 2, seed, r['s'], gs)
        except Exception as e:
            print(f'    skip {tr}: {e}')
            continue
        refined.append({'pair': tr, 'params': th, 'wavg': wavg,
                        'scale': s_best})
    print(f'  refined {len(refined)} candidates')

    # ---- step 5: gamma-domain evaluation ----
    print('[gamma-domain evaluation]')
    evals = []
    for r in refined:
        names = [labels[x] for x in r['pair']]
        I0 = render_model(modes_np, lab2k,
                         {'combo': names, 'params': r['params'],
                          'scale': r['scale']})
        Ig = gamma_render(I0)
        vm = visual_metrics(Ie_np, Ig, pts_e, crop, w)
        # structural check (linear, sigma=0)
        pts_m0 = find_dark_points(I0)
        mt = match_points(pts_e, pts_m0)
        n_ok = abs(len(pts_m0) - len(pts_e)) <= 1
        d_ok = mt['mean'] is not None and mt['mean'] < 12
        gs = global_stats_02(modes, Ie, r['pair'], r['scale'])
        C2, (v0, v2) = c_s02(gs, r['params'], 2)
        evals.append({
            'combo': '+'.join(names), 'pair': r['pair'],
            'params': r['params'], 'scale': r['scale'],
            'wavg': r['wavg'], 'C_s02': C2,
            'struct': 'PASS' if (n_ok and d_ok) else 'fail',
            'cores': len(pts_m0), 'mean_nn': mt['mean'],
            **vm})
        sm = vm['saddle_mean']
        smin = vm['saddle_min']
        print(f'  {evals[-1]["combo"]:30s} lpcc_g='
              f'{vm["local_pcc_gamma"]:.4f} sad={f3(sm)}/min '
              f'{f3(smin)} dep={f3(vm["depth_mean"])} '
              f'IoU={f3(vm["iou"])} cores={vm["cores"]} '
              f'{evals[-1]["struct"]} C_s02={C2:.4f}')

    # incumbent entry
    inc_eval = {'combo': '+'.join(combo0),
                'pair': sorted(lab2k[m] for m in combo0),
                'params': list(params0), 'scale': s0,
                'wavg': None, 'C_s02': None, 'struct': 'PASS',
                'cores': len(find_dark_points(I0_inc)),
                'mean_nn': None, 'is_incumbent': True, **base}
    evals.append(inc_eval)

    # ---- step 6: rank & outputs ----
    passing = [e for e in evals if e['struct'] == 'PASS'
               and not e.get('is_incumbent')]
    pool = passing if passing else [e for e in evals
                                    if not e.get('is_incumbent')]
    champion = max(pool, key=lambda e: e['local_pcc_gamma'])
    print(f'\nCHAMPION (gamma-local-PCC): {champion["combo"]} '
          f'lpcc_g={champion["local_pcc_gamma"]:.4f} '
          f'sad={f3(champion["saddle_mean"])}')

    (OUT / 'candidates.json').write_text(
        json.dumps({'incumbent': inc_eval, 'evals': evals,
                    'champion': champion}, indent=2,
                   ensure_ascii=False, default=str),
        encoding='utf-8')

    # figure: experiment | incumbent | top-4 candidates (gamma display)
    show = [champion] + [e for e in passing[:5]
                         if e is not champion][:2]
    fig, axs = plt.subplots(2, 5, figsize=(24, 9.4))
    axs[0, 0].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
    axs[0, 0].set_title('experiment')
    axs[1, 0].imshow(Ie_np[crop], cmap='inferno', vmin=0, vmax=1)
    axs[1, 0].set_title('experiment (crop)')
    Ig_inc = gamma_render(I0_inc)
    axs[0, 1].imshow(Ig_inc, cmap='inferno', vmin=0, vmax=1)
    axs[0, 1].set_title(f'INCUMBENT {inc_eval["combo"]}\n'
                        f'sad={inc_eval["saddle_mean"]:.3f} '
                        f'dep={inc_eval["depth_mean"]:.3f}')
    axs[1, 1].imshow(Ig_inc[crop], cmap='inferno', vmin=0, vmax=1)
    for k in range(3):
        if k < len(show):
            e = show[k]
            I0 = render_model(modes_np, lab2k,
                             {'combo': e['combo'].split('+'),
                              'params': e['params'],
                              'scale': e['scale']})
            Ig = gamma_render(I0)
            axs[0, k + 2].imshow(Ig, cmap='inferno', vmin=0, vmax=1)
            sm = e['saddle_mean']
            axs[0, k + 2].set_title(
                f'{"C" if e is champion else f"#{k+1}"} {e["combo"]}\n'
                f'lpcc_g={e["local_pcc_gamma"]:.3f} sad={f3(sm)}',
                fontsize=8)
            axs[1, k + 2].imshow(Ig[crop], cmap='inferno', vmin=0,
                                 vmax=1)
        else:
            axs[0, k + 2].axis('off')
            axs[1, k + 2].axis('off')
    for a in axs.ravel():
        a.axis('off')
    fig.suptitle(f'{NAME} stage-2 local search (gamma={GAMMA}, '
                 f'sigma={SIG_DISP}); C = champion', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / '候选对比.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

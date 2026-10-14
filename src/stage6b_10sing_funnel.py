# -*- coding: utf-8 -*-
"""
stage6b_10sing_funnel.py -- 10奇点：全空间四模漏斗搜索（阶段2）
================================================================
全空间 C(131,4) = 11,716,640 个四模组合，两阶段漏斗：

  A. 1/4 分辨率（56²）全空间扫（核加权局部指标，σ={0,0.5}，
     能量网格 35 x 相位网格 125）→ top-800
  B. 1/2 分辨率（112²）重扫 top-800（σ={0,1}，56 x 216）→ top-30
     （canonical 去重）
  C. 每候选：核加权精修 → 标准 W_avg 精修 → 评估（结构验收 +
     C_s02/W_avg/rho56/cmin56）；稳健通过解再做 200 点微扰抛光。
     安全注入：阶段1 冠军族 + 两个已知通过族。

排序 = 稳健结构优先（全核 ≤12px）→ C_s02 → W_avg。
Outputs: 归档_八图分解/10奇点/5_新指标修复与验收/stage2_funnel.json
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from search_three_mode import eval_c_grid
from refine_all_new_method import build_base_fields_N, c_table_N
from rerun_new_metrics import (block_refine_wavg, energy_simplex_N,
                               global_stats_02, c_s02, GRIDS)
from window_metrics import (build_window_stats, w_avg_point,
                            window_pccs)
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats

OUT = ROOT / '归档_八图分解' / '10奇点' / '5_新指标修复与验收'
OUT.mkdir(parents=True, exist_ok=True)
IMGDIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '10Singularities'


def canon(p, labels):
    out = []
    for x in p:
        nm = labels[int(x)]
        kind, rest = nm.split('(')
        pp, ll = rest[:-1].split(',')
        out.append(f'{kind}({pp},{abs(int(ll))})')
    return tuple(sorted(out))


def down_complex(x, size):
    t = torch.view_as_real(x).permute(2, 0, 1)[None]
    s = F.interpolate(t, size=(size, size), mode='bilinear',
                      align_corners=False)
    return torch.view_as_complex(
        s[0].permute(1, 2, 0).contiguous())


def scan_batch(C, stats_fn, qb, B, K):
    Fs = [stats_fn['modes'][qb[:, i]] for i in range(4)]
    G = build_base_fields_N(Fs)
    acc = None
    for sg in stats_fn['sigmas']:
        Gb = gaussian_blur(
            G.reshape(B * K, G.shape[2], G.shape[3]),
            float(sg)).reshape(B, K, G.shape[2], G.shape[3])
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


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device = {dev}')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()

    Ie = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
    Ie_t = torch.tensor(Ie, dtype=torch.float32)
    pts_e = find_dark_points(Ie)
    N_EXP = len(pts_e)
    crop_f = crop_box(pts_e, 224)
    w_f = weight_map(Ie.shape, pts_e)
    print(f'[setup] experiment cores = {N_EXP}; crop {crop_f}')

    # ---------------- stage 1 helpers (same protocol as stage6a) -------
    def struct_check(theta, tr):
        I = render_model(modes_np, lab2k,
                         {'combo': [labels[t] for t in tr],
                          'params': list(theta), 'scale': None})
        pts_m = find_dark_points(I)
        mt = match_points(pts_e, pts_m)
        ok = (abs(len(pts_m) - N_EXP) <= 1
              and mt['mean'] is not None and mt['mean'] < 12)
        return (ok, len(pts_m),
                float(mt['mean']) if mt['mean'] is not None else 99.0,
                float(mt['max']) if mt['max'] is not None else 99.0)

    def full_metrics(theta, tr, gs, ws, meta):
        C2, (v0, v2) = c_s02(gs, theta, 4)
        w = w_avg_point(ws, theta, 4)
        p56 = window_pccs(ws, theta, 4)
        sg2 = [x for x, m in zip(p56, meta) if m['sigma'] == 2]
        return (float(C2), float(w),
                float(np.mean([x < 0.5 for x in sg2])),
                float(np.min(sg2)))

    all_rec = []

    def record(theta, tr, tag, gs, ws, meta):
        ok, cores, mean_nn, max_nn = struct_check(theta, tr)
        if not ok:
            return
        C2, w, rho56, cmin56 = full_metrics(theta, tr, gs, ws, meta)
        all_rec.append({'combo': [labels[t] for t in tr], 'tr': list(tr),
                        'theta': [float(x) for x in theta],
                        'C_s02': C2, 'W_avg': w, 'cores': cores,
                        'mean_nn': mean_nn, 'max_nn': max_nn,
                        'robust': bool(max_nn < 12.0),
                        'rho56': rho56, 'cmin56': cmin56, 'src': tag})

    # ---------------- domains ----------------
    def prep_domain(size, factor, sigmas):
        mnp = torch.stack([down_complex(m, size) for m in modes]).to(dev)
        Ied = F.interpolate(
            Ie_t[None, None], size=(size, size), mode='bilinear',
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

    eA = energy_simplex_N(4, 4)
    axA = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    P2, P3, P4 = np.meshgrid(axA, axA, axA, indexing='ij')
    phA = np.stack([P2.ravel(), P3.ravel(), P4.ravel()], axis=1)
    pmatA = np.concatenate([np.repeat(eA, len(phA), 0),
                            np.tile(phA, (len(eA), 1))], 1)
    CA = c_table_N(pmatA, 4).to(dev)

    eB = energy_simplex_N(4, 5)
    axB = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    Q2, Q3, Q4 = np.meshgrid(axB, axB, axB, indexing='ij')
    phB = np.stack([Q2.ravel(), Q3.ravel(), Q4.ravel()], axis=1)
    pmatB = np.concatenate([np.repeat(eB, len(phB), 0),
                            np.tile(phB, (len(eB), 1))], 1)
    CB = c_table_N(pmatB, 4).to(dev)
    print(f'[setup] gridA={CA.shape[0]} gridB={CB.shape[0]}')

    # ---------------- STAGE A ----------------
    total = 131 * 130 * 129 * 128 // 24
    cache_q = OUT / 'funnel_quadsA.pt'
    if cache_q.exists():
        quadsA = torch.load(cache_q, weights_only=True)
        print(f'[A] loaded cached {quadsA.shape[0]} quads')
    else:
        print(f'[A] full-space scan over {total} quads '
              f'(quarter-res)...')
        t0 = time.time()
        heap = []
        BATCH = 128
        done = 0
        for i in range(128):
            rest = torch.combinations(torch.arange(i + 1, 131), r=3)
            L = rest.shape[0]
            q = torch.cat([torch.full((L, 1), i, dtype=torch.long),
                           rest], 1)
            for b0 in range(0, L, BATCH):
                qb = q[b0:b0 + BATCH].to(dev)
                corr = scan_batch(CA, D4, qb, qb.shape[0], 16)
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
        topA = sorted(heap, reverse=True)
        quadsA = []
        cache_rest = {}
        for v, i, li in topA[:800]:
            if i not in cache_rest:
                cache_rest[i] = torch.combinations(
                    torch.arange(i + 1, 131), r=3)
            quadsA.append([i] + cache_rest[i][li].tolist())
        quadsA = torch.tensor(quadsA, dtype=torch.long)
        torch.save(quadsA, cache_q)
        print(f'[A] decoded {quadsA.shape[0]} quads (cached)')

    # ---------------- STAGE B ----------------
    print('[B] half-res rescan of top-800...')
    t0 = time.time()
    scoresB = torch.zeros(quadsA.shape[0])
    bidxB = torch.zeros(quadsA.shape[0], dtype=torch.long)
    for b0 in range(0, quadsA.shape[0], 64):
        qb = quadsA[b0:b0 + 64].to(dev)
        corr = scan_batch(CB, D2, qb, qb.shape[0], 16)
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

    # safety injections: stage-1 champion family + two known pass families
    s1 = json.load(open(OUT / 'stage1_candidates.json', encoding='utf-8'))
    champ = s1['champion']
    probe = json.load(open(ROOT / '_tmp' / 'probe_10sing_old.json',
                           encoding='utf-8'))
    injections = [
        {'combo': champ['combo'], 'theta': champ['theta'],
         'src': 'stage1_champion'},
        {'combo': ['LG(0,3)', 'LG(2,1)', 'LG(1,3)', 'HG(2,5)'],
         'theta': (probe['rank3 LG(0,3)+LG(2,1)+LG(1,3)+HG(2,5)']
                   ['theta_new']), 'src': 'inject_rank3'},
        {'combo': ['LG(0,3)', 'LG(2,-1)', 'LG(1,3)', 'HG(3,4)'],
         'theta': (probe['rank9 LG(0,3)+LG(2,-1)+LG(1,3)+HG(3,4)']
                   ['theta_new']), 'src': 'inject_rank9'},
    ]
    for inj in injections:
        tr = [lab2k[m] for m in inj['combo']]
        if canon(tr, labels) not in seen:
            seen.add(canon(tr, labels))
            cands.insert(0, {'tr': tr, 'seed': inj['theta'],
                             'scanB': None, 'src': inj['src']})
    print(f'[B] candidates: {len(cands)}')

    # ---------------- STAGE C ----------------
    print('[C] refine (core-weighted -> W_avg) + evaluate...')
    t0 = time.time()
    for c in cands:
        tr = c['tr']
        gs = global_stats_02(modes, Ie_t, tr, None)
        ws, meta, mask = build_window_stats(modes, Ie_t, tr, None)
        ws_cw = weighted_crop_stats(modes, Ie_t, tr, None, crop_f, w_f)
        th_cw, _ = block_refine_wavg(ws_cw, c['seed'], 4,
                                     GRIDS[4]['n_ph'], GRIDS[4]['n_en'],
                                     rounds=5)
        th_w, _ = block_refine_wavg(ws, th_cw, 4, GRIDS[4]['n_ph'],
                                    GRIDS[4]['n_en'], rounds=6)
        record(th_cw, tr, f'{c.get("src","funnel")}/cw', gs, ws, meta)
        record(th_w, tr, f'{c.get("src","funnel")}/w', gs, ws, meta)
    print(f'[C1] {len(all_rec)} pass records  ({time.time()-t0:.0f}s)')

    # micro-polish top-5 robust by C_s02
    def key(r):
        return (-r['C_s02'], -r['W_avg'])
    robust_dedup, seen_keys = [], set()
    for r in sorted([x for x in all_rec if x['robust']], key=key):
        k = tuple(r['combo'])
        if k in seen_keys:
            continue
        seen_keys.add(k)
        robust_dedup.append(r)
    print(f'[C2] robust unique combos: {len(robust_dedup)}; '
          f'micro-polish top-5...')
    t0 = time.time()
    rng = np.random.default_rng(2026)
    for top in robust_dedup[:5]:
        tr = top['tr']
        gs = global_stats_02(modes, Ie_t, tr, None)
        ws, meta, mask = build_window_stats(modes, Ie_t, tr, None)
        for k in range(200):
            th = list(map(float, top['theta']))
            u = np.array(th[:4]) + rng.normal(0, 0.018, 4)
            u = np.clip(u, 0.005, None)
            u = u / u.sum()
            ph = np.array(th[4:]) + rng.normal(0, 0.06, 3)
            record(list(u) + list(ph), tr, 'funnel/polish', gs, ws, meta)
    print(f'[C2] done ({time.time()-t0:.0f}s)')

    # ---------------- rank ----------------
    def dedup(recs):
        out_, seen_ = [], set()
        for r in sorted(recs, key=lambda x: (-x['C_s02'], -x['W_avg'],
                                             x['mean_nn'])):
            k = (tuple(r['combo']),
                 tuple(round(x, 3) for x in r['theta']))
            if k in seen_:
                continue
            seen_.add(k)
            out_.append(r)
        return out_

    final = dedup(all_rec)
    final_robust = [r for r in final if r['robust']]
    print('\n=== STAGE2 ranked (robust first) ===')
    shown = final_robust[:10] + [r for r in final if not r['robust']][:3]
    for k, r in enumerate(shown):
        tag = 'R' if r['robust'] else ' '
        print(f'  {k+1:2d}.{tag} {"+".join(r["combo"]):50s} '
              f'C_s02={r["C_s02"]:.4f} W_avg={r["W_avg"]:.4f} '
              f'cores={r["cores"]}/{N_EXP} mean={r["mean_nn"]:.2f} '
              f'max={r["max_nn"]:.2f} [{r["src"]}]')

    out = {'experiment_cores': N_EXP,
           'n_candidates': len(cands),
           'n_records': len(all_rec),
           'champion': (final_robust or final)[0],
           'ranked': final_robust[:30]
           + [r for r in final if not r['robust']][:10]}
    (OUT / 'stage2_funnel.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8')
    print(f'\n[done] {OUT / "stage2_funnel.json"}')


if __name__ == '__main__':
    main()

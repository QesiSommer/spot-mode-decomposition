# -*- coding: utf-8 -*-
"""
stage8_search.py -- 9奇点 & 11奇点：三/四/五模重分解搜索（阶段1）
===================================================================
对每张图（9Singularities / 11Singularities）依次执行：

  1. 三模全空间：C(131,3)=366,145 粗扫(64px) → top-40 → 双目标精修；
  2. 四模全空间漏斗：C(131,4)=11,716,640，1/4 分辨率核加权全扫
     → top-5000（深前缀）→ 1/2 分辨率复扫 → top-120 精修；
  3. 五模深度引导（严格穷举 2.98 亿不可行，依据"加模单调性+深前缀"）：
     四模 top-3000 前缀 × 全部 127 第五模
     + 三模 top-250 前缀（含现任三模）× 全部模对 C(128,2)=8128
     （约 240 万组，1/4 分辨率核加权评分）→ top-200 精修。

排序 = 稳健结构优先（剔除单个最差核后 ≤14px）→ C_s02 → W_avg。
Outputs: 归档_八图分解/{9奇点|11奇点}_三至五模重分解/stage8_{3,4,5}mode.json
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
from refine_all_new_method import (build_base_fields_N, c_table_N, ols_c,
                                   project_c_N)
from rerun_new_metrics import (block_refine_wavg, energy_simplex_N,
                               global_stats_02, c_s02, GRIDS, COARSE,
                               coarse_grid_params, coarse_scan_N)
from window_metrics import (build_window_stats, w_avg_point, w_avg_batch,
                            window_pccs)
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats

ARCH = ROOT / '归档_八图分解'
IMGDIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
CASES = [{'name': '9Singularities', 'cn': '9奇点',
          'inc_combo': ['HG(3,1)', 'HG(4,0)', 'HG(3,3)']},
         {'name': '11Singularities', 'cn': '11奇点',
          'inc_combo': ['HG(4,0)', 'HG(2,3)', 'HG(4,3)']}]
GRIDS5 = {'n_ph': 11, 'n_en': 9, 'rounds': 5}
BATCH5 = 128


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               default=float), encoding='utf-8')


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


def scan_batch_generic(C, stats_fn, qb, B, K, Nm):
    Fs = [stats_fn['modes'][qb[:, i]] for i in range(Nm)]
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


def scan_quads_best(qb, stats_fn, C, Nm, K, batch=128):
    dev = C.device
    all_c, all_i = [], []
    for b0 in range(0, qb.shape[0], batch):
        q = qb[b0:b0 + batch].to(dev)
        B = q.shape[0]
        corr = scan_batch_generic(C, stats_fn, q, B, K, Nm)
        mx, mi = corr.max(dim=1)
        all_c.append(mx.cpu())
        all_i.append(mi.cpu())
    return torch.cat(all_c), torch.cat(all_i)


def block_refine_hooked(score_fn, p0, Nm, n_ph, n_en, rounds, on_round):
    cur = [float(x) for x in p0]
    w_ph, w_en = 0.5, 0.08
    for r in range(rounds):
        axes = [np.linspace(cur[Nm + j] - w_ph, cur[Nm + j] + w_ph, n_ph)
                for j in range(Nm - 1)]
        mesh = np.meshgrid(*axes, indexing='ij')
        ph_mat = np.stack([m.ravel() for m in mesh], axis=1)
        u_mat = np.tile(cur[:Nm], (ph_mat.shape[0], 1))
        sc = score_fn(np.concatenate([u_mat, ph_mat], 1))
        k = int(torch.argmax(sc))
        for j in range(Nm - 1):
            cur[Nm + j] = float(ph_mat[k, j])
        ax_e = [np.linspace(max(0.0, cur[i] - w_en),
                            min(1.0, cur[i] + w_en), n_en)
                for i in range(Nm - 1)]
        me_ = np.meshgrid(*ax_e, indexing='ij')
        e_mat = np.stack([m.ravel() for m in me_], axis=1)
        e_mat = e_mat[e_mat.sum(axis=1) < 1.0]
        u_mat = np.concatenate(
            [e_mat, (1 - e_mat.sum(axis=1))[:, None]], 1)
        ph_rep = np.tile(cur[Nm:], (len(u_mat), 1))
        sc = score_fn(np.concatenate([u_mat, ph_rep], 1))
        k = int(torch.argmax(sc))
        cur[:Nm] = [float(x) for x in u_mat[k]]
        w_ph *= 0.3
        w_en *= 0.3
        if on_round is not None:
            on_round(r, list(cur))
    return cur


def jitter(theta, rng, Nm, s_en=0.02, s_ph=0.12):
    u = np.array(theta[:Nm]) + rng.normal(0, s_en, Nm)
    u = np.clip(u, 0.005, None)
    u = u / u.sum()
    ph = np.array(theta[Nm:]) + rng.normal(0, s_ph, Nm - 1)
    return list(u) + list(ph)


def make_helpers(pts_e, modes_np, lab2k, labels, N_EXP):
    def struct_eval(theta, tr):
        I = render_model(modes_np, lab2k,
                         {'combo': [labels[t] for t in tr],
                          'params': list(theta), 'scale': None})
        pts_m = find_dark_points(I)
        mt = match_points(pts_e, pts_m)
        n_ok = abs(len(pts_m) - N_EXP) <= 1
        d_ok = mt['mean'] is not None and mt['mean'] < 12
        if not (n_ok and d_ok):
            return None
        dists = np.array(mt['dists'])
        max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.0
        return {'cores': len(pts_m), 'mean_nn': float(mt['mean']),
                'max_nn': float(mt['max']), 'max10': max10,
                'dists': [float(x) for x in dists]}

    def full_metrics(theta, tr, Nm, gs, ws, meta):
        C2, (v0, v2) = c_s02(gs, theta, Nm)
        w = w_avg_point(ws, theta, Nm)
        p56 = window_pccs(ws, theta, Nm)
        sg2 = [x for x, m in zip(p56, meta) if m['sigma'] == 2]
        return (float(C2), float(w),
                float(np.mean([x < 0.5 for x in sg2])),
                float(np.min(sg2)))

    def record(recs, theta, tr, tag, Nm, gs, ws, meta):
        st = struct_eval(theta, tr)
        if st is None:
            return
        C2, w, rho56, cmin56 = full_metrics(theta, tr, Nm, gs, ws, meta)
        recs.append({'combo': [labels[t] for t in tr], 'tr': list(tr),
                     'theta': [float(x) for x in theta],
                     'C_s02': C2, 'W_avg': w,
                     'cores': st['cores'], 'mean_nn': st['mean_nn'],
                     'max_nn': st['max_nn'], 'max10': st['max10'],
                     'robust': bool(st['max10'] <= 14.0),
                     'dists': st['dists'],
                     'rho56': rho56, 'cmin56': cmin56, 'src': tag})

    return struct_eval, full_metrics, record


def prep_domain(modes, Ie_t, pts_e, crop_f, size, factor, sigmas, dev):
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


def run_case(case, dev, labels, lab2k, modes, modes_np):
    NAME, CN = case['name'], case['cn']
    OUT = ARCH / f'{CN}_三至五模重分解'
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'\n############ {CN} ({NAME}) ############')
    Ie = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
    Ie_t = torch.tensor(Ie, dtype=torch.float32)
    pts_e = find_dark_points(Ie)
    N_EXP = len(pts_e)
    crop_f = crop_box(pts_e, 224)
    w_f = weight_map(Ie.shape, pts_e)
    print(f'[{CN}] exp cores={N_EXP}; crop={crop_f}')
    struct_eval, full_metrics, record = make_helpers(
        pts_e, modes_np, lab2k, labels, N_EXP)

    t_case = time.time()

    # ================= 1) three-mode full space =================
    combos = torch.combinations(torch.arange(len(labels)), r=3)
    Ie_small = F.interpolate(Ie_t[None, None], size=(64, 64),
                             mode='bilinear', align_corners=False)[0, 0]
    modes_small = torch.stack([down_complex(m, 64) for m in modes
                               ]).to(dev)
    Ie_small = Ie_small.to(dev)
    cg = COARSE[3]
    pmat3 = coarse_grid_params(3, cg['n_energy'], cg['n_phase'])
    ct3 = c_table_N(pmat3, 3).to(dev)
    t0 = time.time()
    bc, bi = coarse_scan_N(modes_small, Ie_small, combos, ct3, 3)
    print(f'[{CN}][3mode] coarse {combos.shape[0]} done '
          f'{time.time()-t0:.0f}s best={float(bc.max()):.4f}')
    recs3 = []
    vals3, idxs3 = bc.topk(40)
    for ci in range(40):
        tr = combos[idxs3[ci]].cpu().tolist()
        p0 = pmat3[bi[idxs3[ci]].cpu().numpy()].tolist()
        gs = global_stats_02(modes, Ie_t, tr, None)
        ws, meta, mask = build_window_stats(modes, Ie_t, tr, None)

        def w_score(pmat, ws=ws):
            return w_avg_batch(ws, c_table_N(np.asarray(pmat,
                                                        dtype=np.float64), 3))

        def c_score3(pmat, gs=gs):
            C = c_table_N(np.asarray(pmat, dtype=np.float64), 3)
            return (sum(eval_c_grid(C, *s) for s in gs) / len(gs))[0]

        ols, _ = ols_c(gs, sigma_idx=1)
        p_ols = project_c_N(ols, 3)
        rng = np.random.default_rng(800 + ci)
        seeds = [p0] + ([list(p_ols)] if p_ols is not None else [])
        seeds.append(jitter(p0, rng, 3))
        tag0 = '+'.join(labels[t] for t in tr)
        best_w, best_th = -9, None
        for si, sd in enumerate(seeds):
            record(recs3, sd, tr, f'{tag0}/seed{si}', 3, gs, ws, meta)
            th_w = block_refine_hooked(
                w_score, sd, 3, GRIDS[3]['n_ph'], GRIDS[3]['n_en'], 6,
                lambda r, cur, tr=tr, si=si, gs=gs, ws=ws, meta=meta:
                record(recs3, cur, tr, f'{tag0}/s{si}/W{r}', 3, gs, ws,
                       meta))
            vw = w_avg_point(ws, th_w, 3)
            if vw > best_w:
                best_w, best_th = vw, th_w
        block_refine_hooked(
            c_score3, best_th, 3, GRIDS[3]['n_ph'], GRIDS[3]['n_en'], 6,
            lambda r, cur, tr=tr, gs=gs, ws=ws, meta=meta:
            record(recs3, cur, tr, f'{tag0}/C{r}', 3, gs, ws, meta))
    uniq3 = dedup(recs3)
    rob3 = [r for r in uniq3 if r['robust']]
    champ3 = (rob3 or uniq3)[0]
    print(f'[{CN}][3mode] records={len(recs3)} unique={len(uniq3)} '
          f'robust={len(rob3)} champion: {"+".join(champ3["combo"])} '
          f'C_s02={champ3["C_s02"]:.4f} W_avg={champ3["W_avg"]:.4f} '
          f'max10={champ3["max10"]:.2f}')
    save_json(OUT / 'stage8_3mode.json', {
        'image': NAME, 'N': 3, 'n_combos': int(combos.shape[0]),
        'n_records': len(recs3), 'n_unique': len(uniq3),
        'n_robust': len(rob3), 'champion': champ3,
        'ranked': (rob3[:30] + [r for r in uniq3 if not r['robust']][:10])})

    # ================= 2) four-mode full funnel =================
    D4 = prep_domain(modes, Ie_t, pts_e, crop_f, 56, 4, (0.0, 0.5), dev)
    D2 = prep_domain(modes, Ie_t, pts_e, crop_f, 112, 2, (0.0, 1.0), dev)

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

    total4 = 131 * 130 * 129 * 128 // 24
    cache_q = OUT / 'funnel_quadsA_4mode.pt'
    if cache_q.exists():
        quadsA = torch.load(cache_q, weights_only=True)
        print(f'[{CN}][4mode] loaded cached {quadsA.shape[0]} quads')
    else:
        print(f'[{CN}][4mode] full-space scan over {total4} quads...')
        t0 = time.time()
        heap = []
        done = 0
        for i in range(128):
            rest = torch.combinations(torch.arange(i + 1, 131), r=3)
            L = rest.shape[0]
            q = torch.cat([torch.full((L, 1), i, dtype=torch.long),
                           rest], 1)
            for b0 in range(0, L, 128):
                qb = q[b0:b0 + 128].to(dev)
                corr = scan_batch_generic(CA, D4, qb, qb.shape[0], 16, 4)
                mx, mi = corr.max(dim=1)
                k = min(3, mx.numel())
                tv, ti = mx.topk(k)
                for v, ii in zip(tv.tolist(), ti.tolist()):
                    heapq.heappush(heap, (float(v), i, b0 + int(ii)))
                    if len(heap) > 5000:
                        heapq.heappop(heap)
                done += qb.shape[0]
            if i % 10 == 9 or i == 127:
                el = time.time() - t0
                print(f'  [{CN}] shard {i+1}/128 {done}/{total4} '
                      f'({el:.0f}s eta {el*(total4-done)/max(done,1):.0f}s)')
        print(f'[{CN}][4mode] scan done {time.time()-t0:.0f}s')
        topA = sorted(heap, reverse=True)
        quadsA = []
        cache_rest = {}
        for v, i, li in topA:
            if i not in cache_rest:
                cache_rest[i] = torch.combinations(
                    torch.arange(i + 1, 131), r=3)
            quadsA.append([i] + cache_rest[i][li].tolist())
        quadsA = torch.tensor(quadsA, dtype=torch.long)
        torch.save(quadsA, cache_q)
    print(f'[{CN}][4mode] heap quads: {quadsA.shape[0]}')

    t0 = time.time()
    scoresB = torch.zeros(quadsA.shape[0])
    bidxB = torch.zeros(quadsA.shape[0], dtype=torch.long)
    for b0 in range(0, quadsA.shape[0], 64):
        qb = quadsA[b0:b0 + 64].to(dev)
        corr = scan_batch_generic(CB, D2, qb, qb.shape[0], 16, 4)
        mx, mi = corr.max(dim=1)
        scoresB[b0:b0 + qb.shape[0]] = mx.cpu()
        bidxB[b0:b0 + qb.shape[0]] = mi.cpu()
    print(f'[{CN}][4mode] half-res rescan {time.time()-t0:.0f}s')

    vB, iB = scoresB.topk(len(scoresB))
    seen4, prefix4 = set(), []
    for v, k in zip(vB.tolist(), iB.tolist()):
        tr = quadsA[k].tolist()
        c = canon(tr, labels)
        if c in seen4:
            continue
        seen4.add(c)
        prefix4.append({'tr': tr, 'seed': pmatB[bidxB[k]].tolist(),
                        'scanB': v})
    print(f'[{CN}][4mode] unique prefixes: {len(prefix4)}')

    recs4 = []
    for c in prefix4[:120]:
        tr = c['tr']
        gs = global_stats_02(modes, Ie_t, tr, None)
        ws, meta, mask = build_window_stats(modes, Ie_t, tr, None)
        ws_cw = weighted_crop_stats(modes, Ie_t, tr, None, crop_f, w_f)
        th_cw, _ = block_refine_wavg(ws_cw, c['seed'], 4,
                                     GRIDS[4]['n_ph'], GRIDS[4]['n_en'], 5)
        record(recs4, th_cw, tr, 'funnel/cw', 4, gs, ws, meta)
        th_w, _ = block_refine_wavg(ws, th_cw, 4, GRIDS[4]['n_ph'],
                                    GRIDS[4]['n_en'], 6)
        record(recs4, th_w, tr, 'funnel/w', 4, gs, ws, meta)
        record(recs4, c['seed'], tr, 'funnel/seed', 4, gs, ws, meta)

        def c_score4(pmat, gs=gs):
            C = c_table_N(np.asarray(pmat, dtype=np.float64), 4)
            return (sum(eval_c_grid(C, *s) for s in gs) / len(gs))[0]
        block_refine_hooked(
            c_score4, th_w, 4, GRIDS[4]['n_ph'], GRIDS[4]['n_en'], 6,
            lambda r, cur, tr=tr, gs=gs, ws=ws, meta=meta:
            record(recs4, cur, tr, f'funnel/C{r}', 4, gs, ws, meta))
    robust_dedup, seen_k = [], set()
    for r in sorted([x for x in recs4 if x['robust']],
                    key=lambda x: (-x['C_s02'], -x['W_avg'])):
        kk = tuple(r['combo'])
        if kk in seen_k:
            continue
        seen_k.add(kk)
        robust_dedup.append(r)
    rng = np.random.default_rng(8802)
    for top in robust_dedup[:5]:
        tr = top['tr']
        gs = global_stats_02(modes, Ie_t, tr, None)
        ws, meta, mask = build_window_stats(modes, Ie_t, tr, None)
        for k in range(150):
            th = jitter(top['theta'], rng, 4, 0.018, 0.06)
            record(recs4, th, tr, 'funnel/polish', 4, gs, ws, meta)
    uniq4 = dedup(recs4)
    rob4 = [r for r in uniq4 if r['robust']]
    champ4 = (rob4 or uniq4)[0]
    print(f'[{CN}][4mode] records={len(recs4)} unique={len(uniq4)} '
          f'robust={len(rob4)} champion: {"+".join(champ4["combo"])} '
          f'C_s02={champ4["C_s02"]:.4f} W_avg={champ4["W_avg"]:.4f} '
          f'max10={champ4["max10"]:.2f}')
    save_json(OUT / 'stage8_4mode.json', {
        'image': NAME, 'N': 4, 'n_quads_heap': int(quadsA.shape[0]),
        'n_prefix': len(prefix4), 'n_records': len(recs4),
        'n_unique': len(uniq4), 'n_robust': len(rob4),
        'champion': champ4,
        'ranked': (rob4[:30] + [r for r in uniq4 if not r['robust']][:10])})

    # ================= 3) five-mode deep guided =================
    print(f'[{CN}][5mode] building beam quads...')
    t0 = time.time()

    prefix4_by_score = sorted(prefix4, key=lambda x: -x['scanB'])
    beam1_parts = []
    seen_q = set()

    for c in prefix4_by_score[:3000]:
        base = sorted(c['tr'])
        rest = [x for x in range(len(labels)) if x not in base]
        for x in rest:
            q = tuple(sorted(base + [x]))
            if q in seen_q:
                continue
            seen_q.add(q)
            beam1_parts.append(list(q))
    for r in (rob4 or uniq4)[:20]:
        base = sorted(r['tr'])
        rest = [x for x in range(len(labels)) if x not in base]
        for x in rest:
            q = tuple(sorted(base + [x]))
            if q in seen_q:
                continue
            seen_q.add(q)
            beam1_parts.append(list(q))
    print(f'[{CN}][5mode] beam1 quads: {len(beam1_parts)}')

    tri_pool = [sorted(champ3['tr'])]
    for r in uniq3[:250]:
        tri_pool.append(sorted(r['tr']))
    inc_tr = sorted(lab2k[m] for m in case['inc_combo'])
    if inc_tr not in tri_pool:
        tri_pool.insert(0, inc_tr)
    seen_tri = set()
    beam2_tensors = []
    for tr in tri_pool:
        tr = sorted(tr)
        if tuple(tr) in seen_tri:
            continue
        seen_tri.add(tuple(tr))
        rest = torch.tensor([x for x in range(len(labels))
                             if x not in tr], dtype=torch.long)
        pairs = torch.combinations(rest, r=2)
        base = torch.tensor(tr, dtype=torch.long).repeat(len(pairs), 1)
        beam2_tensors.append(torch.cat([base, pairs], 1))
    beam2_quads = torch.cat(beam2_tensors)
    print(f'[{CN}][5mode] beam2 quads: {beam2_quads.shape[0]} '
          f'({len(seen_tri)} triples) ({time.time()-t0:.0f}s)')

    e5 = energy_simplex_N(5, 4)
    ax5 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    M2, M3, M4, M5 = np.meshgrid(ax5, ax5, ax5, ax5, indexing='ij')
    ph5 = np.stack([M2.ravel(), M3.ravel(), M4.ravel(), M5.ravel()],
                   axis=1)
    pmat5 = np.concatenate([np.repeat(e5, len(ph5), 0),
                            np.tile(ph5, (len(e5), 1))], 1)
    C5 = c_table_N(pmat5, 5).to(dev)
    print(f'[{CN}][5mode] grid5 rows={C5.shape[0]}; scoring...')

    cand5 = []
    qq1 = torch.tensor(beam1_parts, dtype=torch.long)
    sc1, si1 = scan_quads_best(qq1, D4, C5, 5, 25, batch=BATCH5)
    bv, bk = sc1.topk(min(150, len(sc1)))
    for v, k in zip(bv.tolist(), bk.tolist()):
        cand5.append({'tr': qq1[k].tolist(),
                      'seed': pmat5[si1[k]].tolist(),
                      'scan5': v, 'src': 'beam1'})
    print(f'[{CN}][5mode] beam1 scored ({time.time()-t0:.0f}s)')
    sc2, si2 = scan_quads_best(beam2_quads, D4, C5, 5, 25,
                               batch=BATCH5)
    bv, bk = sc2.topk(min(150, len(sc2)))
    for v, k in zip(bv.tolist(), bk.tolist()):
        cand5.append({'tr': beam2_quads[k].tolist(),
                      'seed': pmat5[si2[k]].tolist(),
                      'scan5': v, 'src': 'beam2'})
    print(f'[{CN}][5mode] beam2 scored ({time.time()-t0:.0f}s)')

    cand5.sort(key=lambda c: -c['scan5'])
    seen5, final5 = set(), []
    for c in cand5:
        k = canon(c['tr'], labels)
        if k in seen5:
            continue
        seen5.add(k)
        final5.append(c)
        if len(final5) >= 200:
            break
    print(f'[{CN}][5mode] refine candidates: {len(final5)}')

    recs5 = []
    for ci, c in enumerate(final5):
        tr = c['tr']
        gs = global_stats_02(modes, Ie_t, tr, None)
        ws, meta, mask = build_window_stats(modes, Ie_t, tr, None)
        ws_cw = weighted_crop_stats(modes, Ie_t, tr, None, crop_f, w_f)
        th_cw, _ = block_refine_wavg(ws_cw, c['seed'], 5,
                                     GRIDS5['n_ph'], GRIDS5['n_en'], 5)
        record(recs5, th_cw, tr, f'{c["src"]}/cw', 5, gs, ws, meta)
        th_w, _ = block_refine_wavg(ws, th_cw, 5, GRIDS5['n_ph'],
                                    GRIDS5['n_en'], 6)
        record(recs5, th_w, tr, f'{c["src"]}/w', 5, gs, ws, meta)

        def c_score5(pmat, gs=gs):
            C = c_table_N(np.asarray(pmat, dtype=np.float64), 5)
            return (sum(eval_c_grid(C, *s) for s in gs) / len(gs))[0]
        block_refine_hooked(
            c_score5, th_w, 5, GRIDS5['n_ph'], GRIDS5['n_en'], 6,
            lambda r, cur, tr=tr, gs=gs, ws=ws, meta=meta:
            record(recs5, cur, tr, f'{c["src"]}/C{r}', 5, gs, ws, meta))
        if ci % 25 == 24:
            print(f'  [{CN}][5mode] refined {ci+1}/{len(final5)} '
                  f'records={len(recs5)} ({time.time()-t0:.0f}s)')
    uniq5 = dedup(recs5)
    rob5 = [r for r in uniq5 if r['robust']]
    rng = np.random.default_rng(8502)
    for top in rob5[:5]:
        tr = top['tr']
        gs = global_stats_02(modes, Ie_t, tr, None)
        ws, meta, mask = build_window_stats(modes, Ie_t, tr, None)
        for k in range(150):
            th = jitter(top['theta'], rng, 5, 0.016, 0.05)
            record(recs5, th, tr, 'polish', 5, gs, ws, meta)
    uniq5 = dedup(recs5)
    rob5 = [r for r in uniq5 if r['robust']]
    champ5 = (rob5 or uniq5)[0]
    print(f'[{CN}][5mode] records={len(recs5)} unique={len(uniq5)} '
          f'robust={len(rob5)} champion: {"+".join(champ5["combo"])} '
          f'C_s02={champ5["C_s02"]:.4f} W_avg={champ5["W_avg"]:.4f} '
          f'max10={champ5["max10"]:.2f}')
    save_json(OUT / 'stage8_5mode.json', {
        'image': NAME, 'N': 5, 'n_beam1': len(beam1_parts),
        'n_beam2': int(beam2_quads.shape[0]), 'n_refine': len(final5),
        'n_records': len(recs5), 'n_unique': len(uniq5),
        'n_robust': len(rob5), 'champion': champ5,
        'ranked': (rob5[:30] + [r for r in uniq5 if not r['robust']][:10])})
    print(f'[{CN}] case done in {time.time()-t_case:.0f}s')


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={dev}')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()
    for case in CASES:
        run_case(case, dev, labels, lab2k, modes, modes_np)
    print('\n[ALL DONE]')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
stage7b_8sing2_four_mode.py -- 8奇点(2)：四模全空间漏斗（阶段2）
=================================================================
C(131,4) = 11,716,640 四模组合，从零全空间，复用已验证的漏斗引擎：
  A. 1/4 分辨率（56²）核加权（11 实验核）全扫（grid A：35x125）
     → top-800（缓存）
  B. 1/2 分辨率（112²）重扫 top-800（grid B：56x216）→ top-200
     → canonical 去重至 ≤120
  C. 安全注入：现任两模+2 扩展（8256 组）、三模冠军+1 扩展（128 组）
     经 1/4 分辨率核加权评分 → 各取前 10/5
  D. 逐候选：核加权精修 → W_avg 精修 → C_s02 精修（逐轮记录）
  E. 稳健通过解（除至多 1 个离群核外全部 ≤14px）微扰抛光 200 点

注：该图实验检测含 1 个边界离群核 (28,120)（所有已知解均 ~52px
匹配不到）——稳健判据相应放宽为"剔除单个最差核后 max ≤14px"。
Outputs: 归档_八图分解/8奇点2_三模四模重分解/stage7b_four_mode.json
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

OUT = ROOT / '归档_八图分解' / '8奇点2_三模四模重分解'
OUT.mkdir(parents=True, exist_ok=True)
IMGDIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '8Singularities(2)'
N_EXPECT = 4


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


def scan_quads_best(qb, stats_fn, CA, BATCH=64):
    """Best (corr, param-row-index) over grid A for a list of quads."""
    dev = CA.device
    K = 16
    all_c, all_i = [], []
    for b0 in range(0, qb.shape[0], BATCH):
        q = qb[b0:b0 + BATCH].to(dev)
        B = q.shape[0]
        corr = scan_batch(CA, stats_fn, q, B, K)
        mx, mi = corr.max(dim=1)
        all_c.append(mx.cpu())
        all_i.append(mi.cpu())
    return torch.cat(all_c), torch.cat(all_i)


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

    def full_metrics(theta, tr, gs, ws, meta):
        C2, (v0, v2) = c_s02(gs, theta, N_EXPECT)
        w = w_avg_point(ws, theta, N_EXPECT)
        p56 = window_pccs(ws, theta, N_EXPECT)
        sg2 = [x for x, m in zip(p56, meta) if m['sigma'] == 2]
        return (float(C2), float(w),
                float(np.mean([x < 0.5 for x in sg2])),
                float(np.min(sg2)))

    all_rec = []

    def record(theta, tr, tag, gs, ws, meta):
        st = struct_eval(theta, tr)
        if st is None:
            return
        C2, w, rho56, cmin56 = full_metrics(theta, tr, gs, ws, meta)
        all_rec.append({'combo': [labels[t] for t in tr], 'tr': list(tr),
                        'theta': [float(x) for x in theta],
                        'C_s02': C2, 'W_avg': w,
                        'cores': st['cores'], 'mean_nn': st['mean_nn'],
                        'max_nn': st['max_nn'], 'max10': st['max10'],
                        'robust': bool(st['max10'] <= 14.0),
                        'dists': st['dists'],
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

    # ---------------- STAGE A ----------------
    total = 131 * 130 * 129 * 128 // 24
    cache_q = OUT / 'funnel_quadsA.pt'
    if cache_q.exists():
        quadsA = torch.load(cache_q, weights_only=True)
        print(f'[A] loaded cached {quadsA.shape[0]} quads')
    else:
        print(f'[A] full-space scan over {total} quads ...')
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
                      'scanB': v, 'src': 'funnel'})
        if len(cands) >= 120:
            break
    print(f'[B] funnel candidates: {len(cands)}')
    print('[B] top-5:')
    for c in cands[:5]:
        print(f'  {"+".join(labels[x] for x in c["tr"])}  '
              f'B={c["scanB"]:.4f}')

    # ---------------- injections ----------------
    s7a = json.load(open(OUT / 'stage7a_three_mode.json', encoding='utf-8'))
    champ3 = s7a['champion']['tr']
    pair = sorted([lab2k['HG(1,3)'], lab2k['HG(4,1)']])
    rest_all = [x for x in range(len(labels)) if x not in pair]
    q_pair_ext = torch.tensor(
        [pair + [int(x) for x in cc] for cc in
         torch.combinations(torch.tensor(rest_all), r=2)],
        dtype=torch.long)
    rest3 = [x for x in range(len(labels)) if x not in champ3]
    q_champ_ext = torch.tensor(
        [sorted(list(champ3) + [x]) for x in rest3], dtype=torch.long)
    for qq, topk, tag in [(q_pair_ext, 10, 'inj_pair'),
                          (q_champ_ext, 5, 'inj_champ3')]:
        cc, ci = scan_quads_best(qq, D4, CA)
        tv, ti = cc.topk(topk)
        added = 0
        for v, k in zip(tv.tolist(), ti.tolist()):
            tr = qq[k].tolist()
            if canon(tr, labels) in seen:
                continue
            seen.add(canon(tr, labels))
            cands.append({'tr': tr, 'seed': pmatA[ci[k]].tolist(),
                          'scanB': v, 'src': tag})
            added += 1
        print(f'[B] injection {tag}: +{added}')
    print(f'[B] candidates total: {len(cands)}')

    # ---------------- STAGE C/D: refine + record ----------------
    print('[C] refine (core-weighted -> W_avg -> C_s02) + evaluate...')
    t0 = time.time()

    def block_refine_hooked_c(gs, p0, n_ph, n_en, rounds, cb):
        cur = [float(x) for x in p0]
        w_ph, w_en = 0.5, 0.08

        def c_score(pmat):
            C = c_table_N(np.asarray(pmat, dtype=np.float64), 4)
            v = sum(eval_c_grid(C, *s) for s in gs) / len(gs)
            return v[0]
        for r in range(rounds):
            axes = [np.linspace(cur[4 + j] - w_ph, cur[4 + j] + w_ph, n_ph)
                    for j in range(3)]
            mesh = np.meshgrid(*axes, indexing='ij')
            ph_mat = np.stack([m.ravel() for m in mesh], axis=1)
            u_mat = np.tile(cur[:4], (ph_mat.shape[0], 1))
            sc = c_score(np.concatenate([u_mat, ph_mat], 1))
            k = int(torch.argmax(sc))
            for j in range(3):
                cur[4 + j] = float(ph_mat[k, j])
            ax_e = [np.linspace(max(0.0, cur[i] - w_en),
                                min(1.0, cur[i] + w_en), n_en)
                    for i in range(3)]
            me_ = np.meshgrid(*ax_e, indexing='ij')
            e_mat = np.stack([m.ravel() for m in me_], axis=1)
            e_mat = e_mat[e_mat.sum(axis=1) < 1.0]
            u_mat = np.concatenate(
                [e_mat, (1 - e_mat.sum(axis=1))[:, None]], 1)
            ph_rep = np.tile(cur[4:], (len(u_mat), 1))
            sc = c_score(np.concatenate([u_mat, ph_rep], 1))
            k = int(torch.argmax(sc))
            cur[:4] = [float(x) for x in u_mat[k]]
            w_ph *= 0.3
            w_en *= 0.3
            cb(r, list(cur))
        return cur

    for ci_, c in enumerate(cands):
        tr = c['tr']
        gs = global_stats_02(modes, Ie_t, tr, None)
        ws, meta, mask = build_window_stats(modes, Ie_t, tr, None)
        ws_cw = weighted_crop_stats(modes, Ie_t, tr, None, crop_f, w_f)
        src = c['src']
        th_cw, _ = block_refine_wavg(ws_cw, c['seed'], 4,
                                     GRIDS[4]['n_ph'], GRIDS[4]['n_en'],
                                     rounds=5)
        record(th_cw, tr, f'{src}/cw', gs, ws, meta)
        th_w, _ = block_refine_wavg(ws, th_cw, 4, GRIDS[4]['n_ph'],
                                    GRIDS[4]['n_en'], rounds=6)
        record(th_w, tr, f'{src}/w', gs, ws, meta)
        record(c['seed'], tr, f'{src}/seed', gs, ws, meta)
        block_refine_hooked_c(
            gs, th_w, GRIDS[4]['n_ph'], GRIDS[4]['n_en'], 6,
            lambda r, cur, tr=tr, gs=gs, ws=ws, meta=meta, src=src:
            record(cur, tr, f'{src}/C{r}', gs, ws, meta))
        if ci_ % 20 == 19:
            print(f'  [{ci_+1}/{len(cands)}] records={len(all_rec)} '
                  f'({time.time()-t0:.0f}s)')

    print(f'[C] {len(all_rec)} pass records ({time.time()-t0:.0f}s)')

    # ---------------- polish robust top-5 ----------------
    def key(r):
        return (-r['C_s02'], -r['W_avg'])
    robust_dedup, seen_keys = [], set()
    for r in sorted([x for x in all_rec if x['robust']], key=key):
        k = tuple(r['combo'])
        if k in seen_keys:
            continue
        seen_keys.add(k)
        robust_dedup.append(r)
    print(f'[D] robust unique combos: {len(robust_dedup)}; '
          f'micro-polish top-5...')
    t0 = time.time()
    rng = np.random.default_rng(7102)
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
            record(list(u) + list(ph), tr, 'polish', gs, ws, meta)
    print(f'[D] done ({time.time()-t0:.0f}s)')

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
    print('\n=== STAGE7B ranked (robust first) ===')
    shown = final_robust[:10] + [r for r in final if not r['robust']][:5]
    for k, r in enumerate(shown):
        tag = 'R' if r['robust'] else ' '
        print(f'  {k+1:2d}.{tag} {"+".join(r["combo"]):44s} '
              f'C_s02={r["C_s02"]:.4f} W_avg={r["W_avg"]:.4f} '
              f'cores={r["cores"]}/{N_EXP} mean={r["mean_nn"]:.2f} '
              f'max10={r["max10"]:.2f} [{r["src"].split("/")[0]}]')

    out = {'image': NAME, 'N': 4, 'n_candidates': len(cands),
           'n_records': len(all_rec),
           'n_robust': len(final_robust),
           'champion': (final_robust or final)[0],
           'ranked': final_robust[:30]
           + [r for r in final if not r['robust']][:10]}
    (OUT / 'stage7b_four_mode.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8')
    print(f'\n[done] {OUT / "stage7b_four_mode.json"}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
stage7a_8sing2_three_mode.py -- 8奇点(2)：三模全空间搜索（阶段1）
=================================================================
C(131,3) = 366,145 个三模组合，从零全空间：
  1. 64px 粗扫（能量网格 16 x 相位网格 20²）→ top-40
  2. 每候选：OLS 种子 + 粗网格种子 + 抖动种子 → W_avg 块坐标
     下降（逐轮记录）→ C_s02 块坐标下降（逐轮记录）
  3. 稳健通过解（全核 ≤12px）微扰抛光 200 点
排序 = 稳健结构优先 → C_s02 → W_avg。
Outputs: 归档_八图分解/8奇点2_三模四模重分解/stage7a_three_mode.json
"""
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
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from search_three_mode import eval_c_grid
from refine_all_new_method import c_table_N, ols_c, project_c_N
from window_metrics import (build_window_stats, w_avg_point, w_avg_batch,
                            window_pccs)
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from rerun_new_metrics import (global_stats_02, c_s02, GRIDS, COARSE,
                               coarse_grid_params, coarse_scan_N)

OUT = ROOT / '归档_八图分解' / '8奇点2_三模四模重分解'
OUT.mkdir(parents=True, exist_ok=True)
IMGDIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '8Singularities(2)'
N = 3

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[setup] device={dev}')
modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()
Ie_np = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
Ie = torch.tensor(Ie_np, dtype=torch.float32)
pts_e = find_dark_points(Ie_np)
N_EXP = len(pts_e)
print(f'[setup] experiment cores = {N_EXP}')


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
    C2, (v0, v2) = c_s02(gs, theta, N)
    w = w_avg_point(ws, theta, N)
    p56 = window_pccs(ws, theta, N)
    sg2 = [x for x, m in zip(p56, meta) if m['sigma'] == 2]
    return (float(C2), float(w),
            float(np.mean([x < 0.5 for x in sg2])),
            float(np.min(sg2)))


def block_refine_hooked(score_fn, p0, n_ph, n_en, rounds, on_round):
    cur = [float(x) for x in p0]
    w_ph, w_en = 0.5, 0.08
    for r in range(rounds):
        axes = [np.linspace(cur[N + j] - w_ph, cur[N + j] + w_ph, n_ph)
                for j in range(N - 1)]
        mesh = np.meshgrid(*axes, indexing='ij')
        ph_mat = np.stack([m.ravel() for m in mesh], axis=1)
        u_mat = np.tile(cur[:N], (ph_mat.shape[0], 1))
        sc = score_fn(np.concatenate([u_mat, ph_mat], 1))
        k = int(torch.argmax(sc))
        for j in range(N - 1):
            cur[N + j] = float(ph_mat[k, j])
        ax_e = [np.linspace(max(0.0, cur[i] - w_en),
                            min(1.0, cur[i] + w_en), n_en)
                for i in range(N - 1)]
        me_ = np.meshgrid(*ax_e, indexing='ij')
        e_mat = np.stack([m.ravel() for m in me_], axis=1)
        e_mat = e_mat[e_mat.sum(axis=1) < 1.0]
        u_mat = np.concatenate(
            [e_mat, (1 - e_mat.sum(axis=1))[:, None]], 1)
        ph_rep = np.tile(cur[N:], (len(u_mat), 1))
        sc = score_fn(np.concatenate([u_mat, ph_rep], 1))
        k = int(torch.argmax(sc))
        cur[:N] = [float(x) for x in u_mat[k]]
        w_ph *= 0.3
        w_en *= 0.3
        if on_round is not None:
            on_round(r, list(cur))
    return cur


def jitter(theta, rng, s_en=0.02, s_ph=0.12):
    u = np.array(theta[:N]) + rng.normal(0, s_en, N)
    u = np.clip(u, 0.005, None)
    u = u / u.sum()
    ph = np.array(theta[N:]) + rng.normal(0, s_ph, N - 1)
    return list(u) + list(ph)


winners = []


def record(theta, tr, tag, gs, ws, meta):
    ok, cores, mean_nn, max_nn = struct_check(theta, tr)
    if not ok:
        return
    C2, w, rho56, cmin56 = full_metrics(theta, tr, gs, ws, meta)
    winners.append({'combo': [labels[t] for t in tr], 'tr': list(tr),
                    'theta': [float(x) for x in theta],
                    'C_s02': C2, 'W_avg': w, 'cores': cores,
                    'mean_nn': mean_nn, 'max_nn': max_nn,
                    'robust': bool(max_nn < 12.0),
                    'rho56': rho56, 'cmin56': cmin56, 'src': tag})


t_start = time.time()

# ---------------- coarse full-space scan ----------------
def down_complex(x, size):
    t = torch.view_as_real(x).permute(2, 0, 1)[None]
    s = F.interpolate(t, size=(size, size), mode='bilinear',
                      align_corners=False)
    return torch.view_as_complex(
        s[0].permute(1, 2, 0).contiguous())


combos = torch.combinations(torch.arange(len(labels)), r=N)
print(f'[coarse] {combos.shape[0]} triples ...')
Ie_small = F.interpolate(Ie[None, None], size=(64, 64),
                         mode='bilinear', align_corners=False)[0, 0]
modes_small = torch.stack([down_complex(m, 64) for m in modes]).to(dev)
Ie_small = Ie_small.to(dev)
cg = COARSE[N]
params_mat = coarse_grid_params(N, cg['n_energy'], cg['n_phase'])
c_tab = c_table_N(params_mat, N).to(dev)
t0 = time.time()
bc, bi = coarse_scan_N(modes_small, Ie_small, combos, c_tab, N)
print(f'[coarse] done {time.time()-t0:.0f}s, best={float(bc.max()):.4f}')
K_TOP = 40
vals, idxs = bc.topk(K_TOP)
top_combos = combos[idxs.cpu()]

# ---------------- per-candidate refine + record ----------------
for ci in range(K_TOP):
    tr = top_combos[ci].tolist()
    p0 = params_mat[bi[idxs[ci]].cpu().numpy()].tolist()
    gs = global_stats_02(modes, Ie, tr, None)
    ws, meta, mask = build_window_stats(modes, Ie, tr, None)

    def w_score(pmat, ws=ws):
        return w_avg_batch(ws, c_table_N(np.asarray(pmat,
                                                    dtype=np.float64), N))

    def c_score(pmat, gs=gs):
        C = c_table_N(np.asarray(pmat, dtype=np.float64), N)
        v = sum(eval_c_grid(C, *s) for s in gs) / len(gs)
        return v[0]

    ols, _ = ols_c(gs, sigma_idx=1)
    p_ols = project_c_N(ols, N)
    rng = np.random.default_rng(100 + ci)
    seeds = [p0]
    if p_ols is not None:
        seeds.insert(0, list(p_ols))
    seeds.append(jitter(p0, rng))
    tag0 = '+'.join(labels[t] for t in tr)
    best_w, best_th = -9, None
    for si, sd in enumerate(seeds):
        record(sd, tr, f'{tag0}/seed{si}', gs, ws, meta)

        def on_w(r, cur, sd=sd, si=si):
            record(cur, tr, f'{tag0}/s{si}/W{r}', gs, ws, meta)

        th_w = block_refine_hooked(w_score, sd, GRIDS[N]['n_ph'],
                                   GRIDS[N]['n_en'], 6, on_w)
        vw = w_avg_point(ws, th_w, N)
        if vw > best_w:
            best_w, best_th = vw, th_w

    def on_c(r, cur):
        record(cur, tr, f'{tag0}/C{r}', gs, ws, meta)

    block_refine_hooked(c_score, best_th, GRIDS[N]['n_ph'],
                        GRIDS[N]['n_en'], 6, on_c)
    if ci % 8 == 7 or ci == K_TOP - 1:
        print(f'  [{ci+1}/{K_TOP}] winners={len(winners)} '
              f'({time.time()-t_start:.0f}s)')

# ---------------- dedup + micro-polish ----------------
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


uniq = dedup(winners)
robust = [r for r in uniq if r['robust']]
print(f'[rank] {len(winners)} records -> {len(uniq)} unique '
      f'({len(robust)} robust)')

final = []
for top in robust[:5]:
    tr = top['tr']
    gs = global_stats_02(modes, Ie, tr, None)
    ws, meta, mask = build_window_stats(modes, Ie, tr, None)
    rng = np.random.default_rng(777)
    for k in range(200):
        th = jitter(top['theta'], rng, s_en=0.018, s_ph=0.06)
        record(th, tr, 'polish', gs, ws, meta)

merged = dedup(uniq + [w for w in winners if w['src'] == 'polish'])
merged_robust = [r for r in merged if r['robust']]

print('\n=== STAGE7A ranked (robust first) ===')
shown = merged_robust[:10] + [r for r in merged if not r['robust']][:5]
for k, r in enumerate(shown):
    tag = 'R' if r['robust'] else ' '
    print(f'  {k+1:2d}.{tag} {"+".join(r["combo"]):44s} '
          f'C_s02={r["C_s02"]:.4f} W_avg={r["W_avg"]:.4f} '
          f'cores={r["cores"]}/{N_EXP} mean={r["mean_nn"]:.2f} '
          f'max={r["max_nn"]:.2f} [{r["src"].split("/")[-1]}]')

out = {'image': NAME, 'N': N, 'n_combos': int(combos.shape[0]),
       'coarse_best': float(bc.max()),
       'n_records': len(winners), 'n_unique': len(merged),
       'n_robust': len(merged_robust),
       'champion': (merged_robust or merged)[0],
       'ranked': merged_robust[:30]
       + [r for r in merged if not r['robust']][:10]}
(OUT / 'stage7a_three_mode.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print(f'\n[done] {OUT / "stage7a_three_mode.json"} '
      f'({time.time()-t_start:.0f}s)')

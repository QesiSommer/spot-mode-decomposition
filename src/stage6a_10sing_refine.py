# -*- coding: utf-8 -*-
"""
stage6a_10sing_refine.py -- 10奇点：结构约束重精修（新指标修复 阶段1）
=====================================================================
种子 = 5 个旧四模通过解（Sep24 guided search 的 PASS entries，含其
新方法精修参数）。
约束 = 结构验收通过（|核数差| ≤ 1 且 mean_nn < 12px）。
目标 = 双目标块坐标下降（W_avg 与 C_s02 交替，逐轮记录通过解），
再对前三名做随机微扰抛光。
排序口径 = 结构 PASS 优先，然后 C_s02（主口径）> W_avg > 位置偏差。

Outputs: 归档_八图分解/10奇点/5_新指标修复与验收/stage1_candidates.json
"""
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from window_metrics import build_window_stats, w_avg_point, w_avg_batch, \
    window_pccs
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from search_three_mode import eval_c_grid
from refine_all_new_method import c_table_N
from rerun_new_metrics import global_stats_02, c_s02, GRIDS

OUT = ROOT / '归档_八图分解' / '10奇点' / '5_新指标修复与验收'
OUT.mkdir(parents=True, exist_ok=True)
IMGDIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '10Singularities'

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()
Ie_np = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
Ie = torch.tensor(Ie_np, dtype=torch.float32)
pts_e = find_dark_points(Ie_np)
N_EXP = len(pts_e)
print(f'[setup] experiment dark cores = {N_EXP}')

guided = json.load(open(
    ROOT / '归档_八图分解' / '10奇点' / '4_四模搜索与验收'
    / '10Singularities_four_mode_guided.json', encoding='utf-8'))
probe = json.load(open(ROOT / '_tmp' / 'probe_10sing_old.json',
                       encoding='utf-8'))


def struct_check(theta, tr):
    I = render_model(modes_np, lab2k,
                     {'combo': [labels[t] for t in tr],
                      'params': list(theta), 'scale': None})
    pts_m = find_dark_points(I)
    mt = match_points(pts_e, pts_m)
    ok = (abs(len(pts_m) - N_EXP) <= 1
          and mt['mean'] is not None and mt['mean'] < 12)
    return (ok, len(pts_m),
            (float(mt['mean']) if mt['mean'] is not None else 99.0),
            (float(mt['max']) if mt['max'] is not None else 99.0))


def full_metrics(theta, tr, gs, ws, meta):
    C2, (v0, v2) = c_s02(gs, theta, 4)
    w = w_avg_point(ws, theta, 4)
    p56 = window_pccs(ws, theta, 4)
    sg2 = [x for x, m in zip(p56, meta) if m['sigma'] == 2]
    rho56 = float(np.mean([x < 0.5 for x in sg2]))
    cmin56 = float(np.min(sg2))
    return float(C2), float(w), rho56, cmin56


def block_refine_hooked(score_fn, p0, N, n_ph, n_en, rounds, on_round):
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


winners = []


def record(theta, tr, tag, gs, ws, meta):
    ok, cores, mean_nn, max_nn = struct_check(theta, tr)
    if not ok:
        return None
    C2, w, rho56, cmin56 = full_metrics(theta, tr, gs, ws, meta)
    rec = {'combo': [labels[t] for t in tr], 'tr': list(tr),
           'theta': [float(x) for x in theta],
           'C_s02': C2, 'W_avg': w, 'cores': cores,
           'mean_nn': mean_nn, 'max_nn': max_nn,
           'robust': bool(max_nn < 12.0),
           'rho56': rho56, 'cmin56': cmin56,
           'src': tag}
    winners.append(rec)
    return rec


def jitter(theta, rng, s_en=0.02, s_ph=0.12):
    th = list(map(float, theta))
    u = np.array(th[:4]) + rng.normal(0, s_en, 4)
    u = np.clip(u, 0.005, None)
    u = u / u.sum()
    ph = np.array(th[4:]) + rng.normal(0, s_ph, 3)
    return list(u) + list(ph)


t_start = time.time()
pass_entries = [e for e in guided['entries']
                if e.get('verdict') == 'PASS']
print(f'[seeds] {len(pass_entries)} PASS entries')

for e in pass_entries:
    tr = [lab2k[m] for m in e['modes']]
    tag0 = f'rank{e["rank"]}'
    th_old = list(e['energy']) + list(e['phis'])
    key = f'rank{e["rank"]} {"+".join(e["modes"])}'
    th_new = list(probe[key]['theta_new'])

    gs = global_stats_02(modes, Ie, tr, None)
    ws, meta, mask = build_window_stats(modes, Ie, tr, None)

    def w_score(pmat, ws=ws):
        return w_avg_batch(ws, c_table_N(np.asarray(pmat,
                                                    dtype=np.float64), 4))

    def c_score(pmat, gs=gs):
        C = c_table_N(np.asarray(pmat, dtype=np.float64), 4)
        v = sum(eval_c_grid(C, *s) for s in gs) / len(gs)
        return v[0]

    rng = np.random.default_rng(7)
    seeds = [(th_new, 'probe_new'), (th_old, 'sep24_old')]
    for j in range(3):
        seeds.append((jitter(th_new, rng), f'jitter{j}'))

    for th0, sname in seeds:
        record(th0, tr, f'{tag0}/{sname}/seed', gs, ws, meta)

        def on_round_w(r, cur, tr=tr, gs=gs, ws=ws, meta=meta,
                       tag0=tag0, sname=sname):
            record(cur, tr, f'{tag0}/{sname}/W{r}', gs, ws, meta)

        def on_round_c(r, cur, tr=tr, gs=gs, ws=ws, meta=meta,
                       tag0=tag0, sname=sname):
            record(cur, tr, f'{tag0}/{sname}/C{r}', gs, ws, meta)

        block_refine_hooked(w_score, th0, 4, GRIDS[4]['n_ph'],
                            GRIDS[4]['n_en'], 6, on_round_w)
        block_refine_hooked(c_score, th0, 4, GRIDS[4]['n_ph'],
                            GRIDS[4]['n_en'], 6, on_round_c)
    print(f'[{tag0}] winners so far: {len(winners)}  '
          f'({time.time()-t_start:.0f}s)')

# ---- dedup + rank ----
def dedup(recs):
    out, seen = [], set()
    for r in sorted(recs, key=lambda x: (-x['C_s02'], -x['W_avg'],
                                         x['mean_nn'])):
        key = (tuple(r['combo']),
               tuple(round(x, 3) for x in r['theta']))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


uniq = dedup(winners)
robust = [r for r in uniq if r['robust']]
print(f'[rank] {len(winners)} winners -> {len(uniq)} unique '
      f'({len(robust)} robust, i.e. every experiment core matched '
      f'within 12px)')

# ---- micro polish of the top-3 ROBUST candidates ----
final = []
for top in robust[:3]:
    tr = top['tr']
    gs = global_stats_02(modes, Ie, tr, None)
    ws, meta, mask = build_window_stats(modes, Ie, tr, None)
    best = None
    rng = np.random.default_rng(123)
    for k in range(200):
        th = jitter(top['theta'], rng, s_en=0.018, s_ph=0.06)
        ok, cores, mean_nn, max_nn = struct_check(th, tr)
        if not (ok and max_nn < 12.0):
            continue
        C2, w, rho56, cmin56 = full_metrics(th, tr, gs, ws, meta)
        if best is None or C2 > best['C_s02']:
            best = {'combo': top['combo'], 'tr': tr, 'theta': th,
                    'C_s02': C2, 'W_avg': w, 'cores': cores,
                    'mean_nn': mean_nn, 'max_nn': max_nn,
                    'robust': True, 'rho56': rho56,
                    'cmin56': cmin56, 'src': 'micro_polish'}
    final.append(best if best is not None else top)

merged = dedup(uniq + final)
merged_robust = [r for r in merged if r['robust']]

print('\n=== ranked PASS candidates (robust first) ===')
shown = merged_robust[:10] + [r for r in merged if not r['robust']][:4]
for k, r in enumerate(shown):
    tag = 'R' if r['robust'] else ' '
    print(f'  {k+1:2d}.{tag} {"+".join(r["combo"]):50s} '
          f'C_s02={r["C_s02"]:.4f} W_avg={r["W_avg"]:.4f} '
          f'cores={r["cores"]}/{N_EXP} mean={r["mean_nn"]:.2f} '
          f'max={r["max_nn"]:.2f} rho56={r["rho56"]:.3f} '
          f'cmin56={r["cmin56"]:.3f} [{r["src"]}]')

champion = (merged_robust or merged)[0]
print(f'\n[champion] {"+".join(champion["combo"])} '
      f'C_s02={champion["C_s02"]:.4f} W_avg={champion["W_avg"]:.4f} '
      f'cores={champion["cores"]}/{N_EXP} mean={champion["mean_nn"]:.2f} '
      f'max={champion["max_nn"]:.2f} robust={champion["robust"]}')

out = {'experiment_cores': N_EXP,
       'n_seeds_pass': len(pass_entries),
       'n_winners': len(winners),
       'n_robust': len(robust),
       'champion': champion,
       'ranked': (merged_robust[:40]
                  + [r for r in merged if not r['robust']][:10])}
(OUT / 'stage1_candidates.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print(f'[done] {OUT / "stage1_candidates.json"}  '
      f'({time.time()-t_start:.0f}s)')

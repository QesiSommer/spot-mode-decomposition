# -*- coding: utf-8 -*-
"""
stage6e_10sing_minmax.py -- 对照实验：最差窗优化能否压低局部低相关
====================================================================
问题：10奇点交付解的 56px σ2 窗口 PCC 低至 0.56——直接把优化目标
     从"窗口平均(W_avg)"换成"最差3窗均值"（max-min），能否把低窗
     抬起来？代价（平均指标/结构）是什么？增益是物理还是伪影？

设计：
  - 主组：从交付解 LG(0,-3)+LG(1,-3)+LG(2,-1)+HG(3,4) 出发，
    目标 = σ2 窗中最差 3 窗的均值（块坐标下降，网格同既有精修），
    逐轮结构验收记录；最后 200 点微扰抛光；
  - 对照组（反过拟合）：从刻意弱化的种子（相位 ±0.5 rad 级扰动）
    出发做同样优化——若弱化解的"增益"≥ 主组，则增益属伪影级
    （沿承 stage5f 反过拟合标准）。

注：本实验为诊断性质，不改动交付解。
Outputs: 归档_八图分解/10奇点/5_新指标修复与验收/对照实验_最差窗优化/
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from window_metrics import (build_window_stats, window_pccs, w_avg_point,
                            _stack, _eval_c_grid_b)
from refine_all_new_method import c_table_N
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from rerun_new_metrics import global_stats_02, c_s02, GRIDS

OUT = ROOT / '归档_八图分解' / '10奇点' / '5_新指标修复与验收' / '对照实验_最差窗优化'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '10Singularities'
COMSO = ['LG(0,-3)', 'LG(1,-3)', 'LG(2,-1)', 'HG(3,4)']

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()
Ie_np = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
Ie = torch.tensor(Ie_np, dtype=torch.float32)
pts_e = find_dark_points(Ie_np)
N_EXP = len(pts_e)

combo = COMSO
tr = [lab2k[m] for m in combo]
d3 = json.load(open(ROOT / '归档_八图分解' / '10奇点'
                    / '5_新指标修复与验收' / 'stage3_delivery.json',
                    encoding='utf-8'))
theta0 = [float(x) for x in d3['theta']]
print(f'[setup] champion theta loaded: {[round(x,4) for x in theta0]}')

gs = global_stats_02(modes, Ie, tr, None)
ws, meta, mask = build_window_stats(modes, Ie, tr, None)
s2_idx = [i for i, m in enumerate(meta) if m['sigma'] == 2]
print(f'[setup] windows: total={len(meta)} sigma2={len(s2_idx)}')

K_WORST = 3


def worstk_batch(pmat, K=K_WORST):
    C = c_table_N(np.asarray(pmat, dtype=np.float64), 4)
    m, E, Gram, me, ve = _stack(ws, C.device)
    corr = _eval_c_grid_b(C, m, E, Gram, me, ve)     # (Bw, P)
    vals = corr[s2_idx]                              # (nwin, P)
    k = min(K, vals.shape[0])
    return torch.topk(vals, k, largest=False, dim=0).values.mean(dim=0)


def wavg_batch(pmat):
    C = c_table_N(np.asarray(pmat, dtype=np.float64), 4)
    m, E, Gram, me, ve = _stack(ws, C.device)
    corr = _eval_c_grid_b(C, m, E, Gram, me, ve)
    return corr.mean(dim=0)


def eval_all(theta):
    C2, (v0, v2) = c_s02(gs, theta, 4)
    w = w_avg_point(ws, theta, 4)
    p56 = window_pccs(ws, theta, 4)
    s2 = [p56[i] for i in s2_idx]
    I = render_model(modes_np, lab2k,
                     {'combo': combo, 'params': list(theta),
                      'scale': None})
    pts_m = find_dark_points(I)
    mt = match_points(pts_e, pts_m)
    dists = np.array(mt['dists']) if mt['dists'] else np.array([99.0])
    n_ok = abs(len(pts_m) - N_EXP) <= 1
    d_ok = mt['mean'] is not None and mt['mean'] < 12
    struct = bool(n_ok and d_ok)
    max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.0
    srt = np.sort(s2)
    return {'C_s02': float(C2), 'W_avg': float(w),
            's2': [float(x) for x in s2],
            'min_s2': float(srt[0]),
            'worst3': float(np.mean(srt[:min(3, len(srt))])),
            'cores': len(pts_m),
            'mean_nn': float(mt['mean']),
            'max_nn': float(mt['max']), 'max10': max10,
            'struct': struct}


def block_refine(score_fn, p0, n_ph, n_en, rounds, on_round):
    cur = [float(x) for x in p0]
    w_ph, w_en = 0.5, 0.08
    for r in range(rounds):
        axes = [np.linspace(cur[4 + j] - w_ph, cur[4 + j] + w_ph, n_ph)
                for j in range(3)]
        mesh = np.meshgrid(*axes, indexing='ij')
        ph_mat = np.stack([m.ravel() for m in mesh], axis=1)
        u_mat = np.tile(cur[:4], (ph_mat.shape[0], 1))
        sc = score_fn(np.concatenate([u_mat, ph_mat], 1))
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
        sc = score_fn(np.concatenate([u_mat, ph_rep], 1))
        k = int(torch.argmax(sc))
        cur[:4] = [float(x) for x in u_mat[k]]
        w_ph *= 0.3
        w_en *= 0.3
        on_round(r, list(cur))
    return cur


def jitter(theta, rng, s_en=0.015, s_ph=0.05):
    u = np.array(theta[:4]) + rng.normal(0, s_en, 4)
    u = np.clip(u, 0.005, None)
    u = u / u.sum()
    ph = np.array(theta[4:]) + rng.normal(0, s_ph, 3)
    return list(u) + list(ph)


records = []


def run(tag, seed):
    t0 = time.time()
    before = eval_all(seed)
    records.append({'tag': f'{tag}/start', 'theta': list(seed), **before})
    print(f'[{tag}] start: min_s2={before["min_s2"]:.3f} '
          f'worst3={before["worst3"]:.3f} W_avg={before["W_avg"]:.4f} '
          f'C_s02={before["C_s02"]:.4f} struct={before["struct"]} '
          f'max10={before["max10"]:.2f}')

    def on_round(r, cur, tag=tag):
        e = eval_all(cur)
        records.append({'tag': f'{tag}/round{r}', 'theta': list(cur),
                        **e})
        print(f'  [{tag}] r{r}: min_s2={e["min_s2"]:.3f} '
              f'worst3={e["worst3"]:.3f} W_avg={e["W_avg"]:.4f} '
              f'C_s02={e["C_s02"]:.4f} struct={e["struct"]} '
              f'max10={e["max10"]:.2f}')

    th = block_refine(worstk_batch, seed, GRIDS[4]['n_ph'],
                      GRIDS[4]['n_en'], 6, on_round)
    # micro polish
    rng = np.random.default_rng(20261009)
    for k in range(200):
        tj = jitter(th, rng)
        e = eval_all(tj)
        records.append({'tag': f'{tag}/polish', 'theta': list(tj), **e})
        if k % 50 == 49:
            print(f'  [{tag}] polish {k+1}/200 '
                  f'({time.time()-t0:.0f}s)')
    print(f'[{tag}] done {time.time()-t0:.0f}s')


print('\n=== 主组：交付解起点（max-min 优化）===')
run('primary', theta0)

print('\n=== 对照组：弱化种子（反过拟合）===')
seed_c = list(theta0)
seed_c[4] += 0.6
seed_c[5] -= 0.5
seed_c[6] += 0.4
run('control', seed_c)

# ---------------- summarize ----------------
def best(recs, filt=None):
    pool = [r for r in recs if (filt is None or filt(r))]
    return max(pool, key=lambda r: r['worst3']) if pool else None


prim = [r for r in records if r['tag'].startswith('primary')]
ctrl = [r for r in records if r['tag'].startswith('control')]
p0 = prim[0]
pbest_all = best(prim)
pbest_pass = best(prim, lambda r: r['struct'])
c0 = ctrl[0]
cbest_all = best(ctrl)
print('\n=== SUMMARY ===')
for tag, r0, rb in [('primary', p0, pbest_all)]:
    print(f'{tag}: worst3 {r0["worst3"]:.3f} -> {rb["worst3"]:.3f} | '
          f'min {r0["min_s2"]:.3f} -> {rb["min_s2"]:.3f} | '
          f'W_avg {r0["W_avg"]:.4f} -> {rb["W_avg"]:.4f} | '
          f'C_s02 {r0["C_s02"]:.4f} -> {rb["C_s02"]:.4f} | '
          f'struct {r0["struct"]} -> {rb["struct"]} | '
          f'max10 {r0["max10"]:.2f} -> {rb["max10"]:.2f}')
print(f'control: worst3 {c0["worst3"]:.3f} -> {cbest_all["worst3"]:.3f} '
      f'(gain {cbest_all["worst3"]-c0["worst3"]:+.3f}) vs primary gain '
      f'{pbest_all["worst3"]-p0["worst3"]:+.3f}')

# ---------------- figure ----------------
def heatmap(ax, rec, title):
    side = 4
    hm = np.full((side, side), np.nan)
    for i in s2_idx:
        m = meta[i]
        hm[m['row'], m['col']] = rec['s2'][s2_idx.index(i)]
    cmap = matplotlib.colormaps['RdYlGn'].copy()
    cmap.set_bad('0.7')
    ax.imshow(hm, cmap=cmap, vmin=0.3, vmax=1.0)
    for i in range(side):
        for j in range(side):
            if not np.isnan(hm[i, j]):
                ax.text(j, i, f'{hm[i, j]:.2f}', ha='center',
                        va='center', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))
heatmap(axs[0], p0, f'BEFORE (champion)\nworst3={p0["worst3"]:.3f} '
                    f'W_avg={p0["W_avg"]:.4f} C_s02={p0["C_s02"]:.4f}')
heatmap(axs[1], pbest_all, f'AFTER (max-min optimum)\n'
                           f'worst3={pbest_all["worst3"]:.3f} '
                           f'W_avg={pbest_all["W_avg"]:.4f} '
                           f'C_s02={pbest_all["C_s02"]:.4f}')
labels_w = [f'w{i+1}' for i in range(len(s2_idx))]
x = np.arange(len(s2_idx))
axs[2].bar(x - 0.2, p0['s2'], width=0.4, label='before', color='#888')
axs[2].bar(x + 0.2, pbest_all['s2'], width=0.4, label='after (max-min)',
           color='#2a7')
axs[2].axhline(0.5, color='r', ls='--', lw=1, label='fail threshold 0.5')
axs[2].set_xticks(x)
axs[2].set_xticklabels(labels_w)
axs[2].set_ylim(0.3, 1.0)
axs[2].legend(fontsize=9)
axs[2].set_title('per-window PCC (56px, sigma=2)', fontsize=10)
fig.suptitle('10Singularities: max-min (worst-3 windows) optimization '
             'of the delivered solution', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / '最差窗优化对比.png', dpi=140, bbox_inches='tight')
plt.close(fig)

(OUT / 'minmax_records.json').write_text(
    json.dumps({'primary_start': p0, 'primary_best': pbest_all,
                'primary_best_pass': pbest_pass,
                'control_start': c0, 'control_best': cbest_all,
                'records': records},
               ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print(f'\n[done] {OUT}')

# -*- coding: utf-8 -*-
"""
stage13a_4sing_morph_search.py -- A：4奇点形态先验重搜索
============================================================
背景（2026-10-14 诊断）：现解 LG(0,-4)+LG(2,0)+HG(6,4) 的 C_s02 最高
（0.9163），但中央暗区占比 34%（实验 6%）——"高相关、低形态"。

本脚本把形态守门员并入搜索目标：
  最终排序 = 形态 PASS 优先 → C_s02 降序
  形态 PASS = (中央暗区差 ≤ 0.12) AND (cv_r 比 ≥ 0.55)
              AND (形状谱距离 ≤ 0.55)

搜索空间（从零全空间，非旧解扩展）：
  1. 三模全空间 C(131,3)=366,145 粗扫 → top-60 精修 → 形态+C_s02
     双重评估；
  2. 安全注入：旧解（LG(0,1)+LG(1,1)+LG(0,5)）与现解
     （LG(0,-4)+LG(2,0)+HG(6,4)）作为对照基线；
  3. 若形态 PASS 的最优解 C_s02 已 ≥ 0.90 → 判定替换。

Outputs: 归档_八图分解/4奇点_形态重搜索/
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
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from search_three_mode import eval_c_grid
from refine_all_new_method import c_table_N, ols_c, project_c_N
from rerun_new_metrics import (global_stats_02, c_s02, GRIDS, COARSE,
                               coarse_grid_params, coarse_scan_N,
                               block_refine_wavg)
from window_metrics import build_window_stats, w_avg_point, w_avg_batch, \
    window_pccs
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage0_diagnosis import parse_md
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats
from morphology_metrics import morph_compare

OUT = ROOT / '归档_八图分解' / '4奇点_形态重搜索'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '4Singularities'
CN = '4奇点'
N = 3

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[setup] device={dev}')
modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()
Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
Ie_t = torch.tensor(Ie, dtype=torch.float32)
pts_e = find_dark_points(Ie)
N_EXP = len(pts_e)
crop_f = crop_box(pts_e, 224)
w_f = weight_map(Ie.shape, pts_e)
print(f'[setup] exp cores={N_EXP}')

# ---- reference solutions ----
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))
acc_new = V3[CN]['accepted']
combo_new = V3[CN]['combo']
theta_new = [float(x) for x in acc_new['theta']]
combo_old, theta_old, scale_old = parse_md(
    ROOT / '归档_八图分解' / '最新结果' / f'{CN}_参数与指标.md')


def eval_full(I_model, combo, theta, gs):
    """C_s02 + 形态 + 结构 三合一评估。"""
    C2, (v0, v2) = c_s02(gs, theta, len(combo))
    rep = morph_compare(Ie, I_model)
    pts_m = find_dark_points(I_model)
    mt = match_points(pts_e, pts_m)
    dists = np.array(mt['dists']) if mt['dists'] else np.array([99.])
    struct = (abs(len(pts_m) - N_EXP) <= 1
              and mt['mean'] is not None and mt['mean'] < 12)
    max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
    return {'C_s02': float(C2), 'v0': float(v0), 'v2': float(v2),
            'morph': rep, 'morph_pass': rep['morph_pass'],
            'struct': bool(struct), 'cores': len(pts_m),
            'cores_exp': N_EXP, 'mean_nn': float(mt['mean'])
            if mt['mean'] is not None else 99.0, 'max10': max10,
            'central_delta': rep['central_delta'],
            'cv_ratio': rep['cv_ratio'],
            'shape_dist': rep['shape_dist']}


def record_of(combo, theta, tag, gs, note=''):
    I = render_model(modes_np, lab2k,
                     {'combo': combo, 'params': list(theta),
                      'scale': None})
    r = eval_full(I, combo, theta, gs)
    r.update({'combo': list(combo), 'theta': [float(x) for x in theta],
              'src': tag, 'note': note})
    return r


# ---- baselines ----
gs_new = global_stats_02(modes, Ie_t, [lab2k[m] for m in combo_new],
                         None)
base_new = record_of(combo_new, theta_new, 'baseline_current',
                     gs_new, '现解 v3（高相关低形态）')
gs_old = global_stats_02(modes, Ie_t, [lab2k[m] for m in combo_old],
                         scale_old)
I_old = render_model(modes_np, lab2k,
                     {'combo': combo_old, 'params': theta_old,
                      'scale': scale_old})
r_old = eval_full(I_old, combo_old, theta_old, gs_old)
r_old.update({'combo': list(combo_old),
              'theta': [float(x) for x in theta_old],
              'src': 'baseline_historic', 'note': '旧解（形态较像）'})

for tag, r in [('现解', base_new), ('旧解', r_old)]:
    print(f'[{tag}] {"+".join(r["combo"])} C_s02={r["C_s02"]:.4f} '
          f'central_delta={r["central_delta"]:.3f} '
          f'cv_ratio={r["cv_ratio"]:.2f} shape={r["shape_dist"]:.3f} '
          f'morph={"PASS" if r["morph_pass"] else "fail"} '
          f'struct={r["struct"]} cores={r["cores"]}/{N_EXP}')

# ================= three-mode full space =================
def down_complex(x, size):
    t = torch.view_as_real(x).permute(2, 0, 1)[None]
    s = F.interpolate(t, size=(size, size), mode='bilinear',
                      align_corners=False)
    return torch.view_as_complex(s[0].permute(1, 2, 0).contiguous())


combos = torch.combinations(torch.arange(len(labels)), r=N)
print(f'[coarse] {combos.shape[0]} triples ...')
Ie_small = F.interpolate(Ie_t[None, None], size=(64, 64),
                         mode='bilinear', align_corners=False)[0, 0]
modes_small = torch.stack([down_complex(m, 64) for m in modes]).to(dev)
Ie_small = Ie_small.to(dev)
cg = COARSE[N]
pmat3 = coarse_grid_params(N, cg['n_energy'], cg['n_phase'])
ct3 = c_table_N(pmat3, N).to(dev)
t0 = time.time()
bc, bi = coarse_scan_N(modes_small, Ie_small, combos, ct3, N)
print(f'[coarse] done {time.time()-t0:.0f}s best={float(bc.max()):.4f}')

K_TOP = 60
vals, idxs = bc.topk(K_TOP)
records = []
t0 = time.time()
for ci in range(K_TOP):
    tr = combos[idxs[ci]].cpu().tolist()
    p0 = pmat3[bi[idxs[ci]].cpu().numpy()].tolist()
    gs = global_stats_02(modes, Ie_t, tr, None)
    ws, meta, mask = build_window_stats(modes, Ie_t, tr, None)
    ws_cw = weighted_crop_stats(modes, Ie_t, tr, None, crop_f, w_f)

    def w_score(pmat, ws=ws):
        return w_avg_batch(ws, c_table_N(np.asarray(pmat,
                                                    dtype=np.float64), N))

    def c_score(pmat, gs=gs):
        C = c_table_N(np.asarray(pmat, dtype=np.float64), N)
        return (sum(eval_c_grid(C, *s) for s in gs) / len(gs))[0]

    # multi-seed refine: coarse seed + OLS + jitter
    ols, _ = ols_c(gs, sigma_idx=1)
    p_ols = project_c_N(ols, N)
    rng = np.random.default_rng(7000 + ci)
    seeds = [p0] + ([list(p_ols)] if p_ols is not None else [])
    for _ in range(2):
        j = np.array(p0) + rng.normal(0, 0.02, 2 * N - 1)
        j[:N] = np.clip(np.abs(j[:N]), 0.005, None)
        j[:N] = j[:N] / j[:N].sum()
        seeds.append(list(j))

    best_local = None
    for sd in seeds:
        th_w, _ = block_refine_wavg(ws_cw, sd, N, GRIDS[3]['n_ph'],
                                    GRIDS[3]['n_en'], rounds=4)
        th_w, _ = block_refine_wavg(ws, th_w, N, GRIDS[3]['n_ph'],
                                    GRIDS[3]['n_en'], rounds=6)
        # C_s02 refine round
        cur = list(th_w)
        for _ in range(4):
            ax = [np.linspace(cur[N + j] - 0.3, cur[N + j] + 0.3, 21)
                  for j in range(N - 1)]
            mesh = np.meshgrid(*ax, indexing='ij')
            ph = np.stack([m.ravel() for m in mesh], axis=1)
            um = np.tile(cur[:N], (ph.shape[0], 1))
            sc = c_score(np.concatenate([um, ph], 1))
            k = int(torch.argmax(sc))
            for j in range(N - 1):
                cur[N + j] = float(ph[k, j])
            eax = [np.linspace(max(0, cur[i] - 0.05),
                               min(1, cur[i] + 0.05), 11)
                   for i in range(N - 1)]
            me_ = np.meshgrid(*eax, indexing='ij')
            em = np.stack([m.ravel() for m in me_], axis=1)
            em = em[em.sum(axis=1) < 1.0]
            um = np.concatenate(
                [em, (1 - em.sum(axis=1))[:, None]], 1)
            ph_rep = np.tile(cur[N:], (len(um), 1))
            sc = c_score(np.concatenate([um, ph_rep], 1))
            k = int(torch.argmax(sc))
            cur[:N] = [float(x) for x in um[k]]
        r = record_of([labels[t] for t in tr], cur,
                      f'search/seed', gs)
        if best_local is None or r['C_s02'] > best_local['C_s02']:
            best_local = r
    records.append(best_local)
    if ci % 10 == 9:
        npass = sum(1 for r in records if r['morph_pass'])
        print(f'  [{ci+1}/{K_TOP}] morph-pass so far={npass} '
              f'({time.time()-t0:.0f}s)')

# ---- injected baselines as candidates ----
records.append(base_new)
records.append(r_old)

# ---- rank: morphology PASS first, then C_s02 ----
records.sort(key=lambda r: (0 if r['morph_pass'] else 1,
                            -r['C_s02'], -r['v0']))
uniq, seen = [], set()
for r in records:
    k = tuple(r['combo'])
    if k in seen:
        continue
    seen.add(k)
    uniq.append(r)

print('\n=== ranked (morph PASS first, then C_s02) ===')
for k, r in enumerate(uniq[:12]):
    flag = 'M' if r['morph_pass'] else ' '
    print(f'  {k+1:2d}.{flag} {"+".join(r["combo"]):42s} '
          f'C_s02={r["C_s02"]:.4f} centralΔ={r["central_delta"]:.3f} '
          f'cvR={r["cv_ratio"]:.2f} shape={r["shape_dist"]:.3f} '
          f'cores={r["cores"]}/{N_EXP} struct={r["struct"]}')

morph_winners = [r for r in uniq if r['morph_pass'] and r['struct']]
champion = morph_winners[0] if morph_winners else None
if champion is not None:
    print(f'\n[champion] {"+".join(champion["combo"])} '
          f'C_s02={champion["C_s02"]:.4f} '
          f'(vs 现解 {base_new["C_s02"]:.4f}, 旧解 {r_old["C_s02"]:.4f})')
else:
    print('\n[champion] NONE — 无形态通过解')

out = {
    'image': NAME, 'N': N, 'n_combos': int(combos.shape[0]),
    'n_records': len(records),
    'baselines': {'current': base_new, 'historic': r_old},
    'n_morph_pass': sum(1 for r in uniq if r['morph_pass']),
    'champion': champion,
    'ranked': uniq[:40]}
(OUT / 'stage13a_morph_search.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print(f'\n[done] {OUT}')

# -*- coding: utf-8 -*-
"""
probeC_extended_dict.py -- 探针C：扩展字典（N_max 10→14）引导搜索
=====================================================================
问题：把模式字典从 ≤10 阶（131 模）扩到 ≤14 阶（约 240 模），更高阶
     内容能否改善最弱图的拟合？

对象：10 奇点（C_s02 0.8616，且探针A显示其含模型未覆盖的真实
     不对称内容）。
设计（全空间 C(~240,4)≈4.5 亿不可行 → 引导束）：
  beam-A 冠军 4 模 + 每个新增模（5 模）；
  beam-B 冠军 4 模之一被新模替换（4 模）；
  beam-C stage2 漏斗 top-10 稳健 4 模 + 每个新模（5 模）。
  1/4 分辨率核加权评分（4/5 模各自网格）→ 全部候选精修 + 新指标
  验收 → 与现任冠军对比。
Outputs: 归档_八图分解/扩展基评估/probeC_extended_dict.json
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
from refine_all_new_method import c_table_N
from rerun_new_metrics import (energy_simplex_N, global_stats_02,
                               GRIDS, block_refine_wavg)
from window_metrics import build_window_stats
from structural_check import find_dark_points
from calibrate_metrics import render_model
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats
from stage8_search import (save_json, scan_batch_generic,
                           scan_quads_best, block_refine_hooked, jitter,
                           dedup, make_helpers, prep_domain, GRIDS5,
                           down_complex)

OUT = ROOT / '归档_八图分解' / '扩展基评估'
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '10Singularities'
CHAMP = ['LG(0,-3)', 'LG(1,-3)', 'LG(2,-1)', 'HG(3,4)']

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[setup] device={dev}')

modes_old, labels_old = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k_old = {l: k for k, l in enumerate(labels_old)}
print(f'[setup] old dict: {len(labels_old)} modes')

t0 = time.time()
modes_new, labels_new = build_joint_modes(14, 224, 4.0, 'cpu')
lab2k_new = {l: k for k, l in enumerate(labels_new)}
print(f'[setup] new dict: {len(labels_new)} modes '
      f'({time.time()-t0:.0f}s)')
new_labels = [l for l in labels_new if l not in lab2k_old]
print(f'[setup] NEW modes (order 11-14): {len(new_labels)}')

Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
Ie_t = torch.tensor(Ie, dtype=torch.float32)
pts_e = find_dark_points(Ie)
N_EXP = len(pts_e)
crop_f = crop_box(pts_e, 224)
w_f = weight_map(Ie.shape, pts_e)
modes_np_new = modes_new.numpy()
struct_eval, full_metrics, record = make_helpers(
    pts_e, modes_np_new, lab2k_new, labels_new, N_EXP)

D4 = prep_domain(modes_new, Ie_t, pts_e, crop_f, 56, 4, (0.0, 0.5), dev)

# grids
e5 = energy_simplex_N(5, 4)
ax5 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
M2, M3, M4, M5 = np.meshgrid(ax5, ax5, ax5, ax5, indexing='ij')
ph5 = np.stack([M2.ravel(), M3.ravel(), M4.ravel(), M5.ravel()], axis=1)
pmat5 = np.concatenate([np.repeat(e5, len(ph5), 0),
                        np.tile(ph5, (len(e5), 1))], 1)
C5 = c_table_N(pmat5, 5).to(dev)
eA = energy_simplex_N(4, 4)
axA = np.linspace(0, 2 * np.pi, 5, endpoint=False)
P2, P3, P4 = np.meshgrid(axA, axA, axA, indexing='ij')
phA = np.stack([P2.ravel(), P3.ravel(), P4.ravel()], axis=1)
pmatA = np.concatenate([np.repeat(eA, len(phA), 0),
                        np.tile(phA, (len(eA), 1))], 1)
CA = c_table_N(pmatA, 4).to(dev)

# ---- beams ----
champ_idx = [lab2k_new[m] for m in CHAMP]
new_idx = [lab2k_new[l] for l in new_labels]
funnel = json.load(open(
    ROOT / '归档_八图分解' / '10奇点' / '5_新指标修复与验收'
    / 'stage2_funnel.json', encoding='utf-8'))
top4 = []
seen4 = set()
for r in funnel['ranked']:
    k = tuple(r['combo'])
    if k in seen4 or k == tuple(CHAMP):
        continue
    seen4.add(k)
    top4.append(r['tr'])
    if len(top4) >= 10:
        break

beamA = [sorted(champ_idx + [x]) for x in new_idx]
beamB = []
for i in range(4):
    for x in new_idx:
        q = sorted(champ_idx[:i] + champ_idx[i + 1:] + [x])
        beamB.append(q)
beamC = []
for tr in top4:
    for x in new_idx:
        beamC.append(sorted(tr + [x]))
print(f'[beam] A={len(beamA)} B={len(beamB)} C={len(beamC)}')

cands = []
t0 = time.time()
for tag, quads, Nm, C in [('A', beamA, 5, C5), ('B', beamB, 4, CA),
                          ('C', beamC, 5, C5)]:
    qq = torch.tensor(quads, dtype=torch.long)
    sc, si = scan_quads_best(qq, D4, C, Nm, 25 if Nm == 5 else 16,
                             batch=64)
    bv, bk = sc.topk(min(12, len(sc)))
    for v, k in zip(bv.tolist(), bk.tolist()):
        pmat = pmat5 if Nm == 5 else pmatA
        cands.append({'tr': qq[k].tolist(), 'N': Nm,
                      'seed': pmat[si[k]].tolist(), 'scan': v,
                      'src': f'beam{tag}'})
    print(f'[beam{tag}] scored {len(quads)} ({time.time()-t0:.0f}s), '
          f'best scan={float(bv[0]):.4f}')

cands.sort(key=lambda c: -c['scan'])
seen, final = set(), []
for c in cands:
    k = tuple(c['tr'])
    if k in seen:
        continue
    seen.add(k)
    final.append(c)
    if len(final) >= 18:
        break
print(f'[refine] {len(final)} candidates')

recs = []
for ci, c in enumerate(final):
    tr, Nm = c['tr'], c['N']
    gs = global_stats_02(modes_new, Ie_t, tr, None)
    ws, meta, mask = build_window_stats(modes_new, Ie_t, tr, None)
    ws_cw = weighted_crop_stats(modes_new, Ie_t, tr, None, crop_f, w_f)
    g = GRIDS5 if Nm == 5 else GRIDS[4]
    th_cw, _ = block_refine_wavg(ws_cw, c['seed'], Nm, g['n_ph'],
                                 g['n_en'], 5)
    record(recs, th_cw, tr, f'{c["src"]}/cw', Nm, gs, ws, meta)
    th_w, _ = block_refine_wavg(ws, th_cw, Nm, g['n_ph'], g['n_en'], 6)
    record(recs, th_w, tr, f'{c["src"]}/w', Nm, gs, ws, meta)
    if ci % 6 == 5:
        print(f'  [{ci+1}/{len(final)}] records={len(recs)} '
              f'({time.time()-t0:.0f}s)')

uniq = dedup(recs)
rob = [r for r in uniq if r['robust']]
print(f'\n[rank] records={len(recs)} unique={len(uniq)} robust={len(rob)}')
print('\n=== probeC ranked (robust first) ===')
for k, r in enumerate((rob[:8] + [x for x in uniq if not x['robust']][:4])):
    tag = 'R' if r['robust'] else ' '
    newm = [m for m in r['combo'] if m in new_labels]
    print(f'  {k+1:2d}.{tag} {"+".join(r["combo"]):58s} '
          f'N={len(r["combo"])} C_s02={r["C_s02"]:.4f} '
          f'W_avg={r["W_avg"]:.4f} cores={r["cores"]}/{N_EXP} '
          f'max10={r["max10"]:.2f} newmodes={newm}')

champ_rec = None
for r in uniq:
    if tuple(r['combo']) == tuple(CHAMP):
        champ_rec = r
        break
best_new = (rob or uniq)[0]
save_json(OUT / 'probeC_extended_dict.json', {
    'image': NAME, 'n_modes_old': len(labels_old),
    'n_modes_new': len(labels_new), 'n_new_modes': len(new_labels),
    'beams': {'A': len(beamA), 'B': len(beamB), 'C': len(beamC)},
    'baseline_champion': {'combo': CHAMP, 'C_s02': 0.8616},
    'champ_in_new_dict': champ_rec,
    'best_with_new_modes': best_new,
    'ranked': rob[:20] + [x for x in uniq if not x['robust']][:10]})
print(f'\n[champion re-eval in new dict] '
      f'{champ_rec["C_s02"] if champ_rec else None}')
print(f'[best with new modes] {"+".join(best_new["combo"])} '
      f'C_s02={best_new["C_s02"]:.4f} robust={best_new["robust"]}')
print(f'[done] {OUT}')

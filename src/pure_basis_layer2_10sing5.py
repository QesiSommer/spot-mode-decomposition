# -*- coding: utf-8 -*-
r"""
pure_basis_layer2_10sing5.py — 10奇点 五模纯 HG 升级搜索
==========================================================
四模纯 HG 0.9222 已超现任 0.9013，但 9/10 核（差 1 个）。
五模升级目标：拉回 10/10 核 + 更高 C_s02。
策略（复用 stage8 深度引导思路）：四模 top 池 + 单模扩展。
  种子池 = 四模搜索 top-12（结构通过）× 66 个 HG 扩展模 ≈ 800 组合
  + 四模粗筛全局 top 组合扩展。精修后结构门排名。
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
sys.path.insert(0, str(ROOT / 'closed_routes'))  # morph_search_lib 所在目录
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points
import morph_search_lib as L

CN = '10奇点'
NAME = '10Singularities'
N = 5
N_EXP_CORES = 10
CURRENT_C = 0.9013

OUT = ROOT / '归档_八图分解' / '纯基调整_2026-10-16' / '第2层_搜索' \
    / CN
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('building joint modes ...')
modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
modes_np = modes.numpy()
lab2k = {l: k for k, l in enumerate(labels)}

Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
pts_e = find_dark_points(Ie)
print(f'experiment cores = {len(pts_e)}')

hg_idx = [k for k, l in enumerate(labels) if l.startswith('HG')]

# ---- 种子池：四模结果 top 组合扩展 ----
d4 = json.load(open(OUT / '搜索结果.json', encoding='utf-8'))
seeds = []
for b in d4['HG']['best_any'][:12]:
    base = tuple(sorted(lab2k[m] for m in b['combo']))
    seeds.append(base)
# 四模粗筛全局 top 扩展：直接对全部四模组合做一次纯 HG 粗筛
# 已经做过（d4 记录 top5），把 best_any 的 12 个做扩展即可
quads = set()
for base in seeds:
    for x in hg_idx:
        if x in base:
            continue
        quads.add(tuple(sorted(base + (x,))))
print(f'five-mode pool: {len(quads)} combos')
combos = torch.tensor(sorted(quads), dtype=torch.long)

t0 = time.time()
scr = L.screen_pool(modes_np, Ie, combos, N, dev)
corr = scr['corr']
central = scr['central']
central_e = scr['central_exp']
score = corr - 0.5 * np.abs(central - central_e)
order = np.argsort(-score)[:60]
print(f'  screen done {time.time()-t0:.0f}s; top corr='
      f'{corr[order[0]]:.4f}')


def _rec_with_combo(r, combo):
    rec = {k: v for k, v in r.items() if k != '_I'}
    rec['combo'] = list(combo)
    return rec


best_list = []
for rank, qi in enumerate(order[:45]):
    combo = [labels[k] for k in scr['combos'][qi].tolist()]
    p0 = scr['params'][qi]
    ev = L.make_eval(Ie, pts_e, modes_np, lab2k, combo)
    r = L.refine(ev, combo, list(p0), [1.0] * N, [1.0] * N,
                 [0.0, 0.0], [0.0] * N, rounds=2, budget=50,
                 seed=rank, tag=f'HG5#{rank}')
    best_list.append(_rec_with_combo(r, combo))
best_list.sort(key=lambda r: -r['C_s02'])
top = [b for b in best_list if b['struct']][:5]
print(f'  top refined C_s02 = '
      f'{[round(b["C_s02"], 4) for b in best_list[:3]]}')
print(f'  top struct-PASS    = '
      f'{[round(b["C_s02"], 4) for b in top]}')
print(f'  top cores          = {[b["cores"] for b in top]}')

d4['HG5'] = {'n_combos': int(combos.shape[0]),
             'best_any': best_list[:5], 'best_struct': top[:5]}
json.dump(d4, open(OUT / '搜索结果.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

print(f'\n==== 对比（{CN}）====')
print(f'现任 v3 混合解: C_s02 = {CURRENT_C}')
print(f'纯HG 四模:      0.9222 (9/10 核)')
if top:
    b = top[0]
    print(f'纯HG 五模最优:  C_s02 = {b["C_s02"]:.4f}  '
          f'{ "+".join(b["combo"]) }  cores {b["cores"]}/'
          f'{N_EXP_CORES}  mean_nn {b["mean_nn"]:.1f}')
print(f'\nDONE -> {OUT}')

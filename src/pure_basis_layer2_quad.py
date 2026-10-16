# -*- coding: utf-8 -*-
r"""
pure_basis_layer2_quad.py — 第2层：8(1) / 10奇点 四模纯基双轨搜索
====================================================================
与 4奇点脚本同构，四模：
  8(1):  纯 HG C(66,4)=720,720 / 纯 LG C(65,4)=677,040
  10奇点: 同上
现任（标准记号）：
  8(1)  LG(2,0)+HG(1,1)+HG(0,2)+HG(3,3)   C_s02=0.9703
  10奇点 LG(0,3)+LG(1,3)+LG(2,1)+HG(4,3)  C_s02=0.9013
纯 LG 组合锁 sx=sy=1, rot=0（v3 冻结规则），只精修 theta+shift。
四模不够（差距 >0.01）时进五模升级（本脚本先出四模基线）。

用法: python pure_basis_layer2_quad.py 8奇点1|10奇点
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

CN = sys.argv[1] if len(sys.argv) > 1 else '8奇点1'
CONF = {
    '8奇点1': {'name': '8Singularities(1)', 'n_exp': 8,
              'current': 0.9703},
    '10奇点': {'name': '10Singularities', 'n_exp': 10,
              'current': 0.9013},
}[CN]
NAME = CONF['name']
N_EXP_CORES = CONF['n_exp']
CURRENT_C = CONF['current']
N = 4
TOP_K = 40

OUT = ROOT / '归档_八图分解' / '纯基调整_2026-10-16' / '第2层_搜索' / CN
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'building joint modes ... ({CN})')
modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
modes_np = modes.numpy()
lab2k = {l: k for k, l in enumerate(labels)}

Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
pts_e = find_dark_points(Ie)
print(f'experiment cores = {len(pts_e)}')

hg_idx = [k for k, l in enumerate(labels) if l.startswith('HG')]
lg_idx = [k for k, l in enumerate(labels) if l.startswith('LG')]


def _rec_with_combo(r, combo):
    rec = {k: v for k, v in r.items() if k != '_I'}
    rec['combo'] = list(combo)
    return rec


results = {}
for track, fam_idx in [('HG', hg_idx), ('LG', lg_idx)]:
    t0 = time.time()
    combos = torch.combinations(
        torch.tensor(fam_idx, dtype=torch.long), r=N)
    print(f'\n=== track {track}: {combos.shape[0]} combos ===')
    scr = L.screen_pool(modes_np, Ie, combos, N, dev)
    corr = scr['corr']
    central = scr['central']
    central_e = scr['central_exp']
    score = corr - 0.5 * np.abs(central - central_e)
    order = np.argsort(-score)[:TOP_K * 3]
    print(f'  screen done {time.time()-t0:.0f}s; '
          f'top corr={corr[order[0]]:.4f}')

    best_list = []
    for rank, qi in enumerate(order[:TOP_K]):
        combo = [labels[k] for k in scr['combos'][qi].tolist()]
        p0 = scr['params'][qi]
        ev = L.make_eval(Ie, pts_e, modes_np, lab2k, combo)
        r = L.refine(ev, combo, list(p0), [1.0] * N, [1.0] * N,
                     [0.0, 0.0], [0.0] * N, rounds=2, budget=60,
                     seed=rank, tag=f'{track}#{rank}')
        if track == 'LG':
            r2 = ev(r['theta'], [1.0] * N, [1.0] * N, r['shift'],
                    [0.0] * N, True)
            r2['tag'] = r['tag']
            r = r2
        best_list.append(_rec_with_combo(r, combo))
    best_list.sort(key=lambda r: -r['C_s02'])
    top3 = [b for b in best_list if b['struct']][:3]
    print(f'  top refined C_s02 = '
          f'{[round(b["C_s02"],4) for b in best_list[:3]]}')
    print(f'  top struct-PASS    = '
          f'{[round(b["C_s02"],4) for b in top3]}')
    results[track] = {
        'n_combos': int(combos.shape[0]),
        'best_any': best_list[:5],
        'best_struct': top3[:5],
    }
    json.dump(results, open(OUT / '搜索结果.json', 'w',
                            encoding='utf-8'),
              ensure_ascii=False, indent=2)

print(f'\n==== 对比（{CN}）====')
print(f'现任 v3 混合解: C_s02 = {CURRENT_C}')
for track in ('HG', 'LG'):
    bs = results[track]['best_struct']
    if bs:
        b = bs[0]
        print(f'纯{track} 最优结构解: C_s02 = {b["C_s02"]:.4f}  '
              f'{ "+".join(b["combo"]) }  '
              f'cores {b["cores"]}/{N_EXP_CORES}  '
              f'mean_nn {b["mean_nn"]:.1f}')
    else:
        print(f'纯{track}: 无结构通过解（top-40 内）')
json.dump(results, open(OUT / '搜索结果.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print(f'\nDONE -> {OUT}')

# -*- coding: utf-8 -*-
r"""
pure_basis_layer2_4sing.py — 第2层：4奇点 三模纯基双轨全空间搜索
====================================================================
背景：师姐建议选基固定一族。第1层已证混合解可精确改写为纯 LG(3模) /
纯 HG(6模)。本脚本回答：**从零搜索**是否存在稀疏的单一族组合追平
现任混合解 C_s02=0.9163（HG(6,4)+LG(0,-4)+LG(2,0)，内部记号）。

双轨：
  A) 纯 HG 三模：C(66,3) = 45,760 组合
  B) 纯 LG 三模：C(65,3) = 43,680 组合（LG 字典去掉 LG(0,0)）

流程（复用 morph_search_lib）：
  1) screen_pool GPU 粗筛（解析粗扫 3模 (10,12) 网格 + 64px 形态代理）
  2) 按粗筛 C 排序取 top 池（形态代理 central 差 < 0.10 加权）
  3) 每候选 224px 全分辨率精修（refine：theta 抖动 + 平移 +
     逐模 sx/sy + HG-only 旋转；纯 LG 组合锁 sx=sy=1, rot=0——
     v3 规则 LG 冻结，只有 theta+shift 可调）
  4) 结构门 + 排名，与现任对比

输出：归档_八图分解\纯基调整_2026-10-16\第2层_搜索\4奇点\
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

OUT = ROOT / '归档_八图分解' / '纯基调整_2026-10-16' / '第2层_搜索' \
    / '4奇点'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '4Singularities'
CN = '4奇点'
N = 3
N_EXP_CORES = 4            # 该图检测 4 核
TOP_K = 60                 # 精修候选数
CURRENT_C = 0.9163         # 现任 v3 混合解

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('building joint modes ...')
modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
modes_np = modes.numpy()
lab2k = {l: k for k, l in enumerate(labels)}

Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
pts_e = find_dark_points(Ie)
print(f'experiment cores = {len(pts_e)}')

hg_idx = [k for k, l in enumerate(labels) if l.startswith('HG')]
lg_idx = [k for k, l in enumerate(labels) if l.startswith('LG')]

results = {}


def _rec_with_combo(r, combo):
    rec = {k: v for k, v in r.items() if k != '_I'}
    rec['combo'] = list(combo)
    return rec


for track, fam_idx in [('HG', hg_idx), ('LG', lg_idx)]:
    t0 = time.time()
    combos = torch.combinations(torch.tensor(fam_idx, dtype=torch.long),
                                r=N)
    print(f'\n=== track {track}: {combos.shape[0]} combos ===')
    scr = L.screen_pool(modes_np, Ie, combos, N, dev)
    corr = scr['corr']
    central = scr['central']
    central_e = scr['central_exp']
    # 形态加权排序：粗筛 C 为主，中央暗区差惩罚
    score = corr - 0.5 * np.abs(central - central_e)
    order = np.argsort(-score)[:TOP_K * 3]
    print(f'  screen done {time.time()-t0:.0f}s; '
          f'top corr={corr[order[0]]:.4f}')

    # 精修（224px 全分辨率）
    best_list = []
    for rank, qi in enumerate(order[:TOP_K]):
        combo = [labels[k] for k in scr['combos'][qi].tolist()]
        p0 = scr['params'][qi]
        ev = L.make_eval(Ie, pts_e, modes_np, lab2k, combo)
        if track == 'HG':
            r = L.refine(ev, combo, list(p0), [1.0] * N, [1.0] * N,
                         [0.0, 0.0], [0.0] * N, rounds=2, budget=80,
                         seed=rank, tag=f'{track}#{rank}')
        else:
            # 纯 LG：v3 规则冻结 sx/sy/rot，只精修 theta + shift
            r = L.refine(ev, combo, list(p0), [1.0] * N, [1.0] * N,
                         [0.0, 0.0], [0.0] * N, rounds=2, budget=80,
                         seed=rank, tag=f'{track}#{rank}')
            # refine 内部会动 sx/sy —— 重锁后重评
            r2 = ev(r['theta'], [1.0] * N, [1.0] * N, r['shift'],
                    [0.0] * N, True)
            r2['tag'] = r['tag']
            r = r2
        best_list.append(_rec_with_combo(r, combo))
    best_list.sort(key=lambda r: -r['C_s02'])
    # 结构通过的前三 + 全局前三
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

# ---- 对比现任 ----
print('\n==== 对比（4奇点）====')
print(f'现任 v3 混合解: C_s02 = {CURRENT_C}')
for track in ('HG', 'LG'):
    bs = results[track]['best_struct']
    if bs:
        b = bs[0]
        print(f'纯{track} 最优结构解: C_s02 = {b["C_s02"]:.4f}  '
              f'{ "+".join(b["combo"]) }  '
              f'cores {b["cores"]}/{N_EXP_CORES}  '
              f'mean_nn {b["mean_nn"]:.1f}')
json.dump(results, open(OUT / '搜索结果.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print(f'\nDONE -> {OUT}')

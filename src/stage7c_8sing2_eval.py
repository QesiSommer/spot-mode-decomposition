# -*- coding: utf-8 -*-
"""
stage7c_8sing2_eval.py -- 8奇点(2)：统一评估与判定（阶段3）
=============================================================
把三模（stage7a）、四模（stage7b）候选与现任两模解放在同一新指标
口径下逐一评估，按"结构 PASS（含边界核适配稳健判据）→ C_s02 →
W_avg"排序并给出替换判定：

  替换条件：三步解结构 PASS 且 适配稳健（剔除单个最差核后 max ≤14px）
           且 C_s02 >= 现任 且 W_avg >= 现任 - 0.005。

注：该图实验检测含 1 个边界离群核 (28,120)（所有已知解均 ~52px），
    稳健判据为"剔除单个最差核后 max ≤14px"。

Outputs: 归档_八图分解/8奇点2_三模四模重分解/stage7c_decision.json
"""
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from window_metrics import build_window_stats, window_pccs, w_avg_point
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model, angular_distance
from verify_metrics import diff_localization
from rerun_new_metrics import global_stats_02, c_s02
from stage0_diagnosis import parse_md

OUT = ROOT / '归档_八图分解' / '8奇点2_三模四模重分解'
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '8Singularities(2)'
ROBUST_MARGIN = 14.0

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()
Ie_np = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
Ie = torch.tensor(Ie_np, dtype=torch.float32)
pts_e = find_dark_points(Ie_np)


def wstats(pccs, meta):
    sg2 = [p for p, m in zip(pccs, meta) if m['sigma'] == 2]
    return float(np.mean([p < 0.5 for p in sg2])), float(np.min(sg2))


def evaluate(combo, theta, scale, tag):
    tr = [lab2k[m] for m in combo]
    Nm = len(combo)
    gs = global_stats_02(modes, Ie, tr, scale)
    C2, (v0, v2) = c_s02(gs, theta, Nm)
    ws56, meta56, mask = build_window_stats(modes, Ie, tr, scale)
    ws32, meta32, _ = build_window_stats(modes, Ie, tr, scale, W=32,
                                         mask=mask)
    p56 = window_pccs(ws56, theta, Nm)
    p32 = window_pccs(ws32, theta, Nm)
    rho56, cmin56 = wstats(p56, meta56)
    rho32, cmin32 = wstats(p32, meta32)
    wavg = w_avg_point(ws56, theta, Nm)
    I = render_model(modes_np, lab2k,
                     {'combo': combo, 'params': list(theta),
                      'scale': scale})
    pts_m = find_dark_points(I)
    mt = match_points(pts_e, pts_m)
    dists = np.array(mt['dists']) if mt['dists'] else np.array([99.0])
    n_ok = abs(len(pts_m) - len(pts_e)) <= 1
    d_ok = mt['mean'] is not None and mt['mean'] < 12
    struct = bool(n_ok and d_ok)
    max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.0
    robust = bool(struct and max10 <= ROBUST_MARGIN)
    I2 = gaussian_blur(torch.tensor(I, dtype=torch.float32)
                       .unsqueeze(0), 2.0)[0].numpy()
    d_ang, _ = angular_distance(Ie_np, I2)
    conc, loc = diff_localization(Ie_np, I2, mask)
    warns = []
    if rho32 >= 0.2:
        warns.append(f'rho_32px={rho32:.2f}')
    if cmin32 <= 0.3:
        warns.append(f'C_min_32px={cmin32:.2f}')
    return {'tag': tag, 'combo': combo, 'N': Nm, 'theta': list(theta),
            'scale': scale, 'C_s02': float(C2), 'v0': float(v0),
            'v2': float(v2), 'W_avg': float(wavg),
            'rho56': rho56, 'cmin56': cmin56,
            'rho32': rho32, 'cmin32': cmin32,
            'cores': len(pts_m), 'cores_exp': len(pts_e),
            'mean_nn': float(mt['mean']), 'max_nn': float(mt['max']),
            'max10': max10, 'dists': [float(x) for x in dists],
            'struct': struct, 'robust': robust, 'warns': warns,
            'd_ang': float(d_ang),
            'diff_conc': (float(conc) if conc is not None else None),
            'params_ok': True}


# ---------------- incumbent ----------------
combo_i, params_i, scale_i = parse_md(
    ROOT / '归档_八图分解' / '最新结果_新指标' / '8奇点(2)_参数与指标.md')
inc = evaluate(combo_i, params_i, scale_i, 'incumbent(2mode)')
print(f'[incumbent] {"+".join(combo_i)} scale={scale_i} '
      f'C_s02={inc["C_s02"]:.4f} W_avg={inc["W_avg"]:.4f} '
      f'cores={inc["cores"]}/{inc["cores_exp"]} mean={inc["mean_nn"]:.2f} '
      f'max10={inc["max10"]:.2f} warns={inc["warns"]}')

# ---------------- candidates ----------------
s7a = json.load(open(OUT / 'stage7a_three_mode.json', encoding='utf-8'))
s7b = json.load(open(OUT / 'stage7b_four_mode.json', encoding='utf-8'))

seen, sels = set(), []
for src_name, j in [('stage7a', s7a), ('stage7b', s7b)]:
    n_used = 0
    for r in j['ranked']:
        key = '+'.join(r['combo'])
        if key in seen or key == '+'.join(combo_i):
            continue
        seen.add(key)
        sels.append((src_name, r))
        n_used += 1
        if n_used >= 10:
            break
print(f'[select] {len(sels)} unique candidate combos to evaluate')

evals = [inc]
for src_name, r in sels:
    e = evaluate(r['combo'], r['theta'], None, f'{src_name}')
    evals.append(e)

print('\n=== unified evaluation ===')
for e in evals:
    print(f"  {e['tag']:16s} {'+'.join(e['combo']):44s} N={e['N']} "
          f"C_s02={e['C_s02']:.4f} W_avg={e['W_avg']:.4f} "
          f"cores={e['cores']}/{e['cores_exp']} mean={e['mean_nn']:.2f} "
          f"max10={e['max10']:.2f} struct={'P' if e['struct'] else 'F'} "
          f"robust={'Y' if e['robust'] else '-'} "
          f"warns={e['warns']}")

passing = [e for e in evals[1:] if e['struct'] and e['robust']]
passing.sort(key=lambda x: (-x['C_s02'], -x['W_avg']))
winner = passing[0] if passing else None

replace = False
reasons = []
if winner is not None:
    c_ok = winner['C_s02'] >= inc['C_s02']
    w_ok = winner['W_avg'] >= inc['W_avg'] - 0.005
    replace = bool(c_ok and w_ok)
    reasons = [
        f"winner C_s02 {winner['C_s02']:.4f} vs incumbent "
        f"{inc['C_s02']:.4f} (need >=): {'OK' if c_ok else 'NG'}",
        f"winner W_avg {winner['W_avg']:.4f} vs incumbent "
        f"{inc['W_avg']:.4f} (need >= -0.005): {'OK' if w_ok else 'NG'}",
        f"winner cores {winner['cores']}/{winner['cores_exp']} "
        f"max10={winner['max10']:.2f} vs incumbent "
        f"max10={inc['max10']:.2f}",
        f"winner warns {winner['warns']} vs incumbent {inc['warns']}"]
else:
    reasons = ['no passing robust 3/4-mode candidate']

chosen = winner if replace else inc
print('\n=== decision ===')
for r in reasons:
    print('  ' + r)
print(f"  => {'REPLACE with ' + '+'.join(winner['combo']) if replace else 'KEEP incumbent'}")

out = {'image': NAME, 'incumbent': inc, 'evaluations': evals,
       'winner': winner, 'replace': replace, 'reasons': reasons,
       'chosen': chosen,
       'protocol': {
           'struct': '|Δcores|<=1 and mean_nn<12px',
           'robust': f'drop single worst core, remaining max <= {ROBUST_MARGIN}px',
           'replace': 'struct & robust pass, C_s02 >= incumbent, '
                      'W_avg >= incumbent - 0.005'}}
(OUT / 'stage7c_decision.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print(f'\n[done] {OUT / "stage7c_decision.json"}')

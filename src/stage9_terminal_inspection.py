# -*- coding: utf-8 -*-
"""
stage9_terminal_inspection.py -- 终检精修 + 重新验收（八图）
=============================================================
对每张图的交付解做带 **全局平移 (dy,dx) + 逐模束腰 s_i** 的约束精修
（探针 B 结论落地）：
  - 坐标下降：theta 抖动块 + 平移细扫（±3px，粗→细 0.25px）+ 逐模腰
    细扫（±0.08，粗 0.02 → 细 0.01），2 轮；
  - 只采纳：结构 PASS 且 max10 ≤ max(14, 原解 max10)，且 C_s02 提升
    ≥ +0.001（W_avg ≥ 原 −0.005）；
  - 指标全程 numeric 复算（含平移/束腰的渲染），验收=新指标全套。
输出：归档_八图分解/终检精修_2026-10-09/
  - 终检精修报告.md / terminal_inspection.json / 终检精修_总览.png
  - {cn}_终检验收.md （每图；改进图另出 {cn}_拟合对比_终检版.png）
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
from scipy.ndimage import shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from window_metrics import make_windows, _support_mask
from structural_check import find_dark_points, match_points
from calibrate_metrics import angular_distance
from verify_metrics import diff_localization
from stage0_diagnosis import parse_md
from search_two_mode_joint import build_joint_modes  # noqa

ARCH = ROOT / '归档_八图分解'
OUT = ARCH / '终检精修_2026-10-10'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

CASES = [
    {'name': '4Singularities', 'cn': '4奇点',
     'src': ('最新结果_新指标', '4奇点_参数与指标.md')},
    {'name': '5Singularities', 'cn': '5奇点',
     'src': ('局部优化_5及8奇点/最终交付_5及8奇点',
             '5奇点_参数与指标.md')},
    {'name': '6Singularities', 'cn': '6奇点',
     'src': ('最新结果_新指标', '6奇点_参数与指标.md')},
    {'name': '8Singularities(1)', 'cn': '8奇点1', 'json': (
        '8奇点1_HG主导搜索', 'stage5g_accept.json', 'theta')},
    {'name': '8Singularities(2)', 'cn': '8奇点2', 'json': (
        '8奇点2_三模四模重分解', 'stage7d_delivery.json',
        'chosen.theta')},
    {'name': '9Singularities', 'cn': '9奇点', 'json': (
        '9奇点_三至五模重分解', 'stage8_delivery.json', 'chosen.theta')},
    {'name': '10Singularities', 'cn': '10奇点', 'json': (
        '10奇点/5_新指标修复与验收', 'stage3_delivery.json', 'theta')},
    {'name': '11Singularities', 'cn': '11奇点', 'json': (
        '11奇点_三至五模重分解', 'stage8_delivery.json',
        'chosen.theta')},
]

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def blur_sig(I, sg):
    if sg == 0:
        return I
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), float(sg))[0].numpy()


def render_sol(combo, theta, base_scale, scales, shiftv):
    N = len(combo)
    E = np.zeros(modes_np[0].shape, dtype=np.complex128)
    for i, m in enumerate(combo):
        F = modes_np[lab2k[m]]
        s = (base_scale or 1.0) * (scales[i] if scales else 1.0)
        if abs(s - 1.0) > 1e-6:
            F = rescale_lg(torch.from_numpy(F).unsqueeze(0),
                           float(s))[0].numpy()
        phi = 0.0 if i == 0 else theta[N + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * F * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    if shiftv is not None and (abs(shiftv[0]) > 1e-6
                               or abs(shiftv[1]) > 1e-6):
        I = ndshift(I, shiftv, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


def _load_theta(case):
    if 'src' in case:
        folder, fname = case['src']
        combo, theta, scale = parse_md(ARCH / folder / fname)
        return combo, theta, scale
    folder, fname, key = case['json']
    d = json.load(open(ARCH / folder / fname, encoding='utf-8'))
    if key == 'chosen.theta':
        combo = d['chosen']['combo']
        theta = [float(x) for x in d['chosen']['theta']]
    else:
        combo = d['combo'] if isinstance(d.get('combo'), list) \
            else d['combo'].split('+')
        theta = [float(x) for x in d[key]]
    return combo, theta, None


results = {}
for case in CASES:
    CN, NAME = case['cn'], case['name']
    t0 = time.time()
    combo, theta0, base_scale = _load_theta(case)
    N = len(combo)
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    N_EXP = len(pts_e)
    mask = _support_mask(Ie)
    windows, n_side = make_windows(Ie.shape[0], 56)
    w32, _ = make_windows(Ie.shape[0], 32)
    Ie_b = {sg: blur_sig(Ie, sg) for sg in (0, 2)}
    print(f'\n=== {CN} ({NAME}) N={N} '
          f'{"+".join(combo)} ===')

    def win_metrics(Im):
        v56 = []
        Im_b = {sg: blur_sig(Im, sg) for sg in (0, 2)}
        for sl in windows:
            if mask[sl].mean() < 0.5:
                continue
            for sg in (0, 2):
                v56.append((sg, pcc(Ie_b[sg][sl], Im_b[sg][sl])))
        s2 = [v for sg, v in v56 if sg == 2]
        rho56 = float(np.mean([v < 0.5 for v in s2]))
        cmin56 = float(np.min(s2))
        wavg = float(np.mean([v for _, v in v56]))
        v32 = []
        for sl in w32:
            if mask[sl].mean() < 0.5:
                continue
            for sg in (0, 2):
                v32.append((sg, pcc(Ie_b[sg][sl], Im_b[sg][sl])))
        s32 = [v for sg, v in v32 if sg == 2]
        rho32 = float(np.mean([v < 0.5 for v in s32]))
        cmin32 = float(np.min(s32))
        return wavg, rho56, cmin56, rho32, cmin32

    def ev(theta, scales, shiftv):
        Im = render_sol(combo, theta, base_scale, scales, shiftv)
        C2 = 0.5 * (pcc(Ie, Im) + pcc(Ie_b[2], blur_sig(Im, 2)))
        wavg, rho56, cmin56, rho32, cmin32 = win_metrics(Im)
        pts_m = find_dark_points(Im)
        mt = match_points(pts_e, pts_m)
        dists = np.array(mt['dists']) if mt['dists'] else np.array([99.])
        struct = (abs(len(pts_m) - N_EXP) <= 1
                  and mt['mean'] is not None and mt['mean'] < 12)
        max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
        return {'C_s02': C2, 'W_avg': wavg, 'rho56': rho56,
                'cmin56': cmin56, 'rho32': rho32, 'cmin32': cmin32,
                'theta': list(theta),
                'scales': list(scales) if scales else None,
                'shift': list(shiftv) if shiftv else None,
                'cores': len(pts_m), 'mean_nn': float(mt['mean'])
                if mt['mean'] is not None else 99.,
                'max10': max10, 'struct': struct,
                'dists': [float(x) for x in dists]}

    base = ev(theta0, None, None)
    print(f'  baseline: C_s02={base["C_s02"]:.4f} W_avg={base["W_avg"]:.4f} '
          f'cores={base["cores"]}/{N_EXP} max10={base["max10"]:.2f} '
          f'struct={base["struct"]}')
    margin = max(14.0, base['max10'])

    def valid(r):
        return r['struct'] and r['max10'] <= margin

    best = base
    cands = [('base', base)]
    rng = np.random.default_rng(20261010)
    cur = base
    cur_shift = [0.0, 0.0]
    cur_scales = [1.0] * N
    for rnd in range(2):
        # theta jitter block
        for k in range(150):
            u = np.array(cur['theta'][:N]) + rng.normal(0, 0.012, N)
            u = np.clip(u, 0.005, None)
            u = u / u.sum()
            ph = np.array(cur['theta'][N:]) + rng.normal(0, 0.04, N - 1)
            th = list(u) + list(ph)
            r = ev(th, cur_scales, cur_shift)
            cands.append((f'r{rnd}/theta{k}', r))
            if valid(r) and r['C_s02'] > best['C_s02']:
                best = r
        cur = best
        cur_shift = list(best['shift']) if best['shift'] else [0., 0.]
        cur_scales = list(best['scales']) if best['scales'] \
            else [1.0] * N
        # shift coarse -> fine
        for step, rad in [(1.0, 3.0), (0.25, 0.75)]:
            g = np.arange(-rad, rad + 1e-9, step)
            for dy in g:
                for dx in g:
                    r = ev(cur['theta'], cur_scales,
                           [cur_shift[0] + dy, cur_shift[1] + dx])
                    if valid(r) and r['C_s02'] > best['C_s02']:
                        best = r
            cur = best
            cur_shift = list(best['shift']) if best['shift'] \
                else [0., 0.]
        # per-mode scales coarse -> fine
        for step, rad in [(0.02, 0.08), (0.01, 0.03)]:
            for i in range(N):
                g = np.arange(-rad, rad + 1e-9, step)
                for s in g:
                    sc = list(cur_scales)
                    sc[i] = float(np.clip(sc[i] + s, 0.85, 1.15))
                    r = ev(cur['theta'], sc, cur_shift)
                    if valid(r) and r['C_s02'] > best['C_s02']:
                        best = r
                cur = best
                cur_scales = list(best['scales']) if best['scales'] \
                    else [1.0] * N
        print(f'  [round{rnd}] best C_s02={best["C_s02"]:.4f} '
              f'W_avg={best["W_avg"]:.4f} max10={best["max10"]:.2f} '
              f'shift={np.round(cur_shift,2).tolist()} '
              f'scales={np.round(cur_scales,3).tolist()} '
              f'({time.time()-t0:.0f}s)')

    improved = (best is not base
                and best['C_s02'] >= base['C_s02'] + 0.001
                and best['W_avg'] >= base['W_avg'] - 0.005)
    rec = {'name': NAME, 'cn': CN, 'combo': combo, 'N': N,
           'base_scale': base_scale, 'baseline': base, 'refined': best,
           'improved': bool(improved), 'margin': margin,
           'accepted': best if improved else base}
    results[CN] = rec
    print(f'  => {"IMPROVED" if improved else "no change"} '
          f'(C_s02 {base["C_s02"]:.4f} -> {best["C_s02"]:.4f}, '
          f'{time.time()-t0:.0f}s)')

(OUT / 'terminal_inspection.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# ---- summary figure ----
names = list(results.keys())
before = [results[n]['baseline']['C_s02'] for n in names]
after = [results[n]['accepted']['C_s02'] for n in names]
fig, ax = plt.subplots(figsize=(11, 4.8))
x = np.arange(len(names))
ax.bar(x - 0.2, before, width=0.4, label='before (delivered)',
       color='#999')
ax.bar(x + 0.2, after, width=0.4, label='after (terminal inspection)',
       color='#2a7')
for i in range(len(names)):
    d = after[i] - before[i]
    ax.text(i, max(before[i], after[i]) + 0.004,
            f'{d:+.4f}' if abs(d) > 1e-9 else '=', ha='center',
            fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0.75, 1.0)
ax.set_ylabel('C_s02')
ax.legend(fontsize=9)
ax.set_title('Terminal inspection: C_s02 before vs after '
             '(translation + per-mode waist refinement)')
fig.tight_layout()
fig.savefig(OUT / '终检精修_总览.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'\n[done] {OUT}')

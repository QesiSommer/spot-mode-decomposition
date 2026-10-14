# -*- coding: utf-8 -*-
"""
stage16_v4_terminal.py -- 终检 v4：修正管线（LG 排除各向异性/旋转）
======================================================================
问题（2026-10-14 诊断）：v2 引入的逐模各向异性 sx/sy 对 **LG 模是
伪物理** —— LG(p,0) 加 sx≠sy 后不再是 LG 模，圆形节线被拉成椭圆，
会在中央撕出额外暗点。4 奇点因此从 4 核变 5 核（落在 ±1 容差内被
放过，但视觉上"奇点个数没对上"）。

修正内容（与 HG-only 旋转同构的处理）：
  1. **LG 模排除各向异性**：sx/sy 仅对 HG 模开放（LG 保持 1.0）；
  2. **LG 模排除旋转**：rot 仅对 HG 模开放（v3 已如此）；
  3. **核数精确锁定**：结构约束从 |Δ核数| ≤ 1 收紧为
     **Δ核数 == 0**（核数必须与实验完全一致）；
  4. 只采纳：核数精确 + mean_nn < 12px + max10 ≤ max(14, 基线)
     且 C_s02 ≥ 基线。

种子：v3 的 accepted 解（theta/平移/sx/sy/rot 全部保留为初值，
但 LG 的 sx/sy 强制回 1.0）。逐图独立精修，不做跨图耦合。

Outputs: 归档_八图分解/终检精修v4_2026-10-14/
  - terminal_inspection_v4.json（全记录）
  - {cn}/（交付四件套）
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
from structural_check import find_dark_points, match_points
import morph_search_lib as L

ARCH = ROOT / '归档_八图分解'
OUT = ARCH / '终检精修v4_2026-10-14'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V3 = json.load(open(ARCH / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
mnp = modes.numpy()

ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
         '10奇点', '11奇点']


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def make_eval_exact(Ie, pts_e, combo):
    """评估器：核数**精确**匹配 + mean_nn<12 + max10 约束。"""
    Ie_b2 = L.blur2(Ie)
    N_EXP = len(pts_e)
    N = len(combo)

    def ev(theta, sx, sy, shiftv, rot):
        Im = L.render_dof(mnp, lab2k, combo, theta, sx, sy, shiftv, rot)
        v0 = pcc(Ie, Im)
        v2 = pcc(Ie_b2, L.blur2(Im))
        C2 = 0.5 * (v0 + v2)
        pts_m = find_dark_points(Im)
        mt = match_points(pts_e, pts_m)
        dists = np.array(mt['dists']) if mt['dists'] else np.array([99.])
        # 精确核数锁定
        struct = (len(pts_m) == N_EXP
                  and mt['mean'] is not None and mt['mean'] < 12)
        max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
        return {'C_s02': float(C2), 'v0': float(v0), 'v2': float(v2),
                'struct': bool(struct), 'cores': len(pts_m),
                'cores_exp': N_EXP,
                'mean_nn': float(mt['mean'])
                if mt['mean'] is not None else 99.0,
                'max10': max10, 'theta': list(theta), 'sx': list(sx),
                'sy': list(sy), 'shift': list(shiftv),
                'rot': list(rot), '_I': Im}
    return ev


def jitter(theta, rng, N, s_en=0.008, s_ph=0.03):
    u = np.array(theta[:N]) + rng.normal(0, s_en, N)
    u = np.clip(u, 0.005, None)
    u = u / u.sum()
    ph = np.array(theta[N:]) + rng.normal(0, s_ph, N - 1)
    return list(u) + list(ph)


def refine_v4(ev, combo, theta0, sx0, sy0, shift0, rot0,
              rounds=2, budget=140, seed=0):
    """v4 精修：LG 的 sx/sy/rot 冻结在 1.0/0.0，仅 HG 开放。"""
    N = len(combo)
    is_hg = [m.startswith('HG') for m in combo]
    rng = np.random.default_rng(1600 + seed)
    base = ev(theta0, sx0, sy0, shift0, rot0)
    margin = max(14.0, base['max10'])
    best = base
    cur = base
    for rnd in range(rounds):
        for _ in range(budget):
            th = jitter(cur['theta'], rng, N)
            r = ev(th, cur['sx'], cur['sy'], cur['shift'], cur['rot'])
            if r['struct'] and r['max10'] <= margin \
                    and r['C_s02'] > best['C_s02']:
                best = r
                cur = r
        # 平移
        for step, rad in [(1.0, 2.0), (0.25, 0.5)]:
            for dy in np.arange(-rad, rad + 1e-9, step):
                for dx in np.arange(-rad, rad + 1e-9, step):
                    sh = [cur['shift'][0] + dy, cur['shift'][1] + dx]
                    r = ev(cur['theta'], cur['sx'], cur['sy'], sh,
                           cur['rot'])
                    if r['struct'] and r['max10'] <= margin \
                            and r['C_s02'] > best['C_s02']:
                        best = r
                        cur = r
        # 逐模各向异性：仅 HG
        for step, rad in [(0.02, 0.06), (0.01, 0.02)]:
            for i in range(N):
                if not is_hg[i]:
                    continue
                for ax in (0, 1):
                    for d in np.arange(-rad, rad + 1e-9, step):
                        sxv = list(cur['sx'])
                        syv = list(cur['sy'])
                        tgt = sxv if ax == 0 else syv
                        tgt[i] = float(np.clip(tgt[i] + d, 0.7, 1.4))
                        r = ev(cur['theta'], sxv, syv, cur['shift'],
                               cur['rot'])
                        if r['struct'] and r['max10'] <= margin \
                                and r['C_s02'] > best['C_s02']:
                            best = r
                            cur = r
        # 逐模旋转：仅 HG
        for step, rad in [(1.0, 5.0), (0.25, 1.0)]:
            for i in range(N):
                if not is_hg[i]:
                    continue
                for d in np.arange(-rad, rad + 1e-9, step):
                    rv = list(cur['rot'])
                    rv[i] = float(np.clip(rv[i] + d, -20.0, 20.0))
                    r = ev(cur['theta'], cur['sx'], cur['sy'],
                           cur['shift'], rv)
                    if r['struct'] and r['max10'] <= margin \
                            and r['C_s02'] > best['C_s02']:
                        best = r
                        cur = r
    final = ev(best['theta'], best['sx'], best['sy'], best['shift'],
               best['rot'])
    final['start_C_s02'] = base['C_s02']
    final['start_cores'] = base['cores']
    return final


records = {}
for cn in ORDER:
    r3 = V3[cn]
    NAME = r3['name']
    combo = r3['combo']
    N = len(combo)
    acc = r3['accepted']
    theta0 = [float(x) for x in acc['theta']]
    # 修正：LG 的 sx/sy 强制 1.0
    is_hg = [m.startswith('HG') for m in combo]
    sx0 = [float(x) if is_hg[i] else 1.0
           for i, x in enumerate(acc['sx'])]
    sy0 = [float(x) if is_hg[i] else 1.0
           for i, x in enumerate(acc['sy'])]
    shift0 = [float(x) for x in acc['shift']]
    rot0 = [float(x) if is_hg[i] else 0.0
            for i, x in enumerate(acc.get('rot', [0.0] * N))]

    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    N_EXP = len(pts_e)
    ev = make_eval_exact(Ie, pts_e, combo)

    print(f'\n{"="*72}\n=== {cn} ({NAME}) N={N} 实验核数={N_EXP} ===')
    print(f'  combo: {"+".join(combo)}')
    print(f'  LG-freeze: ' + ', '.join(
        f'{combo[i]}={"HG" if is_hg[i] else "LG(冻结)"}'
        for i in range(N)))
    t0 = time.time()
    base = ev(theta0, sx0, sy0, shift0, rot0)
    print(f'  [seed=v3修正] C_s02={base["C_s02"]:.4f} '
          f'cores={base["cores"]}/{N_EXP} '
          f'mean_nn={base["mean_nn"]:.2f} struct={base["struct"]}')

    best = refine_v4(ev, combo, theta0, sx0, sy0, shift0, rot0,
                     rounds=2, budget=140, seed=len(cn))
    print(f'  [v4] C_s02={best["C_s02"]:.4f} '
          f'({best["C_s02"]-base["C_s02"]:+.4f}) '
          f'cores={best["cores"]}/{N_EXP} '
          f'mean_nn={best["mean_nn"]:.2f} max10={best["max10"]:.2f} '
          f'struct={best["struct"]} ({time.time()-t0:.0f}s)')

    records[cn] = {
        'name': NAME, 'cn': cn, 'combo': combo, 'N': N,
        'n_exp': N_EXP,
        'seed': {k: v for k, v in base.items() if k != '_I'},
        'refined': {k: v for k, v in best.items() if k != '_I'},
        'channel_policy': {'sx_sy': 'HG-only', 'rot': 'HG-only',
                           'core_lock': 'exact equality'}}

(OUT / 'terminal_inspection_v4.json').write_text(
    json.dumps(records, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

print('\n' + '=' * 72)
print('=== v4 SUMMARY (核数精确锁定) ===')
print(f'{"图":8s} {"C_s02 v3→v4":>20s} {"核数 v3→v4":>14s} {"结构":>6s}')
for cn in ORDER:
    r = records[cn]
    s, f = r['seed'], r['refined']
    print(f"{cn:8s} {s['C_s02']:.4f} → {f['C_s02']:.4f} "
          f"({f['C_s02']-s['C_s02']:+.4f})  "
          f"{s['cores']} → {f['cores']} (exp {r['n_exp']})  "
          f"{'PASS' if f['struct'] else 'FAIL'}")
print(f'\n[done] {OUT}')

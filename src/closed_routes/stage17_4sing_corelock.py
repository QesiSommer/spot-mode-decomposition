# -*- coding: utf-8 -*-
"""
stage17_4sing_corelock.py -- 4奇点：LG 冻结 + 核数精确锁定 的定向搜索
========================================================================
背景：v3 的"5 核高分解"（C_s02 0.9163）靠的是 LG 各向异性伪物理；
v4 把 LG 冻结后核数修回 4vs4 但 C_s02 掉到 0.8872。

本脚本为 4 奇点找**同时满足**：
  - LG 模 sx=sy=1、rot=0（无伪物理）
  - 核数精确 = 4（与实验一致）
  - C_s02 尽可能高
的解。候选池：
  1. v3 现解组合 LG(0,-4)+LG(2,0)+HG(6,4)（v4 起点）
  2. 形态搜索冠军 LG(0,1)+LG(1,1)+LG(0,5)
  3. Sep-26 参数
  4. 三模全空间粗筛 top-30（用 LG 冻结管线评估）
出口：若找到 4vs4 且 C_s02 > 0.90 的解 → 交付；否则如实报告上限。
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
from stage0_diagnosis import parse_md
import morph_search_lib as L

OUT = ROOT / '归档_八图分解' / '4奇点_核数修正'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
mnp = modes.numpy()
NAME = '4Singularities'
CN = '4奇点'

Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
pts_e = find_dark_points(Ie)
N_EXP = len(pts_e)
print(f'[setup] 实验核数={N_EXP}')


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def make_eval(combo):
    Ie_b2 = L.blur2(Ie)
    N = len(combo)

    def ev(theta, sx, sy, shiftv, rot):
        Im = L.render_dof(mnp, lab2k, combo, theta, sx, sy, shiftv, rot)
        v0 = pcc(Ie, Im)
        v2 = pcc(Ie_b2, L.blur2(Im))
        C2 = 0.5 * (v0 + v2)
        pts_m = find_dark_points(Im)
        mt = match_points(pts_e, pts_m)
        dists = np.array(mt['dists']) if mt['dists'] else np.array([99.])
        struct = (len(pts_m) == N_EXP
                  and mt['mean'] is not None and mt['mean'] < 12)
        max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
        return {'C_s02': float(C2), 'v0': float(v0), 'v2': float(v2),
                'struct': bool(struct), 'cores': len(pts_m),
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


def refine(ev, combo, theta0, sx0, sy0, shift0, rot0, rounds=2,
           budget=160, seed=0):
    N = len(combo)
    is_hg = [m.startswith('HG') for m in combo]
    rng = np.random.default_rng(1700 + seed)
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
    return ev(best['theta'], best['sx'], best['sy'], best['shift'],
              best['rot'])


# ---- 候选池：三模全空间粗筛（LG 冻结语义下评估）----
t0 = time.time()
combos3 = torch.combinations(torch.arange(len(labels)), r=3)
sc = L.screen_pool(mnp, Ie, combos3, 3, torch.device('cpu'))
print(f'[screen] 三模全空间 {combos3.shape[0]} 组合粗筛完成 '
      f'({time.time()-t0:.0f}s)')
top = np.argsort(-sc['corr'])[:30]

cands = []
seen = set()
# injected
for tag, combo, theta in [
        ('v3_现解', V3[CN]['combo'],
         [float(x) for x in V3[CN]['accepted']['theta']]),
        ('形态冠军', ['LG(0,1)', 'LG(1,1)', 'LG(0,5)'], None),
        ('Sep26', None, None)]:
    if combo is None:
        combo, theta, _ = parse_md(ROOT / '归档_八图分解' / '最新结果'
                                   / f'{CN}_参数与指标.md')
    tr = tuple(sorted(lab2k[m] for m in combo))
    if tr in seen:
        continue
    seen.add(tr)
    if theta is None:
        theta = [1 / 3, 1 / 3, 1 / 3, 1.0, 1.0]
    cands.append({'combo': list(combo), 'theta': list(theta),
                  'tag': tag})
# screened
for k in top:
    tr = tuple(sorted(combos3[k].tolist()))
    if tr in seen:
        continue
    seen.add(tr)
    cands.append({'combo': [labels[t] for t in tr],
                  'theta': sc['params'][k].tolist(), 'tag': 'screen'})
print(f'[refine] {len(cands)} 候选（LG 冻结 + 核数精确锁定）')

records = []
t0 = time.time()
for ci, c in enumerate(cands):
    N = len(c['combo'])
    ev = make_eval(c['combo'])
    r = refine(ev, c['combo'], c['theta'], [1.0] * N, [1.0] * N,
               [0.0, 0.0], [0.0] * N, rounds=2, budget=160, seed=ci)
    r['combo'] = c['combo']
    r['tag'] = c['tag']
    records.append(r)
    if ci % 5 == 4 or r['struct']:
        print(f'  [{ci+1}/{len(cands)}] {"+".join(c["combo"]):44s} '
              f'[{c["tag"]:8s}] C_s02={r["C_s02"]:.4f} '
              f'cores={r["cores"]}/{N_EXP} '
              f'mean_nn={r["mean_nn"]:.2f} '
              f'{"PASS" if r["struct"] else ""} '
              f'({time.time()-t0:.0f}s)')

# 排序：结构 PASS 优先 → C_s02
records.sort(key=lambda r: (0 if r['struct'] else 1, -r['C_s02']))
print('\n=== 排名（核数精确 4vs4 优先）===')
for k, r in enumerate(records[:10]):
    print(f'  {k+1:2d}.{"★" if r["struct"] else " "} '
          f'{"+".join(r["combo"]):44s} C_s02={r["C_s02"]:.4f} '
          f'cores={r["cores"]}/{N_EXP} mean_nn={r["mean_nn"]:.2f} '
          f'max10={r["max10"]:.2f}')

usable = [r for r in records if r['struct']]
champ = usable[0] if usable else None

(OUT / 'stage17_corelock.json').write_text(
    json.dumps({'n_exp': N_EXP,
                'v3_reference': {'combo': V3[CN]['combo'],
                                 'C_s02': 0.9163, 'cores': 5},
                'v4_reference': {'C_s02': 0.8872, 'cores': 4},
                'n_usable': len(usable),
                'champion': ({k: v for k, v in champ.items()
                              if k != '_I'} if champ else None),
                'records': [{k: v for k, v in r.items() if k != '_I'}
                            for r in records]},
               ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print(f'\n[champion] ' + (f'{"+".join(champ["combo"])} '
      f'C_s02={champ["C_s02"]:.4f} cores={champ["cores"]}/{N_EXP}'
      if champ else 'NONE'))
print(f'[done] {OUT}')

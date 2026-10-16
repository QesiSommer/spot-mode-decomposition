# -*- coding: utf-8 -*-
"""
check_shijie_20261016.py -- 师姐复核触发的彻底独立检查
==========================================================
背景：师姐用我们公布的分解参数独立叠图（8奇点2），并与她自己
分解的 6 奇点结果对比，发现差异。本脚本不受既有结论约束，做三层
独立核查：

  A. 复现性核查（全部八图）：只用公布的（模式/能量/相位）+ 单一
     全局束腰尺度 s + 全局平移——即"第三方按参数叠图"的口径——
     与实验图和 v3 交付图对比，量化差距；8(2) 与 6 奇点额外做
     theta 重调（纯物理口径下的最优）；
  B. 6 奇点从零纯物理重搜：多尺度两模全空间 C(131,2)=8515 +
     纯物理精修——检验 HG(0,3)+HG(3,1) 是否仍是纯物理最优；
  C. 一致性审计：图名奇点数 vs 我们检测的暗点数；v3 逐模变换
     （sx/sy/rot）在同图内的离散度（同一腔内模式应共享坐标系）。

输出：归档_八图分解/师姐复核检查_2026-10-16/
  - 检查报告.md
  - 复现性_8奇点2.png / 复现性_6奇点.png
  - 6奇点_纯物理重搜.json / 复现性_全八图.json
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
sys.path.insert(0, str(ROOT / 'closed_routes'))  # morph_search_lib 所在目录
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
import morph_search_lib as L

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                         'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = ROOT / '归档_八图分解' / '师姐复核检查_2026-10-16'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
mnp = modes.numpy()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
         '10奇点', '11奇点']
LABEL_N = {'4奇点': 4, '5奇点': 5, '6奇点': 6, '8奇点1': 8,
           '8奇点2': 8, '9奇点': 9, '10奇点': 10, '11奇点': 11}


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def blur2(I):
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy()


def render_pure(combo, theta, s, shiftv):
    """纯物理叠图：公布的模式/能量/相位 + 单一全局尺度 s + 平移。"""
    idx = [lab2k[m] for m in combo]
    ms = mnp[idx]
    if abs(s - 1.0) > 1e-6:
        ms = rescale_lg(torch.from_numpy(ms), float(s)).numpy()
    N = len(combo)
    E = np.zeros((224, 224), dtype=np.complex128)
    for i in range(N):
        phi = 0.0 if i == 0 else theta[N + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * ms[i] * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    if abs(shiftv[0]) > 1e-9 or abs(shiftv[1]) > 1e-9:
        I = ndshift(I, shiftv, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


def C_only(Ie, Ie_b, I):
    return 0.5 * (pcc(Ie, I) + pcc(Ie_b, blur2(I)))


def metrics(Ie, Ie_b, pts_e, I):
    v0 = pcc(Ie, I)
    v2 = pcc(Ie_b, blur2(I))
    pts = find_dark_points(I)
    mt = match_points(pts_e, pts)
    return {'C': 0.5 * (v0 + v2), 'v0': v0, 'v2': v2,
            'cores': len(pts),
            'mean_nn': float(mt['mean']) if mt['mean'] is not None
            else 99.0,
            'max_nn': float(mt['max']) if mt['max'] is not None
            else 99.0}


def refine_pure(Ie, Ie_b, combo, theta0, s0, sh0, rounds=2, budget=150,
                seed=0):
    """纯物理口径下重调 theta + 全局 s + 平移（坐标下降）。"""
    N = len(combo)
    rng = np.random.default_rng(seed)

    def ev(th, s, sh):
        I = render_pure(combo, th, s, sh)
        return C_only(Ie, Ie_b, I)

    cur_th, cur_s, cur_sh = list(theta0), float(s0), tuple(sh0)
    best = {'theta': list(cur_th), 's': cur_s, 'shift': list(cur_sh),
            'C': ev(cur_th, cur_s, cur_sh)}
    for rd in range(rounds):
        for _ in range(budget):
            u = np.array(cur_th[:N]) + rng.normal(0, 0.012, N)
            u = np.clip(u, 0.005, None)
            u = u / u.sum()
            ph = np.array(cur_th[N:]) + rng.normal(0, 0.05, N - 1)
            th = list(u) + list(ph)
            C = ev(th, cur_s, cur_sh)
            if C > best['C']:
                best = {'theta': th, 's': cur_s, 'shift': list(cur_sh),
                        'C': C}
                cur_th = list(th)
        cur_th = list(best['theta'])
        cur_s = best['s']
        cur_sh = tuple(best['shift'])
        for j in range(N - 1):
            for d in np.arange(-0.4, 0.41, 0.1):
                th = list(cur_th)
                th[N + j] += d
                C = ev(th, cur_s, cur_sh)
                if C > best['C']:
                    best = {'theta': th, 's': cur_s,
                            'shift': list(cur_sh), 'C': C}
                    cur_th = list(th)
        for d in np.arange(-0.06, 0.061, 0.015):
            C = ev(cur_th, cur_s + d, cur_sh)
            if C > best['C']:
                best = {'theta': list(cur_th), 's': cur_s + d,
                        'shift': list(cur_sh), 'C': C}
                cur_s = best['s']
        for ax in (0, 1):
            for d in (-1.0, -0.5, 0.5, 1.0):
                sh = list(cur_sh)
                sh[ax] += d
                C = ev(cur_th, cur_s, sh)
                if C > best['C']:
                    best = {'theta': list(cur_th), 's': cur_s,
                            'shift': sh, 'C': C}
                    cur_sh = tuple(sh)
    return best


# ============================================================
# Part A：复现性核查（全八图 quick grid；8(2)/6 加 theta 重调）
# ============================================================
print('==== Part A: pure vs v3 (all 8 images) ====')
repA = {}
deep = {}
for cn in ORDER:
    r = V3[cn]
    acc = r['accepted']
    combo = r['combo']
    N = r['N']
    theta = [float(x) for x in acc['theta']]
    NAME = r['name']
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    Ie_b = blur2(Ie)
    t0 = time.time()

    # v3 交付渲染
    Iv3 = L.render_dof(mnp, lab2k, combo, theta,
                       [float(x) for x in acc['sx']],
                       [float(x) for x in acc['sy']],
                       [float(x) for x in acc['shift']],
                       [float(x) for x in acc.get('rot',
                                                  [0.0] * N)])
    m_v3 = metrics(Ie, Ie_b, pts_e, Iv3)

    # 纯物理 grid（公布 theta，仅调 s 与平移）
    best = None
    for s in np.arange(0.85, 1.301, 0.025):
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                I = render_pure(combo, theta, s, (dy, dx))
                C = C_only(Ie, Ie_b, I)
                if best is None or C > best['C']:
                    best = {'C': C, 's': float(s), 'shift': [dy, dx]}
    Ipu = render_pure(combo, theta, best['s'], best['shift'])
    m_pu = metrics(Ie, Ie_b, pts_e, Ipu)

    repA[cn] = {'name': NAME, 'combo': combo,
                'label_n': LABEL_N[cn], 'detected_cores': len(pts_e),
                'v3': m_v3,
                'pure_pub': {**m_pu, 's': best['s'],
                             'shift': best['shift']},
                'gap': m_v3['C'] - m_pu['C']}
    print(f'  {cn:8s} v3={m_v3["C"]:.4f} pure={m_pu["C"]:.4f} '
          f'gap={m_v3["C"]-m_pu["C"]:+.4f} '
          f'cores v3/pure/exp={m_v3["cores"]}/{m_pu["cores"]}/'
          f'{len(pts_e)} ({time.time()-t0:.0f}s)')

    if cn in ('8奇点2', '6奇点'):
        rb = refine_pure(Ie, Ie_b, combo, theta, best['s'],
                         best['shift'], rounds=2, budget=150,
                         seed=hash(cn) % 10000)
        Irf = render_pure(combo, rb['theta'], rb['s'], rb['shift'])
        m_rf = metrics(Ie, Ie_b, pts_e, Irf)
        deep[cn] = {'refined': {**m_rf, 's': rb['s'],
                                'shift': rb['shift'],
                                'theta': rb['theta']}}
        print(f'           pure+refine={m_rf["C"]:.4f} '
              f'cores={m_rf["cores"]}/{len(pts_e)}')

(OUT / '复现性_全八图.json').write_text(
    json.dumps({'A': repA, 'deep': deep}, ensure_ascii=False,
               indent=2, default=float), encoding='utf-8')

# ---- 复现性对比图（8(2) 与 6）----
for cn in ('8奇点2', '6奇点'):
    r = V3[cn]
    acc = r['accepted']
    combo = r['combo']
    N = r['N']
    theta = [float(x) for x in acc['theta']]
    NAME = r['name']
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    Ie_b = blur2(Ie)
    Iv3 = L.render_dof(mnp, lab2k, combo, theta,
                       [float(x) for x in acc['sx']],
                       [float(x) for x in acc['sy']],
                       [float(x) for x in acc['shift']],
                       [float(x) for x in acc.get('rot', [0.0] * N)])
    pp = repA[cn]['pure_pub']
    Ipu = render_pure(combo, theta, pp['s'], pp['shift'])
    dr = deep[cn]['refined']
    Irf = render_pure(combo, dr['theta'], dr['s'], dr['shift'])

    fig, axs = plt.subplots(1, 4, figsize=(19, 5.4))
    panels = [('实验图', Ie, len(pts_e)),
              ('纯物理叠图\n(公布参数+全局尺度)', Ipu,
               pp['cores']),
              ('纯物理重调 θ\n(同组合纯口径最优)', Irf,
               dr['cores']),
              ('v3 交付拟合图\n(含逐模拉伸/旋转)', Iv3,
               repA[cn]['v3']['cores'])]
    for k, (tag, I, nc) in enumerate(panels):
        axs[k].imshow(blur2(I), cmap='inferno', vmin=0, vmax=1)
        for y, x, _ in find_dark_points(I):
            axs[k].plot(x, y, 'o', mfc='none', mec='cyan', ms=9)
        axs[k].set_title(f'{tag}\ncores={nc} '
                         f'(exp={len(pts_e)})', fontsize=10)
        axs[k].axis('off')
    fig.suptitle(f'{NAME}（{cn}）：第三方纯物理叠图 vs 交付图 —— '
                 f'C_s02: 纯物理 {pp["C"]:.4f} / 纯物理重调 '
                 f'{dr["C"]:.4f} / v3 交付 '
                 f'{repA[cn]["v3"]["C"]:.4f}', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f'复现性_{cn}.png', dpi=125,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig] 复现性_{cn}.png')

# ============================================================
# Part B：6 奇点从零纯物理两模重搜（多尺度全空间）
# ============================================================
print('\n==== Part B: 6Sing from-scratch pure 2-mode search ====')
cn = '6奇点'
NAME = V3[cn]['name']
Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
pts_e = find_dark_points(Ie)
Ie_b = blur2(Ie)
Ie_t = torch.tensor(Ie, dtype=torch.float32)

scales = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
pairs = torch.combinations(torch.arange(len(labels)), r=2)
pmat = L.coarse_grid_params(2, 16, 120)
ct = c_tab = L.c_table_N(pmat, 2).to(dev)

Ie_small = torch.nn.functional.interpolate(
    Ie_t[None, None], size=(64, 64), mode='bilinear',
    align_corners=False)[0, 0].to(dev)

cands = []
t0 = time.time()
for s in scales:
    ms = rescale_lg(torch.from_numpy(mnp), float(s))
    ms_small = torch.stack(
        [L.down_complex(m, 64) for m in ms]).to(dev)
    bc, bi = L._coarse_scan(ms_small, Ie_small, pairs, ct, 2)
    vals, idxs = bc.topk(15)
    for v, k in zip(vals.tolist(), idxs.tolist()):
        cands.append({'pair': pairs[k].tolist(),
                      'theta': pmat[bi[k]].tolist(), 's': s,
                      'corr': v})
    print(f'  scale={s:.2f} best={float(bc.max()):.4f} '
          f'({time.time()-t0:.0f}s)')

# 去重（同 pair 保最优）
best_by_pair = {}
for c in cands:
    k = tuple(sorted(c['pair']))
    if k not in best_by_pair or c['corr'] > best_by_pair[k]['corr']:
        best_by_pair[k] = c
pool = sorted(best_by_pair.values(), key=lambda c: -c['corr'])[:30]
# 注入 v3 组合
v3pair = sorted(lab2k[m] for m in V3[cn]['combo'])
pool.append({'pair': v3pair, 'theta': [float(x) for x in
                                       V3[cn]['accepted']['theta']],
             's': 1.0, 'corr': None, 'tag': 'v3combo'})
print(f'  refine pool: {len(pool)}')

records = []
for ci, c in enumerate(pool):
    combo = [labels[t] for t in c['pair']]
    rb = refine_pure(Ie, Ie_b, combo, c['theta'], c['s'], (0, 0),
                     rounds=2, budget=120, seed=ci)
    I = render_pure(combo, rb['theta'], rb['s'], rb['shift'])
    m = metrics(Ie, Ie_b, pts_e, I)
    records.append({'combo': combo, 'theta': rb['theta'],
                    's': rb['s'], 'shift': rb['shift'],
                    'C': m['C'], 'cores': m['cores'],
                    'mean_nn': m['mean_nn'],
                    'struct': bool(abs(m['cores'] - len(pts_e)) <= 1
                                   and m['mean_nn'] < 12),
                    'tag': c.get('tag', 'search')})
    if ci % 6 == 5:
        print(f'  [{ci+1}/{len(pool)}] best so far '
              f'{max(r["C"] for r in records):.4f}')

records.sort(key=lambda r: (0 if r['struct'] else 1, -r['C']))
print('\n  top-8 (struct-first, pure physical):')
for k, rr in enumerate(records[:8]):
    print(f'   {k+1}. {"+".join(rr["combo"]):34s} C={rr["C"]:.4f} '
          f'cores={rr["cores"]}/{len(pts_e)} '
          f'mean_nn={rr["mean_nn"]:.1f} '
          f'[{rr.get("tag","")}]')

(OUT / '6奇点_纯物理重搜.json').write_text(
    json.dumps({'records': records,
                'v3_ref': {'combo': V3[cn]['combo'],
                           'C': repA[cn]['v3']['C']}},
               ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# 搜索前三对比图
fig, axs = plt.subplots(1, 4, figsize=(19, 5.2))
axs[0].imshow(blur2(Ie), cmap='inferno', vmin=0, vmax=1)
axs[0].set_title(f'EXPERIMENT ({len(pts_e)} cores)')
for k, rr in enumerate(records[:3]):
    I = render_pure(rr['combo'], rr['theta'], rr['s'], rr['shift'])
    axs[k + 1].imshow(blur2(I), cmap='inferno', vmin=0, vmax=1)
    axs[k + 1].set_title(f'pure #{k+1}: {"+".join(rr["combo"])}\n'
                         f'C={rr["C"]:.4f} '
                         f'cores={rr["cores"]}/{len(pts_e)} '
                         f'nn={rr["mean_nn"]:.1f}px', fontsize=9)
for a in axs:
    a.axis('off')
fig.suptitle(f'{NAME}：纯物理口径从零两模搜索 top-3（vs v3 交付 '
             f'C={repA[cn]["v3"]["C"]:.4f}）', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / '6奇点_纯物理重搜_top3.png', dpi=125,
            bbox_inches='tight')
plt.close(fig)

# ============================================================
# Part C：一致性审计
# ============================================================
print('\n==== Part C: consistency audit ====')
audit = []
for cn in ORDER:
    r = V3[cn]
    acc = r['accepted']
    sx = [float(x) for x in acc['sx']]
    sy = [float(x) for x in acc['sy']]
    rot = [float(x) for x in acc.get('rot', [0.0] * r['N'])]
    hg_rot = [rot[i] for i, m in enumerate(r['combo'])
              if m.startswith('HG')]
    audit.append({
        'cn': cn, 'label_n': LABEL_N[cn],
        'detected': repA[cn]['detected_cores'],
        'label_eq_detected': LABEL_N[cn] == repA[cn]['detected_cores'],
        'sx': sx, 'sy': sy, 'rot': rot,
        'sx_spread': max(sx) - min(sx),
        'sy_spread': max(sy) - min(sy),
        'rot_spread_hg': (max(hg_rot) - min(hg_rot))
        if hg_rot else 0.0,
        'v3_C': repA[cn]['v3']['C'],
        'pure_C': repA[cn]['pure_pub']['C'],
        'gap': repA[cn]['gap']})
    print(f'  {cn:8s} 标签{LABEL_N[cn]:2d}vs检测'
          f'{repA[cn]["detected_cores"]:2d} '
          f'sx散布={max(sx)-min(sx):.3f} '
          f'sy散布={max(sy)-min(sy):.3f} '
          f'rot散布(HG)={max(hg_rot)-min(hg_rot) if hg_rot else 0:.2f}'
          f'  v3-pure gap={repA[cn]["gap"]:+.4f}')

(OUT / '一致性审计.json').write_text(
    json.dumps(audit, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print(f'\n[done] {OUT}')

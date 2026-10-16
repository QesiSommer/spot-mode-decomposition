# -*- coding: utf-8 -*-
"""
check_shijie2_20261016.py -- 师姐复核彻底检查（v2，含叠加式裁决）
====================================================================
背景：师姐 ①用公布参数独立叠 8奇点(2) 觉得"参数不太对"；
     ②自己分解 6奇点 = HG(1,3)+1.2·e^{iπ/2}·HG(3,0)，认为与我们
       的叠加"相反"，要求检查输出参数。

已查明的事实：
  - 我方 HG 基（hg_torch.py:90 indexing='ij'）：HG(m,n) 的 m 沿数组
    轴0（imshow 竖直）、n 沿轴1（水平）——与 MATLAB/标准光学约定
    （m 沿水平 x）互为转置；
  - v3 交付渲染（render_dof）含逐模 sx/sy/rot + 全局 shift，公布的
    能量/相位表不含这些 → 第三方纯物理叠图必有差距。

三层检查：
  S. 叠加式裁决（6奇点）：把师姐公式按 (i) 标准约定逐字、(ii) 我方
     约定逐字、(iii) 转置映射（她的 HG(1,3)→我方 HG(3,1) 等）三种
     解释在实验图上拟合（全局尺度+平移），C_s02 定胜负；
  A. 复现性核查（全八图）：纯物理叠图（公布 theta，仅调 s/平移）
     vs v3 交付渲染 的差距表——解释 8(2) "参数叠出来不太对"；
  B. 6奇点从零纯物理两模重搜（多尺度全空间 8515×6）——确认
     HG(0,3)+HG(3,1) 是否纯物理最优（排除拟合管线偏差）；
  C. 一致性审计：图名奇点数 vs 检测暗点数、逐模 sx/sy/rot 散布。

附带产出：给师姐的 MATLAB 直用叠加式（标准约定，含 w0/像素标定）
          + 每图复场文件 .npz。
Outputs: 归档_八图分解/师姐复核检查_2026-10-16/
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
from scipy.special import hermite as hermite
from scipy.special import factorial

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'closed_routes'))  # morph_search_lib 所在目录
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from refine_all_new_method import c_table_N
from rerun_new_metrics import coarse_grid_params, coarse_scan_N
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

ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
         '10奇点', '11奇点']
LABEL_N = {'4奇点': 4, '5奇点': 5, '6奇点': 6, '8奇点1': 8,
           '8奇点2': 8, '9奇点': 9, '10奇点': 10, '11奇点': 11}
SIZE, EXTENT = 224, 4.0
PX_W0 = SIZE / (2 * EXTENT)          # 1 个束腰 w0 = 28 px


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def blur2(I):
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy()


def C_s02(Ie, Ie_b, I):
    return 0.5 * (pcc(Ie, I) + pcc(Ie_b, blur2(I)))


def normI(I):
    I = np.asarray(I, dtype=np.float64)
    return I / max(I.max(), 1e-12)


# ---------------- standard-convention HG (x horizontal) ----------------
def std_hg(m, n, size=SIZE, extent=EXTENT):
    """MATLAB 约定：HG(m,n)=H_m(x)H_n(y)，x 水平（axis1）、y 竖直。"""
    x = np.linspace(-extent, extent, size)
    y = np.linspace(-extent, extent, size)
    Y, X = np.meshgrid(y, x, indexing='ij')      # rows=y, cols=x
    norm = np.sqrt(2.0 / (np.pi * factorial(m, exact=True)
                          * factorial(n, exact=True)
                          * (2.0 ** (m + n))))
    return (norm * hermite(m)(X * np.sqrt(2))
            * hermite(n)(Y * np.sqrt(2))
            * np.exp(-(X ** 2 + Y ** 2))).astype(np.complex128)


def our_hg(m, n):
    return mnp[lab2k[f'HG({m},{n})']]


def render_stack(mode_arrays, amps, phis, s, sh):
    E = np.zeros((SIZE, SIZE), dtype=np.complex128)
    for F, a, ph in zip(mode_arrays, amps, phis):
        Fs = rescale_lg(torch.from_numpy(F)[None], float(s))[0].numpy() \
            if abs(s - 1.0) > 1e-9 else F
        E = E + a * Fs * np.exp(1j * ph)
    I = normI(np.abs(E) ** 2)
    if abs(sh[0]) > 1e-9 or abs(sh[1]) > 1e-9:
        I = ndshift(I, sh, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


def fit_stack(Ie, Ie_b, mode_arrays, amps, phis,
              s_grid=None, sh_range=3):
    """对一组叠加做 全局尺度+平移 网格拟合，返回最优 (C, s, sh)。"""
    if s_grid is None:
        s_grid = np.arange(0.70, 1.45, 0.05)
    best = None
    for s in s_grid:
        for dy in range(-sh_range, sh_range + 1):
            for dx in range(-sh_range, sh_range + 1):
                I = render_stack(mode_arrays, amps, phis, s, (dy, dx))
                C = C_s02(Ie, Ie_b, I)
                if best is None or C > best['C']:
                    best = {'C': C, 's': float(s), 'sh': [dy, dx]}
    # 细化
    s0 = best['s']
    for s in np.arange(s0 - 0.04, s0 + 0.041, 0.01):
        for dy in np.arange(best['sh'][0] - 1, best['sh'][0] + 1.01,
                            0.5):
            for dx in np.arange(best['sh'][1] - 1,
                                best['sh'][1] + 1.01, 0.5):
                I = render_stack(mode_arrays, amps, phis, s, (dy, dx))
                C = C_s02(Ie, Ie_b, I)
                if C > best['C']:
                    best = {'C': C, 's': float(s),
                            'sh': [float(dy), float(dx)]}
    return best


def cores_of(I):
    return find_dark_points(I)


# ============================================================
# Part S：叠加式裁决（6奇点）
# ============================================================
print('==== Part S: shijie stack formula adjudication (6Sing) ====')
cn = '6奇点'
NAME = V3[cn]['name']
r6 = V3[cn]
src6 = r6['accepted'] if 'theta' in r6.get('accepted', {}) \
    else r6['refined_v3']
combo6 = r6['combo']
N6 = r6['N']
th6 = [float(x) for x in src6['theta']]
print(f'  v3 {cn}: combo={combo6}')
print(f'  energies={[round(x,4) for x in th6[:N6]]} '
      f'(sum={sum(th6[:N6]):.4f})')
print(f'  phases={[round(x,4) for x in th6[N6:]]}')

Ie6 = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
Ie6_b = blur2(Ie6)
pts6 = find_dark_points(Ie6)
print(f'  experiment cores={len(pts6)}')

# --- 约定映射检验：我方 HG(m,n) 与标准 HG(a,b) 谁等价？ ---
print('\n  [convention map] our HG <-> std HG (max |pcc|):')
conv_map = {}
for (m, n) in [(0, 3), (3, 1), (1, 3), (3, 0)]:
    F_our = our_hg(m, n)
    best = None
    for a in range(0, 11):
        for b in range(0, 11 - a):
            c = pcc(F_our.real, std_hg(a, b).real)
            if best is None or abs(c) > abs(best[2]):
                best = (a, b, c)
    conv_map[f'HG({m},{n})'] = best
    print(f'    our HG({m},{n})  ~  std HG({best[0]},{best[1]})  '
          f'pcc={best[2]:+.4f}')

# --- 三种解释 + 我方公布参数 ---
HER = {'amps': [1.0, 1.2], 'phis': [0.0, np.pi / 2]}
results_S = {}

# S1: 标准约定逐字（她 MATLAB 里那一套）
ms1 = [std_hg(1, 3), std_hg(3, 0)]
f1 = fit_stack(Ie6, Ie6_b, ms1, HER['amps'], HER['phis'])
I1 = render_stack(ms1, HER['amps'], HER['phis'], f1['s'], f1['sh'])
results_S['S1_her_literal_std'] = {**f1,
    'cores': len(cores_of(I1))}
print(f'\n  S1 她的公式·标准约定逐字: C={f1["C"]:.4f} '
      f's={f1["s"]:.2f} cores={len(cores_of(I1))}/{len(pts6)}')

# S2: 我方约定逐字（把她的下标直接当我们的标签）
ms2 = [our_hg(1, 3), our_hg(3, 0)]
f2 = fit_stack(Ie6, Ie6_b, ms2, HER['amps'], HER['phis'])
I2 = render_stack(ms2, HER['amps'], HER['phis'], f2['s'], f2['sh'])
results_S['S2_her_literal_ourconv'] = {**f2,
    'cores': len(cores_of(I2))}
print(f'  S2 她的公式·我方约定逐字: C={f2["C"]:.4f} '
      f's={f2["s"]:.2f} cores={len(cores_of(I2))}/{len(pts6)}')

# S3: 转置映射（她的 HG(1,3)=我方 HG(3,1)，她的 HG(3,0)=我方 HG(0,3)）
ms3 = [our_hg(3, 1), our_hg(0, 3)]
f3 = fit_stack(Ie6, Ie6_b, ms3, HER['amps'], HER['phis'])
I3 = render_stack(ms3, HER['amps'], HER['phis'], f3['s'], f3['sh'])
results_S['S3_her_transposed_ourconv'] = {**f3,
    'cores': len(cores_of(I3))}
print(f'  S3 她的公式·转置映射到我方: C={f3["C"]:.4f} '
      f's={f3["s"]:.2f} cores={len(cores_of(I3))}/{len(pts6)}')

# S4: 我方公布参数纯物理叠图（v3 theta，拟合 s/平移）
ms4 = [mnp[lab2k[m]] for m in combo6]
amps4 = [np.sqrt(max(x, 0)) for x in th6[:N6]]
phis4 = [0.0] + list(th6[N6:])
f4 = fit_stack(Ie6, Ie6_b, ms4, amps4, phis4)
I4 = render_stack(ms4, amps4, phis4, f4['s'], f4['sh'])
results_S['S4_our_published_pure'] = {**f4, 'cores': len(cores_of(I4))}
print(f'  S4 我方公布参数·纯物理:    C={f4["C"]:.4f} '
      f's={f4["s"]:.2f} cores={len(cores_of(I4))}/{len(pts6)}')

# S5: 她的模对（映射后）+ 自由能量/相位精修（同模对天花板）
rng = np.random.default_rng(61016)
best5 = dict(f3)
cur = {'amps': list(HER['amps']), 'phis': list(HER['phis'])}


def ev5(amps, phis, s, sh):
    return C_s02(Ie6, Ie6_b,
                 render_stack(ms3, amps, phis, s, sh))


for it in range(120):
    a = np.array(cur['amps']) * np.exp(rng.normal(0, 0.04, 2))
    a = a / a.max()
    ph = np.array(cur['phis']) + rng.normal(0, 0.10, 2)
    C = ev5(a, ph, best5['s'], best5['sh'])
    if C > best5['C']:
        best5 = {**best5, 'C': C}
        cur = {'amps': list(a), 'phis': list(ph)}
for it in range(60):
    s_ = best5['s'] + rng.normal(0, 0.01)
    sh_ = [best5['sh'][0] + rng.normal(0, 0.4),
           best5['sh'][1] + rng.normal(0, 0.4)]
    C = ev5(cur['amps'], cur['phis'], s_, sh_)
    if C > best5['C']:
        best5 = {'C': C, 's': float(s_), 'sh': [float(sh_[0]),
                                                float(sh_[1])]}
I5 = render_stack(ms3, cur['amps'], cur['phis'], best5['s'],
                  best5['sh'])
results_S['S5_her_pair_refined'] = {
    'C': best5['C'], 's': best5['s'], 'sh': best5['sh'],
    'amps': cur['amps'], 'phis': cur['phis'],
    'cores': len(cores_of(I5))}
print(f'  S5 她的模对·能量相位精修:  C={best5["C"]:.4f} '
      f'amps={[round(x,3) for x in cur["amps"]]} '
      f'cores={len(cores_of(I5))}/{len(pts6)}')

# S6: v3 交付渲染（含逐模自由度）参考
Iv3_6 = L.render_dof(mnp, lab2k, combo6, th6,
                     [float(x) for x in src6['sx']],
                     [float(x) for x in src6['sy']],
                     [float(x) for x in src6['shift']],
                     [float(x) for x in src6.get('rot',
                                                 [0.0] * N6)])
c_v3 = C_s02(Ie6, Ie6_b, Iv3_6)
results_S['S6_our_v3_full'] = {'C': c_v3,
                               'cores': len(cores_of(Iv3_6))}
print(f'  S6 我方 v3 交付渲染:        C={c_v3:.4f} '
      f'cores={len(cores_of(Iv3_6))}/{len(pts6)}')

# --- 裁决图 ---
fig, axs = plt.subplots(1, 6, figsize=(24, 4.6))
panels = [('实验图', Ie6, len(pts6)),
          ('S1 她公式·标准约定', I1, results_S['S1_her_literal_std']['cores']),
          ('S2 她公式·我方约定', I2, results_S['S2_her_literal_ourconv']['cores']),
          ('S3 她公式·转置映射', I3, results_S['S3_her_transposed_ourconv']['cores']),
          ('S4 我方公布参数·纯物理', I4, results_S['S4_our_published_pure']['cores']),
          ('S6 我方 v3 交付', Iv3_6, results_S['S6_our_v3_full']['cores'])]
for k, (tag, I, nc) in enumerate(panels):
    axs[k].imshow(blur2(normI(I)), cmap='inferno', vmin=0, vmax=1)
    axs[k].set_title(f'{tag}\nC={C_s02(Ie6, Ie6_b, I):.4f} '
                     f'cores={nc}', fontsize=9)
    axs[k].axis('off')
fig.suptitle(f'{NAME}（6奇点）：师姐叠加式的三种约定解释 vs 我方解 '
             '（全部对实验图打分）', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / 'S_叠加式裁决_6奇点.png', dpi=125,
            bbox_inches='tight')
plt.close(fig)
print('  [fig] S_叠加式裁决_6奇点.png')

# ============================================================
# Part A：复现性核查（全八图）
# ============================================================
print('\n==== Part A: pure published vs v3 (all 8) ====')
repA = {}
for cn in ORDER:
    r = V3[cn]
    src = r['accepted'] if 'theta' in r.get('accepted', {}) \
        else r['refined_v3']
    combo = r['combo']
    N_ = r['N']
    th = [float(x) for x in src['theta']]
    NAME = r['name']
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    Ie_b = blur2(Ie)
    pts_e = find_dark_points(Ie)
    Iv3 = L.render_dof(mnp, lab2k, combo, th,
                       [float(x) for x in src['sx']],
                       [float(x) for x in src['sy']],
                       [float(x) for x in src['shift']],
                       [float(x) for x in src.get('rot', [0.0] * N_)])
    C_v3 = C_s02(Ie, Ie_b, Iv3)
    ms = [mnp[lab2k[m]] for m in combo]
    amps = [np.sqrt(max(x, 0)) for x in th[:N_]]
    phis = [0.0] + list(th[N_:])
    fb = fit_stack(Ie, Ie_b, ms, amps, phis, sh_range=2)
    Ipu = render_stack(ms, amps, phis, fb['s'], fb['sh'])
    C_pu = fb['C']
    repA[cn] = {'name': NAME, 'combo': combo, 'label_n': LABEL_N[cn],
                'detected': len(pts_e),
                'v3_C': C_v3, 'pure_C': C_pu, 'gap': C_v3 - C_pu,
                'pure_s': fb['s'], 'pure_sh': fb['sh'],
                'pure_cores': len(cores_of(Ipu)),
                'v3_cores': len(cores_of(Iv3))}
    print(f'  {cn:8s} v3={C_v3:.4f}  pure={C_pu:.4f}  '
          f'gap={C_v3-C_pu:+.4f}  cores v3/pure/exp='
          f'{len(cores_of(Iv3))}/{len(cores_of(Ipu))}/{len(pts_e)}')

# 复现性图：8(2) 与 6奇点
for cn in ('8奇点2', '6奇点'):
    r = V3[cn]
    src = r['accepted'] if 'theta' in r.get('accepted', {}) \
        else r['refined_v3']
    combo = r['combo']
    N_ = r['N']
    th = [float(x) for x in src['theta']]
    NAME = r['name']
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    Ie_b = blur2(Ie)
    pts_e = find_dark_points(Ie)
    Iv3 = L.render_dof(mnp, lab2k, combo, th,
                       [float(x) for x in src['sx']],
                       [float(x) for x in src['sy']],
                       [float(x) for x in src['shift']],
                       [float(x) for x in src.get('rot', [0.0] * N_)])
    a = repA[cn]
    ms = [mnp[lab2k[m]] for m in combo]
    amps = [np.sqrt(max(x, 0)) for x in th[:N_]]
    phis = [0.0] + list(th[N_:])
    Ipu = render_stack(ms, amps, phis, a['pure_s'], a['pure_sh'])
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for k, (tag, I) in enumerate([('实验图', Ie),
                                  ('纯物理叠图(公布参数)', Ipu),
                                  ('v3 交付(含逐模自由度)', Iv3)]):
        axs[k].imshow(blur2(normI(I)), cmap='inferno', vmin=0, vmax=1)
        axs[k].set_title(f'{tag}\nC={C_s02(Ie, Ie_b, I):.4f}',
                         fontsize=10)
        axs[k].axis('off')
    fig.suptitle(f'{NAME}（{cn}）：为什么按公布参数叠图与交付图有差距'
                 f'（gap={a["gap"]:+.4f}）', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f'A_复现性_{cn}.png', dpi=125,
                bbox_inches='tight')
    plt.close(fig)
print('  [fig] A_复现性_8奇点2.png / A_复现性_6奇点.png')

# ============================================================
# Part B：6奇点从零纯物理两模重搜
# ============================================================
print('\n==== Part B: 6Sing from-scratch pure 2-mode search ====')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Ie6_t = torch.tensor(Ie6, dtype=torch.float32)
Ie6_small = torch.nn.functional.interpolate(
    Ie6_t[None, None], size=(64, 64), mode='bilinear',
    align_corners=False)[0, 0].to(dev)
pairs = torch.combinations(torch.arange(len(labels)), r=2)
pmat = coarse_grid_params(2, 16, 120)
ct = c_table_N(pmat, 2).to(dev)

cands = []
t0 = time.time()
for s in [0.80, 0.90, 1.00, 1.10, 1.20, 1.30]:
    ms_full = rescale_lg(modes, float(s))
    ms_small = torch.stack(
        [L.down_complex(m, 64) for m in ms_full]).to(dev)
    bc, bi = coarse_scan_N(ms_small, Ie6_small, pairs, ct, 2)
    vals, idxs = bc.topk(15)
    for v, k in zip(vals.tolist(), idxs.tolist()):
        cands.append({'pair': pairs[k].tolist(),
                      'theta': pmat[bi[k]].tolist(), 's': s,
                      'corr': v})
    print(f'  scale={s:.2f} best={float(bc.max()):.4f} '
          f'({time.time()-t0:.0f}s)')

best_by_pair = {}
for c in cands:
    k = tuple(sorted(c['pair']))
    if k not in best_by_pair or c['corr'] > best_by_pair[k]['corr']:
        best_by_pair[k] = c
pool = sorted(best_by_pair.values(), key=lambda c: -c['corr'])[:30]
v3pair = sorted(lab2k[m] for m in combo6)
pool.append({'pair': v3pair,
             'theta': [th6[0], th6[1], th6[2]],
             's': 1.0, 'corr': None, 'tag': 'v3combo'})
print(f'  refine pool: {len(pool)}')


def refine_pair(Ie, Ie_b, combo, th0, s0, seed):
    """N=2 纯物理精修：能量/相位/尺度/平移。"""
    rng = np.random.default_rng(seed)
    ms = [mnp[lab2k[m]] for m in combo]
    cur = {'amps': [float(x) for x in th0[:2]],
           'phis': [0.0, float(th0[2])], 's': float(s0),
           'sh': [0.0, 0.0]}
    best = {'C': C_s02(Ie, Ie_b, render_stack(
        ms, cur['amps'], cur['phis'], cur['s'], cur['sh']))}
    for it in range(150):
        a = np.array(cur['amps']) * np.exp(rng.normal(0, 0.05, 2))
        a = list(a / a.max())
        ph = [0.0, float(cur['phis'][1] + rng.normal(0, 0.12))]
        C = C_s02(Ie, Ie_b, render_stack(ms, a, ph, cur['s'],
                                         cur['sh']))
        if C > best['C']:
            best = {'C': C}
            cur['amps'], cur['phis'] = a, ph
    for it in range(80):
        s_ = cur['s'] + rng.normal(0, 0.015)
        C = C_s02(Ie, Ie_b, render_stack(ms, cur['amps'],
                                         cur['phis'], s_,
                                         cur['sh']))
        if C > best['C']:
            best = {'C': C}
            cur['s'] = float(s_)
    for it in range(80):
        sh = [cur['sh'][0] + rng.normal(0, 0.5),
              cur['sh'][1] + rng.normal(0, 0.5)]
        C = C_s02(Ie, Ie_b, render_stack(ms, cur['amps'],
                                         cur['phis'], cur['s'], sh))
        if C > best['C']:
            best = {'C': C}
            cur['sh'] = [float(sh[0]), float(sh[1])]
    return cur, best['C']


records = []
for ci, c in enumerate(pool):
    combo = [labels[t] for t in c['pair']]
    cur, C = refine_pair(Ie6, Ie6_b, combo, c['theta'], c['s'],
                         seed=ci)
    I = render_stack([mnp[lab2k[m]] for m in combo], cur['amps'],
                     cur['phis'], cur['s'], cur['sh'])
    nc = len(cores_of(I))
    mt = match_points(pts6, cores_of(I))
    records.append({'combo': combo, 'C': C, 'cores': nc,
                    'mean_nn': float(mt['mean'])
                    if mt['mean'] is not None else 99.0,
                    's': cur['s'], 'sh': cur['sh'],
                    'amps': cur['amps'], 'phi': cur['phis'][1],
                    'struct': bool(abs(nc - len(pts6)) <= 1
                                   and mt['mean'] is not None
                                   and mt['mean'] < 12),
                    'tag': c.get('tag', 'search')})
    if ci % 6 == 5:
        print(f'  [{ci+1}/{len(pool)}] best='
              f'{max(r_["C"] for r_ in records):.4f}')

records.sort(key=lambda r_: (0 if r_['struct'] else 1, -r_['C']))
print('\n  top-8 (struct-first, pure physical):')
for k, rr in enumerate(records[:8]):
    print(f'   {k+1}. {"+".join(rr["combo"]):30s} C={rr["C"]:.4f} '
          f'cores={rr["cores"]}/{len(pts6)} '
          f'nn={rr["mean_nn"]:.1f} [{rr["tag"]}]')

fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
axs[0].imshow(blur2(Ie6), cmap='inferno', vmin=0, vmax=1)
axs[0].set_title(f'EXPERIMENT ({len(pts6)} cores)')
for k, rr in enumerate(records[:3]):
    I = render_stack([mnp[lab2k[m]] for m in rr['combo']],
                     rr['amps'], [0.0, rr['phi']], rr['s'], rr['sh'])
    axs[k + 1].imshow(blur2(I), cmap='inferno', vmin=0, vmax=1)
    axs[k + 1].set_title(
        f'pure #{k+1}: {"+".join(rr["combo"])}\nC={rr["C"]:.4f} '
        f'cores={rr["cores"]}', fontsize=9)
for a_ in axs:
    a_.axis('off')
fig.suptitle(f'{NAME}：纯物理口径从零两模搜索 top-3', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / 'B_6奇点_纯物理重搜_top3.png', dpi=125,
            bbox_inches='tight')
plt.close(fig)

# ============================================================
# Part C：一致性审计
# ============================================================
print('\n==== Part C: consistency audit ====')
audit = []
for cn in ORDER:
    r = V3[cn]
    src = r['accepted'] if 'theta' in r.get('accepted', {}) \
        else r['refined_v3']
    sx = [float(x) for x in src['sx']]
    sy = [float(x) for x in src['sy']]
    rot = [float(x) for x in src.get('rot', [0.0] * r['N'])]
    hg_rot = [rot[i] for i, m in enumerate(r['combo'])
              if m.startswith('HG')]
    a = repA[cn]
    audit.append({
        'cn': cn, 'label_n': LABEL_N[cn], 'detected': a['detected'],
        'label_eq_detected': LABEL_N[cn] == a['detected'],
        'sx': sx, 'sy': sy, 'rot': rot,
        'sx_spread': max(sx) - min(sx),
        'sy_spread': max(sy) - min(sy),
        'rot_spread_hg': (max(hg_rot) - min(hg_rot))
        if hg_rot else 0.0,
        'v3_C': a['v3_C'], 'pure_C': a['pure_C'], 'gap': a['gap']})
    print(f'  {cn:8s} 标签{LABEL_N[cn]:2d} vs 检测{a["detected"]:2d}'
          f'  sx散布={max(sx)-min(sx):.3f}'
          f'  sy散布={max(sy)-min(sy):.3f}'
          f'  rot散布(HG)={max(hg_rot)-min(hg_rot) if hg_rot else 0:.2f}'
          f'  gap={a["gap"]:+.4f}')

# ============================================================
# 给师姐的 MATLAB 直用叠加式（HG-only 图）+ 复场 npz
# ============================================================
def std_label_for(our_label):
    """数值映射：我方标签 → 标准约定标签。"""
    if not our_label.startswith('HG'):
        return None
    F = mnp[lab2k[our_label]]
    best = None
    for a in range(0, 11):
        for b in range(0, 11 - a):
            c = pcc(F.real, std_hg(a, b).real)
            if best is None or abs(c) > abs(best[2]):
                best = (a, b, c)
    return best


formula_lines = ['# 师姐可直接使用的叠加式（标准 MATLAB 约定：',
                 '#   HG(m,n) = H_m(x)·H_n(y)·G，x 水平；',
                 '#   归一化强度；w0(束腰) = 像素尺度/28，',
                 '#   即坐标 x,y 以像素为单位后除以 28（224px 视场 ±4w0）。',
                 '# 相位约定：E = Σ a_i·exp(i·φ_i)·HG_i，强度=|E|²。',
                 '# 注意：下标顺序与项目内部记号互为转置，此处已换算。',
                 '']
for cn in ORDER:
    r = V3[cn]
    src = r['accepted'] if 'theta' in r.get('accepted', {}) \
        else r['refined_v3']
    combo = r['combo']
    N_ = r['N']
    th = [float(x) for x in src['theta']]
    amps = [np.sqrt(max(x, 0)) for x in th[:N_]]
    phis = [0.0] + list(th[N_:])
    a = repA[cn]
    formula_lines.append(f'## {r["name"]}（{cn}，'
                         f'C_s02 v3={a["v3_C"]:.4f}）')
    if any(m.startswith('LG') for m in combo):
        formula_lines.append('- 含 LG 模式：请使用附带的复场文件 '
                             f'`field_{r["name"]}.npz`'
                             '（E 场 + 参数），LG 的角向相位约定'
                             '需另行核对，不建议手写叠加式。')
    else:
        terms = []
        for i, m in enumerate(combo):
            mp = std_label_for(m)
            mm, nn = mp[0], mp[1]
            conv = '（转置换算）' if (mm, nn) != (
                int(m[3]), int(m[5])) else ''
            if i == 0:
                terms.append(f'{amps[i]:.4f}*HG({mm},{nn},x,y)'
                             f'{conv}')
            else:
                terms.append(f'{amps[i]:.4f}*exp(1i*{phis[i]:.4f})'
                             f'*HG({mm},{nn},x,y){conv}')
        formula_lines.append('- E = ' + ' + '.join(terms))
        formula_lines.append(f'- 逐模微调（提升用）：sx={src["sx"]}, '
                             f'sy={src["sy"]}, shift={src["shift"]}, '
                             f'rot={src.get("rot")}')
    formula_lines.append('')
(OUT / '师姐用叠加式_标准约定.md').write_text(
    '\n'.join(formula_lines), encoding='utf-8')

# 复场 npz（全八图）
for cn in ORDER:
    r = V3[cn]
    src = r['accepted'] if 'theta' in r.get('accepted', {}) \
        else r['refined_v3']
    combo = r['combo']
    N_ = r['N']
    th = [float(x) for x in src['theta']]
    Iv3 = L.render_dof(mnp, lab2k, combo, th,
                       [float(x) for x in src['sx']],
                       [float(x) for x in src['sy']],
                       [float(x) for x in src['shift']],
                       [float(x) for x in src.get('rot',
                                                  [0.0] * N_)])
    np.savez_compressed(
        OUT / f'field_{r["name"]}.npz',
        intensity=Iv3,
        combo=np.array(combo), theta=np.array(th),
        sx=np.array(src['sx']), sy=np.array(src['sy']),
        shift=np.array(src['shift']),
        rot=np.array(src.get('rot', [0.0] * N_)))

# ---------------- save all ----------------
(OUT / '检查结果_20261016.json').write_text(json.dumps({
    'S': results_S, 'A': repA,
    'B_top': records[:15], 'C_audit': audit,
    'convention_map': {k: list(v) for k, v in conv_map.items()}},
    ensure_ascii=False, indent=2, default=float), encoding='utf-8')

print('\n=== VERDICT ===')
print(f"  S1 她公式·标准约定逐字   C={results_S['S1_her_literal_std']['C']:.4f}")
print(f"  S2 她公式·我方约定逐字   C={results_S['S2_her_literal_ourconv']['C']:.4f}")
print(f"  S3 她公式·转置映射       C={results_S['S3_her_transposed_ourconv']['C']:.4f}")
print(f"  S4 我方公布参数·纯物理   C={results_S['S4_our_published_pure']['C']:.4f}")
print(f"  S5 她的模对·精修天花板   C={results_S['S5_her_pair_refined']['C']:.4f}")
print(f"  S6 我方 v3 交付          C={results_S['S6_our_v3_full']['C']:.4f}")
print(f"  B 最优纯物理两模: {records[0]['combo']} C={records[0]['C']:.4f}")
print(f'\n[done] {OUT}')

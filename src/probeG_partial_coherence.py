# -*- coding: utf-8 -*-
"""
probeG_partial_coherence.py -- 探针G：部分相干（相干矩阵 μ）
==============================================================
在终检解上把光强模型 I=|ΣUᵢ|² 推广为 I=Σᵢⱼ μᵢⱼ Re[UᵢUⱼ*]（μ诊断
=1 退化为完全相干；4 模共 6 个 μ 参数）。快评：I(μ)=I_coh − Σ(1−
μᵢⱼ)·Rᵢⱼ（Rᵢⱼ=2Re[UᵢUⱼ*] 预计算，μ 扫描近零成本）。

对象：10 奇点（其 ~2.3 真实不对称奇阶是主要动机）。
- 主臂：μ 坐标下降（结构+稳健约束，C_s02 目标）+ 轻量 theta 抖动；
- 对照臂：同预算纯 theta 重搜；
- 防过拟合对照：弱化种子（相位扰动）同样做 μ 优化，比较增益；
- 角谱核查：μ 解是否吃掉奇数 m 阶残差。
诊断性质，不改动交付解。
Outputs: 归档_八图分解/物理模型评估/
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
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from calibrate_metrics import angular_distance

OUT = ROOT / '归档_八图分解' / '物理模型评估'
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '10Singularities'
CN = '10奇点'
TI = json.load(open(ROOT / '归档_八图分解' / '终检精修_2026-10-10'
                    / 'terminal_inspection.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()

rec = TI[CN]
combo, N = rec['combo'], rec['N']
acc = rec['accepted']
theta0 = [float(x) for x in acc['theta']]
base_scale = rec['base_scale']
iso = acc['scales'] if acc['scales'] else [1.0] * N
gshift = acc['shift'] if acc['shift'] else [0.0, 0.0]
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
print(f'[setup] {CN} N={N} pairs={len(PAIRS)}')

Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
pts_e = find_dark_points(Ie)
N_EXP = len(pts_e)


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def blur_sig(I, sg):
    return I if sg == 0 else gaussian_blur(
        torch.tensor(I, dtype=torch.float32).unsqueeze(0),
        float(sg))[0].numpy()


from scipy.ndimage import shift as ndshift


def build_U(theta):
    Us = []
    for i, m in enumerate(combo):
        Fm = modes_np[lab2k[m]]
        s = base_scale or 1.0
        u = Fm
        if abs(s * iso[i] - 1.0) > 1e-6:
            u = rescale_lg(torch.from_numpy(Fm).unsqueeze(0),
                           float(s * iso[i]))[0].numpy()
        phi = 0.0 if i == 0 else theta[N + i - 1]
        Us.append(np.sqrt(max(theta[i], 0)) * u * np.exp(1j * phi))
    return Us


Ie_b2 = blur_sig(Ie, 2)


def make_eval(theta):
    Us = build_U(theta)
    Icoh = np.abs(sum(Us)) ** 2
    R = {p: 2 * np.real(Us[p[0]] * np.conj(Us[p[1]])) for p in PAIRS}

    def apply_shift_norm(I):
        I = np.clip(I, 0, None)
        I = I / max(I.max(), 1e-12)
        if abs(gshift[0]) > 1e-6 or abs(gshift[1]) > 1e-6:
            I = ndshift(I, gshift, order=1, mode='nearest')
            I = np.clip(I, 0, None)
        return I

    def render(mu):
        I = Icoh.copy()
        for p in PAIRS:
            I = I - (1.0 - mu.get(p, 1.0)) * R[p]
        return apply_shift_norm(I)

    def ev(mu, with_struct=True):
        I = render(mu)
        C2 = 0.5 * (pcc(Ie, I) + pcc(Ie_b2, blur_sig(I, 2)))
        return I, C2

    return render, ev


font_dummy = None
rng = np.random.default_rng(77)

render0, ev0 = make_eval(theta0)
I0, C20 = ev0({})
pts_m0 = find_dark_points(I0)
mt0 = match_points(pts_e, pts_m0)
d0 = np.array(mt0['dists'])
struct0 = (abs(len(pts_m0) - N_EXP) <= 1 and mt0['mean'] < 12)
max10_0 = float(np.sort(d0)[-2])
margin = max(14.0, max10_0)
print(f'[baseline] C_s02={C20:.4f} cores={len(pts_m0)}/{N_EXP} '
      f'max10={max10_0:.2f} struct={struct0}')

ANG = angular_distance


def odd_amp_of(I):
    _, (a, _) = ANG(np.zeros_like(I) + 0, I)
    return float(np.sum(a[1::2]))


# note: angular_distance(Ie, Im) returns (spec(Ie), spec(Im))
_, (ae, am0) = ANG(Ie, I0)
print(f'[baseline] odd-amp: exp={float(np.sum(ae[1::2])):.3f} '
      f'model={float(np.sum(am0[1::2])):.3f}')


def check(I):
    pts_m = find_dark_points(I)
    mt = match_points(pts_e, pts_m)
    dists = np.array(mt['dists']) if mt['dists'] else np.array([99.])
    struct = (abs(len(pts_m) - N_EXP) <= 1 and mt['mean'] is not None
              and mt['mean'] < 12)
    max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
    return struct, max10, len(pts_m)


def run_mu_opt(theta_seed, tag):
    render, ev = make_eval(theta_seed)
    mu = {p: 1.0 for p in PAIRS}
    I, C2 = ev(mu)
    struct, max10, nc = check(I)
    best = {'C_s02': C2, 'mu': dict(mu), 'struct': struct,
            'max10': max10, 'cores': nc}
    t0 = time.time()
    for rnd in range(2):
        for p in PAIRS:
            for step, rad in [(0.05, 0.35), (0.02, 0.10)]:
                for d in np.arange(-rad, rad + 1e-9, step):
                    muv = dict(best['mu'])
                    muv[p] = float(np.clip(best['mu'][p] + d, 0.2, 1.0))
                    I, C2 = ev(muv)
                    s_, m10, nc_ = check(I)
                    if s_ and m10 <= margin and C2 > best['C_s02']:
                        best = {'C_s02': C2, 'mu': muv, 'struct': s_,
                                'max10': m10, 'cores': nc_}
        print(f'  [{tag}] round{rnd}: C_s02={best["C_s02"]:.4f} '
              f'mu={ {f"{p[0]}-{p[1]}": round(v,2) for p,v in best["mu"].items()} }'
              f' ({time.time()-t0:.0f}s)')
    return best


print('\n=== 主臂：终检解起点，μ 优化（结构约束）===')
best_main = run_mu_opt(theta0, 'main')
I_main = make_eval(theta0)[0](best_main['mu'])
_, (_, am_new) = ANG(Ie, I_main)
print(f'[main] C_s02 {C20:.4f} -> {best_main["C_s02"]:.4f} '
      f'({best_main["C_s02"]-C20:+.4f}); struct={best_main["struct"]} '
      f'max10={best_main["max10"]:.2f}')
print(f'[main] model odd-amp: {float(np.sum(am0[1::2])):.3f} -> '
      f'{float(np.sum(am_new[1::2])):.3f} (exp '
      f'{float(np.sum(ae[1::2])):.3f})')

print('\n=== 对照臂：同预算纯 theta 重搜 ===')
best_ctrl = dict(best_main)
best_ctrl = {'C_s02': C20, 'mu': None}
for k in range(250):
    u = np.array(theta0[:N]) + rng.normal(0, 0.010, N)
    u = np.clip(u, 0.005, None)
    u = u / u.sum()
    ph = np.array(theta0[N:]) + rng.normal(0, 0.035, N - 1)
    th = list(u) + list(ph)
    render, ev = make_eval(th)
    I, C2 = ev({})
    s_, m10, nc_ = check(I)
    if s_ and m10 <= margin and C2 > best_ctrl['C_s02']:
        best_ctrl = {'C_s02': C2, 'mu': None, 'struct': s_,
                     'max10': m10, 'cores': nc_}
print(f'[control] C_s02 {C20:.4f} -> {best_ctrl["C_s02"]:.4f} '
      f'({best_ctrl["C_s02"]-C20:+.4f}) struct='
      f'{best_ctrl["struct"]} max10={best_ctrl["max10"]:.2f}')

print('\n=== 防过拟合对照：弱化种子同样做 μ 优化 ===')
th_c = list(theta0)
th_c[4] += 0.6
th_c[5] -= 0.5
th_c[6] += 0.4
render_c, ev_c = make_eval(th_c)
Ic, C2c = ev_c({})
sc_, m10c, _ = check(Ic)
print(f'[corrupt] start C_s02={C2c:.4f} struct={sc_} max10={m10c:.2f}')
best_corrupt = run_mu_opt(th_c, 'corrupt')
print(f'[corrupt] C_s02 {C2c:.4f} -> {best_corrupt["C_s02"]:.4f} '
      f'(gain {best_corrupt["C_s02"]-C2c:+.4f})')

summary = {
    'image': NAME,
    'baseline': {'C_s02': float(C20), 'mu': {str(p): 1.0
                                             for p in PAIRS},
                 'struct': bool(struct0), 'max10': float(max10_0)},
    'main_mu_opt': {'C_s02': float(best_main['C_s02']),
                    'mu': {str(p): float(v)
                           for p, v in best_main['mu'].items()},
                    'struct': bool(best_main['struct']),
                    'max10': float(best_main['max10']),
                    'gain': float(best_main['C_s02'] - C20),
                    'model_odd_before': float(np.sum(am0[1::2])),
                    'model_odd_after': float(np.sum(am_new[1::2])),
                    'exp_odd': float(np.sum(ae[1::2]))},
    'control_theta': {'C_s02': float(best_ctrl['C_s02']),
                      'gain': float(best_ctrl['C_s02'] - C20)},
    'corrupt_control': {'start': float(C2c),
                        'after_mu_opt': float(best_corrupt['C_s02']),
                        'gain': float(best_corrupt['C_s02'] - C2c)}}
(OUT / 'probeG_partial_coherence.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# figure: metric comparison
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels_ = ['baseline\n(terminal)', 'control\n(theta only)',
           'main\n(mu opt)', 'corrupt seed\n(mu opt, anti-overfit)']
vals = [C20, best_ctrl['C_s02'], best_main['C_s02'],
        best_corrupt['C_s02']]
starts = [C20, C20, C20, C2c]
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(4)
ax.bar(x - 0.2, starts, width=0.4, label='start', color='#999')
ax.bar(x + 0.2, vals, width=0.4, label='after', color='#27a')
for i in range(4):
    ax.text(i, max(starts[i], vals[i]) + 0.004,
            f'{vals[i]-starts[i]:+.4f}', ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels_, fontsize=9)
ax.set_ylim(0.7, 1.0)
ax.set_ylabel('C_s02')
ax.legend()
ax.set_title('Probe G: partial coherence (mu) on 10Sing - gains vs '
             'controls')
fig.tight_layout()
fig.savefig(OUT / 'probeG_部分相干.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'\n[done] {OUT}')

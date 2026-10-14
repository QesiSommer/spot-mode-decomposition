# -*- coding: utf-8 -*-
"""
probeH2_extra_dofs_v3.py -- ②③⑤⑥ 变体探针（10奇点）
======================================================
在 v2 终检解上分别测试五个新自由度（独立臂 + 共用对照臂）：

  z     ：逐模二次相位（离轴传播/焦面误差）c_i·r^2；
  rotg  ：全局旋转角（残差解剖显示"切向偶极子"指纹→旋转缺席）；
  rotp  ：逐模旋转角 θ_i；
  bg    ：低阶多项式背景（杂散光/平场，5 项系数）；
  det   ：探测器非线性（强度 gamma + 软饱和）。

协议：每臂 = theta 抖动 300 点 + 该自由度坐标下降（粗→细，2 轮），
     只采纳 结构 PASS 且 max10 ≤ margin 且 C_s02 提升的解；
     对照臂 = 同预算纯 theta 重搜（隔离搜索噪声）；
     记录各臂终点模型的奇阶残差（odd-amp）变化。
Outputs: 归档_八图分解/10奇点残差专项/probeH2_*
"""
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import shift as ndshift
from scipy.ndimage import rotate as ndrotate

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from calibrate_metrics import angular_distance

OUT = ROOT / '归档_八图分解' / '10奇点残差专项'
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V2 = json.load(open(ROOT / '归档_八图分解' / '终检精修v2_2026-10-11'
                    / 'terminal_inspection_v2.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()

CN = '10奇点'
rec = V2[CN]
combo, N = rec['combo'], rec['N']
acc = rec['accepted']
theta0 = [float(x) for x in acc['theta']]
sx0 = [float(x) for x in acc['sx']]
sy0 = [float(x) for x in acc['sy']]
shift0 = [float(x) for x in acc['shift']]

Ie = load_experiment(f'{IMG}/{rec["name"]}.png', 'nipy_spectral', 'isot')
pts_e = find_dark_points(Ie)
N_EXP = len(pts_e)
Ie_b2 = gaussian_blur(torch.tensor(Ie, dtype=torch.float32)
                      .unsqueeze(0), 2.0)[0].numpy()

H = Ie.shape[0]
yy, xx = np.mgrid[0:H, 0:H]
cy = cx = (H - 1) / 2.0
g2 = ((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float64)
xn = (xx - cx) / 112.0
yn = (yy - cy) / 112.0
POLY = {'x': xn, 'y': yn, 'x2': xn ** 2, 'y2': yn ** 2, 'xy': xn * yn}


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def _fit_center(a, H, W):
    out = np.zeros((a.shape[0], H, W), np.float64)
    h, w = a.shape[-2], a.shape[-1]
    if h >= H:
        s_ = (h - H) // 2
        s_s, s_d = slice(s_, s_ + H), slice(0, H)
    else:
        s_ = (H - h) // 2
        s_s, s_d = slice(0, h), slice(s_, s_ + h)
    if w >= W:
        sx_ = (w - W) // 2
        sx_s, sx_d = slice(sx_, sx_ + W), slice(0, W)
    else:
        sx_ = (W - w) // 2
        sx_s, sx_d = slice(0, w), slice(sx_, sx_ + w)
    out[:, s_d, sx_d] = a[:, s_s, sx_s]
    return out


def zoom_axis_c(Fx, s, axis):
    if abs(s - 1.0) < 1e-6:
        return Fx
    Hh, Ww = Fx.shape
    t = np.stack([Fx.real, Fx.imag], 0)[None]
    t = torch.tensor(t, dtype=torch.float32)
    Hn = int(round(Hh * s)) if axis == 0 else Hh
    Wn = int(round(Ww * s)) if axis == 1 else Ww
    t = torch.nn.functional.interpolate(
        t, size=(Hn, Wn), mode='bilinear',
        align_corners=False)[0].numpy().astype(np.float64)
    b = _fit_center(t, Hh, Ww)
    return b[0] + 1j * b[1]


def render(theta, extra):
    """extra: dict with optional c_i (list), rot_g, rot_i (list),
    bg (dict name->coef), gamma, sat."""
    E = np.zeros(modes_np[0].shape, dtype=np.complex128)
    c = extra.get('c')
    rg = extra.get('rot_g', 0.0)
    ri = extra.get('rot_i')
    for i, m in enumerate(combo):
        Fm = modes_np[lab2k[m]]
        if abs(sx0[i] - 1) > 1e-6 or abs(sy0[i] - 1) > 1e-6:
            Fm = zoom_axis_c(Fm, sx0[i], 1)
            Fm = zoom_axis_c(Fm, sy0[i], 0)
        if c is not None and abs(c[i]) > 1e-12:
            Fm = Fm * np.exp(1j * c[i] * g2)
        ang = rg + (ri[i] if ri is not None else 0.0)
        if abs(ang) > 1e-9:
            Fm = ndrotate(Fm.real, ang, reshape=False, order=1) + \
                1j * ndrotate(Fm.imag, ang, reshape=False, order=1)
        phi = 0.0 if i == 0 else theta[N + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi)
    I = np.abs(E) ** 2
    gam = extra.get('gamma', 1.0)
    if abs(gam - 1.0) > 1e-9:
        I = np.clip(I, 0, None) ** gam
    sat = extra.get('sat', 0.0)
    if sat > 1e-9:
        Im_ = I / max(I.max(), 1e-12)
        I = Im_ / (1.0 + sat * Im_)
    bg = extra.get('bg')
    if bg:
        I = I + sum(co * POLY[k] for k, co in bg.items()) * I.max()
        I = np.clip(I, 0, None)
    I = I / max(I.max(), 1e-12)
    if abs(shift0[0]) > 1e-6 or abs(shift0[1]) > 1e-6:
        I = ndshift(I, shift0, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


def ev(theta, extra):
    Im = render(theta, extra)
    C2 = 0.5 * (pcc(Ie, Im) + pcc(Ie_b2, gaussian_blur(
        torch.tensor(Im, dtype=torch.float32).unsqueeze(0), 2.0
    )[0].numpy()))
    pts_m = find_dark_points(Im)
    mt = match_points(pts_e, pts_m)
    dists = np.array(mt['dists']) if mt['dists'] else np.array([99.])
    struct = (abs(len(pts_m) - N_EXP) <= 1
              and mt['mean'] is not None and mt['mean'] < 12)
    max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
    return {'C_s02': C2, 'struct': struct, 'cores': len(pts_m),
            'max10': max10, 'mean_nn': float(mt['mean'])
            if mt['mean'] is not None else 99., 'extra': extra,
            '_I': Im}


base = ev(theta0, {})
margin = max(14.0, base['max10'])
_, (_, am0) = angular_distance(Ie, base['_I'])
print(f'[baseline v2] C_s02={base["C_s02"]:.4f} '
      f'cores={base["cores"]}/{N_EXP} max10={base["max10"]:.2f} '
      f'odd-amp={float(np.sum(am0[1::2])):.2f}')

rng = np.random.default_rng(31337)


def theta_jit(th):
    u = np.array(th[:N]) + rng.normal(0, 0.010, N)
    u = np.clip(u, 0.005, None)
    u = u / u.sum()
    ph = np.array(th[N:]) + rng.normal(0, 0.035, N - 1)
    return list(u) + list(ph)


def run_arm(name, init_extra, dof_scans, n_theta=300):
    t0 = time.time()
    best = dict(base)
    best.update({'theta': list(theta0), 'extra': dict(init_extra)})
    cur = best
    for rnd in range(2):
        # theta jitter block
        for k in range(n_theta // 2):
            th = theta_jit(cur['theta'])
            r = ev(th, cur['extra'])
            if r['struct'] and r['max10'] <= margin \
                    and r['C_s02'] > best['C_s02']:
                best = {'C_s02': r['C_s02'], 'struct': r['struct'],
                        'cores': r['cores'], 'max10': r['max10'],
                        'mean_nn': r['mean_nn'], '_I': r['_I'],
                        'theta': list(th),
                        'extra': {k_: (list(v_) if isinstance(v_, list)
                                       else v_)
                                  for k_, v_ in cur['extra'].items()}}
                cur = best
        # DOF scans
        for (key, idx, step, rad) in dof_scans:
            curv = cur['extra'].get(key)
            is_list = isinstance(curv, list)
            is_dict = isinstance(curv, dict)
            if is_list:
                base_v = float(curv[idx])
            elif is_dict:
                base_v = float(curv.get(idx, 0.0))
            else:
                base_v = float(curv if curv is not None else 0.0)
            for d in np.arange(-rad, rad + 1e-9, step):
                ex = {k_: (list(v_) if isinstance(v_, list)
                           else dict(v_) if isinstance(v_, dict) else v_)
                      for k_, v_ in cur['extra'].items()}
                if is_list:
                    ex.setdefault(key, [0.0] * N)
                    ex[key][idx] = base_v + float(d)
                elif is_dict:
                    ex.setdefault(key, {})
                    ex[key][idx] = base_v + float(d)
                else:
                    ex[key] = base_v + float(d)
                r = ev(cur['theta'], ex)
                if r['struct'] and r['max10'] <= margin \
                        and r['C_s02'] > best['C_s02']:
                    best = {'C_s02': r['C_s02'], 'struct': r['struct'],
                            'cores': r['cores'], 'max10': r['max10'],
                            'mean_nn': r['mean_nn'], '_I': r['_I'],
                            'theta': list(cur['theta']),
                            'extra': ex}
                    cur = best
            cur = best
        def fmt(v):
            if isinstance(v, list):
                return np.round(v, 4).tolist()
            if isinstance(v, dict):
                return {k2: round(float(v2), 5) for k2, v2 in v.items()}
            return round(float(v), 5)

        print(f'  [{name}] round{rnd}: C_s02={best["C_s02"]:.4f} '
              f'({time.time()-t0:.0f}s) extra='
              f'{ {k_: fmt(v_) for k_, v_ in best["extra"].items()} }')
    return best


results = {'baseline': {'C_s02': base['C_s02'],
                        'max10': base['max10'],
                        'odd_amp': float(np.sum(am0[1::2]))}}

# control arm
print('\n=== control (theta only) ===')
ctrl = run_arm('control', {}, [])
_, (_, amc) = angular_distance(Ie, ctrl['_I'])
results['control'] = {'C_s02': ctrl['C_s02'],
                      'gain': ctrl['C_s02'] - base['C_s02'],
                      'max10': ctrl['max10'],
                      'odd_amp': float(np.sum(amc[1::2]))}
print(f'  control: {base["C_s02"]:.4f} -> {ctrl["C_s02"]:.4f} '
      f'({ctrl["C_s02"]-base["C_s02"]:+.4f})')

# ② per-mode quadratic phase
print('\n=== z (per-mode quadratic phase) ===')
scans = [('c', i, 1e-4, 6e-4) for i in range(N)]
r = run_arm('z', {'c': [0.0] * N}, scans)
_, (_, amz) = angular_distance(Ie, r['_I'])
results['z'] = {'C_s02': r['C_s02'], 'gain': r['C_s02'] - base['C_s02'],
                'max10': r['max10'], 'c': [float(x) for x in r['extra']['c']],
                'odd_amp': float(np.sum(amz[1::2]))}

# ③a global rotation
print('\n=== rotg (global rotation) ===')
r = run_arm('rotg', {'rot_g': 0.0},
            [('rot_g', None, 0.5, 3.0), ('rot_g', None, 0.25, 0.75)])
_, (_, amg) = angular_distance(Ie, r['_I'])
results['rotg'] = {'C_s02': r['C_s02'],
                   'gain': r['C_s02'] - base['C_s02'],
                   'max10': r['max10'],
                   'rot_g': float(r['extra']['rot_g']),
                   'odd_amp': float(np.sum(amg[1::2]))}

# ③b per-mode rotation
print('\n=== rotp (per-mode rotation) ===')
scans = [('rot_i', i, 1.0, 6.0) for i in range(N)]
r = run_arm('rotp', {'rot_i': [0.0] * N}, scans)
_, (_, amp) = angular_distance(Ie, r['_I'])
results['rotp'] = {'C_s02': r['C_s02'],
                   'gain': r['C_s02'] - base['C_s02'],
                   'max10': r['max10'],
                   'rot_i': [float(x) for x in r['extra']['rot_i']],
                   'odd_amp': float(np.sum(amp[1::2]))}

# ⑤ polynomial background
print('\n=== bg (polynomial background) ===')
scans = [('bg', k, 0.01, 0.04) for k in POLY]
scans += [('bg', k, 0.005, 0.015) for k in POLY]
r = run_arm('bg', {'bg': {}}, scans)
_, (_, amb) = angular_distance(Ie, r['_I'])
results['bg'] = {'C_s02': r['C_s02'], 'gain': r['C_s02'] - base['C_s02'],
                 'max10': r['max10'],
                 'bg': {k: float(v) for k, v in
                        r['extra'].get('bg', {}).items()},
                 'odd_amp': float(np.sum(amb[1::2]))}

# ⑥ detector nonlinearity
print('\n=== det (detector nonlinearity) ===')
scans = [('gamma', None, 0.02, 0.08), ('sat', None, 0.05, 0.2)]
scans += [('gamma', None, 0.01, 0.03), ('sat', None, 0.025, 0.075)]
r = run_arm('det', {'gamma': 1.0, 'sat': 0.0}, scans)
_, (_, amd) = angular_distance(Ie, r['_I'])
results['det'] = {'C_s02': r['C_s02'],
                  'gain': r['C_s02'] - base['C_s02'],
                  'max10': r['max10'],
                  'gamma': float(r['extra']['gamma']),
                  'sat': float(r['extra']['sat']),
                  'odd_amp': float(np.sum(amd[1::2]))}

(OUT / 'probeH2_extra_dofs_v3.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

print('\n=== SUMMARY (10奇点, baseline '
      f'{base["C_s02"]:.4f}, odd {float(np.sum(am0[1::2])):.2f}) ===')
for k in ['control', 'z', 'rotg', 'rotp', 'bg', 'det']:
    rr = results[k]
    print(f'  {k:7s} C_s02={rr["C_s02"]:.4f} '
          f'(net {rr["gain"]-results["control"]["gain"]:+.4f}) '
          f'max10={rr["max10"]:.2f} odd={rr["odd_amp"]:.2f}')
print(f'[done] {OUT}')

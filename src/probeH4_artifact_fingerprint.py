# -*- coding: utf-8 -*-
"""
probeH4_artifact_fingerprint.py -- ④ 伪影合成鉴别（10奇点）
=============================================================
把 v2 模型渲染当"理想图"，叠加四类系统伪影，检验哪种伪影能复制
实验图的"残差指纹"：

实验图指纹（v2 管线）:  raw odd=4.80 / 拟合后模型 odd=2.19 /
                         拟合后残差 odd=4.39 / 拟合 C_s02=0.8929

流程（每个伪影配置）:
  1. 合成 I_syn = artifact(I_model)（归一化×max，同预处理）；
  2. 对 I_syn 做轻量重拟合（theta 抖动 + 平移 + sx/sy 一轮）；
  3. 输出 (C_s02, 模型 odd, 残差 odd, raw odd)，与实验指纹对照。
判定：能同时逼近 (raw odd≈4.8, 模型 odd≈2.2, 残差 odd≈4.4) 的
      伪影族 = 残差的系统级候选解释。
Outputs: 归档_八图分解/10奇点残差专项/probeH4_*
"""
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import shift as ndshift

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
H = Ie.shape[0]
yy, xx = np.mgrid[0:H, 0:H]
cy = cx = (H - 1) / 2.0
xn = (xx - cx) / 112.0
yn = (yy - cy) / 112.0
POLY = {'x': xn, 'y': yn, 'x2': xn ** 2, 'y2': yn ** 2, 'xy': xn * yn}


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def _fit_center(a, Hh, Ww):
    out = np.zeros((a.shape[0], Hh, Ww), np.float64)
    h, w = a.shape[-2], a.shape[-1]
    if h >= Hh:
        s_ = (h - Hh) // 2
        s_s, s_d = slice(s_, s_ + Hh), slice(0, Hh)
    else:
        s_ = (Hh - h) // 2
        s_s, s_d = slice(0, h), slice(s_, s_ + h)
    if w >= Ww:
        sx_ = (w - Ww) // 2
        sx_s, sx_d = slice(sx_, sx_ + Ww), slice(0, Ww)
    else:
        sx_ = (Ww - w) // 2
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


def render(theta, sx, sy, shiftv):
    E = np.zeros(modes_np[0].shape, dtype=np.complex128)
    for i, m in enumerate(combo):
        Fm = modes_np[lab2k[m]]
        if abs(sx[i] - 1) > 1e-6 or abs(sy[i] - 1) > 1e-6:
            Fm = zoom_axis_c(Fm, sx[i], 1)
            Fm = zoom_axis_c(Fm, sy[i], 0)
        phi = 0.0 if i == 0 else theta[N + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    if shiftv is not None and (abs(shiftv[0]) > 1e-6
                               or abs(shiftv[1]) > 1e-6):
        I = ndshift(I, shiftv, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


def odd_amp(a):
    return float(np.sum(a[1::2]))


def finger(I_meas):
    """light refit of the model to I_meas; return signature."""
    Ie_b2m = gaussian_blur(torch.tensor(I_meas, dtype=torch.float32)
                           .unsqueeze(0), 2.0)[0].numpy()

    def ev(theta, sx, sy, shiftv):
        Im = render(theta, sx, sy, shiftv)
        C2 = 0.5 * (pcc(I_meas, Im) + pcc(Ie_b2m, gaussian_blur(
            torch.tensor(Im, dtype=torch.float32).unsqueeze(0), 2.0
        )[0].numpy()))
        return C2, Im

    best = {'C2': -9}
    rng = np.random.default_rng(9)

    def consider(theta, sx, sy, shiftv):
        nonlocal best
        C2, Im = ev(theta, sx, sy, shiftv)
        if C2 > best['C2']:
            best = {'C2': C2, 'I': Im}

    consider(theta0, sx0, sy0, shift0)
    for k in range(60):
        u = np.array(theta0[:N]) + rng.normal(0, 0.012, N)
        u = np.clip(u, 0.005, None)
        u = u / u.sum()
        ph = np.array(theta0[N:]) + rng.normal(0, 0.05, N - 1)
        consider(list(u) + list(ph), sx0, sy0, shift0)
    for step, rad in [(1.0, 2.0), (0.25, 0.5)]:
        for dy in np.arange(-rad, rad + 1e-9, step):
            for dx in np.arange(-rad, rad + 1e-9, step):
                consider(theta0, sx0, sy0,
                         [shift0[0] + dy, shift0[1] + dx])
    for i in range(N):
        for sgn in (-1, 1):
            for k2 in (0.04, 0.08):
                sxv = list(sx0)
                syv = list(sy0)
                sxv[i] *= (1 + sgn * k2)
                consider(theta0, sxv, sy0, shift0)
                syv[i] *= (1 + sgn * k2)
                consider(theta0, sx0, syv, shift0)
    I_fit = best['I']
    _, (_, am) = angular_distance(I_meas, I_fit)
    _, (_, ar) = angular_distance(I_meas, I_meas - I_fit)
    return {'C_s02': best['C2'], 'model_odd': odd_amp(am),
            'resid_odd': odd_amp(ar)}


# ---- reference: the real experiment signature (recompute) ----
I_real_fit = render(theta0, sx0, sy0, shift0)
_, (ae, amr) = angular_distance(Ie, I_real_fit)
_, (_, arr) = angular_distance(Ie, Ie - I_real_fit)
ref = {'C_s02': 0.5 * (pcc(Ie, I_real_fit) + pcc(
           gaussian_blur(torch.tensor(Ie, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy(),
           gaussian_blur(torch.tensor(I_real_fit, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy())),
       'raw_odd': odd_amp(ae), 'model_odd': odd_amp(amr),
       'resid_odd': odd_amp(arr)}
print(f'[REAL fingerprint] {json.dumps({k: round(v,3) for k,v in ref.items()})}')

# ---- synthetic base ----
I_gt = I_real_fit  # ideal model render
print(f'[synthetic base] CG residual odd of perfect fit should be ~0')

configs = []
for fam, params in [
        ('gain_x', [0.05, 0.1, 0.15, -0.05, -0.1, -0.15]),
        ('gain_y', [0.05, 0.1, 0.15, -0.05, -0.1, -0.15]),
        ('poly_x2', [0.03, 0.06, -0.03, -0.06]),
        ('poly_xy', [0.03, 0.06, -0.03, -0.06]),
        ('poly_x', [0.03, 0.06, -0.03, -0.06]),
        ('gamma', [0.9, 0.95, 1.05, 1.1]),
        ('sat', [0.15, 0.3, 0.5])]:
    for p in params:
        configs.append((fam, p))

rows = []
t0 = time.time()
for fam, p in configs:
    I = I_gt.copy()
    if fam == 'gain_x':
        I = I * (1 + p * xn)
    elif fam == 'gain_y':
        I = I * (1 + p * yn)
    elif fam.startswith('poly_'):
        I = I + p * POLY[fam[5:]] * I.max()
    elif fam == 'gamma':
        I = np.clip(I, 0, None) ** p
    elif fam == 'sat':
        Im_ = I / max(I.max(), 1e-12)
        I = Im_ / (1 + p * Im_)
    I = np.clip(I, 0, None)
    I = I / max(I.max(), 1e-12)
    _, (asyn, _) = angular_distance(I, I)
    raw_odd = odd_amp(asyn)
    sig = finger(I)
    row = {'family': fam, 'param': p, 'raw_odd': raw_odd, **sig}
    rows.append(row)
    print(f'  [{fam:8s} {p:+.2f}] raw_odd={raw_odd:5.2f} '
          f'refit C_s02={sig["C_s02"]:.4f} model_odd='
          f'{sig["model_odd"]:5.2f} resid_odd={sig["resid_odd"]:5.2f} '
          f'({time.time()-t0:.0f}s)')

(OUT / 'probeH4_artifact_fingerprint.json').write_text(
    json.dumps({'real_fingerprint': ref, 'configs': rows},
               ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# ---- match score ranking ----
def score(r):
    return (abs(r['raw_odd'] - ref['raw_odd']) / max(ref['raw_odd'], 1)
            + abs(r['model_odd'] - ref['model_odd'])
            / max(ref['model_odd'], 1)
            + abs(r['resid_odd'] - ref['resid_odd'])
            / max(ref['resid_odd'], 1))

rows.sort(key=score)
print('\n=== best fingerprint matches (lower=closer to real) ===')
for r in rows[:8]:
    print(f"  {r['family']:8s} {r['param']:+.2f}  score={score(r):.2f} "
          f"raw={r['raw_odd']:.2f} model={r['model_odd']:.2f} "
          f"resid={r['resid_odd']:.2f} C02={r['C_s02']:.4f}")
print(f'[done] {OUT}')

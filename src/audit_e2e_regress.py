# -*- coding: utf-8 -*-
"""audit_e2e_regress.py — 端到端八图回归复核（不经 convert_labels，
直接用标准记号组合 + 换算 theta 全 dof 渲染，C_s02 应=交付值）"""
import json
import re
import sys
import numpy as np
import torch
import os
from pathlib import Path
from scipy.ndimage import rotate as ndrotate, shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes


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


def render_dof(fields, theta, sx, sy, shiftv, rot):
    E = np.zeros((224, 224), dtype=np.complex128)
    n = len(fields)
    for i, Fm in enumerate(fields):
        if abs(sx[i] - 1.0) > 1e-9:
            Fm = zoom_axis_c(Fm, sx[i], 1)
        if abs(sy[i] - 1.0) > 1e-9:
            Fm = zoom_axis_c(Fm, sy[i], 0)
        if abs(rot[i]) > 1e-9:
            Fm = ndrotate(Fm.real, rot[i], reshape=False, order=1) \
                + 1j * ndrotate(Fm.imag, rot[i], reshape=False, order=1)
        phi = 0.0 if i == 0 else theta[n + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-30)
    I = ndshift(I, shiftv, order=1, mode='nearest')
    return np.clip(I, 0, None)


V3 = json.load(open(ROOT / '归档_八图分解/终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))
EXPECT = {'4奇点': 0.9163, '5奇点': 0.9656, '6奇点': 0.9746,
          '8奇点1': 0.9703, '8奇点2': 0.9384, '9奇点': 0.9480,
          '10奇点': 0.9013, '11奇点': 0.9474}
STD = {'4奇点': ['LG(0,4)', 'LG(2,0)', 'HG(4,6)'],
       '5奇点': ['LG(1,1)', 'LG(0,-3)'],
       '6奇点': ['HG(3,0)', 'HG(1,3)'],
       '8奇点1': ['LG(2,0)', 'HG(1,1)', 'HG(0,2)', 'HG(3,3)'],
       '8奇点2': ['HG(4,0)', 'HG(3,1)', 'HG(1,3)', 'HG(1,5)'],
       '9奇点': ['HG(1,3)', 'HG(0,4)', 'HG(3,3)', 'HG(1,5)', 'HG(1,7)'],
       '10奇点': ['LG(0,3)', 'LG(1,3)', 'LG(2,1)', 'HG(4,3)'],
       '11奇点': ['HG(3,2)', 'HG(1,4)', 'HG(3,4)', 'HG(3,0)', 'HG(4,3)']}
modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
modes_np = modes.numpy()
lab2k = {l: k for k, l in enumerate(labels)}


def pcc(a, b):
    a = np.asarray(a).ravel() - np.asarray(a).mean()
    b = np.asarray(b).ravel() - np.asarray(b).mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else -1.0


def l_of(lab):
    if lab.startswith('HG'):
        return 0
    return int(re.findall(r'-?\d+', lab)[1])


print('端到端回归（新基 + 标准记号组合 + 换算 theta 全 dof 渲染）:')
fails = 0
for cn, r3 in V3.items():
    src3 = r3['accepted']
    N = r3['N']
    theta0 = [float(x) for x in src3['theta']]
    combo_int = r3['combo']
    l0 = l_of(combo_int[0])
    theta_conv = list(theta0)
    for i in range(1, N):
        theta_conv[N + i - 1] = theta0[N + i - 1] \
            + (l_of(combo_int[i]) - l0) * np.pi / 2
    fields = [modes_np[lab2k[m]] for m in STD[cn]]
    sx = [float(x) for x in src3['sx']]
    sy = [float(x) for x in src3['sy']]
    shiftv = [float(x) for x in src3['shift']]
    rot = [float(x) for x in src3['rot']]
    I = render_dof(fields, theta_conv, sx, sy, shiftv, rot)
    Ie = load_experiment(
        os.path.join(IMG, f"{r3['name']}.png"),
        'nipy_spectral', 'isot')
    Ieb = gaussian_blur(torch.tensor(Ie, dtype=torch.float32)
                        .unsqueeze(0), 2.0)[0].numpy()
    Ib = gaussian_blur(torch.tensor(I, dtype=torch.float32)
                       .unsqueeze(0), 2.0)[0].numpy()
    C = 0.5 * (pcc(Ie, I) + pcc(Ieb, Ib))
    ok = abs(C - EXPECT[cn]) < 5e-4
    fails += not ok
    print(f'  {cn}: C_s02={C:.4f} 期望 {EXPECT[cn]}  '
          f'{"PASS" if ok else "FAIL"}')
print(f'总判据: {"ALL PASS" if fails == 0 else f"{fails} FAIL"}')

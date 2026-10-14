# -*- coding: utf-8 -*-
"""
stage10_terminal_inspection_v2.py -- 终检 v2：并入逐模像散
============================================================
依据探针 F 结论（逐模像散真实有效，净 +0.008~+0.017），把逐模各向
异性缩放 (sx_i, sy_i) 并入终检管线：

- 种子 = v1 终检合格解（theta + 平移 + 各向同性逐模束腰，作为
  sx=sy 初值）；
- 坐标下降：theta 抖动 + 平移细扫（±3px，0.25px）+ 逐模 sx/sy 细扫
  （±0.08，0.02 → ±0.03，0.01），2 轮；
- 只采纳：结构 PASS 且 max10 ≤ max(14, 基线 max10) 且 C_s02 提升
  ≥ +0.001（W_avg ≥ 基线 − 0.005）；
- 渲染统一走"先 x 轴后 y 轴"各向异性缩放路径（含内置自检：
  与 rescale_lg 在 isotropic 情形的差异）。
Outputs: 归档_八图分解/终检精修v2_2026-10-11/terminal_inspection_v2.json
"""
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from scipy.ndimage import shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from window_metrics import make_windows, _support_mask
from structural_check import find_dark_points, match_points

ARCH = ROOT / '归档_八图分解'
OUT = ARCH / '终检精修v2_2026-10-11'
OUT.mkdir(parents=True, exist_ok=True)
PREV = json.load(open(ARCH / '终检精修_2026-10-10'
                      / 'terminal_inspection.json', encoding='utf-8'))
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def blur_sig(I, sg):
    return I if sg == 0 else gaussian_blur(
        torch.tensor(I, dtype=torch.float32).unsqueeze(0),
        float(sg))[0].numpy()


def _fit_center(a, H, W):
    out = np.zeros((a.shape[0], H, W), np.float64)
    h, w = a.shape[-2], a.shape[-1]
    if h >= H:
        sy = (h - H) // 2
        sys_, syd = slice(sy, sy + H), slice(0, H)
    else:
        sy = (H - h) // 2
        sys_, syd = slice(0, h), slice(sy, sy + h)
    if w >= W:
        sx = (w - W) // 2
        sxs, sxd = slice(sx, sx + W), slice(0, W)
    else:
        sx = (W - w) // 2
        sxs, sxd = slice(0, w), slice(sx, sx + w)
    out[:, syd, sxd] = a[:, sys_, sxs]
    return out


def zoom_axis_c(F, s, axis):
    if abs(s - 1.0) < 1e-6:
        return F
    H, W = F.shape
    t = np.stack([F.real, F.imag], 0)[None]
    t = torch.tensor(t, dtype=torch.float32)
    Hn = int(round(H * s)) if axis == 0 else H
    Wn = int(round(W * s)) if axis == 1 else W
    t = TF.interpolate(t, size=(Hn, Wn), mode='bilinear',
                       align_corners=False)[0].numpy().astype(np.float64)
    b = _fit_center(t, H, W)
    return b[0] + 1j * b[1]


def render_v2(combo, theta, sx, sy, shiftv):
    N = len(combo)
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


# ---- self-check: isotropic aniso-path vs rescale_lg ----
F0 = modes_np[lab2k['HG(2,0)']]
s_test = 1.08
a = rescale_lg(torch.from_numpy(F0).unsqueeze(0), s_test)[0].numpy()
b = zoom_axis_c(zoom_axis_c(F0, s_test, 1), s_test, 0)
print(f'[selfcheck] isotropic s=1.08: max|aniso-rescale| = '
      f'{np.abs(a-b).max():.4f}, rel = '
      f'{np.abs(a-b).max()/np.abs(a).max():.3f}')

results = {}
for CN in ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2',
           '9奇点', '10奇点', '11奇点']:
    rec0 = PREV[CN]
    NAME, combo, N = rec0['name'], rec0['combo'], rec0['N']
    acc = rec0['accepted']
    theta0 = [float(x) for x in acc['theta']]
    iso = acc['scales'] if acc['scales'] else [1.0] * N
    base_scale = rec0['base_scale'] or 1.0
    iso_total = [base_scale * s for s in iso]
    gshift = [float(x) for x in (acc['shift'] or [0.0, 0.0])]
    t0 = time.time()
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    N_EXP = len(pts_e)
    Ie_b = {sg: blur_sig(Ie, sg) for sg in (0, 2)}
    mask = _support_mask(Ie)
    windows, _ = make_windows(Ie.shape[0], 56)

    def win_metrics(Im):
        Im_b = {sg: blur_sig(Im, sg) for sg in (0, 2)}
        v56 = []
        for sl in windows:
            if mask[sl].mean() < 0.5:
                continue
            for sg in (0, 2):
                v56.append((sg, pcc(Ie_b[sg][sl], Im_b[sg][sl])))
        s2 = [v for sg, v in v56 if sg == 2]
        return (float(np.mean([v for _, v in v56])),
                float(np.mean([v < 0.5 for v in s2])),
                float(np.min(s2)))

    def ev(theta, sx, sy, shiftv):
        Im = render_v2(combo, theta, sx, sy, shiftv)
        C2 = 0.5 * (pcc(Ie, Im) + pcc(Ie_b[2], blur_sig(Im, 2)))
        wavg, rho56, cmin56 = win_metrics(Im)
        pts_m = find_dark_points(Im)
        mt = match_points(pts_e, pts_m)
        dists = np.array(mt['dists']) if mt['dists'] else np.array([99.])
        struct = (abs(len(pts_m) - N_EXP) <= 1
                  and mt['mean'] is not None and mt['mean'] < 12)
        max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
        return {'C_s02': C2, 'W_avg': wavg, 'rho56': rho56,
                'cmin56': cmin56, 'theta': list(theta),
                'sx': list(sx), 'sy': list(sy),
                'shift': list(shiftv), 'cores': len(pts_m),
                'mean_nn': float(mt['mean']) if mt['mean'] is not None
                else 99., 'max10': max10, 'struct': struct,
                'dists': [float(x) for x in dists]}

    print(f'\n=== {CN} ({NAME}) N={N} ===')
    base = ev(theta0, list(iso_total), list(iso_total), gshift)
    margin = max(14.0, base['max10'])
    print(f'  v1 terminal (v2 pipeline): C_s02={base["C_s02"]:.4f} '
          f'W_avg={base["W_avg"]:.4f} cores={base["cores"]}/{N_EXP} '
          f'max10={base["max10"]:.2f}')
    best = base
    rng = np.random.default_rng(1111)
    cur = base
    for rnd in range(2):
        # theta jitter
        for k in range(150):
            u = np.array(cur['theta'][:N]) + rng.normal(0, 0.010, N)
            u = np.clip(u, 0.005, None)
            u = u / u.sum()
            ph = np.array(cur['theta'][N:]) + rng.normal(0, 0.035,
                                                         N - 1)
            r = ev(list(u) + list(ph), cur['sx'], cur['sy'],
                   cur['shift'])
            if r['struct'] and r['max10'] <= margin \
                    and r['C_s02'] > best['C_s02']:
                best = r
        cur = best
        # global shift coarse->fine
        for step, rad in [(1.0, 3.0), (0.25, 0.75)]:
            for dy in np.arange(-rad, rad + 1e-9, step):
                for dx in np.arange(-rad, rad + 1e-9, step):
                    sh = [cur['shift'][0] + dy, cur['shift'][1] + dx]
                    r = ev(cur['theta'], cur['sx'], cur['sy'], sh)
                    if r['struct'] and r['max10'] <= margin \
                            and r['C_s02'] > best['C_s02']:
                        best = r
            cur = best
        # per-mode anisotropic scans
        for step, rad in [(0.02, 0.08), (0.01, 0.03)]:
            for i in range(N):
                for ax in (0, 1):
                    key = 'sx' if ax == 0 else 'sy'
                    for d in np.arange(-rad, rad + 1e-9, step):
                        sxv = list(cur['sx'])
                        syv = list(cur['sy'])
                        (sxv if ax == 0 else syv)[i] = float(
                            np.clip(cur[key][i] + d, 0.7, 1.4))
                        r = ev(cur['theta'], sxv, syv, cur['shift'])
                        if r['struct'] and r['max10'] <= margin \
                                and r['C_s02'] > best['C_s02']:
                            best = r
                    cur = best
        print(f'  [round{rnd}] C_s02={best["C_s02"]:.4f} '
              f'W_avg={best["W_avg"]:.4f} max10={best["max10"]:.2f} '
              f'shift=({best["shift"][0]:+.2f},{best["shift"][1]:+.2f}) '
              f'({time.time()-t0:.0f}s)')

    improved = (best['C_s02'] >= base['C_s02'] + 0.001
                and best['W_avg'] >= base['W_avg'] - 0.005)
    accepted = best if improved else base
    results[CN] = {'name': NAME, 'cn': CN, 'combo': combo, 'N': N,
                   'baseline_v2': base, 'refined': best,
                   'improved': bool(improved), 'margin': margin,
                   'accepted': accepted}
    print(f'  => {"IMPROVED" if improved else "no change"} '
          f'({base["C_s02"]:.4f} -> {best["C_s02"]:.4f}, '
          f'{time.time()-t0:.0f}s)')

(OUT / 'terminal_inspection_v2.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# summary
print('\n[SUMMARY]')
for CN, r in results.items():
    b, a = r['baseline_v2'], r['accepted']
    print(f"  {CN:8s} {b['C_s02']:.4f} -> {a['C_s02']:.4f} "
          f"({a['C_s02']-b['C_s02']:+.4f}) max10 {b['max10']:.1f}->"
          f"{a['max10']:.1f} struct={a['struct']}")
print(f'\n[done] {OUT}')

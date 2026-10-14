# -*- coding: utf-8 -*-
"""
probeH2b_mode7.py -- ③的管线化验证：给冠军加 N=7 伴模
=======================================================
③ 发现 HG(3,4) 需旋转 ~-3°；旋转在 HG-N7 子空间内可精确表达
（旋转 HG(3,4) = Σ_{k} c_k·HG_k（m+n=7））。标准管线中可用
“1 个伴模”近似：冠军 + 每个 N=7 模 逐个测试（5 模），
精修后与 ③ 的旋转臂（0.9038）及基线（0.8929）对比。
Outputs: 归档_八图分解/10奇点残差专项/probeH2b_mode7.json
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
from search_two_mode_joint import build_joint_modes
from refine_all_new_method import c_table_N
from rerun_new_metrics import (global_stats_02, c_s02, GRIDS,
                               block_refine_wavg, energy_simplex_N,
                               coarse_scan_N)
from search_three_mode import eval_c_grid
from window_metrics import build_window_stats, w_avg_point
from structural_check import find_dark_points, match_points
from stage2_5sing_local import crop_box, weight_map, weighted_crop_stats
from stage8_search import block_refine_hooked, prep_domain

OUT = ROOT / '归档_八图分解' / '10奇点残差专项'
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V2 = json.load(open(ROOT / '归档_八图分解' / '终检精修v2_2026-10-11'
                    / 'terminal_inspection_v2.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()

rec = V2['10奇点']
combo0, N0 = rec['combo'], rec['N']
acc = rec['accepted']
theta0 = [float(x) for x in acc['theta']]
sx0 = [float(x) for x in acc['sx']]
sy0 = [float(x) for x in acc['sy']]
shift0 = [float(x) for x in acc['shift']]

Ie = load_experiment(f'{IMG}/10Singularities.png', 'nipy_spectral',
                     'isot')
Ie_t = torch.tensor(Ie, dtype=torch.float32)
pts_e = find_dark_points(Ie)
N_EXP = len(pts_e)
crop_f = crop_box(pts_e, 224)
w_f = weight_map(Ie.shape, pts_e)

PARTNERS = ['HG(0,7)', 'HG(1,6)', 'HG(2,5)', 'HG(4,3)', 'HG(5,2)',
            'HG(6,1)', 'HG(7,0)']
print(f'[setup] champion {"+".join(combo0)}; testing {len(PARTNERS)} '
      f'N=7 partners')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def struct_eval(I):
    pts_m = find_dark_points(I)
    mt = match_points(pts_e, pts_m)
    dists = np.array(mt['dists']) if mt['dists'] else np.array([99.])
    struct = (abs(len(pts_m) - N_EXP) <= 1
              and mt['mean'] is not None and mt['mean'] < 12)
    max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
    return struct, len(pts_m), float(mt['mean']) \
        if mt['mean'] is not None else 99., max10


# ---- rotation seed: rotate HG(3,4) by -3.25°, project on N=7 modes ----
from scipy.ndimage import rotate as ndrotate
g34 = modes_np[lab2k['HG(3,4)']]
g34r = ndrotate(g34.real, -3.25, reshape=False, order=1) + \
    1j * ndrotate(g34.imag, -3.25, reshape=False, order=1)
n7 = [l for l in labels if l.startswith('HG') and
      sum(map(int, l[3:-1].split(','))) == 7]
Bm = np.stack([modes_np[lab2k[l]].ravel() for l in n7], axis=1)
coef, *_ = np.linalg.lstsq(Bm, g34r.ravel(), rcond=None)
rot_mix = {n7[k]: float(coef[k]) for k in range(len(n7))}
print('[rot seed] rotate HG(3,4) by -3.25deg -> N7 proj: '
      + ', '.join(f'{k}:{v:+.3f}' for k, v in rot_mix.items()))

rows = []
for part in PARTNERS:
    combo = combo0 + [part]
    N = N0 + 1
    tr = [lab2k[m] for m in combo]
    # seed: partner carries a small energy; rest scaled from champion
    u0 = theta0[:N0]
    part_e = 0.10 if abs(rot_mix.get(part, 0.0)) > 0.5 else 0.04
    u5 = (1 - part_e) * np.asarray(u0)
    seed = list(u5) + [part_e] + list(theta0[N0:]) + [0.0]
    seed = list(np.array(seed[:N]) / np.sum(seed[:N])) \
        + list(seed[N:])
    gs = global_stats_02(modes, Ie_t, tr, None)
    ws, meta, mask = build_window_stats(modes, Ie_t, tr, None)
    ws_cw = weighted_crop_stats(modes, Ie_t, tr, None, crop_f, w_f)
    g = GRIDS[4]

    def c_score5(pmat, gs=gs):
        C = c_table_N(np.asarray(pmat, dtype=np.float64), N)
        return (sum(eval_c_grid(C, *s) for s in gs) / len(gs))[0]

    t0 = time.time()
    th_cw, _ = block_refine_wavg(ws_cw, seed, N, 11, 9, rounds=5)
    th_w, _ = block_refine_wavg(ws, th_cw, N, 11, 9, rounds=6)
    best = {'th': th_w, 'val': None}

    def rec_eval(cur):
        C2, (v0, v2) = c_s02(gs, cur, N)
        if C2 > (best['val'] or -9):
            best['th'] = list(cur)
            best['val'] = C2

    rec_eval(th_w)
    block_refine_hooked(c_score5, best['th'], N, 11, 9, 6,
                        lambda r, cur: rec_eval(cur))
    th = best['th']
    C2, (v0, v2) = c_s02(gs, th, N)

    # render + structure check (with v2 transforms)
    E = np.zeros(modes_np[0].shape, dtype=np.complex128)
    for i, m in enumerate(combo):
        Fm = modes_np[lab2k[m]]
        s_i = sx0[i] if i < N0 else 1.0
        sy_i = sy0[i] if i < N0 else 1.0
        if abs(s_i - 1) > 1e-6 or abs(sy_i - 1) > 1e-6:
            Fm = zoom_axis_c(Fm, s_i, 1)
            Fm = zoom_axis_c(Fm, sy_i, 0)
        phi = 0.0 if i == 0 else th[N + i - 1]
        E = E + np.sqrt(max(th[i], 0)) * Fm * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    from scipy.ndimage import shift as ndshift
    I = ndshift(I, shift0, order=1, mode='nearest')
    struct, nc, mean_nn, max10 = struct_eval(I)
    e5 = th[N0]
    rows.append({'partner': part, 'C_s02': float(C2),
                 'partner_energy': float(e5), 'struct': struct,
                 'cores': nc, 'mean_nn': mean_nn, 'max10': max10,
                 'time': round(time.time() - t0, 1)})
    print(f'  +{part:9s} C_s02={C2:.4f} e5={e5*100:.1f}% struct={struct} '
          f'cores={nc}/{N_EXP} max10={max10:.2f} ({time.time()-t0:.0f}s)')

rows.sort(key=lambda r: -r['C_s02'])
(OUT / 'probeH2b_mode7.json').write_text(
    json.dumps({'baseline': 0.8929, 'rotp_arm': 0.9038,
                'rot_seed_mix': rot_mix, 'results': rows},
               ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print('\n=== ranked ===')
for r in rows:
    print(f"  {r['partner']:9s} C_s02={r['C_s02']:.4f} "
          f"(e5={r['partner_energy']*100:.1f}%) struct={r['struct']}")
print(f"[best] {rows[0]['partner']} {rows[0]['C_s02']:.4f} vs "
      f"rot-arm 0.9038 vs baseline 0.8929")

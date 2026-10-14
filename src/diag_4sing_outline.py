# -*- coding: utf-8 -*-
"""
diag_4sing_outline.py -- 4奇点外轮廓差异定量诊断
====================================================
回答："4奇点模型在大体轮廓上为什么不像实验图？"
拆解四个层面：
  D1 角向半径函数 r(θ)：从质心出发的轮廓半径曲线 + 傅里叶分解
     （实验多瓣有机形 vs 模型星形——谁在 m=4/5/6 上有分量）
  D2 中央暗洞：模型中央方形暗区面积/深度 vs 实验中央区
  D3 边界弥散度：轮廓处径向梯度（实验柔和渐变 vs 模型锐利）
  D4 新旧两个组合的并排定量对比（旧 LG(0,1)+LG(1,1)+LG(0,5)
     vs 新 LG(0,-4)+LG(2,0)+HG(6,4)）——验证"换组合是否换对了轮廓"

Outputs: 归档_八图分解/4奇点_轮廓诊断/
"""
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, binary_fill_holes

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from calibrate_metrics import render_model
from structural_check import find_dark_points

OUT = ROOT / '归档_八图分解'
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()

# ---- load images ----
Ie = load_experiment(f'{IMG}/4Singularities.png', 'nipy_spectral', 'isot')

# new (v3 accepted = v2 baseline since no improvement)
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))
acc = V3['4奇点']['accepted']
theta = [float(x) for x in acc['theta']]
# NOTE: v3 render pipeline applies per-mode aniso zoom; use v2-path
# values from the record (sx/sy) — for the outline diagnosis we render
# with the SAME simple path as render_model for the two raw combos,
# and additionally with v2 aniso via stage12c helpers is overkill here.
# We render three variants:
#   new_raw  : LG(0,-4)+LG(2,0)+HG(6,4), theta from v3 accepted
#   old_raw  : LG(0,1)+LG(1,1)+LG(0,5), theta from 最新结果 folder
from stage0_diagnosis import parse_md
combo_old, theta_old, scale_old = parse_md(
    ROOT / '归档_八图分解' / '最新结果' / '4奇点_参数与指标.md')
combo_new = V3['4奇点']['combo']

I_new = render_model(modes_np, lab2k,
                     {'combo': combo_new, 'params': theta,
                      'scale': None})
I_old = render_model(modes_np, lab2k,
                     {'combo': combo_old, 'params': theta_old,
                      'scale': scale_old})


def centroid(I):
    ys, xs = np.mgrid[0:I.shape[0], 0:I.shape[1]]
    w = np.clip(I, 0, None)
    t = w.sum()
    return (w * ys).sum() / t, (w * xs).sum() / t


def outline_r_theta(I, frac=0.18, nt=720):
    """r(θ): for each angle, the radius of the outermost bright ridge
    (last local maximum above frac*max along the ray). Robust to deep
    interior dark holes which break threshold-crossing trackers."""
    cy, cx = centroid(I)
    thetas = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    thr = frac * I.max()
    rs = []
    rmax = 120.0
    for th in thetas:
        rr = np.arange(0, rmax, 0.5)
        ys = cy + rr * np.sin(th)
        xs = cx + rr * np.cos(th)
        vals = map_coordinates(I, [ys, xs], order=1, mode='constant',
                               cval=0)
        # outermost radius where the ray is still bright (crossing down)
        above = vals >= thr
        if not above.any():
            rs.append(np.nan)
            continue
        idx = np.where(above)[0][-1]
        rs.append(rr[idx])
    return thetas, np.array(rs), (cy, cx)


def fourier_r(r):
    n = len(r)
    F = np.fft.fft(r - r.mean()) / n
    return np.abs(F)


def blur(I, sg):
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), float(sg))[0].numpy()


def boundary_stats(I, lo=0.1, hi=0.5):
    """edge softness: radial gradient magnitude along the contour band."""
    cy, cx = centroid(I)
    ys, xs = np.mgrid[0:I.shape[0], 0:I.shape[1]]
    rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    band = (I > lo * I.max()) & (I < hi * I.max())
    gy, gx = np.gradient(I)
    g = np.sqrt(gy ** 2 + gx ** 2)
    # normalized gradient: how fast intensity crosses the band
    width = np.sqrt(band.sum())  # ~ characteristic band width
    return {'band_area_px': int(band.sum()),
            'mean_grad_in_band': float(g[band].mean()),
            'rms_grad_in_band': float(np.sqrt((g[band] ** 2).mean()))}


def central_hole_stats(I, r_in=30):
    cy, cx = centroid(I)
    ys, xs = np.mgrid[0:I.shape[0], 0:I.shape[1]]
    rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    cen = rr <= r_in
    return {'center_mean': float(I[cen].mean()),
            'center_min': float(I[cen].min()),
            'center_frac_dark10': float((I[cen] < 0.1 * I.max()).mean())}


results = {}
for tag, I in [('experiment', Ie), ('model_new', I_new),
               ('model_old', I_old)]:
    th, r, c = outline_r_theta(I)
    Fr = fourier_r(r)
    bs = boundary_stats(I)
    ch = central_hole_stats(I)
    pts = find_dark_points(I)
    results[tag] = {
        'r_theta': r, 'thetas': th, 'centroid': c, 'Fr': Fr,
        'mean_r': float(r.mean()), 'cv_r': float(r.std() / r.mean()),
        'boundary': bs, 'central': ch, 'n_dark': len(pts),
        'Fr_ratio': [float(Fr[m] / Fr[0].clip(min=1e-9))
                     for m in range(1, 13)]}

print('=== r(θ) 统计 ===')
for tag in results:
    r = results[tag]
    print(f"{tag:12s} mean_r={r['mean_r']:6.1f}px "
          f"cv_r={r['cv_r']:.3f} "
          f"中心: mean={r['central']['center_mean']:.3f} "
          f"dark10%={r['central']['center_frac_dark10']:.2f} "
          f"| 边界带面积={r['boundary']['band_area_px']} "
          f"grad={r['boundary']['mean_grad_in_band']:.4f}")
print()
print('=== 轮廓傅里叶分量 |A_m|/mean_r ===')
hdr = '      ' + ''.join(f'{m:7d}' for m in range(1, 13))
print(hdr)
for tag in results:
    r = results[tag]
    rr = r['r_theta']
    vals = [np.abs(np.fft.fft(rr - rr.mean()))[m] / rr.mean()
            for m in range(1, 13)]
    print(f'{tag:12s}' + ''.join(f'{v:7.3f}' for v in vals))

# dark-core positions for context
for tag, I in [('experiment', Ie), ('model_new', I_new),
               ('model_old', I_old)]:
    pts = find_dark_points(I)
    pos = [(int(y), int(x)) for y, x, _ in pts]
    print(f'{tag}: {len(pts)} dark cores at {pos}')

# ---- figure ----
fig, axs = plt.subplots(2, 4, figsize=(21, 10.5))
imgs = [('experiment', Ie), ('model_new', I_new), ('model_old', I_old)]
for j, (tag, I) in enumerate(imgs):
    axs[0, j].imshow(blur(I, 2), cmap='inferno', vmin=0, vmax=1)
    th = results[tag]['thetas']
    r = results[tag]['r_theta']
    cy, cx = results[tag]['centroid']
    axs[0, j].plot(cx + r * np.cos(th), cy + r * np.sin(th), 'c-',
                   lw=1.2)
    pts = find_dark_points(I)
    for y, x, _ in pts:
        axs[0, j].plot(x, y, 'o', mfc='none', mec='cyan', ms=10)
    axs[0, j].set_title(f'{tag} (sigma=2) + outline(18%) + cores')
    axs[0, j].axis('off')
# r(theta) polar
axp = fig.add_subplot(2, 4, 4, projection='polar')
for tag, col in [('experiment', 'red'), ('model_new', 'blue'),
                 ('model_old', 'gray')]:
    axp.plot(results[tag]['thetas'], results[tag]['r_theta'], col,
             label=tag)
axp.set_title('outline r(θ)')
axp.legend(fontsize=8, loc='lower right')
# fourier bars
axf = axs[1, 0]
w = 0.27
m_arr = np.arange(1, 13)
for k, (tag, col) in enumerate([('experiment', 'red'),
                                ('model_new', 'blue'),
                                ('model_old', 'gray')]):
    rr = results[tag]['r_theta']
    vals = [np.abs(np.fft.fft(rr - rr.mean()))[m] / rr.mean()
            for m in m_arr]
    axf.bar(m_arr + (k - 1) * w, vals, width=w, color=col, label=tag)
axf.set_xlabel('angular harmonic m')
axf.set_ylabel('|A_m| / mean_r')
axf.set_title('outline shape spectrum')
axf.legend(fontsize=8)
# radial profiles
axr = axs[1, 1]
for tag, col in [('experiment', 'red'), ('model_new', 'blue'),
                 ('model_old', 'gray')]:
    I = dict(experiment=Ie, model_new=I_new, model_old=I_old)[tag]
    cy, cx = results[tag]['centroid']
    ys, xs = np.mgrid[0:224, 0:224]
    rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    bins = np.arange(0, 115, 4)
    prof = [I[(rr >= b) & (rr < b + 4)].mean()
            if ((rr >= b) & (rr < b + 4)).any() else np.nan
            for b in bins]
    axr.plot(bins + 2, prof, col, label=tag)
axr.set_xlabel('r (px)')
axr.set_ylabel('mean I')
axr.set_title('radial profile')
axr.legend(fontsize=8)
# central crop comparison
crop = (slice(70, 155), slice(70, 155))
for j, (tag, I) in enumerate([('experiment', Ie), ('model_new', I_new)]):
    axs[1, 2 + j].imshow(blur(I, 2)[crop], cmap='inferno', vmin=0,
                         vmax=1)
    axs[1, 2 + j].set_title(f'{tag} center crop')
    axs[1, 2 + j].axis('off')
fig.suptitle('4Singularities outline diagnosis: experiment vs '
             'model_new (LG(0,-4)+LG(2,0)+HG(6,4)) vs model_old '
             '(LG(0,1)+LG(1,1)+LG(0,5))', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / '诊断_4奇点_轮廓诊断.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)

# dump json (arrays shortened)
dump = {t: {k: v for k, v in results[t].items()
            if k not in ('r_theta', 'thetas', 'Fr')}
        for t in results}
(OUT / '诊断_4奇点_轮廓诊断.json').write_text(
    json.dumps(dump, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print(f'\n[done] {OUT}')

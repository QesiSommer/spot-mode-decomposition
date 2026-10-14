# -*- coding: utf-8 -*-
"""
probeH1_residual_anatomy.py -- ① 残差解剖
============================================
把 10 奇点 v2 终检解的残差 |Ie - Im| 当作研究对象：
  a. 空间分布图（含误差定位）；
  b. 径向功率剖面（残差集中在哪个半径带）；
  c. 角向谱（奇/偶 m 阶、与实验奇阶 4.8 的对照）；
  d. 基投影：把残差线性投影到 {|Ui|^2, 2Re[UiUj*]} 基与低阶多项式上，
     看残差"想表达"什么；
  e. 横向对照：8(2) 与 11 奇点的残差奇阶（该残差是否为 10 奇点独有）。
Outputs: 归档_八图分解/10奇点残差专项/probeH1_*
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from calibrate_metrics import angular_distance
from verify_metrics import diff_localization
from window_metrics import _support_mask

OUT = ROOT / '归档_八图分解' / '10奇点残差专项'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V2 = json.load(open(ROOT / '归档_八图分解' / '终检精修v2_2026-10-11'
                    / 'terminal_inspection_v2.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()


def _fit_center(a, H, W):
    out = np.zeros((a.shape[0], H, W), np.float64)
    h, w = a.shape[-2], a.shape[-1]
    if h >= H:
        sys_ = (h - H) // 2
        sys_s, sys_d = slice(sys_, sys_ + H), slice(0, H)
    else:
        sys_ = (H - h) // 2
        sys_s, sys_d = slice(0, h), slice(sys_, sys_ + h)
    if w >= W:
        sxs_ = (w - W) // 2
        sxs_s, sxs_d = slice(sxs_, sxs_ + W), slice(0, W)
    else:
        sxs_ = (W - w) // 2
        sxs_s, sxs_d = slice(0, w), slice(sxs_, sxs_ + w)
    out[:, sys_d, sxs_d] = a[:, sys_s, sxs_s]
    return out


def zoom_axis_c(Fx, s, axis):
    if abs(s - 1.0) < 1e-6:
        return Fx
    H, W = Fx.shape
    t = np.stack([Fx.real, Fx.imag], 0)[None]
    t = torch.tensor(t, dtype=torch.float32)
    Hn = int(round(H * s)) if axis == 0 else H
    Wn = int(round(W * s)) if axis == 1 else W
    t = torch.nn.functional.interpolate(
        t, size=(Hn, Wn), mode='bilinear',
        align_corners=False)[0].numpy().astype(np.float64)
    b = _fit_center(t, H, W)
    return b[0] + 1j * b[1]


def mode_fields(combo, theta, sx, sy):
    """per-mode complex fields U_i with anisotropic zoom (no shift)."""
    N = len(combo)
    Us = []
    for i, m in enumerate(combo):
        Fm = modes_np[lab2k[m]]
        if abs(sx[i] - 1) > 1e-6 or abs(sy[i] - 1) > 1e-6:
            Fm = zoom_axis_c(Fm, sx[i], 1)
            Fm = zoom_axis_c(Fm, sy[i], 0)
        phi = 0.0 if i == 0 else theta[N + i - 1]
        Us.append(np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi))
    return Us


def render_v2(combo, theta, sx, sy, shiftv):
    E = sum(mode_fields(combo, theta, sx, sy))
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    if shiftv is not None and (abs(shiftv[0]) > 1e-6
                               or abs(shiftv[1]) > 1e-6):
        from scipy.ndimage import shift as ndshift
        I = ndshift(I, shiftv, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


def odd_amp(a):
    return float(np.sum(a[1::2]))


# ---------------- main: 10-sing anatomy ----------------
CN = '10奇点'
rec = V2[CN]
combo, N = rec['combo'], rec['N']
acc = rec['accepted']
theta = [float(x) for x in acc['theta']]
sx = [float(x) for x in acc['sx']]
sy = [float(x) for x in acc['sy']]
shiftv = [float(x) for x in acc['shift']]

Ie = load_experiment(f'{IMG}/{rec["name"]}.png', 'nipy_spectral',
                     'isot')
Im = render_v2(combo, theta, sx, sy, shiftv)
R = Ie - Im
mask = _support_mask(Ie)

_, (ae, am) = angular_distance(Ie, Im)
_, (_, ar) = angular_distance(Ie, R)
conc, loc = diff_localization(Ie, blur := gaussian_blur(
    torch.tensor(Im, dtype=torch.float32).unsqueeze(0), 2.0)[0].numpy(),
    mask)

# radial profile of |R|
H = Ie.shape[0]
yy, xx = np.mgrid[0:H, 0:H]
cy = cx = (H - 1) / 2.0
rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
bins = np.arange(0, 112, 8)
rad_prof = []
for b0, b1 in zip(bins[:-1], bins[1:]):
    sel = (rr >= b0) & (rr < b1)
    if sel.any():
        rad_prof.append((float(b0), float(np.abs(R[sel]).mean()),
                         float(Ie[sel].mean()), float(Im[sel].mean())))

# basis projection: U_i products + polynomial terms
Us = mode_fields(combo, theta, sx, sy)
basis, names_ = [], []
for i in range(N):
    basis.append(np.abs(Us[i]) ** 2)
    names_.append(f'|U{i+1}|^2')
for i in range(N):
    for j in range(i + 1, N):
        basis.append(2 * np.real(Us[i] * np.conj(Us[j])))
        names_.append(f'X{i+1}{j+1}')
xn = (xx - cx) / 112.0
yn = (yy - cy) / 112.0
for nm, img in [('x', xn), ('y', yn), ('x2', xn**2), ('y2', yn**2),
                ('xy', xn * yn), ('x3', xn**3), ('y3', yn**3),
                ('x2y', xn**2 * yn), ('xy2', xn * yn**2)]:
    basis.append(img)
    names_.append('P_' + nm)
B = np.stack([b[mask] for b in basis], axis=1)
rvec = R[mask]
coef, *_ = np.linalg.lstsq(B, rvec, rcond=None)
proj = B @ coef
r2 = 1 - ((rvec - proj) ** 2).sum() / (rvec ** 2).sum()
order = np.argsort(-np.abs(coef))

print(f'[10奇点] residual anatomy:')
print(f'  residual rms={np.sqrt((R**2).mean()):.4f} max={np.abs(R).max():.4f}')
print(f'  exp odd-amp={odd_amp(ae):.3f} model odd-amp={odd_amp(am):.3f}')
print(f'  residual odd-amp={odd_amp(ar):.3f}')
print(f'  err concentration(top3)={conc*100:.1f}%')
print('  top basis projections:')
for k in order[:8]:
    print(f'    {names_[k]:8s} coef={coef[k]:+.4f}')
print(f'  basis+poly R^2 (variance explained) = {r2:.3f}')

# cross-image comparison of residual odd-amp
cross = {}
for cn2 in ['8奇点2', '11奇点']:
    r2_ = V2[cn2]
    a2 = r2_['accepted']
    Ie2 = load_experiment(f'{IMG}/{r2_["name"]}.png', 'nipy_spectral',
                          'isot')
    Im2 = render_v2(r2_['combo'], [float(x) for x in a2['theta']],
                    [float(x) for x in a2['sx']],
                    [float(x) for x in a2['sy']],
                    [float(x) for x in a2['shift']])
    _, (ae2, am2) = angular_distance(Ie2, Im2)
    _, (_, ar2) = angular_distance(Ie2, Ie2 - Im2)
    cross[cn2] = {'exp_odd': odd_amp(ae2), 'model_odd': odd_amp(am2),
                  'resid_odd': odd_amp(ar2)}
    print(f'[{cn2}] exp odd={odd_amp(ae2):.2f} model odd='
          f'{odd_amp(am2):.2f} resid odd={odd_amp(ar2):.2f}')

summary = {
    'image': rec['name'], 'combo': combo,
    'resid_rms': float(np.sqrt((R ** 2).mean())),
    'resid_max': float(np.abs(R).max()),
    'exp_odd': odd_amp(ae), 'model_odd': odd_amp(am),
    'resid_odd': odd_amp(ar),
    'exp_spec': [float(x) for x in ae],
    'model_spec': [float(x) for x in am],
    'resid_spec': [float(x) for x in ar],
    'radial_profile': rad_prof,
    'err_concentration_top3': conc,
    'err_locations': loc,
    'basis_coefs': {names_[k]: float(coef[k]) for k in order},
    'basis_poly_R2': float(r2),
    'cross_images': cross}
(OUT / 'probeH1_residual_anatomy.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# ---------------- figure ----------------
fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
axs[0, 0].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
axs[0, 0].set_title('experiment')
axs[0, 1].imshow(blur, cmap='inferno', vmin=0, vmax=1)
axs[0, 1].set_title('model v2 (sigma=2)')
im2 = axs[0, 2].imshow(R, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
axs[0, 2].set_title('signed residual (exp - model)')
plt.colorbar(im2, ax=axs[0, 2], fraction=0.046)
m = np.arange(11)
axs[1, 0].bar(m - 0.2, ae, width=0.4, label='experiment', color='#c66')
axs[1, 0].bar(m + 0.2, am, width=0.4, label='model', color='#68a')
axs[1, 0].set_title(f'azimuthal spectra (exp odd='
                    f'{odd_amp(ae):.2f}, model odd={odd_amp(am):.2f})')
axs[1, 0].legend(fontsize=8)
axs[1, 0].set_xlabel('m')
bp = np.array([p[1] for p in rad_prof])
b0s = np.array([p[0] for p in rad_prof])
axs[1, 1].plot(b0s + 4, bp, 'o-', color='#b33')
axs[1, 1].set_title('radial profile of mean |residual|')
axs[1, 1].set_xlabel('radius (px)')
k_show = order[:8][::-1]
axs[1, 2].barh([names_[k] for k in k_show], [coef[k] for k in k_show],
               color='#27a')
axs[1, 2].set_title(f'residual projection on basis (R2={r2:.2f})')
for a_ in axs.ravel():
    a_.tick_params(labelsize=8)
fig.suptitle('10Singularities residual anatomy (v2 solution)', fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'probeH1_残差解剖.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'[done] {OUT}')

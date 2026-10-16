# -*- coding: utf-8 -*-
r"""
refit_8sing2_std.py — 8奇点(2) 标准记号新基重新拟合（2026-10-16）
====================================================================
背景：约定对齐（基轴序修正）后，8奇点(2) 的标准记号解 =
    HG(4,0)+HG(3,1)+HG(1,3)+HG(1,5)   （内部记号 HG(0,4)+HG(1,3)+HG(3,1)+HG(5,1)）

本脚本在修正后的新基下：
  A) 校验渲染：换算后的 v3 全参数（theta+逐模 sx/sy/rot/shift）经
     render_dof 渲染，C_s02 应逐位复现交付值 0.9384；
  B) 真重拟合：从换算种子出发，对全部自由度（能量/相位/逐模
     sx/sy/rot/全局平移）做坐标下降精修（C_s02 目标），
     确认收敛到同一最优（基修正仅是标签映射，物理场不变）；
  C) 产出交付口径五联图 + 参数明细图 + JSON。

输出：归档_八图分解\8奇点2_标准记号重拟合_2026-10-16\
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
from scipy.ndimage import zoom as ndzoom, rotate as ndrotate, shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = ROOT / '归档_八图分解' / '8奇点2_标准记号重拟合_2026-10-16'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '8Singularities(2)'
CN = '8奇点2'
SIZE, EXTENT = 224, 4.0
COMBO_STD = ['HG(4,0)', 'HG(3,1)', 'HG(1,3)', 'HG(1,5)']
COMBO_INT = ['HG(0,4)', 'HG(1,3)', 'HG(3,1)', 'HG(5,1)']
N = 4
C_EXPECT = 0.9384

V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))
r3 = V3[CN]
src3 = r3['accepted'] if 'theta' in r3.get('accepted', {}) \
    else r3['refined_v3']
theta0 = [float(x) for x in src3['theta']]
sx0 = [float(x) for x in src3['sx']]
sy0 = [float(x) for x in src3['sy']]
shift0 = [float(x) for x in src3['shift']]          # [dy, dx]
rot0 = [float(x) for x in src3.get('rot', [0.0] * N)]
# 全 HG 组合：换算不改任何数值（相位补偿 = 0）

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('building joint modes (post-fix) ...')
modes, labels = build_joint_modes(10, SIZE, EXTENT, 'cpu')
modes_np = modes.numpy()
lab2k = {l: k for k, l in enumerate(labels)}
fields0 = [modes_np[lab2k[m]] for m in COMBO_STD]

Ie_np = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
Ie_t = torch.tensor(Ie_np, dtype=torch.float32)
Ie_b_np = gaussian_blur(Ie_t.unsqueeze(0), 2.0)[0].numpy()
pts_e = find_dark_points(Ie_np)
print(f'  experiment cores: {len(pts_e)}')


def pcc(a, b):
    a = np.asarray(a).ravel() - np.asarray(a).mean()
    b = np.asarray(b).ravel() - np.asarray(b).mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else -1.0


def render_dof(fields, theta, sx, sy, shiftv, rot):
    E = np.zeros((SIZE, SIZE), dtype=np.complex128)
    n = len(fields)
    for i, Fm in enumerate(fields):
        if abs(sx[i] - 1.0) > 1e-9:
            Fm = ndzoom(Fm, (1.0, sx[i]), order=1) if False else \
                _zoom_axis(Fm, sx[i], axis=1)
        if abs(sy[i] - 1.0) > 1e-9:
            Fm = _zoom_axis(Fm, sy[i], axis=0)
        if abs(rot[i]) > 1e-9:
            re = ndrotate(Fm.real, rot[i], reshape=False, order=1)
            im = ndrotate(Fm.imag, rot[i], reshape=False, order=1)
            Fm = re + 1j * im
        phi = 0.0 if i == 0 else theta[n + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-30)
    I = ndshift(I, shiftv, order=1, mode='nearest')
    return np.clip(I, 0, None)


def _fit_center(a, H, W):
    """同 stage12c_v3_hgonly.py:61-77 —— 缩放后中心裁剪/补零回 (H,W)。"""
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


def _zoom_axis(F, s, axis):
    """同 stage12c_v3_hgonly.py:80-92 zoom_axis_c —— 双线性缩放到
    round(H*s) 再中心裁剪回 224（交付渲染同径路径）。"""
    if abs(s - 1.0) < 1e-6:
        return F
    Hh, Ww = F.shape
    t = np.stack([F.real, F.imag], 0)[None]
    t = torch.tensor(t, dtype=torch.float32)
    Hn = int(round(Hh * s)) if axis == 0 else Hh
    Wn = int(round(Ww * s)) if axis == 1 else Ww
    t = torch.nn.functional.interpolate(
        t, size=(Hn, Wn), mode='bilinear',
        align_corners=False)[0].numpy().astype(np.float64)
    b = _fit_center(t, Hh, Ww)
    return b[0] + 1j * b[1]


def C_s02_of(I):
    return 0.5 * (pcc(Ie_np, I) + pcc(Ie_b_np,
                 gaussian_blur(torch.tensor(I, dtype=torch.float32)
                               .unsqueeze(0), 2.0)[0].numpy()))


def full_C(params):
    th, sx, sy, sh, ro = params
    return C_s02_of(render_dof(fields0, th, sx, sy, sh, ro))


# ---- A) 校验渲染：应逐位复现 0.9384 ----
params0 = (list(theta0), list(sx0), list(sy0), list(shift0), list(rot0))
C_check = full_C(params0)
print(f'[check] 换算解+全dof 渲染 C_s02 = {C_check:.4f} '
      f'[交付值 {C_EXPECT}]  {"OK" if abs(C_check - C_EXPECT) < 5e-4 else "MISMATCH!"}')

# ---- B) 坐标下降重拟合（C_s02 目标，全自由度） ----
print('[refit] coordinate descent on (theta, sx, sy, shift, rot) ...')
t0 = time.time()
cur = tuple(list(v) for v in params0)
best = full_C(cur)
print(f'  seed C_s02 = {best:.4f}')


def try_axis(cur, best, which, idx, deltas):
    """对参数 (which, idx) 试一组增量，取最优。
    阈值 5e-4：拒绝噪声级改善（会破坏临界暗点结构）。"""
    th, sx, sy, sh, ro = cur
    base = {'th': th, 'sx': sx, 'sy': sy, 'sh': sh, 'ro': ro}[which]
    for d in deltas:
        v = base[idx] + d
        if which in ('sx', 'sy') and not (0.7 <= v <= 1.4):
            continue
        if which == 'ro' and not (-12 <= v <= 12):
            continue
        if which == 'sh' and not (-6 <= v <= 6):
            continue
        cand = [list(x) for x in cur]
        cand[{'th': 0, 'sx': 1, 'sy': 2, 'sh': 3, 'ro': 4}[which]][idx] = v
        c = full_C(tuple(cand))
        if c > best + 5e-4:
            return tuple(cand), c, True
    return cur, best, False


ROUNDS = 3
for rnd in range(ROUNDS):
    improved = False
    # 1) 全局平移（dy, dx）
    for idx in range(2):
        cur, best, ch = try_axis(cur, best, 'sh', idx,
                                 [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0])
        improved |= ch
    # 2) 逐模能量（保持归一化：调 u_i 后重归一）
    for idx in range(N):
        th, sx, sy, sh, ro = cur
        for d in (-0.03, -0.015, 0.015, 0.03):
            u = list(th)
            u[idx] = max(u[idx] + d, 0.01)
            s = sum(u[:N])
            u = [x / s for x in u[:N]] + u[N:]
            cand = (u, sx, sy, sh, ro)
            c = full_C(cand)
            if c > best + 5e-4:
                cur, best, improved = cand, c, True
                break
    # 3) 逐模相位
    for idx in range(1, N):
        cur, best, ch = try_axis(cur, best, 'th', N + idx - 1,
                                 [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20])
        improved |= ch
    # 4) 逐模 sx / sy / rot
    for idx in range(N):
        cur, best, ch = try_axis(cur, best, 'sx', idx,
                                 [-0.04, -0.02, 0.02, 0.04])
        improved |= ch
        cur, best, ch = try_axis(cur, best, 'sy', idx,
                                 [-0.04, -0.02, 0.02, 0.04])
        improved |= ch
        cur, best, ch = try_axis(cur, best, 'ro', idx,
                                 [-1.0, -0.5, 0.5, 1.0])
        improved |= ch
    print(f'  round {rnd+1}: C_s02 = {best:.4f} '
          f'({"improved" if improved else "converged"}) '
          f'[{time.time()-t0:.0f}s]')
    if not improved:
        break

th_r, sx_r, sy_r, sh_r, ro_r = cur
I_model = render_dof(fields0, th_r, sx_r, sy_r, sh_r, ro_r)
C_ref = C_s02_of(I_model)
print(f'[refit] final C_s02 = {C_ref:.4f} (check {C_check:.4f}, '
      f'delta {C_ref - C_check:+.4f})')

# ---- 结构验收 ----
pts_m = find_dark_points(I_model)
mt = match_points(pts_e, pts_m)
print(f'  cores: model {len(pts_m)} vs exp {len(pts_e)}; '
      f'mean_nn={mt["mean"]:.2f} max={mt["max"]:.2f}')


# ---- 56px 窗 PCC 热图（σ2，直接从最终渲染图计算） ----
def window_pcc_map(I, W=56):
    sup = Ie_np > 0.05 * Ie_np.max()
    sup = np.pad(sup, 8, constant_values=False)[8:-8, 8:-8]
    from scipy.ndimage import binary_dilation
    sup = binary_dilation(sup, iterations=8)
    I_b = gaussian_blur(torch.tensor(I, dtype=torch.float32)
                        .unsqueeze(0), 2.0)[0].numpy()
    side = SIZE // W
    hm = np.full((side, side), np.nan)
    for i in range(side):
        for j in range(side):
            sl = np.s_[i * W:(i + 1) * W, j * W:(j + 1) * W]
            if not sup[sl].any():
                continue
            hm[i, j] = pcc(Ie_b_np[sl], I_b[sl])
    return hm


hm = window_pcc_map(I_model)
hm_v3 = window_pcc_map(render_dof(fields0, *params0))

# ---- C) 五联图 ----
raw = plt.imread(f'{IMG}/{NAME}.png')
if raw.ndim == 3 and raw.shape[2] == 4:
    raw = raw[:, :, :3]
fig, axs = plt.subplots(1, 5, figsize=(21.5, 4.4))
axs[0].imshow(raw)
axs[0].set_title('实验原图')
axs[1].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
axs[1].set_title('实验（处理口径）')
axs[2].imshow(I_model, cmap='inferno', vmin=0, vmax=1)
axs[2].set_title(f'模型拟合（标准记号新基）\n'
                 f'{"+".join(COMBO_STD)}\n'
                 f'C_s02={C_ref:.4f}')
axs[3].imshow(np.abs(Ie_np - I_model), cmap='viridis', vmin=0, vmax=0.6)
axs[3].set_title('|diff|（固定标尺 0~0.6）')
cmap = matplotlib.colormaps['RdYlGn'].copy()
cmap.set_bad('0.7')
im = axs[4].imshow(hm, cmap=cmap, vmin=0.3, vmax=1.0)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        if not np.isnan(hm[i, j]):
            axs[4].text(j, i, f'{hm[i, j]:.2f}', ha='center',
                        va='center', fontsize=9)
axs[4].set_title('56px 窗 PCC（σ=2，支持区）')
for a in axs:
    a.axis('off')
fig.suptitle(f'{NAME}（{CN}）标准记号新基重拟合 — '
             f'核数 {len(pts_m)} vs {len(pts_e)}，'
             f'重拟合 C_s02={C_ref:.4f}（校验 {C_check:.4f}）',
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / f'{CN}_标准记号重拟合.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)
print('  [fig] 五联图 done')

# ---- 参数明细图 ----
fig2, axs2 = plt.subplots(1, 2, figsize=(12.5, 4.4))
ener = [u * 100 for u in th_r[:N]]
amps = [np.sqrt(max(u, 0)) for u in th_r[:N]]
bars = axs2[0].bar(range(N), ener, color='#4C72B0')
axs2[0].set_xticks(range(N))
axs2[0].set_xticklabels(COMBO_STD, rotation=12)
axs2[0].set_ylabel('能量占比 %')
axs2[0].set_title('各模式能量占比（重拟合后）')
for b, e, a in zip(bars, ener, amps):
    axs2[0].text(b.get_x() + b.get_width() / 2,
                 b.get_height() + 0.4, f'{e:.2f}%\n(a={a:.4f})',
                 ha='center', fontsize=9)
phs = [0.0] + list(th_r[N:])
axs2[1].bar(range(N), [np.degrees(p) for p in phs], color='#DD8452')
axs2[1].set_xticks(range(N))
axs2[1].set_xticklabels(COMBO_STD, rotation=12)
axs2[1].set_ylabel('相对相位 °')
axs2[1].set_title('相对相位（模式 1 = 0）')
for i, p in enumerate(phs):
    axs2[1].text(i, np.degrees(p) + (3 if p >= 0 else -10),
                 f'{np.degrees(p):+.1f}°', ha='center', fontsize=9)
fig2.suptitle(f'{CN} 标准记号重拟合 — 参数明细', fontsize=12)
fig2.tight_layout()
fig2.savefig(OUT / f'{CN}_重拟合参数明细.png', dpi=140,
             bbox_inches='tight')
plt.close(fig2)

# ---- JSON ----
rep = {
    'cn': CN, 'name': NAME,
    'basis': 'post-fix standard convention (hg_torch axis order fixed)',
    'combo_std': COMBO_STD, 'combo_internal': COMBO_INT,
    'check_C_s02_v3_params': C_check, 'expected': C_EXPECT,
    'refit_C_s02': C_ref,
    'refit_theta': th_r, 'refit_sx': sx_r, 'refit_sy': sy_r,
    'refit_shift_dydx': sh_r, 'refit_rot_deg': ro_r,
    'energy_pct': ener, 'amps': amps,
    'phi_deg': [np.degrees(p) for p in phs],
    'cores_model': len(pts_m), 'cores_exp': len(pts_e),
    'mean_nn': mt['mean'], 'max_nn': mt['max'],
    'core_dists': mt['dists'],
    'window_pcc_sigma2': {f'r{i}c{j}': (None if np.isnan(hm[i, j])
                          else round(float(hm[i, j]), 4))
                          for i in range(hm.shape[0])
                          for j in range(hm.shape[1])
                          if not np.isnan(hm[i, j])},
}
json.dump(rep, open(OUT / '重拟合结果.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print(f'\nDONE -> {OUT}')

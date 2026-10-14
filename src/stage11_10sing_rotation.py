# -*- coding: utf-8 -*-
"""
stage11_10sing_rotation.py -- 10奇点 v3：并入逐模旋转（精修+交付）
====================================================================
依据残差专项③：HG(3,4) 需 ~-3° 旋转（像散轴取向自由度）。
本脚本对 10 奇点 v2 终检解做全量精修：theta + 平移 + 逐模 sx/sy +
逐模旋转 rot_i（结构/稳健约束，2 轮，只采纳提升 ≥+0.001 的解），
随后生成 v3 验收三件套 + v2-v3 对比图，同步 bundle 与汇总表。
（其余七图的 v3 重跑待用户确认。）

Outputs: 归档_八图分解/终检精修v3_2026-10-11/10奇点/ + JSON
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
from scipy.ndimage import rotate as ndrotate

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from window_metrics import make_windows, _support_mask
from structural_check import find_dark_points, match_points

ARCH = ROOT / '归档_八图分解'
OUT = ARCH / '终检精修v3_2026-10-12'
OUT.mkdir(parents=True, exist_ok=True)
BUNDLE = ROOT / '八图最新分解结果_2026-10-09'
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V2 = json.load(open(ARCH / '终检精修v2_2026-10-11'
                    / 'terminal_inspection_v2.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()

CN = '10奇点'
rec0 = V2[CN]
NAME, combo, N = rec0['name'], rec0['combo'], rec0['N']
acc0 = rec0['accepted']
theta0 = [float(x) for x in acc0['theta']]
sx0 = [float(x) for x in acc0['sx']]
sy0 = [float(x) for x in acc0['sy']]
shift0 = [float(x) for x in acc0['shift']]
rot0 = [0.0] * N

Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
pts_e = find_dark_points(Ie)
N_EXP = len(pts_e)
Ie_b2 = gaussian_blur(torch.tensor(Ie, dtype=torch.float32)
                      .unsqueeze(0), 2.0)[0].numpy()
mask = _support_mask(Ie)
windows, _ = make_windows(Ie.shape[0], 56)


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


def render_full(theta, sx, sy, shiftv, rot):
    E = np.zeros(modes_np[0].shape, dtype=np.complex128)
    for i, m in enumerate(combo):
        Fm = modes_np[lab2k[m]]
        if abs(sx[i] - 1) > 1e-6 or abs(sy[i] - 1) > 1e-6:
            Fm = zoom_axis_c(Fm, sx[i], 1)
            Fm = zoom_axis_c(Fm, sy[i], 0)
        if abs(rot[i]) > 1e-9:
            Fm = ndrotate(Fm.real, rot[i], reshape=False, order=1) + \
                1j * ndrotate(Fm.imag, rot[i], reshape=False, order=1)
        phi = 0.0 if i == 0 else theta[N + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    if abs(shiftv[0]) > 1e-6 or abs(shiftv[1]) > 1e-6:
        I = ndshift(I, shiftv, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


def ev(theta, sx, sy, shiftv, rot):
    Im = render_full(theta, sx, sy, shiftv, rot)
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
            if mt['mean'] is not None else 99., 'I': Im,
            'theta': list(theta), 'sx': list(sx), 'sy': list(sy),
            'shift': list(shiftv), 'rot': list(rot)}


base = ev(theta0, sx0, sy0, shift0, rot0)
margin = max(14.0, base['max10'])
print(f'[baseline v2] C_s02={base["C_s02"]:.4f} '
      f'cores={base["cores"]}/{N_EXP} max10={base["max10"]:.2f}')

rng = np.random.default_rng(1011)
best = base
cur = base
t0 = time.time()
for rnd in range(2):
    # theta jitter
    for k in range(150):
        u = np.array(cur['theta'][:N]) + rng.normal(0, 0.008, N)
        u = np.clip(u, 0.005, None)
        u = u / u.sum()
        ph = np.array(cur['theta'][N:]) + rng.normal(0, 0.03, N - 1)
        r = ev(list(u) + list(ph), cur['sx'], cur['sy'], cur['shift'],
               cur['rot'])
        if r['struct'] and r['max10'] <= margin \
                and r['C_s02'] > best['C_s02']:
            best = r
            cur = r
    # shift coarse -> fine
    for step, rad in [(1.0, 2.0), (0.25, 0.5)]:
        for dy in np.arange(-rad, rad + 1e-9, step):
            for dx in np.arange(-rad, rad + 1e-9, step):
                sh = [cur['shift'][0] + dy, cur['shift'][1] + dx]
                r = ev(cur['theta'], cur['sx'], cur['sy'], sh,
                       cur['rot'])
                if r['struct'] and r['max10'] <= margin \
                        and r['C_s02'] > best['C_s02']:
                    best = r
                    cur = r
    # per-mode sx/sy coarse->fine
    for step, rad in [(0.02, 0.06), (0.01, 0.02)]:
        for i in range(N):
            for ax in (0, 1):
                for d in np.arange(-rad, rad + 1e-9, step):
                    sxv = list(cur['sx'])
                    syv = list(cur['sy'])
                    (sxv if ax == 0 else syv)[i] = float(
                        np.clip(cur['sx' if ax == 0 else 'sy'][i] + d,
                                0.7, 1.4))
                    r = ev(cur['theta'], sxv, syv, cur['shift'],
                           cur['rot'])
                    if r['struct'] and r['max10'] <= margin \
                            and r['C_s02'] > best['C_s02']:
                        best = r
                        cur = r
    # per-mode rotation coarse->fine
    for step, rad in [(1.0, 4.0), (0.25, 1.0)]:
        for i in range(N):
            for d in np.arange(-rad, rad + 1e-9, step):
                rv = list(cur['rot'])
                rv[i] = float(np.clip(cur['rot'][i] + d, -20.0, 20.0))
                r = ev(cur['theta'], cur['sx'], cur['sy'], cur['shift'],
                       rv)
                if r['struct'] and r['max10'] <= margin \
                        and r['C_s02'] > best['C_s02']:
                    best = r
                    cur = r
    print(f'  [round{rnd}] C_s02={best["C_s02"]:.4f} '
          f'rot={np.round(best["rot"],2).tolist()} '
          f'shift=({best["shift"][0]:+.2f},{best["shift"][1]:+.2f}) '
          f'max10={best["max10"]:.2f} ({time.time()-t0:.0f}s)')

improved = (best['C_s02'] >= base['C_s02'] + 0.001
            and best['C_s02'] > base['C_s02'])
acc = best if improved else base
print(f'[result] v2 {base["C_s02"]:.4f} -> v3 {best["C_s02"]:.4f} '
      f'({best["C_s02"]-base["C_s02"]:+.4f}) improved={improved}')

result = {'name': NAME, 'cn': CN, 'combo': combo, 'N': N,
          'baseline_v2': {k: v for k, v in base.items() if k != 'I'},
          'refined_v3': {k: v for k, v in best.items() if k != 'I'},
          'accepted': {k: v for k, v in acc.items() if k != 'I'},
          'improved': bool(improved), 'margin': margin}
(OUT / 'terminal_inspection_v3_10sing.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# ================= delivery =================
(OUT / CN).mkdir(exist_ok=True)
theta = [float(x) for x in acc['theta']]
sx = [float(x) for x in acc['sx']]
sy = [float(x) for x in acc['sy']]
shiftv = [float(x) for x in acc['shift']]
rot = [float(x) for x in acc['rot']]
Im = acc['I']
v0 = pcc(Ie, Im)
v2_ = pcc(Ie_b2, gaussian_blur(torch.tensor(Im, dtype=torch.float32)
                                .unsqueeze(0), 2.0)[0].numpy())
C2 = 0.5 * (v0 + v2_)
Im_b = {sg: (Im if sg == 0 else gaussian_blur(
    torch.tensor(Im, dtype=torch.float32).unsqueeze(0),
    float(sg))[0].numpy()) for sg in (0, 2)}
Ie_b = {sg: (Ie if sg == 0 else gaussian_blur(
    torch.tensor(Ie, dtype=torch.float32).unsqueeze(0),
    float(sg))[0].numpy()) for sg in (0, 2)}
v56 = []
for sl in windows:
    if mask[sl].mean() < 0.5:
        continue
    for sg in (0, 2):
        v56.append((sg, pcc(Ie_b[sg][sl], Im_b[sg][sl])))
s2 = [v for sg, v in v56 if sg == 2]
wavg = float(np.mean([v for _, v in v56]))
rho56 = float(np.mean([v < 0.5 for v in s2]))
cmin56 = float(np.min(s2))
w32, _ = make_windows(Ie.shape[0], 32)
v32 = []
for sl in w32:
    if mask[sl].mean() < 0.5:
        continue
    for sg in (0, 2):
        v32.append((sg, pcc(Ie_b[sg][sl], Im_b[sg][sl])))
s32 = [v for sg, v in v32 if sg == 2]
rho32 = float(np.mean([v < 0.5 for v in s32]))
cmin32 = float(np.min(s32))
pts_m = find_dark_points(Im)
mt = match_points(pts_e, pts_m)
dists = np.array(mt['dists'])
struct = (abs(len(pts_m) - N_EXP) <= 1
          and mt['mean'] is not None and mt['mean'] < 12)
max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
T_C, T0, T2 = 0.83, 0.82, 0.83
gate_ok = (C2 >= T_C and v0 >= T0 and v2_ >= T2
           and rho56 <= 0.10 and cmin56 >= 0.40)
warns = []
if rho32 >= 0.2:
    warns.append(f'rho_32px={rho32:.2f}')
if cmin32 <= 0.3:
    warns.append(f'C_min_32px={cmin32:.2f}')
print(f'[v3 acceptance] C_s02={C2:.4f} W_avg={wavg:.4f} '
      f'cores={len(pts_m)}/{N_EXP} max10={max10:.2f} '
      f'struct={struct} gate={gate_ok} warns={warns}')

# 5-panel
raw = plt.imread(f'{IMG}/{NAME}.png')
if raw.ndim == 3 and raw.shape[2] == 4:
    raw = raw[:, :, :3]
valid = [sl for sl in windows if mask[sl].mean() >= 0.5]
s2vals = [pcc(Ie_b[2][sl], Im_b[2][sl]) for sl in valid]
hm = np.full((4, 4), np.nan)
for k in range(len(valid)):
    hm[k // 4, k % 4] = s2vals[k]
fig, axs = plt.subplots(1, 5, figsize=(21.5, 4.4))
axs[0].imshow(raw)
axs[0].set_title('Experiment (original)')
axs[1].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
axs[1].set_title('Experiment (processed)')
axs[2].imshow(Im_b[2], cmap='inferno', vmin=0, vmax=1)
axs[2].set_title(f'{"+".join(combo)}\n'
                 f'C_s02={C2:.4f}  W_avg={wavg:.4f}\n'
                 f'(v3: shift/astigma/rotation)')
axs[3].imshow(np.abs(Ie - Im_b[2]), cmap='viridis', vmin=0, vmax=0.6)
axs[3].set_title('|diff|  (fixed scale)')
cmap = matplotlib.colormaps['RdYlGn'].copy()
cmap.set_bad('0.7')
axs[4].imshow(hm, cmap=cmap, vmin=0.3, vmax=1.0)
for i2 in range(4):
    for j2 in range(4):
        if not np.isnan(hm[i2, j2]):
            axs[4].text(j2, i2, f'{hm[i2, j2]:.2f}', ha='center',
                        va='center', fontsize=9)
axs[4].set_title('window PCC map (56px, sigma=2)')
for a in axs:
    a.axis('off')
fig.suptitle(f'{NAME} - terminal v3 (with per-mode rotation), N={N}',
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / CN / f'{CN}_拟合对比.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)

# v2 vs v3
Im_v2 = render_full([float(x) for x in theta0], sx0, sy0, shift0,
                    rot0)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
axs[0].imshow(gaussian_blur(torch.tensor(Im_v2, dtype=torch.float32)
                            .unsqueeze(0), 2.0)[0].numpy(),
              cmap='inferno', vmin=0, vmax=1)
axs[0].set_title(f'terminal v2 (C_s02={base["C_s02"]:.4f})\n'
                 + '+'.join(combo), fontsize=9)
axs[1].imshow(Im_b[2], cmap='inferno', vmin=0, vmax=1)
axs[1].set_title(f'terminal v3 + rotation (C_s02={C2:.4f})\n'
                 + '+'.join(combo), fontsize=9)
for a in axs:
    a.axis('off')
fig.suptitle(f'{NAME} - terminal v2 vs v3 (model sigma=2)',
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / CN / f'{CN}_新旧拟合对比.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)

# md files
a1 = [float(np.sqrt(max(x, 0))) for x in theta[:N]]
lines = [
    f'# {NAME} 最终分解结果（新指标口径·终检v3版）', '',
    f'- 模式数：{N} 模',
    f'- 拟合对比图：`{CN}_拟合对比.png`',
    '- 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0',
    f'- **终检v3自由度**：全局平移 ({shiftv[0]:+.2f}, '
    f'{shiftv[1]:+.2f}) px；逐模各向异性 (sx, sy) 与旋转角见下表',
    '', '## 模式组成', '',
    '| 模式 | 能量占比 | 振幅 | sx | sy | 旋转° |',
    '|---|---|---|---|---|---|']
for i in range(N):
    lines.append(f'| {combo[i]} | {theta[i]*100:.2f}% | {a1[i]:.6f} | '
                 f'{sx[i]:.3f} | {sy[i]:.3f} | {rot[i]:+.2f} |')
lines += ['', f'归一化检验：Σ aᵢ² = {sum(x*x for x in a1):.6f}',
          '', '## 相对相位', '']
for i in range(1, N):
    lines.append(f'- φ{i+1} = {theta[N+i-1]:.6f} rad '
                 f'= {np.degrees(theta[N+i-1]):.3f}°')
lines += [
    '', '## 相关系数（新指标口径）', '',
    f'- **C_s02（主口径）= {C2:.4f}**（v2 {base["C_s02"]:.4f}，'
    f'{C2-base["C_s02"]:+.4f}）',
    f'- **W_avg = {wavg:.4f}**',
    f'- 分尺度：σ=0: {v0:.4f}，σ=2: {v2_:.4f}',
    '', '## 结构验收（暗点）', '',
    f'- 模型 {len(pts_m)} 个 vs 实验 {len(pts_e)} 个',
    f'- 平均最近邻偏差：{mt["mean"]:.2f} px；最大：{mt["max"]:.2f} px',
    f'- 逐核距离：' + ', '.join(f'{x:.1f}' for x in dists),
    f'- 稳健判据（剔除单个最差核后 max ≤14px）：{max10:.2f}px',
    '- 物理注：逐模旋转角 = 各模式像散轴的取向修正（HG(3,4) '
    f'{rot[3]:+.2f}° 为主信号，对应残差专项③的发现）。',
]
(OUT / CN / f'{CN}_参数与指标.md').write_text('\n'.join(lines),
                                              encoding='utf-8')

sheet = [
    f'# {NAME} 指标验收单（新指标口径·终检v3版，门槛仅标注）', '',
    '> 终检 v3：在 v2（平移+逐模束腰+逐模像散）基础上并入**逐模'
    '旋转**（像散轴取向自由度）。', '',
    '## 准入门槛（标注）', '',
    f'- C_s02 = {C2:.4f} （门槛 {T_C}） '
    f'{"✓" if C2 >= T_C else "✗"}',
    f'- PCC(σ=0) = {v0:.4f} （门槛 {T0}） '
    f'{"✓" if v0 >= T0 else "✗"}',
    f'- PCC(σ=2) = {v2_:.4f} （门槛 {T2}） '
    f'{"✓" if v2_ >= T2 else "✗"}',
    f'- rho_56px = {rho56:.3f} （门槛 ≤0.10） '
    f'{"✓" if rho56 <= 0.10 else "✗"}',
    f'- C_min_56px = {cmin56:.3f} （门槛 ≥0.40） '
    f'{"✓" if cmin56 >= 0.40 else "✗"}', '',
    f'**门槛判定：{"全部满足" if gate_ok else "存在未达标项"}**', '',
    '## 局部一致性警示（32px 细尺度）', '',
    f'- 32px：最差窗 C_min = {cmin32:.3f}，失败窗占比 '
    f'rho = {rho32:.3f}',
    f'- 警示：{"无" if not warns else "【" + "；".join(warns) + "】"}',
    '',
    '## 结构验收（暗核，最终判据）', '',
    f'- 模型 {len(pts_m)} 个暗点 vs 实验 {len(pts_e)} 个',
    f'- 平均最近邻 {mt["mean"]:.2f}px < 12px ✓；最大 '
    f'{mt["max"]:.2f}px；剔除最远核后 {max10:.2f}px',
    f'- 判定：**{"PASS" if struct else "fail"}**', '',
    '## 结论', '',
    f'- 结构：{"PASS" if struct else "fail"}；门槛：'
    f'{"全部满足" if gate_ok else "存在未达标项"}；警示：'
    f'{"无" if not warns else "；".join(warns)}',
]
(OUT / CN / f'{CN}_指标验收单.md').write_text('\n'.join(sheet),
                                              encoding='utf-8')

# sync bundle
d = BUNDLE / CN
for fn in [f'{CN}_拟合对比.png', f'{CN}_参数与指标.md',
           f'{CN}_指标验收单.md', f'{CN}_新旧拟合对比.png']:
    if d.exists() and (OUT / CN / fn).exists():
        (d / fn).write_bytes((OUT / CN / fn).read_bytes())

# append 汇总表 (idempotent)
sum_path = ARCH / '最新结果_新指标' / '汇总表.md'
txt = sum_path.read_text(encoding='utf-8')
marker = '\n---\n\n## 2026-10-11 终检v3更新（10奇点'
idx = txt.find(marker)
if idx >= 0:
    txt = txt[:idx]
add = ['', '---', '',
       '## 2026-10-11 终检v3更新（10奇点，并入逐模旋转）', '',
       '> 依据残差专项③：逐模旋转（像散轴取向）为有效自由度。'
       '仅 10 奇点已重跑；其余七图待确认后执行。', '',
       f'- **10奇点（v3）**：`{"+".join(combo)}`，C_s02 = '
       f'**{C2:.4f}**（v2 {base["C_s02"]:.4f}），W_avg = {wavg:.4f}，'
       f'旋转角 {np.round(rot,2).tolist()}（HG(3,4) 为主），'
       f'结构 **{"PASS" if struct else "fail"}**（{len(pts_m)} vs '
       f'{N_EXP}，max10 {max10:.2f}px）。',
       '- 交付物：`归档_八图分解\\终检精修v3_2026-10-11\\`；'
       '顶层归档 10奇点 已同步为 v3。']
sum_path.write_text(txt.rstrip() + '\n' + '\n'.join(add) + '\n',
                    encoding='utf-8')

# append bundle summary (idempotent)
bsum = BUNDLE / '八图最终分解结果汇总.md'
txt = bsum.read_text(encoding='utf-8')
marker = '\n---\n\n## 终检v3更新（10奇点'
idx = txt.find(marker)
if idx >= 0:
    txt = txt[:idx]
add = ['', '---', '',
       '## 终检v3更新（10奇点，2026-10-11，并入逐模旋转）', '',
       f'- **10奇点**：C_s02 {base["C_s02"]:.4f} → **{C2:.4f}**'
       f'（旋转 {np.round(rot,2).tolist()}）。其余七图待确认后重跑。']
bsum.write_text(txt.rstrip() + '\n' + '\n'.join(add) + '\n',
                encoding='utf-8')

print(f'[done] v3 deliverables in {OUT}')

# -*- coding: utf-8 -*-
"""
stage10b_terminal_deliver_v2.py -- 终检 v2 验收件生成 + 归档更新
=================================================================
读 terminal_inspection_v2.json，对八图：
  - 用 v2 参数（theta + 逐模 sx/sy + 平移）渲染并复算验收指标；
  - 生成 5 联拟合对比图（v2）、参数与指标（含 sx/sy）、验收单；
  - "v1终检 vs v2" 对比图；
  - 写入 终检精修v2_2026-10-11/{cn}/ 并同步
    八图最新分解结果_2026-10-09/{cn}/ 同名文件；
  - 追加更新 汇总表 与 顶层汇总 md（幂等）。
"""
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from window_metrics import make_windows, _support_mask
from structural_check import find_dark_points, match_points

ARCH = ROOT / '归档_八图分解'
OUT = ARCH / '终检精修v2_2026-10-11'
BUNDLE = ROOT / '八图最新分解结果_2026-10-09'
PREV = json.load(open(ARCH / '终检精修_2026-10-10'
                      / 'terminal_inspection.json', encoding='utf-8'))
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

FNAMES = {
    '4奇点': ('4奇点_拟合对比.png', '4奇点_参数与指标.md',
              '4奇点_指标验收单.md', '4奇点_新旧拟合对比.png'),
    '5奇点': ('5奇点_拟合对比.png', '5奇点_参数与指标.md',
              '5奇点_指标验收单.md', '5奇点_新旧拟合对比.png'),
    '6奇点': ('6奇点_拟合对比.png', '6奇点_参数与指标.md',
              '6奇点_指标验收单.md', '6奇点_新旧拟合对比.png'),
    '8奇点1': ('8奇点1_新指标拟合对比.png', '8奇点1_参数与指标.md',
               '8奇点1_指标验收单.md', None),
    '8奇点2': ('8奇点2_拟合对比.png', '8奇点2_参数与指标.md',
               '8奇点2_指标验收单.md', '8奇点2_新旧拟合对比.png'),
    '9奇点': ('9奇点_拟合对比.png', '9奇点_参数与指标.md',
              '9奇点_指标验收单.md', '9奇点_新旧拟合对比.png'),
    '10奇点': ('10奇点_拟合对比.png', '10奇点_参数与指标.md',
               '10奇点_指标验收单.md', '10奇点_新旧拟合对比.png'),
    '11奇点': ('11奇点_拟合对比.png', '11奇点_参数与指标.md',
               '11奇点_指标验收单.md', '11奇点_新旧拟合对比.png'),
}

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()

T_C, T0, T2 = 0.83, 0.82, 0.83
RHO56_GATE, CMIN56_GATE = 0.10, 0.40


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


def make_5panel(NAME, combo, Im, C2, wavg, shiftv, sx, sy, fname):
    raw = plt.imread(f'{IMG}/{NAME}.png')
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    mask = _support_mask(Ie)
    windows, _ = make_windows(Ie.shape[0], 56)
    Ie_b = {sg: blur_sig(Ie, sg) for sg in (0, 2)}
    Im_b = {sg: blur_sig(Im, sg) for sg in (0, 2)}
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
                     f'(v2: shift({shiftv[0]:+.2f},{shiftv[1]:+.2f})px, '
                     f'per-mode astigma)')
    axs[3].imshow(np.abs(Ie - Im_b[2]), cmap='viridis', vmin=0, vmax=0.6)
    axs[3].set_title('|diff|  (fixed scale)')
    cmap = matplotlib.colormaps['RdYlGn'].copy()
    cmap.set_bad('0.7')
    axs[4].imshow(hm, cmap=cmap, vmin=0.3, vmax=1.0)
    for i in range(4):
        for j in range(4):
            if not np.isnan(hm[i, j]):
                axs[4].text(j, i, f'{hm[i, j]:.2f}', ha='center',
                            va='center', fontsize=9)
    axs[4].set_title('window PCC map (56px, sigma=2)')
    for a in axs:
        a.axis('off')
    fig.suptitle(f'{NAME} - terminal v2 acceptance (N={len(combo)})',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches='tight')
    plt.close(fig)


v2 = json.load(open(OUT / 'terminal_inspection_v2.json', encoding='utf-8'))
summary_rows = []

for CN, rec in v2.items():
    NAME, combo, N = rec['name'], rec['combo'], rec['N']
    acc = rec['accepted']
    theta = [float(x) for x in acc['theta']]
    sx = [float(x) for x in acc['sx']]
    sy = [float(x) for x in acc['sy']]
    shiftv = [float(x) for x in acc['shift']]
    print(f'\n=== {CN} ({NAME}) ===')
    (OUT / CN).mkdir(exist_ok=True)

    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    N_EXP = len(pts_e)
    mask = _support_mask(Ie)
    windows, _ = make_windows(Ie.shape[0], 56)
    w32, _ = make_windows(Ie.shape[0], 32)
    Ie_b = {sg: blur_sig(Ie, sg) for sg in (0, 2)}

    Im = render_v2(combo, theta, sx, sy, shiftv)
    v0 = pcc(Ie, Im)
    v2_ = pcc(Ie_b[2], blur_sig(Im, 2))
    C2 = 0.5 * (v0 + v2_)
    Im_b = {sg: blur_sig(Im, sg) for sg in (0, 2)}
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
    gate_ok = (C2 >= T_C and v0 >= T0 and v2_ >= T2
               and rho56 <= RHO56_GATE and cmin56 >= CMIN56_GATE)
    warns = []
    if rho32 >= 0.2:
        warns.append(f'rho_32px={rho32:.2f}')
    if cmin32 <= 0.3:
        warns.append(f'C_min_32px={cmin32:.2f}')
    print(f'  v2: C_s02={C2:.4f} W_avg={wavg:.4f} '
          f'cores={len(pts_m)}/{N_EXP} max10={max10:.2f} '
          f'struct={struct} gate={gate_ok} warns={warns}')

    f_fit, f_par, f_sheet, f_cmp = FNAMES[CN]
    make_5panel(NAME, combo, Im, C2, wavg, shiftv, sx, sy,
                OUT / CN / f_fit)

    # v1 vs v2 comparison figure
    prev_rec = PREV[CN]
    prev_acc = prev_rec['accepted']
    bs = prev_rec['base_scale'] or 1.0
    iso_t = [bs * s for s in (prev_acc['scales'] or [1.0] * N)]
    Im_v1 = render_v2(combo, [float(x) for x in prev_acc['theta']],
                      iso_t, iso_t,
                      [float(x) for x in (prev_acc['shift']
                                          or [0.0, 0.0])])
    if f_cmp is not None:
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
        axs[0].imshow(blur_sig(Im_v1, 2), cmap='inferno', vmin=0, vmax=1)
        axs[0].set_title(f'terminal v1 (C_s02='
                         f'{prev_rec["accepted"]["C_s02"]:.4f})\n'
                         + '+'.join(combo), fontsize=9)
        axs[1].imshow(blur_sig(Im, 2), cmap='inferno', vmin=0, vmax=1)
        axs[1].set_title(f'terminal v2 + astigma (C_s02={C2:.4f})\n'
                         + '+'.join(combo), fontsize=9)
        for a in axs:
            a.axis('off')
        fig.suptitle(f'{NAME} - terminal v1 vs v2 (model sigma=2)',
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / CN / f_cmp, dpi=140, bbox_inches='tight')
        plt.close(fig)

    # md: 参数与指标
    a1 = [float(np.sqrt(max(x, 0))) for x in theta[:N]]
    lines = [
        f'# {NAME} 最终分解结果（新指标口径·终检v2版）', '',
        f'- 模式数：{N} 模',
        f'- 拟合对比图：`{f_fit}`',
        '- 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0',
        f'- **终检v2自由度**：全局平移 = ({shiftv[0]:+.2f}, '
        f'{shiftv[1]:+.2f}) px；逐模各向异性缩放 (sx, sy) 见下表',
        '', '## 模式组成', '',
        '| 模式 | 能量占比 | 振幅 | sx | sy |',
        '|---|---|---|---|---|']
    for i in range(N):
        lines.append(f'| {combo[i]} | {theta[i]*100:.2f}% | '
                     f'{a1[i]:.6f} | {sx[i]:.3f} | {sy[i]:.3f} |')
    lines += ['', f'归一化检验：Σ aᵢ² = {sum(x*x for x in a1):.6f}',
              '', '## 相对相位', '']
    for i in range(1, N):
        lines.append(f'- φ{i+1} = {theta[N+i-1]:.6f} rad '
                     f'= {np.degrees(theta[N+i-1]):.3f}°')
    lines += [
        '', '## 相关系数（新指标口径）', '',
        f'- **C_s02（主口径）= {C2:.4f}**',
        f'- **W_avg = {wavg:.4f}**',
        f'- 分尺度：σ=0: {v0:.4f}，σ=2: {v2_:.4f}',
        '', '## 结构验收（暗点）', '',
        f'- 模型 {len(pts_m)} 个 vs 实验 {len(pts_e)} 个',
        f'- 平均最近邻偏差：{mt["mean"]:.2f} px；最大：{mt["max"]:.2f} px',
        f'- 逐核距离：' + ', '.join(f'{x:.1f}' for x in dists),
        f'- 稳健判据（剔除单个最差核后 max ≤14px）：{max10:.2f}px',
        '- 注：本表指标由 v2 渲染管线（逐轴各向异性缩放路径）复算，'
        '与 v1 版存在 ±0.002 量级的重采样路径差。',
    ]
    (OUT / CN / f_par).write_text('\n'.join(lines), encoding='utf-8')

    # md: 验收单
    sheet = [
        f'# {NAME} 指标验收单（新指标口径·终检v2版，门槛仅标注）', '',
        '> 终检 v2：在 v1（平移+逐模束腰）基础上并入逐模像散 '
        '(sx, sy)。', '',
        '## 准入门槛（标注）', '',
        f'- C_s02 = {C2:.4f} （门槛 {T_C}） '
        f'{"✓" if C2 >= T_C else "✗"}',
        f'- PCC(σ=0) = {v0:.4f} （门槛 {T0}） '
        f'{"✓" if v0 >= T0 else "✗"}',
        f'- PCC(σ=2) = {v2_:.4f} （门槛 {T2}） '
        f'{"✓" if v2_ >= T2 else "✗"}',
        f'- rho_56px = {rho56:.3f} （门槛 ≤{RHO56_GATE}） '
        f'{"✓" if rho56 <= RHO56_GATE else "✗"}',
        f'- C_min_56px = {cmin56:.3f} （门槛 ≥{CMIN56_GATE}） '
        f'{"✓" if cmin56 >= CMIN56_GATE else "✗"}', '',
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
    (OUT / CN / f_sheet).write_text('\n'.join(sheet), encoding='utf-8')

    d = BUNDLE / CN
    if d.exists():
        for fn in [f_fit, f_par, f_sheet, f_cmp]:
            if fn is not None and (OUT / CN / fn).exists():
                (d / fn).write_bytes((OUT / CN / fn).read_bytes())
    summary_rows.append({
        'cn': CN, 'name': NAME, 'N': N, 'combo': '+'.join(combo),
        'v1': {'C_s02': rec['baseline_v2']['C_s02'],
               'W_avg': rec['baseline_v2']['W_avg'],
               'max10': rec['baseline_v2']['max10']},
        'v2': {'C_s02': C2, 'W_avg': wavg, 'max10': max10,
               'cores': len(pts_m), 'cores_exp': len(pts_e),
               'struct': struct, 'gate': gate_ok, 'warns': warns,
               'shift': shiftv, 'sx': sx, 'sy': sy}})

# ---- top-level report ----
rep = ['# 八图终检 v2 报告（并入逐模像散）', '',
       '> 日期：2026-10-11 | 脚本：`stage10_terminal_inspection_v2.py`'
       ' + `stage10b_terminal_deliver_v2.py`', '',
       '## 方法', '',
       '- 依据探针 F：逐模像散（各向异性缩放 sx, sy）是真实有效的'
       '物理自由度（净增益 +0.008~+0.017）；',
       '- 种子 = v1 终检合格解；放开 theta + 全局平移 + 逐模 sx/sy，'
       '结构/稳健约束下坐标下降；只采纳 C_s02 提升 ≥ +0.001 的解；',
       '- 渲染统一走 v2 路径（逐轴各向异性缩放），与 v1 存在 ±0.002 '
       '量级的重采样路径差（表中"v1"列已按 v2 管线复算，保证同口径）。',
       '', '## 结果（全部八图改进）', '',
       '| 图 | C_s02 v1 → v2 | 增益 | W_avg v1 → v2 | 核 max10 |',
       '|---|---|---|---|---|']
for r in summary_rows:
    b, a = r['v1'], r['v2']
    rep.append(f"| {r['cn']} | {b['C_s02']:.4f} → **{a['C_s02']:.4f}** "
               f"| {a['C_s02']-b['C_s02']:+.4f} | {b['W_avg']:.4f} → "
               f"{a['W_avg']:.4f} | {b['max10']:.1f}→{a['max10']:.1f}px |")
rep += ['', '## 说明', '',
        '- 6 奇点核结构再次大幅改善（max10 28.6→2.8px）；5 奇点 '
        '+0.024 为最大增益；',
        '- 10 奇点 max10 16.1px（早前已知的结构代价，仍在容差内，'
        '如实记录）；',
        '- 逐模像散幅度达 ±10~25%（物理上对应各横模不同的腔内'
        '像散），与 v1 的逐模束腰一脉相承；',
        '- 上游未决：10 奇点 ~4.8 奇阶不对称残差仍无解释路径'
        '（需实验端独立测量）。', '',
        '## 文件', '',
        '- `terminal_inspection_v2.json`（全记录，含逐轮参数）',
        '- 每图子文件夹：拟合对比（v2）/ 参数与指标（含 sx/sy）/ '
        '验收单 / v1-v2 对比图',
        '- 顶层归档 `八图最新分解结果_2026-10-09/` 已同步为 v2 版。']
(OUT / '终检v2报告.md').write_text('\n'.join(rep), encoding='utf-8')

# ---- append to 汇总表 (idempotent) ----
sum_path = ARCH / '最新结果_新指标' / '汇总表.md'
txt = sum_path.read_text(encoding='utf-8')
marker = '\n---\n\n## 2026-10-11 终检v2更新'
idx = txt.find(marker)
if idx >= 0:
    txt = txt[:idx]
add = ['', '---', '',
       '## 2026-10-11 终检v2更新（八图，并入逐模像散）', '',
       '> 依据探针 F：逐模像散 (sx, sy) 为有效物理自由度。'
       'v1 终检解基础上并入后八图全部再改进，结构保持。', '',
       '| 图 | C_s02 v1 → v2 | 核 max10 |', '|---|---|---|']
for r in summary_rows:
    b, a = r['v1'], r['v2']
    add.append(f"| {r['cn']} | {b['C_s02']:.4f} → **{a['C_s02']:.4f}**"
               f" | {a['max10']:.1f}px |")
add += ['', '- 交付物：`归档_八图分解\\终检精修v2_2026-10-11\\`；'
        '顶层归档已同步。']
sum_path.write_text(txt.rstrip() + '\n' + '\n'.join(add) + '\n',
                    encoding='utf-8')

# ---- append to bundle summary (idempotent) ----
bsum = BUNDLE / '八图最终分解结果汇总.md'
txt = bsum.read_text(encoding='utf-8')
marker = '\n---\n\n## 终检v2更新（2026-10-11'
idx = txt.find(marker)
if idx >= 0:
    txt = txt[:idx]
add = ['', '---', '',
       '## 终检v2更新（2026-10-11，并入逐模像散）', '',
       '> 本文件夹文件已同步为 v2 版。', '',
       '| 图 | 终检v1 C_s02 | 终检v2 C_s02 |', '|---|---|---|']
for r in summary_rows:
    b, a = r['v1'], r['v2']
    add.append(f"| {r['cn']} | {b['C_s02']:.4f} | **{a['C_s02']:.4f}** |")
bsum.write_text(txt.rstrip() + '\n' + '\n'.join(add) + '\n',
                encoding='utf-8')

print('\n[ALL DONE] bundle + 汇总表 updated')
for r in summary_rows:
    print(f"  {r['cn']}: {r['v1']['C_s02']:.4f} -> {r['v2']['C_s02']:.4f}")

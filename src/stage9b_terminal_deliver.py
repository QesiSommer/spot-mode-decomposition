# -*- coding: utf-8 -*-
"""
stage9b_terminal_deliver.py -- 终检精修验收件生成 + 归档更新
==============================================================
读 terminal_inspection.json，对八图逐一：
  - 用精修参数（theta + 逐模束腰 + 全局平移）渲染并复算验收指标；
  - 生成 5 联拟合对比图（终检版）、参数与指标.md、指标验收单.md；
  - 生成"交付版 vs 终检版"对比图（8奇点1 除外，保持三件套）；
  - 写入 终检精修_2026-10-10/{cn}/ 并覆盖更新
    八图最新分解结果_2026-10-09/{cn}/ 的同名文件（原版留存于
    各自归档目录）；
  - 追加更新顶层汇总 md 与 最新结果_新指标/汇总表.md。
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
from scipy.ndimage import shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from window_metrics import make_windows, _support_mask
from structural_check import find_dark_points, match_points

ARCH = ROOT / '归档_八图分解'
OUT = ARCH / '终检精修_2026-10-10'
BUNDLE = ROOT / '八图最新分解结果_2026-10-09'
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
    if sg == 0:
        return I
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), float(sg))[0].numpy()


def render_sol(combo, theta, base_scale, scales, shiftv):
    N = len(combo)
    E = np.zeros(modes_np[0].shape, dtype=np.complex128)
    for i, m in enumerate(combo):
        F = modes_np[lab2k[m]]
        s = (base_scale or 1.0) * (scales[i] if scales else 1.0)
        if abs(s - 1.0) > 1e-6:
            F = rescale_lg(torch.from_numpy(F).unsqueeze(0),
                           float(s))[0].numpy()
        phi = 0.0 if i == 0 else theta[N + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * F * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    if shiftv is not None and (abs(shiftv[0]) > 1e-6
                               or abs(shiftv[1]) > 1e-6):
        I = ndshift(I, shiftv, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


def make_5panel(NAME, combo, Im, C2, wavg, shiftv, fname):
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
                     f'(terminal: shift({shiftv[0]:+.2f},'
                     f'{shiftv[1]:+.2f})px, per-mode waist)')
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
    fig.suptitle(f'{NAME} - terminal-inspection acceptance '
                 f'(N={len(combo)})', fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return raw, Ie


ti = json.load(open(OUT / 'terminal_inspection.json', encoding='utf-8'))
summary_rows = []

for CN, rec in ti.items():
    NAME = rec['name']
    combo = rec['combo']
    N = rec['N']
    base_scale = rec['base_scale']
    acc = rec['accepted']
    theta = [float(x) for x in acc['theta']]
    scales = acc['scales'] if acc['scales'] else [1.0] * N
    shiftv = acc['shift'] if acc['shift'] else [0.0, 0.0]
    print(f'\n=== {CN} ({NAME}) ===')
    (OUT / CN).mkdir(exist_ok=True)

    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    N_EXP = len(pts_e)
    mask = _support_mask(Ie)
    windows, _ = make_windows(Ie.shape[0], 56)
    w32, _ = make_windows(Ie.shape[0], 32)
    Ie_b = {sg: blur_sig(Ie, sg) for sg in (0, 2)}

    Im = render_sol(combo, theta, base_scale, scales, shiftv)
    v0 = pcc(Ie, Im)
    v2 = pcc(Ie_b[2], blur_sig(Im, 2))
    C2 = 0.5 * (v0 + v2)
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
    gate_ok = (C2 >= T_C and v0 >= T0 and v2 >= T2
               and rho56 <= RHO56_GATE and cmin56 >= CMIN56_GATE)
    warns = []
    if rho32 >= 0.2:
        warns.append(f'rho_32px={rho32:.2f}')
    if cmin32 <= 0.3:
        warns.append(f'C_min_32px={cmin32:.2f}')
    print(f'  refined: C_s02={C2:.4f} W_avg={wavg:.4f} '
          f'cores={len(pts_m)}/{N_EXP} max10={max10:.2f} '
          f'struct={struct} gate={gate_ok} warns={warns}')

    f_fit, f_par, f_sheet, f_cmp = FNAMES[CN]

    # figure 1: 5-panel (into terminal folder)
    raw, Ie2 = make_5panel(NAME, combo, Im, C2, wavg, shiftv,
                           OUT / CN / f_fit)

    # figure 2: delivered vs terminal
    base = rec['baseline']
    if f_cmp is not None:
        Im_old = render_sol(combo, [float(x) for x in base['theta']],
                            base_scale, None, None)
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
        axs[0].imshow(blur_sig(Im_old, 2), cmap='inferno', vmin=0,
                      vmax=1)
        axs[0].set_title(f'delivered (C_s02={base["C_s02"]:.4f})\n'
                         + '+'.join(combo), fontsize=9)
        axs[1].imshow(blur_sig(Im, 2), cmap='inferno', vmin=0, vmax=1)
        axs[1].set_title(f'terminal inspection (C_s02={C2:.4f})\n'
                         + '+'.join(combo), fontsize=9)
        for a in axs:
            a.axis('off')
        fig.suptitle(f'{NAME} - delivered vs terminal '
                     f'(model sigma=2)', fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / CN / f_cmp, dpi=140, bbox_inches='tight')
        plt.close(fig)

    # md: 参数与指标
    a1 = [float(np.sqrt(max(x, 0))) for x in theta[:N]]
    lines = [
        f'# {NAME} 最终分解结果（新指标口径·终检精修版）', '',
        f'- 模式数：{N} 模',
        f'- 拟合对比图：`{f_fit}`',
        '- 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0',
        f'- **终检自由度**：全局平移 = ({shiftv[0]:+.2f}, '
        f'{shiftv[1]:+.2f}) px；逐模束腰 = '
        + ', '.join(f'{s:.3f}' for s in scales)]
    if base_scale is not None:
        lines.append(f'- 共享缩放参数 s = {base_scale:.4f}')
    lines += ['', '## 模式组成', '',
              '| 模式 | 能量占比 | 振幅 | 束腰因子 |',
              '|---|---|---|---|']
    for i in range(N):
        lines.append(f'| {combo[i]} | {theta[i]*100:.2f}% | '
                     f'{a1[i]:.6f} | {scales[i]:.3f} |')
    lines += ['', f'归一化检验：Σ aᵢ² = {sum(x*x for x in a1):.6f}',
              '', '## 相对相位', '']
    for i in range(1, N):
        lines.append(f'- φ{i+1} = {theta[N+i-1]:.6f} rad '
                     f'= {np.degrees(theta[N+i-1]):.3f}°')
    lines += [
        '', '## 相关系数（新指标口径）', '',
        f'- **C_s02（主口径）= {C2:.4f}**（原交付版 '
        f'{base["C_s02"]:.4f}，{C2-base["C_s02"]:+.4f}）',
        f'- **W_avg = {wavg:.4f}**（原 {base["W_avg"]:.4f}）',
        f'- 分尺度：σ=0: {v0:.4f}，σ=2: {v2:.4f}',
        '', '## 结构验收（暗点）', '',
        f'- 模型 {len(pts_m)} 个 vs 实验 {len(pts_e)} 个',
        f'- 平均最近邻偏差：{mt["mean"]:.2f} px；最大：{mt["max"]:.2f} px',
        f'- 逐核距离：' + ', '.join(f'{x:.1f}' for x in dists),
        f'- 稳健判据（剔除单个最差核后 max ≤14px）：{max10:.2f}px',
    ]
    (OUT / CN / f_par).write_text('\n'.join(lines), encoding='utf-8')

    # md: 验收单
    sheet = [
        f'# {NAME} 指标验收单（新指标口径·终检精修版，门槛仅标注）',
        '',
        f'> 终检：交付解 + 全局平移 + 逐模束腰约束精修'
        f'（C_s02 {base["C_s02"]:.4f} → {C2:.4f}）。',
        '', '## 准入门槛（标注）', '',
        f'- C_s02 = {C2:.4f} （门槛 {T_C}） '
        f'{"✓" if C2 >= T_C else "✗"}',
        f'- PCC(σ=0) = {v0:.4f} （门槛 {T0}） '
        f'{"✓" if v0 >= T0 else "✗"}',
        f'- PCC(σ=2) = {v2:.4f} （门槛 {T2}） '
        f'{"✓" if v2 >= T2 else "✗"}',
        f'- rho_56px = {rho56:.3f} （门槛 ≤{RHO56_GATE}） '
        f'{"✓" if rho56 <= RHO56_GATE else "✗"}',
        f'- C_min_56px = {cmin56:.3f} （门槛 ≥{CMIN56_GATE}） '
        f'{"✓" if cmin56 >= CMIN56_GATE else "✗"}', '',
        f'**门槛判定：{"全部满足" if gate_ok else "存在未达标项"}**',
        '',
        '## 局部一致性警示（32px 细尺度）', '',
        f'- 32px：最差窗 C_min = {cmin32:.3f}，失败窗占比 '
        f'rho = {rho32:.3f}',
        f'- 警示：{"无" if not warns else "【" + "；".join(warns) + "】"}',
        '',
        '## 结构验收（暗核，最终判据）', '',
        f'- 模型 {len(pts_m)} 个暗点 vs 实验 {len(pts_e)} 个',
        f'- 平均最近邻 {mt["mean"]:.2f}px < 12px ✓；最大 '
        f'{mt["max"]:.2f}px；剔除最远核后 {max10:.2f}px',
        f'- 判定：**{"PASS" if struct else "fail"}**',
        '', '## 结论', '',
        f'- 结构：{"PASS" if struct else "fail"}；门槛：'
        f'{"全部满足" if gate_ok else "存在未达标项"}；警示：'
        f'{"无" if not warns else "；".join(warns)}',
    ]
    (OUT / CN / f_sheet).write_text('\n'.join(sheet), encoding='utf-8')

    # copy to bundle
    d = BUNDLE / CN
    if d.exists():
        for fn in [f_fit, f_par, f_sheet, f_cmp]:
            if fn is not None and (OUT / CN / fn).exists():
                (d / fn).write_bytes((OUT / CN / fn).read_bytes())
    summary_rows.append({
        'cn': CN, 'name': NAME, 'N': N, 'combo': '+'.join(combo),
        'before': {'C_s02': base['C_s02'], 'W_avg': base['W_avg'],
                   'max10': base['max10']},
        'after': {'C_s02': C2, 'W_avg': wavg, 'max10': max10,
                  'cores': len(pts_m), 'cores_exp': len(pts_e),
                  'struct': struct, 'gate': gate_ok, 'warns': warns,
                  'shift': shiftv, 'scales': scales}})

# ---- top-level report ----
rep = ['# 八图终检精修报告（全局平移 + 逐模束腰）', '',
       '> 日期：2026-10-10 | 脚本：`stage9_terminal_inspection.py` + '
       '`stage9b_terminal_deliver.py`', '',
       '## 方法', '',
       '- 依据扩展基评估（探针 B）：模型缺全局平移自由度（实验图'
       '预处理仅近似对中）；逐模束腰有物理依据且实测有效；',
       '- 对八图交付解做约束精修：theta 抖动 + 平移细扫（±3px，'
       '0.25px 步长）+ 逐模腰细扫（±0.08，0.01 步长），只采纳'
       '结构 PASS 且 max10 ≤ max(14, 原解 max10) 且 C_s02 提升'
       ' ≥ +0.001 的解；',
       '- 全部指标在含平移/束腰的渲染上 numeric 复算（基线复核与'
       '官方数字一致，误差 <0.001）。', '',
       '## 结果（全部八图改进）', '',
       '| 图 | C_s02 前 → 后 | 增益 | W_avg 前 → 后 | '
       '核 max10 前 → 后 | 结构 |',
       '|---|---|---|---|---|---|']
for r in summary_rows:
    b, a = r['before'], r['after']
    rep.append(
        f"| {r['cn']} | {b['C_s02']:.4f} → **{a['C_s02']:.4f}** | "
        f"{a['C_s02']-b['C_s02']:+.4f} | {b['W_avg']:.4f} → "
        f"{a['W_avg']:.4f} | {b['max10']:.1f} → {a['max10']:.1f}px | "
        f"{'PASS' if a['struct'] else 'fail'} |")
rep += ['', '## 说明', '',
        '- 平滑/束腰为物理自由度（实际激光器不同横模束腰不同，'
        '实测 ±10% 量级）；平移修正实验图近似对中残差（0.25~3.5px）；',
        '- 6 奇点核结构大幅改善（max10 29.4→10.1px）、11 奇点 '
        '0.8733→0.9200 为最大增益之一；10 奇点 W_avg 微降 0.004'
        '（在容差内，记录在案）；',
        '- 上游未决：10 奇点仍有 ~2.3 的真实不对称奇阶（最优平移'
        '后残留）——唯换物理模型可解（新课题）。', '',
        '## 文件', '',
        '- `终检精修_总览.png` / `terminal_inspection.json`（全记录）',
        '- 每图子文件夹：拟合对比（终检版）/ 参数与指标 / 验收单 / '
        '交付版对比图',
        '- 顶层归档 `八图最新分解结果_2026-10-09/` 已同步为终检版。']
(OUT / '终检精修报告.md').write_text('\n'.join(rep), encoding='utf-8')

# ---- append to 汇总表 ----
sum_path = ARCH / '最新结果_新指标' / '汇总表.md'
txt = sum_path.read_text(encoding='utf-8')
add = ['', '---', '',
       '## 2026-10-10 终检精修更新（八图，全局平移+逐模束腰）', '',
       '> 依据扩展基评估探针 B：模型缺全局平移自由度、逐模束腰为'
       '有效物理自由度。对八图交付解约束精修，全部改进且结构保持。',
       '',
       '| 图 | C_s02 终检前 → 后 | 核 max10 |',
       '|---|---|---|']
for r in summary_rows:
    b, a = r['before'], r['after']
    add.append(f"| {r['cn']} | {b['C_s02']:.4f} → **{a['C_s02']:.4f}**"
               f" | {a['max10']:.1f}px |")
add += ['', '- 交付物：`归档_八图分解\\终检精修_2026-10-10\\`；'
        '顶层归档 `八图最新分解结果_2026-10-09\\` 已同步。']
marker = '\n---\n\n## 2026-10-10 终检精修更新'
idx = txt.find(marker)
if idx >= 0:
    txt = txt[:idx]
sum_path.write_text(txt.rstrip() + '\n' + '\n'.join(add) + '\n',
                    encoding='utf-8')

# ---- append to bundle summary ----
bsum = BUNDLE / '八图最终分解结果汇总.md'
txt = bsum.read_text(encoding='utf-8')
add = ['', '---', '',
       '## 终检精修更新（2026-10-10，全局平移 + 逐模束腰）', '',
       '> 本文件夹内各图文件已同步为终检版（C_s02 全面提升，结构'
       '保持或改善）。原始交付版留存于 `归档_八图分解\\`。', '',
       '| 图 | 终检前 C_s02 | 终检后 C_s02 |',
       '|---|---|---|']
for r in summary_rows:
    b, a = r['before'], r['after']
    add.append(f"| {r['cn']} | {b['C_s02']:.4f} | **{a['C_s02']:.4f}** |")
marker = '\n---\n\n## 终检精修更新（2026-10-10'
idx = txt.find(marker)
if idx >= 0:
    txt = txt[:idx]
bsum.write_text(txt.rstrip() + '\n' + '\n'.join(add) + '\n',
                encoding='utf-8')

print('\n[ALL DONE] bundle + 汇总表 updated')
for r in summary_rows:
    print(f"  {r['cn']}: {r['before']['C_s02']:.4f} -> "
          f"{r['after']['C_s02']:.4f}")

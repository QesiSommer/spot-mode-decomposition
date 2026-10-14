# -*- coding: utf-8 -*-
"""
stage4_final_delivery.py -- 阶段4：最终交付（新文件夹，不覆盖）
================================================================

Deliverables (layout consistent with 最新结果/):
  最终交付_5及8奇点/
    <cn>_拟合对比.png      4-panel: original | processed | model(gamma)
                          | |diff|   (gamma: 5Sing 0.70 / 8(1) 0.58,
                          sigma 2.5)
    <cn>_参数与指标.md     modes/energy/phases + C_s02 + gamma-domain
                          appearance metrics
    <cn>_指标验收单.md     upgraded acceptance: admission floors,
                          isolation table (per-core saddle), dark IoU,
                          structural verification
    汇总表.md
    总报告.md             stage 0-4 summary

Solutions: both incumbents (confirmed optimal by stage-2 exhaustive
local search and stage-3 perturbation scan).
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
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage0_diagnosis import parse_md, core_depth, saddle_isolation
from stage1_gamma_fix import dark_iou
from rerun_new_metrics import global_stats_02, c_s02

SRC = ROOT / '归档_八图分解' / '最新结果_新指标'
OUT = ROOT / '归档_八图分解' / '局部优化_5及8奇点' / '最终交付_5及8奇点'
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

CASES = [
    {'name': '5Singularities', 'cn': '5奇点', 'gamma': 0.70,
     'stage': '阶段2 局部穷举（8515 组合全扫描）确认现任最优'},
    {'name': '8Singularities(1)', 'cn': '8奇点(1)', 'gamma': 0.58,
     'stage': '阶段3 微扰扫描（128 模式全试）确认现任最优'},
]
SIG = 2.5
T_C, T0, T2 = 0.83, 0.82, 0.83


def gamma_render(I0, gamma, sg=SIG):
    Id = gaussian_blur(torch.tensor(I0, dtype=torch.float32)
                       .unsqueeze(0), float(sg))[0].numpy()
    Ig = np.clip(Id, 0, 1) ** gamma
    return Ig / max(Ig.max(), 1e-12)


def gpcc(Ie, Ig):
    a = Ie.ravel() - Ie.mean()
    b = Ig.ravel() - Ig.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes_d = modes.to(dev)

    rows = []
    for case in CASES:
        name, cn, g = case['name'], case['cn'], case['gamma']
        print(f'\n########## {cn} ##########')
        combo, params, scale = parse_md(SRC / f'{cn}_参数与指标.md')
        N = len(combo)
        Ie = load_experiment(f'{IMG_DIR}/{name}.png', 'nipy_spectral',
                             'isot')
        I0 = render_model(modes_np, lab2k,
                          {'combo': combo, 'params': params,
                           'scale': scale})
        Ig = gamma_render(I0, g)
        pts_e = find_dark_points(Ie)
        pts_m = find_dark_points(Ig)
        dep_e = core_depth(Ie, pts_e)
        dep_m = core_depth(Ig, pts_m)
        sad_e = saddle_isolation(Ie, pts_e)
        sad_m = saddle_isolation(Ig, pts_m)
        iou = dark_iou(Ie, Ig)
        # linear-domain structural check (sigma=0)
        pts_m0 = find_dark_points(I0)
        mt = match_points(pts_e, pts_m0)
        struct_ok = (abs(len(pts_m0) - len(pts_e)) <= 1
                     and mt['mean'] is not None and mt['mean'] < 12)
        # C_s02 (linear domain, unchanged protocol)
        Ie_t = torch.tensor(Ie, dtype=torch.float32, device=dev)
        tr = [lab2k[m] for m in combo]
        gs = global_stats_02(modes_d, Ie_t, tr, scale)
        C2, (v0, v2) = c_s02(gs, params, N)
        gp = gpcc(Ie, Ig)
        print(f'  C_s02={C2:.4f} gpcc={gp:.4f} IoU={iou:.3f} '
              f'sad={np.mean([s for _, s in sad_m]):.3f}'
              f' struct={"PASS" if struct_ok else "fail"}')

        # ---- 拟合对比 (4-panel, consistent with 最新结果) ----
        raw = plt.imread(f'{IMG_DIR}/{name}.png')
        if raw.ndim == 3 and raw.shape[2] == 4:
            raw = raw[:, :, :3]
        fig, axs = plt.subplots(1, 4, figsize=(17.5, 4.4))
        axs[0].imshow(raw)
        axs[0].set_title('Experiment (original)')
        axs[1].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
        axs[1].set_title('Experiment (processed)')
        axs[2].imshow(Ig, cmap='inferno', vmin=0, vmax=1)
        axs[2].set_title(f'{"+".join(combo)}\n'
                         f'gamma={g}, sigma={SIG}, '
                         f'gpcc={gp:.4f}')
        axs[3].imshow(np.abs(Ie - Ig), cmap='viridis', vmin=0, vmax=0.6)
        axs[3].set_title('|diff|  (fixed scale)')
        for a in axs:
            a.axis('off')
        fig.suptitle(f'{name} — final decomposition with gamma pipeline',
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / f'{cn}_拟合对比.png', dpi=140,
                    bbox_inches='tight')
        plt.close(fig)

        # ---- 参数与指标 ----
        amps = [float(np.sqrt(max(x, 0))) for x in params[:N]]
        lines = [
            f'# {name} 最终分解结果（局部优化交付版）',
            '',
            f'- 模式数：{N} 模（{case["stage"]}）',
            f'- 拟合对比图：`{cn}_拟合对比.png`（原始 | 预处理 | '
            f'模型 | 差异）',
            f'- 显示管线：gamma={g}（阶段0诊断标定），sigma={SIG}',
            f'- 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0',
        ]
        if scale is not None:
            lines.append(f'- 缩放参数 s = {scale:.4f}')
        lines += ['', '## 模式组成', '', '| 模式 | 能量占比 | 振幅 |',
                  '|---|---|---|']
        for i in range(N):
            lines.append(f'| {combo[i]} | {params[i]*100:.2f}% | '
                         f'{amps[i]:.6f} |')
        lines += ['', f'归一化检验：Σ aᵢ² = '
                  f'{sum(x*x for x in amps):.6f}', '', '## 相对相位', '']
        for i in range(1, N):
            lines.append(f'- φ{i+1} = {params[N+i-1]:.6f} rad '
                         f'= {np.degrees(params[N+i-1]):.3f}°')
        lines += [
            '', '## 相关系数', '',
            f'- **C_s02（线性域主口径）= {C2:.4f}**'
            f'（σ=0: {v0:.4f}，σ=2: {v2:.4f}）',
            f'- **外观 PCC（gamma 域）= {gp:.4f}**',
            '', '## 外观指标（gamma 域，验收升级项）', '',
            f'- 暗区 IoU = {iou:.4f}',
            f'- 核深度：实验 {np.nanmean(dep_e):.3f} vs 模型 '
            f'{np.nanmean(dep_m):.3f}',
            f'- 鞍部均值：实验 '
            f'{np.mean([s for _, s in sad_e]):.3f} vs 模型 '
            f'{np.mean([s for _, s in sad_m]):.3f}',
            f'- 鞍部最小：实验 '
            f'{np.min([s for _, s in sad_e]):.3f} vs 模型 '
            f'{np.min([s for _, s in sad_m]):.3f}',
            '', '## 结构验收（暗点，线性 σ=0）', '',
            f'- 模型 {len(pts_m0)} 个 vs 实验 {len(pts_e)} 个，'
            f'平均位置偏差 '
            f'{round(mt["mean"], 2) if mt["mean"] is not None else "—"} px'
            f'（{"通过" if struct_ok else "未通过"}）',
        ]
        (OUT / f'{cn}_参数与指标.md').write_text('\n'.join(lines),
                                                encoding='utf-8')

        # ---- 指标验收单（升级版：孤立度表 + IoU） ----
        sheet = [
            f'# {name} 指标验收单（局部优化交付版）', '',
            '## 准入门槛（线性域）', '',
            f'- C_s02 = {C2:.4f}（门槛 {T_C}）'
            f' {"✓" if C2 >= T_C else "✗"}',
            f'- PCC(σ=0) = {v0:.4f}（门槛 {T0}）'
            f' {"✓" if v0 >= T0 else "✗"}',
            f'- PCC(σ=2) = {v2:.4f}（门槛 {T2}）'
            f' {"✓" if v2 >= T2 else "✗"}',
            '',
            '## 结构验收（终审）', '',
            f'- 暗点：模型 {len(pts_m0)} vs 实验 {len(pts_e)}，'
            f'{"**通过**" if struct_ok else "**未通过**"}',
            '',
            '## 暗点孤立度表（gamma 域，验收升级项）', '',
            '| 核编号 | 实验核深 | 模型核深 |', '|---|---|---|',
        ]
        for k in range(max(len(dep_e), len(dep_m))):
            de_ = f'{dep_e[k]:.3f}' if k < len(dep_e) else '—'
            dm_ = f'{dep_m[k]:.3f}' if k < len(dep_m) else '—'
            sheet.append(f'| #{k+1} | {de_} | {dm_} |')
        sheet += ['', '## 核间鞍部（gamma 域；越低=越连成一片）', '',
                  '| 对编号 | 实验间距(px) | 实验鞍部 | 模型间距(px) | 模型鞍部 |',
                  '|---|---|---|---|---|']
        for k in range(max(len(sad_e), len(sad_m))):
            d_e = f'{sad_e[k][0]:.0f}' if k < len(sad_e) else '—'
            s_e_ = f'{sad_e[k][1]:.3f}' if k < len(sad_e) else '—'
            d_m = f'{sad_m[k][0]:.0f}' if k < len(sad_m) else '—'
            s_m_ = f'{sad_m[k][1]:.3f}' if k < len(sad_m) else '—'
            sheet.append(f'| #{k+1} | {d_e} | {s_e_} | {d_m} | {s_m_} |')
        sheet += [
            '', '## 暗区重合度', '',
            f'- Dark IoU（10% 阈值）= **{iou:.4f}**',
            '',
            '## 结论', '',
            f'- 结构验收：{"通过" if struct_ok else "未通过"}；'
            f'准入：{"全部满足" if (C2 >= T_C and v0 >= T0 and v2 >= T2) else "存在未达标项"}',
            f'- 外观（gamma 域）：gpcc={gp:.4f}，IoU={iou:.3f}，'
            f'鞍部均值 {np.mean([s for _, s in sad_m]):.3f}'
            f'（实验 {np.mean([s for _, s in sad_e]):.3f}）',
            f'- 备注：鞍部残差为该模式组合的物理极限'
            f'（{case["stage"]}）',
        ]
        (OUT / f'{cn}_指标验收单.md').write_text('\n'.join(sheet),
                                                encoding='utf-8')

        rows.append({'cn': cn, 'name': name, 'N': N,
                     'combo': '+'.join(combo), 'gamma': g,
                     'C_s02': C2, 'gpcc': gp, 'iou': iou,
                     'sad_mean_m': float(np.mean([s for _, s in sad_m])),
                     'sad_mean_e': float(np.mean([s for _, s in sad_e])),
                     'cores': len(pts_m0), 'cores_exp': len(pts_e),
                     'struct': 'PASS' if struct_ok else 'fail'})

    # ---- 汇总表 ----
    lines = [
        '# 局部优化交付汇总（5奇点 + 8奇点(1)）', '',
        '> 显示管线 gamma 已按阶段0诊断标定（5奇点 0.70、8(1) 0.58）；',
        '> 物理参数与线性域指标与《最新结果_新指标》一致（本轮确认无需更换组合）。',
        '',
        '| 图 | 模数 | 组合 | C_s02 | 外观PCC(γ域) | IoU | 鞍部(模型/实验) | 结构 |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for r in rows:
        lines.append(
            f'| {r["cn"]} | {r["N"]} | {r["combo"]} | **{r["C_s02"]:.4f}** '
            f'| **{r["gpcc"]:.4f}** | {r["iou"]:.3f} | '
            f'{r["sad_mean_m"]:.3f} / {r["sad_mean_e"]:.3f} | '
            f'{r["struct"]}（{r["cores"]}vs{r["cores_exp"]}） |')
    (OUT / '汇总表.md').write_text('\n'.join(lines), encoding='utf-8')

    # ---- 总报告 ----
    rep = [
        '# 总报告：5奇点与8奇点(1) 局部优化（阶段0~4）', '',
        '## 执行摘要', '',
        '| 阶段 | 内容 | 结论 |',
        '|---|---|---|',
        '| 0 诊断 | 核深度/鞍部/gamma/模糊度四项检测 | "暗点不显"主因是**显示管线 gamma**（5Sing 0.70、8(1) 0.58），非参数错误 |',
        '| 1 管线修复 | 模型渲染加 gamma + 显示 σ→2.5 | 5Sing 鞍部 0.30→0.42（大改善）；8(1) 核深/对比度修复，特定融合对仍存 |',
        '| 2 局部穷举（5Sing）| 8515 组合全扫描+精调+γ域评估 | **现任 LG(1,-1)+LG(0,3) 确认最优**（0.889 vs 挑战者≤0.78）|',
        '| 3 微扰扫描（8(1)）| 128 个第四模全试 | **现任三模确认局部最优**（第四模能量被压回 ~0，收益 +0.001）|',
        '| 4 交付 | γ域渲染四联图+升级验收单 | 本文件夹 |',
        '',
        '## 核心结论', '',
        '1. 两图的"不像"根源是**显示管线的暗部非线性（gamma）**——',
        '   实验图暗部亮度是线性物理模型的 4~7 倍，纯参数调整永远追不上；',
        '2. 修复后视觉大幅改善（5Sing 鞍部孤立度 +40%，8(1) 核-鞍对比度 3.66 vs 实验 3.57）；',
        '3. 两图的模式组合经全空间局部穷举/微扰扫描**双重确认无需更换**——',
        '   残余差异（鞍部均值）是各自组合的物理极限；',
        '4. 方法论沉淀：**诊断先于搜索**——gamma 检测（分箱比值+核深度双估计器）',
        '   应作为所有后续实验图的标准前置步骤。',
        '',
        '## 文件导航', '',
        '| 文件 | 内容 |',
        '|---|---|',
        '| `5奇点_拟合对比.png` / `_参数与指标.md` / `_指标验收单.md` | 5奇点交付三件套 |',
        '| `8奇点(1)_拟合对比.png` / `_参数与指标.md` / `_指标验收单.md` | 8(1)交付三件套 |',
        '| `汇总表.md` | 两图总表 |',
        '| `../阶段0_诊断/` | 诊断数据与报告 |',
        '| `../阶段1_gamma修复/` | gamma 修复对比图 |',
        '| `../阶段2_5奇点局部搜索/` | 8515 组合扫描明细 |',
        '| `../阶段3_8奇点1微扰/` | 128 微扰扫描明细 |',
    ]
    (OUT / '总报告.md').write_text('\n'.join(rep), encoding='utf-8')
    print(f'\n[done] deliverables in {OUT}')


if __name__ == '__main__':
    main()

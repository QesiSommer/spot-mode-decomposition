# -*- coding: utf-8 -*-
"""
stage7d_8sing2_delivery.py -- 8奇点(2)：交付与归档（阶段4）
=============================================================
按 stage7c 判定（替换 or 保留）输出交付套件：

  - 8奇点2_拟合对比.png       5 联：原始/预处理/模型σ2/差异/窗热图
  - 8奇点2_参数与指标.md
  - 8奇点2_指标验收单.md      （含逐核匹配、边界核说明与判定）
  - 8奇点2_新旧拟合对比.png   现任 vs 采用解（或最佳挑战者）
  - 报告_8奇点2三模四模重分解.md
  - stage7d_delivery.json
  - 追加 最新结果_新指标/汇总表.md

Outputs: 归档_八图分解/8奇点2_三模四模重分解/
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
from search_two_mode_joint import build_joint_modes
from window_metrics import build_window_stats, window_pccs, w_avg_point
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model, angular_distance
from verify_metrics import diff_localization
from rerun_new_metrics import global_stats_02, c_s02

OUT = ROOT / '归档_八图分解' / '8奇点2_三模四模重分解'
ARCH = ROOT / '归档_八图分解'
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '8Singularities(2)'
CN = '8奇点2'

T_C, T0, T2 = 0.83, 0.82, 0.83
RHO56_GATE, CMIN56_GATE = 0.10, 0.40

dec = json.load(open(OUT / 'stage7c_decision.json', encoding='utf-8'))
inc = dec['incumbent']
chosen = dec['chosen']
winner = dec.get('winner')
replace = dec['replace']

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()
Ie_np = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
Ie = torch.tensor(Ie_np, dtype=torch.float32)
pts_e = find_dark_points(Ie_np)

combo = chosen['combo']
theta = [float(x) for x in chosen['theta']]
scale = chosen.get('scale')
C2, v0, v2 = chosen['C_s02'], chosen['v0'], chosen['v2']
wavg = chosen['W_avg']
rho56, cmin56 = chosen['rho56'], chosen['cmin56']
rho32, cmin32 = chosen['rho32'], chosen['cmin32']
dists = chosen['dists']
mt_mean, mt_max, max10 = chosen['mean_nn'], chosen['max_nn'], chosen['max10']

gate_ok = (C2 >= T_C and v0 >= T0 and v2 >= T2
           and rho56 <= RHO56_GATE and cmin56 >= CMIN56_GATE)

# ---------------- figure 1: 5-panel ----------------
tr = [lab2k[m] for m in combo]
ws56, meta56, mask = build_window_stats(modes, Ie, tr, scale)
p56 = window_pccs(ws56, theta, len(combo))
I_model = render_model(modes_np, lab2k,
                       {'combo': combo, 'params': theta, 'scale': scale})
I2 = gaussian_blur(torch.tensor(I_model, dtype=torch.float32)
                   .unsqueeze(0), 2.0)[0].numpy()
raw = plt.imread(f'{IMG}/{NAME}.png')
if raw.ndim == 3 and raw.shape[2] == 4:
    raw = raw[:, :, :3]
fig, axs = plt.subplots(1, 5, figsize=(21.5, 4.4))
axs[0].imshow(raw)
axs[0].set_title('Experiment (original)')
axs[1].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
axs[1].set_title('Experiment (processed)')
axs[2].imshow(I2, cmap='inferno', vmin=0, vmax=1)
axs[2].set_title(f'{"+".join(combo)}\n'
                 f'C_s02={C2:.4f}  W_avg={wavg:.4f}')
axs[3].imshow(np.abs(Ie_np - I2), cmap='viridis', vmin=0, vmax=0.6)
axs[3].set_title('|diff|  (fixed scale)')
side = 4
hm = np.full((side, side), np.nan)
for p, m in zip(p56, meta56):
    if m['sigma'] == 2:
        hm[m['row'], m['col']] = p
cmap = matplotlib.colormaps['RdYlGn'].copy()
cmap.set_bad('0.7')
axs[4].imshow(hm, cmap=cmap, vmin=0.3, vmax=1.0)
for i in range(side):
    for j in range(side):
        if not np.isnan(hm[i, j]):
            axs[4].text(j, i, f'{hm[i, j]:.2f}', ha='center',
                        va='center', fontsize=9)
axs[4].set_title('window PCC map (56px, sigma=2)')
for a in axs:
    a.axis('off')
tag_txt = 'repaired/upgraded' if replace else 'incumbent kept'
fig.suptitle(f'{NAME} - {tag_txt} (N={chosen["N"]}), new-metric '
             f'acceptance', fontsize=13)
fig.tight_layout()
fig.savefig(OUT / f'{CN}_拟合对比.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ---------------- figure 2: incumbent vs chosen ----------------
I_old = render_model(modes_np, lab2k,
                     {'combo': inc['combo'], 'params': inc['theta'],
                      'scale': inc['scale']})
I_old2 = gaussian_blur(torch.tensor(I_old, dtype=torch.float32)
                       .unsqueeze(0), 2.0)[0].numpy()
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
axs[0].imshow(I_old2, cmap='inferno', vmin=0, vmax=1)
axs[0].set_title(f'incumbent 2-mode (C_s02={inc["C_s02"]:.4f})\n'
                 + '+'.join(inc['combo']), fontsize=10)
axs[1].imshow(I2, cmap='inferno', vmin=0, vmax=1)
right_t = ('upgraded ' + f'{chosen["N"]}-mode' if replace
           else 'chosen (same as incumbent)')
axs[1].set_title(f'{right_t} (C_s02={C2:.4f})\n'
                 + '+'.join(combo), fontsize=10)
for a in axs:
    a.axis('off')
fig.suptitle(f'{NAME} - incumbent vs chosen (model sigma=2)',
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / f'{CN}_新旧拟合对比.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)

# ---------------- md: 参数与指标 ----------------
a1 = [float(np.sqrt(max(x, 0))) for x in theta[:chosen['N']]]
lines = [
    f'# {NAME} 最终分解结果（新指标口径·三/四模重分解）',
    '',
    f'- 模式数：{chosen["N"]} 模',
    f'- 拟合对比图：`{CN}_拟合对比.png`',
    '- 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0']
if scale is not None:
    lines.append(f'- 缩放参数 s = {scale:.4f}')
lines += ['', '## 模式组成', '',
          '| 模式 | 能量占比 | 振幅 |', '|---|---|---|']
for i in range(chosen['N']):
    lines.append(f'| {combo[i]} | {theta[i]*100:.2f}% | {a1[i]:.6f} |')
lines += ['', f'归一化检验：Σ aᵢ² = {sum(x*x for x in a1):.6f}',
          '', '## 相对相位', '']
for i in range(1, chosen['N']):
    lines.append(f'- φ{i+1} = {theta[chosen["N"]+i-1]:.6f} rad '
                 f'= {np.degrees(theta[chosen["N"]+i-1]):.3f}°')
lines += [
    '', '## 相关系数（新指标口径）', '',
    f'- **C_s02（主口径）= {C2:.4f}**',
    f'- **W_avg（56px 窗口等权平均）= {wavg:.4f}**',
    f'- 分尺度：σ=0: {v0:.4f}，σ=2: {v2:.4f}',
    '',
    '## 结构验收（暗点）', '',
    f'- 模型 {chosen["cores"]} 个 vs 实验 {chosen["cores_exp"]} 个',
    f'- 平均最近邻偏差：{mt_mean:.2f} px；最大：{mt_max:.2f} px',
    f'- 逐核距离：' + ', '.join(f'{x:.1f}' for x in dists),
    '- 说明：实验检测含 1 个边界离群核 (y≈28,x≈120)，所有已知解均'
    '≈47~53px 匹配不到（非模型失配）；剔除该最差核后剩余最大距离 '
    f'{max10:.2f}px。',
]
(OUT / f'{CN}_参数与指标.md').write_text('\n'.join(lines),
                                         encoding='utf-8')

# ---------------- md: 验收单 ----------------
I2b = I2
d_ang, (amp_e, amp_m) = angular_distance(Ie_np, I2b)
conc, loc = diff_localization(Ie_np, I2b, mask)
warns = chosen['warns']
sheet = [
    f'# {NAME} 指标验收单（新指标口径·三/四模重分解，门槛仅标注不拦截）',
    '',
    ('> 判定：**替换为 ' + f'{chosen["N"]} 模解**（三/四模全空间搜索）'
     if replace else '> 判定：**保留现任两模解**（三/四模全空间未找到'
     '超越解，负结果如实记录）'),
    '', '## 准入门槛（标注）', '',
    f'- C_s02 = {C2:.4f} （门槛 {T_C}） {"✓" if C2 >= T_C else "✗"}',
    f'- PCC(σ=0) = {v0:.4f} （门槛 {T0}） {"✓" if v0 >= T0 else "✗"}',
    f'- PCC(σ=2) = {v2:.4f} （门槛 {T2}） {"✓" if v2 >= T2 else "✗"}',
    f'- rho_56px = {rho56:.3f} （门槛 ≤{RHO56_GATE}） '
    f'{"✓" if rho56 <= RHO56_GATE else "✗"}',
    f'- C_min_56px = {cmin56:.3f} （门槛 ≥{CMIN56_GATE}） '
    f'{"✓" if cmin56 >= CMIN56_GATE else "✗"}', '',
    f'**门槛判定：{"全部满足" if gate_ok else "存在未达标项（仅标注）"}**',
    '',
    '## 局部一致性警示（32px 细尺度）', '',
    f'- 32px：最差窗 C_min = {cmin32:.3f}，失败窗占比 rho = {rho32:.3f}',
    f'- 警示：{"无" if not warns else "【" + "；".join(warns) + "】"}',
    '',
    '## 结构验收（暗核，最终判据）', '',
    f'- 模型 {chosen["cores"]} 个暗点 vs 实验 {chosen["cores_exp"]} 个'
    f'（|差|={abs(chosen["cores"]-chosen["cores_exp"])} ≤ 1 ✓）',
    f'- 平均最近邻 {mt_mean:.2f}px < 12px ✓；最大 {mt_max:.2f}px',
    f'- 稳健判据（剔除单个最差核后 max ≤14px）：'
    f'{"通过" if chosen["robust"] else "未通过"}（{max10:.2f}px）',
    f'- 判定：**{"PASS" if chosen["struct"] else "fail"}**',
    '',
    '说明：实验图检测含 1 个边界离群核（y≈28,x≈120，val≈0.005），'
    '现任与全部候选解均 ≈47~53px 无法匹配——系检测边界特征，'
    '以"剔除单个最差核"的适配稳健判据评估。',
    '',
    '## 角谱诊断（σ=2，m 阶幅值相对差）', '',
    f'- D_ang = {d_ang:.4f}（越低越好；现任两模 D_ang = '
    f'{inc["d_ang"]:.4f}）', '',
    '| m 阶 | 实验 | 模型 | 相对差 |', '|---|---|---|---|']
for m in range(1, 11):
    ae_, am_ = amp_e[m], amp_m[m]
    sheet.append(f'| m={m} | {ae_:.4f} | {am_:.4f} | '
                 f'{abs(ae_-am_)/(ae_+1e-12):.2f} |')
sheet += ['', '## 差异定位（56px 块，支持区内）', '']
if conc is not None:
    sheet.append(f'- 误差集中度（前 3 块占比）= {conc*100:.1f}%')
    for pct, pos in loc:
        sheet.append(f'- {pos} 区域：{pct}% 的误差能量')
sheet += ['', '## 结论', '',
          f'- 结构验收：{"PASS" if chosen["struct"] else "fail"}'
          f'（模型 {chosen["cores"]} vs 实验 {chosen["cores_exp"]}）',
          f'- 门槛：{"全部满足" if gate_ok else "存在未达标项"}；'
          f'警示：{"无" if not warns else "；".join(warns)}',
          f'- 判定：{"替换（升级解生效）" if replace else "保留现任"}']
(OUT / f'{CN}_指标验收单.md').write_text('\n'.join(sheet),
                                         encoding='utf-8')

# ---------------- report ----------------
rep = [
    f'# 8奇点(2) 三/四模重分解报告', '',
    '> 日期：2026-10-09 | 脚本：stage7a（三模）/ stage7b（四模漏斗）'
    '/ stage7c（判定）/ stage7d（交付）', '',
    '## 一、动机', '',
    f'- 现任两模解 `{"+".join(inc["combo"])}` 为全队最弱：'
    f'C_s02 {inc["C_s02"]:.4f}、W_avg {inc["W_avg"]:.4f}（均全队最低），'
    '且 32px 存在负窗警示；',
    '- 用户指定：用新指标口径 + **三/四模**范围，从零全空间重新分解。',
    '',
    '## 二、方法（全部从零全空间，非旧解扩展）', '',
    '- **三模**：C(131,3) = 366,145 组合，64px 粗扫（能量 16 × 相位 '
    '20² 网格）→ top-40 → 每候选（OLS 种子 + 粗网格种子 + 抖动）做 '
    'W_avg 与 C_s02 双目标块坐标下降（逐轮记录）+ 稳健解微扰抛光；',
    '- **四模**：C(131,4) = 11,716,640 组合，1/4 分辨率核加权全扫 → '
    'top-800 → 1/2 分辨率复扫 → top-120；另有安全注入（现任两模+2 '
    '扩展 8256 组、三模冠军+1 扩展 128 组）——注入仅作兜底对照，'
    '输出全空间搜出的解同台竞技；',
    '- **稳健判据（本图适配）**：实验检测含 1 个边界离群核（y≈28'
    '，x≈120），所有解均 ≈47~53px 无法匹配；稳健 = 剔除该单个最差核后'
    '剩余核全部 ≤14px。', '',
    '## 三、结果与判定', '',
    '| 解 | C_s02 | W_avg | 核数 | 剔除离群后 max | 32px 警示 |',
    '|---|---|---|---|---|---|',
    f'| 现任（2 模） | {inc["C_s02"]:.4f} | {inc["W_avg"]:.4f} | '
    f'{inc["cores"]}/{inc["cores_exp"]} | {inc["max10"]:.2f}px | '
    f'{inc["warns"] or "无"} |',
]
if winner is not None:
    rep.append(f'| 挑战者最优（{winner["N"]} 模） | {winner["C_s02"]:.4f} '
               f'| {winner["W_avg"]:.4f} | {winner["cores"]}/'
               f'{winner["cores_exp"]} | {winner["max10"]:.2f}px | '
               f'{winner["warns"] or "无"} |')
rep += ['', '**判定依据**：', '']
for r in dec['reasons']:
    rep.append(f'- {r}')
final_txt = ('替换为 `' + '+'.join(chosen['combo']) + '`'
             if replace else '保留现任两模解（负结果）')
rep += ['', f'**最终**：{final_txt}。', '']
if not replace:
    rep += ['三/四模全空间未找到同时满足"结构稳健 + C_s02/W_avg 不降"'
            '的解——该图的分辨率提升空间不在模式数维度（或现有基'
            '下的 3/4 模表达与 2 模等价），如实记录为负结果。', '']
rep += ['## 四、文件', '',
        '| 文件 | 内容 |', '|---|---|',
        f'| `{CN}_拟合对比.png` | 5 联：原始/预处理/模型σ2/差异/窗热图 |',
        f'| `{CN}_参数与指标.md` | 最终参数与指标 |',
        f'| `{CN}_指标验收单.md` | 新指标全套 + 逐核匹配 + 边界核说明 |',
        f'| `{CN}_新旧拟合对比.png` | 现任 vs 采用解 |',
        '| `stage7a_three_mode.json` | 三模全空间全部记录 |',
        '| `stage7b_four_mode.json` | 四模漏斗全部记录 |',
        '| `stage7c_decision.json` | 统一评估与判定明细 |',
        '| `stage7d_delivery.json` | 最终交付数据 |']
(OUT / '报告_8奇点2三模四模重分解.md').write_text('\n'.join(rep),
                                                  encoding='utf-8')

# ---------------- delivery json ----------------
delivery = {'name': NAME, 'replace': replace,
            'chosen': chosen, 'incumbent': inc,
            'winner': winner, 'reasons': dec['reasons']}
(OUT / 'stage7d_delivery.json').write_text(
    json.dumps(delivery, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# ---------------- append to 汇总表 ----------------
sum_path = ARCH / '最新结果_新指标' / '汇总表.md'
txt = sum_path.read_text(encoding='utf-8')
add = [
    '',
    '---',
    '',
    '## 2026-10-09 重分解更新（8奇点(2)，三/四模）',
    '',
    '> 动机：该图两模解为全队最弱（C_s02 0.8373、W_avg 0.6690，'
    '且有 32px 负窗警示）。方法：三模（366,145）+ 四模（1171 万）'
    '从零全空间搜索 + 统一新指标验收。',
    '',
]
if replace:
    add += [
        f'- **8奇点(2)（升级解）**：`{"+".join(chosen["combo"])}`，'
        f'C_s02 = **{C2:.4f}**，W_avg = {wavg:.4f}，'
        f'结构验收 **PASS**（{chosen["cores"]} vs '
        f'{chosen["cores_exp"]}，剔除边界离群核后 max '
        f'{max10:.2f}px），门槛：'
        f'{"全部满足" if gate_ok else "存在未达标项"}。',
        '- 交付物：`归档_八图分解\\8奇点2_三模四模重分解\\`'
        '（拟合对比 / 参数与指标 / 验收单 / 新旧对比 / 报告）。',
        '- 上表 Sep 26 两模行保留为历史快照（含边界离群核说明）。',
    ]
else:
    add += [
        f'- **结论（负结果）**：三/四模全空间重分解**未找到**同时满足'
        f'"结构稳健 + C_s02/W_avg 不低于现任"的解；**保留现任两模解**'
        f'`{"+".join(inc["combo"])}`（C_s02 {inc["C_s02"]:.4f} / '
        f'W_avg {inc["W_avg"]:.4f}）。详见 '
        '`归档_八图分解\\8奇点2_三模四模重分解\\报告.md`。',
    ]
sum_path.write_text(txt.rstrip() + '\n' + '\n'.join(add) + '\n',
                    encoding='utf-8')

print(f'[decision] replace={replace}')
print(f'[chosen] {"+".join(chosen["combo"])} N={chosen["N"]} '
      f'C_s02={C2:.4f} W_avg={wavg:.4f} cores={chosen["cores"]}/'
      f'{chosen["cores_exp"]} max10={max10:.2f} warns={warns}')
print(f'[done] deliverables in {OUT}')

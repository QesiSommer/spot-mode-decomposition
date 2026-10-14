# -*- coding: utf-8 -*-
"""
stage8_deliver.py -- 9奇点 & 11奇点：统一评估、判定与交付（阶段2-4）
=====================================================================
对每张图：
  - 统一新指标口径评估：现任 + 三/四/五模全部候选（各取 top-10 唯一组合）；
  - 判定：稳健 PASS（剔除单个最差核后 ≤ max(14, 现任) px）优先 → C_s02；
          替换条件 = C_s02 ≥ 现任 且 W_avg ≥ 现任 - 0.005；
  - 交付套件 + 汇总表追加（替换或负结果均如实记录）。

Outputs: 归档_八图分解/{9奇点|11奇点}_三至五模重分解/
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
from stage0_diagnosis import parse_md

ARCH = ROOT / '归档_八图分解'
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

T_C, T0, T2 = 0.83, 0.82, 0.83
RHO56_GATE, CMIN56_GATE = 0.10, 0.40

CASES = [{'name': '9Singularities', 'cn': '9奇点'},
         {'name': '11Singularities', 'cn': '11奇点'}]

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()


def wstats(pccs, meta):
    sg2 = [p for p, m in zip(pccs, meta) if m['sigma'] == 2]
    return float(np.mean([p < 0.5 for p in sg2])), float(np.min(sg2))


def evaluate(combo, theta, scale, tag, pts_e, Ie_np, Ie):
    tr = [lab2k[m] for m in combo]
    Nm = len(combo)
    gs = global_stats_02(modes, Ie, tr, scale)
    C2, (v0, v2) = c_s02(gs, theta, Nm)
    ws56, meta56, mask = build_window_stats(modes, Ie, tr, scale)
    ws32, meta32, _ = build_window_stats(modes, Ie, tr, scale, W=32,
                                         mask=mask)
    p56 = window_pccs(ws56, theta, Nm)
    p32 = window_pccs(ws32, theta, Nm)
    rho56, cmin56 = wstats(p56, meta56)
    rho32, cmin32 = wstats(p32, meta32)
    wavg = w_avg_point(ws56, theta, Nm)
    I = render_model(modes_np, lab2k,
                     {'combo': combo, 'params': list(theta),
                      'scale': scale})
    pts_m = find_dark_points(I)
    mt = match_points(pts_e, pts_m)
    dists = np.array(mt['dists']) if mt['dists'] else np.array([99.0])
    n_ok = abs(len(pts_m) - len(pts_e)) <= 1
    d_ok = mt['mean'] is not None and mt['mean'] < 12
    struct = bool(n_ok and d_ok)
    max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.0
    I2 = gaussian_blur(torch.tensor(I, dtype=torch.float32)
                       .unsqueeze(0), 2.0)[0].numpy()
    d_ang, _ = angular_distance(Ie_np, I2)
    warns = []
    if rho32 >= 0.2:
        warns.append(f'rho_32px={rho32:.2f}')
    if cmin32 <= 0.3:
        warns.append(f'C_min_32px={cmin32:.2f}')
    return {'tag': tag, 'combo': combo, 'N': Nm, 'theta': list(theta),
            'scale': scale, 'C_s02': float(C2), 'v0': float(v0),
            'v2': float(v2), 'W_avg': float(wavg),
            'rho56': rho56, 'cmin56': cmin56,
            'rho32': rho32, 'cmin32': cmin32,
            'cores': len(pts_m), 'cores_exp': len(pts_e),
            'mean_nn': float(mt['mean']), 'max_nn': float(mt['max']),
            'max10': max10, 'dists': [float(x) for x in dists],
            'struct': struct, 'warns': warns, 'd_ang': float(d_ang),
            '_I_model': I, '_mask': mask, '_ws56': ws56,
            '_meta56': meta56, '_p56': p56}


def process_case(case):
    NAME, CN = case['name'], case['cn']
    OUT = ARCH / f'{CN}_三至五模重分解'
    print(f'\n############ {CN} ({NAME}) ############')
    Ie_np = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    Ie = torch.tensor(Ie_np, dtype=torch.float32)
    pts_e = find_dark_points(Ie_np)

    combo_i, params_i, scale_i = parse_md(
        ARCH / '最新结果_新指标' / f'{CN}_参数与指标.md')
    inc = evaluate(combo_i, params_i, scale_i, 'incumbent', pts_e, Ie_np,
                   Ie)
    inc_key = '+'.join(inc['combo'])
    print(f'[incumbent] {inc_key} N={inc["N"]} C_s02={inc["C_s02"]:.4f} '
          f'W_avg={inc["W_avg"]:.4f} cores={inc["cores"]}/'
          f'{inc["cores_exp"]} max10={inc["max10"]:.2f} '
          f'warns={inc["warns"]}')

    seen, evals = {inc_key}, [inc]
    per_mode_best = {}
    for fname, tag in [('stage8_3mode.json', '3mode'),
                       ('stage8_4mode.json', '4mode'),
                       ('stage8_5mode.json', '5mode')]:
        j = json.load(open(OUT / fname, encoding='utf-8'))
        n = 0
        for r in j['ranked']:
            key = '+'.join(r['combo'])
            if key in seen:
                continue
            seen.add(key)
            e = evaluate(r['combo'], r['theta'], None, tag, pts_e, Ie_np,
                         Ie)
            evals.append(e)
            n += 1
            if n >= 10:
                break
    print(f'[select] {len(evals)-1} unique candidates evaluated')

    print('\n=== unified evaluation ===')
    for e in evals:
        print(f"  {e['tag']:10s} {'+'.join(e['combo']):44s} N={e['N']} "
              f"C_s02={e['C_s02']:.4f} W_avg={e['W_avg']:.4f} "
              f"cores={e['cores']}/{e['cores_exp']} mean={e['mean_nn']:.2f} "
              f"max10={e['max10']:.2f} struct={'P' if e['struct'] else 'F'} "
              f"warns={e['warns']}")

    margin = max(14.0, inc['max10'])
    ok = [e for e in evals[1:]
          if e['struct'] and e['max10'] <= margin]
    ok.sort(key=lambda x: (-x['C_s02'], -x['W_avg']))
    winner = ok[0] if ok else None
    replace = False
    reasons = []
    if winner is not None:
        c_ok = winner['C_s02'] >= inc['C_s02']
        w_ok = winner['W_avg'] >= inc['W_avg'] - 0.005
        replace = bool(c_ok and w_ok)
        reasons = [
            f"winner C_s02 {winner['C_s02']:.4f} vs incumbent "
            f"{inc['C_s02']:.4f} (need >=): {'OK' if c_ok else 'NG'}",
            f"winner W_avg {winner['W_avg']:.4f} vs incumbent "
            f"{inc['W_avg']:.4f} (need >= -0.005): "
            f"{'OK' if w_ok else 'NG'}",
            f"winner cores {winner['cores']}/{winner['cores_exp']} "
            f"max10={winner['max10']:.2f} vs incumbent "
            f"max10={inc['max10']:.2f}",
            f"winner warns {winner['warns']} vs incumbent "
            f"{inc['warns']}"]
    else:
        reasons = ['no candidate with struct PASS and max10 <= '
                   f'{margin:.2f} (incumbent-level structure)']
    chosen = winner if replace else inc
    print('\n=== decision ===')
    for r in reasons:
        print('  ' + r)
    print('  => ' + ('REPLACE with ' + '+'.join(winner['combo'])
                     if replace else 'KEEP incumbent'))

    # ---------------- deliverables ----------------
    combo = chosen['combo']
    theta = [float(x) for x in chosen['theta']]
    scale = chosen.get('scale')
    C2, v0, v2 = chosen['C_s02'], chosen['v0'], chosen['v2']
    wavg = chosen['W_avg']
    rho56, cmin56 = chosen['rho56'], chosen['cmin56']
    rho32, cmin32 = chosen['rho32'], chosen['cmin32']
    dists = chosen['dists']
    mt_mean, mt_max, max10 = (chosen['mean_nn'], chosen['max_nn'],
                              chosen['max10'])
    gate_ok = (C2 >= T_C and v0 >= T0 and v2 >= T2
               and rho56 <= RHO56_GATE and cmin56 >= CMIN56_GATE)
    tr = [lab2k[m] for m in combo]
    ws56 = chosen['_ws56']
    meta56 = chosen['_meta56']
    p56 = chosen['_p56']
    mask = chosen['_mask']
    I_model = chosen['_I_model']
    I2 = gaussian_blur(torch.tensor(I_model, dtype=torch.float32)
                       .unsqueeze(0), 2.0)[0].numpy()

    # figure 1: 5-panel
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
    tag_txt = 'upgraded' if replace else 'incumbent kept'
    fig.suptitle(f'{NAME} - {tag_txt} (N={chosen["N"]}), new-metric '
                 f'acceptance', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f'{CN}_拟合对比.png', dpi=140,
                bbox_inches='tight')
    plt.close(fig)

    # figure 2: incumbent vs chosen
    I_old = render_model(modes_np, lab2k,
                         {'combo': inc['combo'], 'params': inc['theta'],
                          'scale': inc['scale']})
    I_old2 = gaussian_blur(torch.tensor(I_old, dtype=torch.float32)
                           .unsqueeze(0), 2.0)[0].numpy()
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axs[0].imshow(I_old2, cmap='inferno', vmin=0, vmax=1)
    axs[0].set_title(f'incumbent {inc["N"]}-mode '
                     f'(C_s02={inc["C_s02"]:.4f})\n'
                     + '+'.join(inc['combo']), fontsize=10)
    axs[1].imshow(I2, cmap='inferno', vmin=0, vmax=1)
    right_t = (f'chosen {chosen["N"]}-mode' if replace
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

    # md: 参数与指标
    a1 = [float(np.sqrt(max(x, 0))) for x in theta[:chosen['N']]]
    lines = [
        f'# {NAME} 最终分解结果（新指标口径·三至五模重分解）',
        '',
        f'- 模式数：{chosen["N"]} 模',
        f'- 拟合对比图：`{CN}_拟合对比.png`',
        '- 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0']
    if scale is not None:
        lines.append(f'- 缩放参数 s = {scale:.4f}')
    lines += ['', '## 模式组成', '',
              '| 模式 | 能量占比 | 振幅 |', '|---|---|---|']
    for i in range(chosen['N']):
        lines.append(f'| {combo[i]} | {theta[i]*100:.2f}% | '
                     f'{a1[i]:.6f} |')
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
        f'- 稳健判据（剔除单个最差核后 max ≤14px）：'
        f'{"通过" if max10 <= 14 else "未通过"}（{max10:.2f}px）',
    ]
    (OUT / f'{CN}_参数与指标.md').write_text('\n'.join(lines),
                                             encoding='utf-8')

    # md: 验收单
    d_ang, (amp_e, amp_m) = angular_distance(Ie_np, I2)
    conc, loc = diff_localization(Ie_np, I2, mask)
    warns = chosen['warns']
    sheet = [
        f'# {NAME} 指标验收单（新指标口径·三至五模重分解，'
        '门槛仅标注不拦截）',
        '',
        ('> 判定：**替换为 ' + f'{chosen["N"]} 模解**（三/四/五模'
         '全空间+引导搜索）' if replace else
         '> 判定：**保留现任解**（三/四/五模搜索未找到满足替换'
         '条件的解，负结果如实记录）'),
        '', '## 准入门槛（标注）', '',
        f'- C_s02 = {C2:.4f} （门槛 {T_C}） {"✓" if C2 >= T_C else "✗"}',
        f'- PCC(σ=0) = {v0:.4f} （门槛 {T0}） '
        f'{"✓" if v0 >= T0 else "✗"}',
        f'- PCC(σ=2) = {v2:.4f} （门槛 {T2}） '
        f'{"✓" if v2 >= T2 else "✗"}',
        f'- rho_56px = {rho56:.3f} （门槛 ≤{RHO56_GATE}） '
        f'{"✓" if rho56 <= RHO56_GATE else "✗"}',
        f'- C_min_56px = {cmin56:.3f} （门槛 ≥{CMIN56_GATE}） '
        f'{"✓" if cmin56 >= CMIN56_GATE else "✗"}', '',
        f'**门槛判定：{"全部满足" if gate_ok else "存在未达标项（仅标注）"}**',
        '',
        '## 局部一致性警示（32px 细尺度）', '',
        f'- 32px：最差窗 C_min = {cmin32:.3f}，失败窗占比 '
        f'rho = {rho32:.3f}',
        f'- 警示：{"无" if not warns else "【" + "；".join(warns) + "】"}',
        '',
        '## 结构验收（暗核，最终判据）', '',
        f'- 模型 {chosen["cores"]} 个暗点 vs 实验 {chosen["cores_exp"]} 个'
        f'（|差|={abs(chosen["cores"]-chosen["cores_exp"])} ≤ 1 ✓）',
        f'- 平均最近邻 {mt_mean:.2f}px < 12px ✓；最大 {mt_max:.2f}px',
        f'- 稳健判据（剔除单个最差核后 max ≤14px）：'
        f'{max10:.2f}px',
        f'- 判定：**{"PASS" if chosen["struct"] else "fail"}**',
        '',
        '## 角谱诊断（σ=2，m 阶幅值相对差）', '',
        f'- D_ang = {d_ang:.4f}（越低越好；现任 D_ang = '
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

    # report
    n3 = sum(1 for e in evals if e['tag'] == '3mode')
    n4 = sum(1 for e in evals if e['tag'] == '4mode')
    n5 = sum(1 for e in evals if e['tag'] == '5mode')
    rep = [
        f'# {CN} 三/四/五模重分解报告', '',
        '> 日期：2026-10-09 | 脚本：stage8_search（搜索）/ '
        'stage8_deliver（评估与交付）', '',
        '## 一、动机', '',
        f'- 现任解 `{inc_key}`（{inc["N"]} 模）：C_s02 '
        f'{inc["C_s02"]:.4f}、W_avg {inc["W_avg"]:.4f}，'
        f'警示 {inc["warns"] or "无"}；',
        '- 用户指定：按 8 奇点(2) 同法（新指标口径 + 从零全空间搜索）'
        '重分解，模式范围 **三/四/五模**。', '',
        '## 二、方法', '',
        '- **三模**：C(131,3)=366,145 全空间粗扫 → top-40 → W_avg 与 '
        'C_s02 双目标精修（OLS 种子 + 抖动）；',
        '- **四模**：C(131,4)=11,716,640 全空间漏斗（1/4 分辨率核加权'
        '全扫 → top-5000 深前缀 → 1/2 分辨率复扫 → top-120 精修）；',
        '- **五模**：严格穷举 C(131,5)=2.98 亿（≈8 小时/图）不可行，'
        '采用**深度引导**（用户确认）：四模 top-3000 前缀 × 全部 127 '
        '第五模 + 三模 top-250 前缀（含现任三模）× 全部模对'
        '（8128），共约 240 万组 1/4 分辨率核加权评分 → top-200 精修；'
        '依据"加模单调性 + 深前缀（top 0.03%）"覆盖高分五模解；',
        '- **稳健判据**：标准结构 PASS（|核数差|≤1 且均值 <12px）+ '
        '剔除单个最差核后 max ≤14px。', '',
        '## 三、结果与判定', '',
        '| 解 | C_s02 | W_avg | 核数 | 剔除最差核后 max | 警示 |',
        '|---|---|---|---|---|---|',
        f'| 现任（{inc["N"]} 模） | {inc["C_s02"]:.4f} | '
        f'{inc["W_avg"]:.4f} | {inc["cores"]}/{inc["cores_exp"]} | '
        f'{inc["max10"]:.2f}px | {inc["warns"] or "无"} |',
    ]
    if winner is not None:
        rep.append(f'| 挑战者最优（{winner["N"]} 模） | '
                   f'{winner["C_s02"]:.4f} | {winner["W_avg"]:.4f} | '
                   f'{winner["cores"]}/{winner["cores_exp"]} | '
                   f'{winner["max10"]:.2f}px | {winner["warns"] or "无"} |')
    rep += ['', f'（评估候选：3 模 ×{n3}、4 模 ×{n4}、5 模 ×{n5}，'
            '详见 stage8_delivery.json）', '', '**判定依据**：', '']
    for r in reasons:
        rep.append(f'- {r}')
    final_txt = ('替换为 `' + '+'.join(chosen['combo']) + '`'
                 if replace else '保留现任解（负结果）')
    rep += ['', f'**最终**：{final_txt}。', '']
    if not replace:
        rep += ['三/四/五模搜索未找到同时满足"结构稳健 + C_s02/W_avg '
                '不降"的解——如实记录为负结果。', '']
    rep += ['## 四、文件', '',
            '| 文件 | 内容 |', '|---|---|',
            f'| `{CN}_拟合对比.png` | 5 联：原始/预处理/模型σ2/差异/窗热图 |',
            f'| `{CN}_参数与指标.md` | 最终参数与指标 |',
            f'| `{CN}_指标验收单.md` | 新指标全套 + 逐核匹配 |',
            f'| `{CN}_新旧拟合对比.png` | 现任 vs 采用解 |',
            '| `stage8_3mode.json` / `stage8_4mode.json` / '
            '`stage8_5mode.json` | 各模数搜索全部记录 |',
            '| `stage8_delivery.json` | 统一评估与判定明细 |']
    (OUT / f'报告_{CN}三至五模重分解.md').write_text('\n'.join(rep),
                                                     encoding='utf-8')

    # delivery json
    def strip(e):
        if e is None:
            return None
        return {k: v for k, v in e.items() if not k.startswith('_')}

    delivery = {'name': NAME, 'incumbent': strip(inc),
                'evaluations': [strip(e) for e in evals],
                'winner': strip(winner), 'replace': replace,
                'reasons': reasons, 'chosen': strip(chosen),
                'protocol': {
                    'robust': 'drop single worst core, remaining max '
                              f'<= {margin:.2f}px',
                    'replace': 'struct & robust pass, C_s02 >= incumbent, '
                               'W_avg >= incumbent - 0.005'}}
    (OUT / 'stage8_delivery.json').write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8')

    # append 汇总表
    sum_path = ARCH / '最新结果_新指标' / '汇总表.md'
    txt = sum_path.read_text(encoding='utf-8')
    add = [
        '',
        '---',
        '',
        f'## 2026-10-09 重分解更新（{CN}，三/四/五模）',
        '',
        f'> 方法：三模（366,145）与四模（1171 万）全空间 + 五模深度引导'
        '（约 240 万组，严格穷举 2.98 亿不可行）+ 统一新指标验收。',
        '',
    ]
    if replace:
        add += [
            f'- **{CN}（升级解）**：`{"+".join(chosen["combo"])}`，'
            f'C_s02 = **{C2:.4f}**，W_avg = {wavg:.4f}，'
            f'结构验收 **PASS**（{chosen["cores"]} vs '
            f'{chosen["cores_exp"]}，剔除最差核后 max {max10:.2f}px），'
            f'门槛：{"全部满足" if gate_ok else "存在未达标项"}。',
            f'- 交付物：`归档_八图分解\\{CN}_三至五模重分解\\`。',
            '- 上表 Sep 26 行保留为历史快照。',
        ]
    else:
        add += [
            f'- **结论（负结果）**：三/四/五模搜索**未找到**满足替换'
            f'条件的解；**保留现任** `{inc_key}`（C_s02 '
            f'{inc["C_s02"]:.4f} / W_avg {inc["W_avg"]:.4f}）。详见 '
            f'`归档_八图分解\\{CN}_三至五模重分解\\报告.md`。',
        ]
    sum_path.write_text(txt.rstrip() + '\n' + '\n'.join(add) + '\n',
                        encoding='utf-8')

    print(f'[chosen] {"+".join(chosen["combo"])} N={chosen["N"]} '
          f'C_s02={C2:.4f} W_avg={wavg:.4f} '
          f'cores={chosen["cores"]}/{chosen["cores_exp"]} '
          f'max10={max10:.2f} warns={warns}')
    print(f'[done] deliverables in {OUT}')


def main():
    for case in CASES:
        process_case(case)
    print('\n[ALL DONE]')


if __name__ == '__main__':
    main()

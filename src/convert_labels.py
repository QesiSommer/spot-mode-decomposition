# -*- coding: utf-8 -*-
"""
convert_labels.py — 旧结果记号换算工具（内部记号 → 标准记号）
================================================================
背景（2026-10-16）：修正前的工作区基（hg_torch.py 等 3 个文件的
indexing='ij' 轴序）与学术/MATLAB 标准约定互为转置。代码已修正，
历史 JSON/报告中的旧记号需经本工具换算后方可对外使用。

换算规则（数值验证 pcc=1.000000、八图回归 ALL PASS）：
  1. 模式标签：  HG(m,n) → HG(n,m)
                LG(p,l) → LG(p,−l)
  2. 相对相位（以模式 0 为基准，φ₀≡0）：E 整体提出公因子
     e^{+i·l₀·π/2}（不影响强度），故
                φ_k → φ_k + (l_k − l₀)·π/2   （k≥1；l 为【内部】
                记号的角量子数，HG 记 l=0；l₀ 为模式 0 的 l）
  3. 其余参数（能量占比、振幅、sx、sy、旋转、全局平移）全部不变
     ——它们作用于数组空间，与标签转置无关。

用法：
  from convert_labels import convert_combo, convert_solution
  std_combo, std_theta = convert_solution(combo, theta)
  std_label = convert_combo(['LG(0,-3)', 'HG(3,4)'])
"""
import numpy as np


def parse_mode(s):
    """'LG(0,-3)' -> ('LG', 0, -3)；'HG(3,4)' -> ('HG', 3, 4)"""
    s = s.strip()
    fam = s[:2]
    a, b = (int(v) for v in s[3:-1].split(','))
    return fam, a, b


def convert_combo(combo):
    """内部记号组合 → 标准记号组合（仅标签）。"""
    out = []
    for s in combo:
        fam, a, b = parse_mode(s)
        out.append(f'HG({b},{a})' if fam == 'HG' else f'LG({a},{-b})')
    return out


def _l_of(s):
    fam, _a, b = parse_mode(s)
    return b if fam == 'LG' else 0


def convert_solution(combo, theta):
    """
    完整换算：模式组合 + theta 向量（前 N 能量、后 N−1 相位）。

    返回 (std_combo, std_theta)：
      std_combo : 标准记号标签列表
      std_theta : 长度不变；能量段原样，相位段按
                  φ'_k = φ_k + (l_k − l₀)·π/2 修正并折回 (−π, π]
    """
    combo = list(combo)
    N = len(combo)
    theta = list(theta)
    if len(theta) != 2 * N - 1:
        raise ValueError(f'theta 长度 {len(theta)} ≠ 2N−1={2*N-1}')
    l0 = _l_of(combo[0])
    std_theta = list(theta)
    for i in range(1, N):
        phi = theta[N + i - 1] + (_l_of(combo[i]) - l0) * np.pi / 2
        std_theta[N + i - 1] = (phi + np.pi) % (2 * np.pi) - np.pi
    return convert_combo(combo), std_theta


def std_expr(combo, theta):
    """
    生成 MATLAB 可直用的叠加式字符串（标准约定）。
    例：'E = 0.4992*LG(0,3)(x,y) + 0.3711*exp(1i*4.2799)*LG(1,3)(x,y)'
    """
    combo_std, theta_std = convert_solution(combo, theta)
    N = len(combo)
    terms = []
    for i, lab in enumerate(combo_std):
        amp = float(np.sqrt(max(theta_std[i], 0)))
        phi = 0.0 if i == 0 else float(theta_std[N + i - 1])
        t = f'{amp:.4f}*{lab}(x,y)'
        if i > 0:
            t = f'{amp:.4f}*exp(1i*{phi:+.4f})*{lab}(x,y)'
        terms.append(t)
    return 'E = ' + ' + '.join(terms)


if __name__ == '__main__':
    # 自检：用 v3 JSON 的八图实测解跑一遍换算并打印
    import json
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent
    V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                        / 'terminal_inspection_v3.json',
                        encoding='utf-8'))
    for cn in ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
               '10奇点', '11奇点']:
        r = V3[cn]
        src = r['accepted'] if 'theta' in r.get('accepted', {}) \
            else r['refined_v3']
        combo_std, theta_std = convert_solution(r['combo'],
                                                src['theta'])
        print(f'{cn:8s} {"+".join(combo_std)}')
        print(f'         {std_expr(r["combo"], src["theta"])}')

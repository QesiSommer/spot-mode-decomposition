# -*- coding: utf-8 -*-
"""
conv_shijie_doc.py — 师姐文档补齐（4/5/8(1)/10 图 LG 叠加式）
================================================================
背景：师姐用叠加式_标准约定.md 中 4/5/8(1)/10 四图只给了
"请用复场文件"的占位（当时 LG 角向相位约定未定）。现约定已
裁决（LG(p,l)→LG(p,−l)、φ+l·π/2 相对补偿，回归 ALL PASS），
补齐四图的手写叠加式 + 修正既有 HG 图的 dof 换算说明。

dof 换算结论（回归证明）：sx/sy/shift 数值不变；rot 仅 HG 模
有意义且数值不变；LG 相位按 φ'_k=φ_k+(l_k−l0)·π/2 换算。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))
OUT = ROOT / '归档_八图分解' / '师姐复核检查_2026-10-16'

ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
         '10奇点', '11奇点']


def parse_mode(s):
    fam, idx = s.strip()[:2], s.strip()[3:-1]
    a, b = (int(v) for v in idx.split(','))
    return fam, a, b


lines = [
    '# 师姐可直接使用的叠加式（完整版，标准 MATLAB 约定）',
    '',
    '**约定**（2026-10-16 全部对齐后）:',
    '- HG(m,n) = H_m(x)·H_n(y)·G，x 水平、y 竖直（m=水平阶数）。',
    '- LG(p,l)：θ = atan2(y,x)（y 竖直、x 水平），l 为拓扑荷。',
    '- 坐标：x,y 以像素为单位后除以 28（224px 视场 = ±4 个束腰 w0）。',
    '- 相位约定：E = Σ aᵢ·exp(iφᵢ)·Fᵢ，φ₁ = 0（相对相位）；'
    '强度 = |E|²。',
    '- 标签换算（我方内部 → 标准）：HG(m,n)→HG(n,m)；'
    'LG(p,l)→LG(p,−l)。',
    '- 相位换算：相对相位 φ_k(标准) = φ_k(内部) + (l_k−l₀)·π/2，'
    'l₀ = 模式 1 的角量子数（HG 模记 l=0）。',
    '',
    '---',
    '',
]

for cn in ORDER:
    r = V3[cn]
    src = r['accepted'] if 'theta' in r.get('accepted', {}) \
        else r['refined_v3']
    combo = r['combo']
    N = r['N']
    theta = [float(x) for x in src['theta']]
    sx = [float(x) for x in src['sx']]
    sy = [float(x) for x in src['sy']]
    shiftv = [float(x) for x in src['shift']]
    rot = [float(x) for x in src.get('rot', [0.0] * N)]
    c_val = float(src.get('C_s02', r['delivery']['C_s02']))

    # 换算
    l_of = [parse_mode(s)[2] if parse_mode(s)[0] == 'LG' else 0
            for s in combo]
    l0 = l_of[0]
    combo_std, ph_std = [], [0.0]
    for i, s in enumerate(combo):
        fam, a, b = parse_mode(s)
        if fam == 'HG':
            combo_std.append((f'HG({b},{a})', np.sqrt(theta[i]), None))
        else:
            combo_std.append((f'LG({a},{-b})', np.sqrt(theta[i]), a))
        if i > 0:
            ph = (theta[N + i - 1] + (l_of[i] - l0) * np.pi / 2
                  + np.pi) % (2 * np.pi) - np.pi
            ph_std.append(ph)

    lines.append(f'## {r["name"]}（{cn}，C_s02 v3={c_val:.4f}）')
    lines.append('')
    # 叠加式
    terms = []
    for i, (lab, amp, _p) in enumerate(combo_std):
        if i == 0:
            terms.append(f'{amp:.4f}*{lab}(x,y)')
        else:
            terms.append(f'{amp:.4f}*exp(1i*{ph_std[i]:+.4f})*'
                         f'{lab}(x,y)')
    lines.append('- E = ' + ' + '.join(terms))
    lines.append('- LG 用 MATLAB 标准式 '
                 'LG(p,l)=(√2r)^|l|·L_p^|l|(2r²)·e^{−r²}·e^{i·l·θ}，'
                 'θ=atan2(y,x)。')
    # dof
    lines.append('- 逐模微调（提升用，可省略）：'
                 f'sx={sx}, sy={sy}, shift={shiftv}, '
                 f'rot={rot}（rot 仅 HG 模有意义）')
    lines.append(f'- 相位换算说明：模式 1 的 l₀={l0}，'
                 'LG 模相位已含 (l−l₀)·π/2 补偿。')
    lines.append('')

md = '\n'.join(lines)
target = OUT / '师姐用叠加式_标准约定_完整版.md'
target.write_text(md, encoding='utf-8')
print(f'DONE -> {target}')
print(md[:1500])

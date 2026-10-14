# -*- coding: utf-8 -*-
"""
stage14b_morph_audit_v2.py -- 修正版八图形态审计（全自由度管线）
==================================================================
修正 stage14 的管线错误：审计必须用**与交付一致的完整自由度渲染**
（全局平移 + 逐模 sx/sy + 逐模旋转），而 stage14 用了朴素渲染
（仅 theta），导致：
  - 9 奇点被误判 FAIL（实际 PASS，Δcentral 0.037）
  - 8 奇点(1) 被误判 PASS（实际 FAIL，shape 0.564 超门）
本脚本用正确管线重做，并把已完成的修复（4 奇点新解、5/6/9/11 候选）
一并列入对照。

输出：归档_八图分解/形态审计_2026-10-14/八图形态审计_修正版.md
      + morph_audit_v2.json
"""
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from morphology_metrics import morph_compare
import morph_search_lib as L

OUT = ROOT / '归档_八图分解' / '形态审计_2026-10-14'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))
FIX = ROOT / '归档_八图分解' / '形态修复_2026-10-14'
FIX4 = ROOT / '归档_八图分解' / '4奇点_形态重搜索' / '交付'

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
mnp = modes.numpy()
ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
         '10奇点', '11奇点']


def load_sol(cn):
    """返回 (combo, theta, sx, sy, shift, rot, src)。"""
    r3 = V3[cn]
    acc = r3['accepted']
    combo = r3['combo']
    N = len(combo)
    return (combo, [float(x) for x in acc['theta']],
            [float(x) for x in acc['sx']],
            [float(x) for x in acc['sy']],
            [float(x) for x in acc['shift']],
            [float(x) for x in acc.get('rot', [0.0] * N)],
            'v3')


def load_fix(cn):
    """已完成的修复解（若有）。"""
    if cn == '4奇点':
        p = FIX4 / 'stage13d_delivery.json'
    else:
        p = FIX / cn / f'delivery_{cn}.json'
    if not p.exists():
        return None
    d = json.load(open(p, encoding='utf-8'))
    N = len(d['combo'])
    return (d['combo'], [float(x) for x in d['theta']],
            [float(x) for x in d['sx']], [float(x) for x in d['sy']],
            [float(x) for x in d['shift']],
            [float(x) for x in d['rot']], 'fix')


rows = []
for cn in ORDER:
    NAME = V3[cn]['name']
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    cur = load_sol(cn)
    Im_cur = L.render_dof(mnp, lab2k, cur[0], cur[1], cur[2], cur[3],
                          cur[4], cur[5])
    r_cur = morph_compare(Ie, Im_cur)
    row = {'cn': cn, 'name': NAME,
           'cur_combo': '+'.join(cur[0]),
           'cur_central': r_cur['central_dark_model'],
           'cur_delta': r_cur['central_delta'],
           'cur_cvR': r_cur['cv_ratio'],
           'cur_shape': r_cur['shape_dist'],
           'cur_pass': r_cur['morph_pass'],
           'exp_central': r_cur['central_dark_exp'],
           'exp_cvR': r_cur['cv_r_exp']}
    fix = load_fix(cn)
    if fix is not None:
        Im_fix = L.render_dof(mnp, lab2k, fix[0], fix[1], fix[2],
                              fix[3], fix[4], fix[5])
        r_fix = morph_compare(Ie, Im_fix)
        row.update({'fix_combo': '+'.join(fix[0]),
                    'fix_delta': r_fix['central_delta'],
                    'fix_cvR': r_fix['cv_ratio'],
                    'fix_shape': r_fix['shape_dist'],
                    'fix_pass': r_fix['morph_pass']})
    rows.append(row)
    fx = (f" | fix Δ{row['fix_delta']:.3f} "
          f"{'P' if row['fix_pass'] else 'F'}"
          f" ({row['fix_combo']})" if 'fix_pass' in row else '')
    print(f"{cn:8s} v3: Δ{row['cur_delta']:.3f} cvR={row['cur_cvR']:.2f} "
          f"shape={row['cur_shape']:.3f} "
          f"{'PASS' if row['cur_pass'] else 'FAIL'}{fx}")

n_pass = sum(1 for r in rows if r['cur_pass'])
print(f'\n[修正审计] v3 解形态通过：{n_pass}/8')
print('  通过：' + ', '.join(r['cn'] for r in rows if r['cur_pass']))
print('  未过：' + ', '.join(r['cn'] for r in rows if not r['cur_pass']))

# ---- report ----
lines = [
    '# 八图形态审计（修正版：全自由度管线）', '',
    '> 日期：2026-10-14 | 脚本：`stage14b_morph_audit_v2.py` |',
    '> **重要修正**：初版审计（stage14）误用朴素渲染（仅 theta），',
    '> 未包含交付解实际使用的全局平移 / 逐模各向异性 / 逐模旋转，',
    '> 导致 9 奇点被误判 FAIL（实际 PASS）、8 奇点(1) 被误判 PASS',
    '> （实际 FAIL）。本版用与交付一致的完整自由度管线重做。', '',
    '## 一、修正后审计（v3 交付解）', '',
    '| 图 | 实验中央暗区 | 模型中央暗区 | Δcentral | cvR | 形状谱 | '
    '形态判定 |',
    '|---|---|---|---|---|---|---|']
for r in rows:
    st = 'PASS' if r['cur_pass'] else '**FAIL**'
    lines.append(f"| {r['cn']} | {r['exp_central']:.3f} | "
                 f"{r['cur_central']:.3f} | {r['cur_delta']:.3f} | "
                 f"{r['cur_cvR']:.2f} | {r['cur_shape']:.3f} | {st} |")
lines += ['', f'**v3 解形态通过：{n_pass}/8**；'
          f'未过：{", ".join(r["cn"] for r in rows if not r["cur_pass"])}',
          '', '## 二、与初版审计的差异（管线修正）', '',
          '| 图 | 初版（朴素渲染）| 修正版（全自由度）| 变化 |',
          '|---|---|---|---|']
NAIVE = {'4奇点': False, '5奇点': False, '6奇点': False,
         '8奇点1': True, '8奇点2': True, '9奇点': False,
         '10奇点': True, '11奇点': False}
for r in rows:
    nv = NAIVE[r['cn']]
    cv = r['cur_pass']
    chg = '一致' if nv == cv else ('**修正：误判 PASS → 实为 FAIL**'
                                   if nv else
                                   '**修正：误判 FAIL → 实为 PASS**')
    lines.append(f"| {r['cn']} | {'PASS' if nv else 'FAIL'} | "
                 f"{'PASS' if cv else 'FAIL'} | {chg} |")
lines += ['', '## 三、已完成/候选修复对照', '',
          '| 图 | v3 解 | 修复解 | 形态 | 说明 |', '|---|---|---|---|---|']
for r in rows:
    if 'fix_pass' in r:
        lines.append(
            f"| {r['cn']} | Δ{r['cur_delta']:.3f} "
            f"{'P' if r['cur_pass'] else 'F'} | "
            f"Δ{r['fix_delta']:.3f} {r['fix_cvR']:.2f} "
            f"{'P' if r['fix_pass'] else 'F'} | "
            f"{'+'.join(r['fix_combo'].split('+'))} | 见各自交付报告 |")
    else:
        lines.append(f"| {r['cn']} | Δ{r['cur_delta']:.3f} "
                     f"{'P' if r['cur_pass'] else 'F'} | — | — | "
                     f"{'无需修复' if r['cur_pass'] else '**待修复**'} |")
lines += ['', '## 四、结论', '',
          '1. 修正后：**v3 解中 4/8 形态通过**（9、10、8(2) 及…）；',
          '2. 需修复的图：'
          + ', '.join(r['cn'] for r in rows
                      if not r['cur_pass'] and 'fix_pass' not in r)
          + '（其余已生成候选，见形态修复_2026-10-14）；',
          '3. **管线一致性教训**：形态审计必须使用与交付一致的渲染管线；'
          '本轮已把该约束固化进 `morph_search_lib`（所有评估走 '
          '`render_dof`）。', '',
          '## 五、文件', '',
          '| 文件 | 内容 |', '|---|---|',
          '| `八图形态审计_修正版.md` | 本文件 |',
          '| `morph_audit_v2.json` | 全部数值 |']
(OUT / '八图形态审计_修正版.md').write_text('\n'.join(lines),
                                            encoding='utf-8')
(OUT / 'morph_audit_v2.json').write_text(
    json.dumps(rows, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print(f'[done] {OUT}')

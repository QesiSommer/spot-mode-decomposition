# -*- coding: utf-8 -*-
"""
stage12b_v3_bundle.py -- v3 顶层归档（另存）+ 汇总表合并更新
=============================================================
1. 汇总表：把 10 奇点-only 的 v3 节替换为八图完整 v3 节（幂等）；
2. 新建顶层归档 `八图最新分解结果_v3_2026-10-12/`（不覆盖旧的
   2026-10-09 版）：
   - 每图子文件夹：拟合对比（v3）/ 参数与指标 / 验收单 / v2-v3 对比
     （8奇点1 为三件套）；
   - 顶层：新版《八图最终分解结果汇总.md》（v3 总表 + 历轮纪要 +
     指针）+ 汇总表副本。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCH = ROOT / '归档_八图分解'
V3DIR = ARCH / '终检精修v3_2026-10-12'
BUNDLE = ROOT / '八图最新分解结果_v3_2026-10-12'
BUNDLE.mkdir(parents=True, exist_ok=True)

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
ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
         '10奇点', '11奇点']
NAMES = {'4奇点': '4Singularities', '5奇点': '5Singularities',
         '6奇点': '6Singularities', '8奇点1': '8Singularities(1)',
         '8奇点2': '8Singularities(2)', '9奇点': '9Singularities',
         '10奇点': '10Singularities', '11奇点': '11Singularities'}

v3 = json.load(open(V3DIR / 'terminal_inspection_v3.json',
                    encoding='utf-8'))

# ---- copy files into the new bundle ----
for cn in ORDER:
    d = BUNDLE / cn
    d.mkdir(exist_ok=True)
    for fn in FNAMES[cn]:
        if fn is None:
            continue
        src = V3DIR / cn / fn
        if src.exists():
            (d / fn).write_bytes(src.read_bytes())
    print('bundled', cn)

# ---- update master 汇总表 (consolidate v3 section to all 8) ----
sum_path = ARCH / '最新结果_新指标' / '汇总表.md'
txt = sum_path.read_text(encoding='utf-8')
marker = '\n---\n\n## 2026-10-12 终检v3更新'
idx = txt.find(marker)
if idx >= 0:
    txt = txt[:idx]
add = ['', '---', '',
       '## 2026-10-12 终检v3更新（八图，并入逐模旋转）', '',
       '> 依据残差专项③：逐模旋转（像散轴取向）为有效自由度。'
       '八图在 v2 基础上全部重跑。', '',
       '| 图 | C_s02 v2 → v3 | 核 max10 | 主要旋转角 |',
       '|---|---|---|---|']
for cn in ORDER:
    r = v3[cn]
    b = r['baseline_v2']['C_s02']
    a = r['delivery']['C_s02']
    rot = r['accepted']['rot']
    rot_txt = ', '.join(f'{v:+.2f}' for v in rot
                        if abs(v) > 0.5) or '（全部 <0.5°）'
    add.append(f"| {cn} | {b:.4f} → **{a:.4f}** | "
               f"{r['delivery']['max10']:.1f}px | {rot_txt} |")
add += ['', '- 交付物：`归档_八图分解\\终检精修v3_2026-10-12\\`；'
        '顶层归档另存为 `八图最新分解结果_v3_2026-10-12\\`'
        '（2026-10-09 版保留未覆盖）。']
sum_path.write_text(txt.rstrip() + '\n' + '\n'.join(add) + '\n',
                    encoding='utf-8')
(BUNDLE / '汇总表_含全部更新记录.md').write_bytes(
    sum_path.read_bytes())

# ---- fresh bundle summary ----
lines = [
    '# 八图最终分解结果汇总（v3 版，2026-10-12）', '',
    '> 本文件夹为八张实验图（4~11 奇点）**终检 v3** 最终结果的统一'
    '归档（另存版；2026-10-09 版顶层归档保留未覆盖）。',
    '> v3 = v2（全局平移 + 逐模束腰 + 逐模像散 sx/sy）**＋ 逐模旋转'
    '（像散轴取向）**。',
    '> 口径：C_s02（主口径）=[PCC(σ=0)+PCC(σ=2)]/2；W_avg=56px '
    '窗口等权平均；结构验收（暗核匹配）为最终判据。', '',
    '## 一、总表（v3）', '',
    '| 图 | 模式数 | 最终组合 | C_s02 | W_avg | 结构验收 | 警示 |',
    '|---|---|---|---|---|---|---|']
for cn in ORDER:
    r = v3[cn]
    d_ = r['delivery']
    combo = '+'.join(r['combo'])
    warns = '；'.join(d_['warns']) if d_['warns'] else '无'
    lines.append(f"| {cn} | {r['N']} | {combo} | "
                 f"**{d_['C_s02']:.4f}** | {d_['W_avg']:.4f} | "
                 f"PASS（{d_['cores']}vs{d_['cores_exp']}） | {warns} |")
lines += [
    '', '## 二、历轮纪要', '',
    '- **2026-09-26 基线**：新指标管线全量复跑（口径定型：C_s02/'
    'W_avg/56px·32px 警示/结构验收）；',
    '- **2026-10-09**：10 奇点修复（管线选择偏差）、8(1) 接受决议、'
    '8(2)/9/11 三至五模重分解升级；',
    '- **2026-10-10 终检**：并入全局平移（对中残差 0.25~3.5px）'
    '+ 逐模束腰；',
    '- **2026-10-11 终检 v2**：并入逐模像散 (sx, sy)（探针 F 验证，'
    '±10~25% 各向异性）；',
    '- **2026-10-12 终检 v3**：并入逐模旋转（残差专项③，'
    '10 奇点 HG(3,4) -3.5° 为首个突破 0.90 的主信号）。', '',
    '## 三、文件', '',
    '- 每图子文件夹：`{图}_拟合对比.png`（v3 五联）/ `{图}_参数与'
    '指标.md`（含 sx/sy/旋转角）/ `{图}_指标验收单.md` / '
    '`{图}_新旧拟合对比.png`（v2-vs-v3；8奇点1 为三件套）；',
    '- `汇总表_含全部更新记录.md`：主汇总表全量（历轮更新节）；',
    '- 过程材料（搜索/精修/探针脚本与全记录 JSON）：保留在 '
    '`归档_八图分解\\` 对应文件夹。']
(BUNDLE / '八图最终分解结果汇总.md').write_text('\n'.join(lines),
                                                 encoding='utf-8')
print('[done] new bundle:', BUNDLE)

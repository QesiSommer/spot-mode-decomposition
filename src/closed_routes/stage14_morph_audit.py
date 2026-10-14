# -*- coding: utf-8 -*-
"""
stage14_morph_audit.py -- C：八图形态审计 + 验收模板升级
============================================================
用 morphology_metrics 对八图当前交付解（v3）逐一做形态体检，
输出：
  - 归档_八图分解/形态审计_2026-10-14/八图形态审计.md
  - 八图形态审计.png（每图：实验 | 模型 | 中央暗区/轮廓谱对比）
  - 形态审计.json（供后续验收单引用）

判定：central 差 ≤0.12、cv 比 ≥0.55、形状谱距离 ≤0.55 —— 三项须全过。
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
from calibrate_metrics import render_model
from stage0_diagnosis import parse_md
from morphology_metrics import morph_compare, morph_table_row

OUT = ROOT / '归档_八图分解' / '形态审计_2026-10-14'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
mnp = modes.numpy()

SOURCES = {
    '4奇点': ('最新结果_新指标', '4奇点_参数与指标.md'),
    '5奇点': ('局部优化_5及8奇点/最终交付_5及8奇点',
             '5奇点_参数与指标.md'),
    '6奇点': ('最新结果_新指标', '6奇点_参数与指标.md'),
}
ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
         '10奇点', '11奇点']

rows, reports = [], {}
for cn in ORDER:
    r3 = V3[cn]
    NAME = r3['name']
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    acc = r3['accepted']
    combo = r3['combo']
    theta = [float(x) for x in acc['theta']]
    scale = None
    Im = render_model(mnp, lab2k, {'combo': combo, 'params': theta,
                                   'scale': scale})
    rep = morph_compare(Ie, Im)
    reports[cn] = {'name': NAME, 'combo': combo, 'N': r3['N'],
                   **{k: v for k, v in rep.items()}}
    row = morph_table_row(cn, Ie, Im)
    rows.append(row)
    flags = row['flags']
    print(f'{cn:8s} central exp/gen = '
          f'{row["central_exp"]:.3f}/{row["central_model"]:.3f} '
          f'(Δ{row["central_delta"]:.3f})  cvR={row["cv_ratio"]:.2f}  '
          f'shape={row["shape_dist"]:.3f}  '
          f'{"PASS" if row["morph_pass"] else "FAIL " + str(flags)}')

n_pass = sum(1 for r in rows if r['morph_pass'])
print(f'\n[summary] morphology PASS: {n_pass}/8')

# ---- report md ----
lines = ['# 八图形态审计（C：新增形态守门员）', '',
         '> 日期：2026-10-14 | 脚本：`stage14_morph_audit.py` | '
         '指标模块：`morphology_metrics.py`',
         '',
         '## 背景', '',
         '4 奇点诊断发现：现解的 C_s02 达标（0.9163）但**形态严重不符**'
         '（中央暗区占比 34% vs 实验 6%）。说明既有验收体系对"轮廓/'
         '中央形态"没有守门员。本审计用三个形态指标对八图现行交付解'
         '做统一体检。', '',
         '## 形态指标定义（三项须全过）', '',
         '| 指标 | 定义 | 门槛 |',
         '|---|---|---|',
         '| 中央暗区差 Δcentral | \|模型中央(r≤30px)暗像素占比 − '
         '实验\| | ≤ 0.12 |',
         '| 轮廓起伏比 cvR | 模型 cv_r / 实验 cv_r（轮廓半径起伏度）'
         '| ≥ 0.55 |',
         '| 形状谱距离 | r(θ) 角向谐波 L1 距离（归一化） | ≤ 0.55 |',
         '',
         '## 审计结果', '',
         '| 图 | 中央暗区 实验/模型 | Δcentral | cvR | 形状谱距离 | '
         '形态判定 |',
         '|---|---|---|---|---|---|']
for r in rows:
    st = 'PASS' if r['morph_pass'] else '**FAIL**'
    lines.append(f"| {r['cn']} | {r['central_exp']:.3f} / "
                 f"{r['central_model']:.3f} | "
                 f"{r['central_delta']:.3f} | {r['cv_ratio']:.2f} | "
                 f"{r['shape_dist']:.3f} | {st} |")
lines += ['', f'**通过：{n_pass}/8**', '',
          '## 结论与建议', '']
fails = [r for r in rows if not r['morph_pass']]
if fails:
    lines.append('需人工复核的图（形态未通过）：')
    for r in fails:
        lines.append(f"- **{r['cn']}**：未通过项 = {r['flags']}"
                     f"（中央 Δ{r['central_delta']:.3f}、"
                     f"cvR {r['cv_ratio']:.2f}、"
                     f"形状 {r['shape_dist']:.3f}）")
    lines.append('')
    lines.append('建议：对这些图按 4 奇点同法（形态先验重搜索）处理。')
else:
    lines.append('八图形态全部通过——现行交付解在轮廓与中央形态上'
                 '均与实验一致。')
lines += ['', '## 验收模板升级（后续所有图适用）', '',
          '在原有 `C_s02 / PCC(σ=0,2) / rho / C_min / 结构验收` 之外，'
          '**新增形态三项**：中央暗区差、轮廓起伏比、形状谱距离；'
          '任一项未过 → 标记"形态待复核"，不得仅凭 C_s02 定稿。', '',
          '## 文件', '',
          '| 文件 | 内容 |', '|---|---|',
          '| `八图形态审计.png` | 八图中央区与轮廓谱对照（2×8）|',
          '| `形态审计.json` | 全部数值（可被验收单引用）|',
          '| `morphology_metrics.py` | 形态指标模块（工作区根目录）|']
(OUT / '八图形态审计.md').write_text('\n'.join(lines), encoding='utf-8')
(OUT / '形态审计.json').write_text(
    json.dumps(reports, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')

# ---- figure: 2 rows x 8 figs (central crop + spectrum bars) ----
fig, axs = plt.subplots(4, 8, figsize=(26, 12),
                        gridspec_kw={'height_ratios': [2, 1, 2, 1]})
for j, cn in enumerate(ORDER):
    r3 = V3[cn]
    NAME = r3['name']
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    acc = r3['accepted']
    Im = render_model(mnp, lab2k,
                      {'combo': r3['combo'],
                       'params': [float(x) for x in acc['theta']],
                       'scale': None})
    cy, cx = 111.5, 111.5
    sl = (slice(int(cy) - 42, int(cy) + 42),
          slice(int(cx) - 42, int(cx) + 42))
    axs[0, j].imshow(Ie[sl], cmap='inferno', vmin=0, vmax=1)
    axs[0, j].set_title(f'{cn} exp', fontsize=9)
    axs[1, j].imshow(Im[sl], cmap='inferno', vmin=0, vmax=1)
    rep = reports[cn]
    ok = rep['morph_pass']
    axs[1, j].set_title(f'model {"✓" if ok else "✗"}', fontsize=9)
    # spectrum bars
    sp_e = np.array(rep['spectrum_exp'])
    sp_m = np.array(rep['spectrum_model'])
    m = np.arange(1, len(sp_e) + 1)
    axs[2, j].bar(m - 0.2, sp_e, width=0.4, color='#c66',
                  label='exp')
    axs[2, j].bar(m + 0.2, sp_m, width=0.4, color='#68a',
                  label='model')
    axs[2, j].set_ylim(0, max(0.25, sp_e.max() * 1.1))
    axs[2, j].set_title(f'Δc={rep["central_delta"]:.2f} '
                        f'cvR={rep["cv_ratio"]:.2f} '
                        f'sh={rep["shape_dist"]:.2f}', fontsize=8)
    axs[2, j].tick_params(labelsize=6)
    axs[3, j].axis('off')
    axs[3, j].text(0.5, 0.5, 'PASS' if ok else 'FAIL',
                   ha='center', va='center', fontsize=14,
                   color='green' if ok else 'red',
                   transform=axs[3, j].transAxes)
for a in (axs[0, :].tolist() + axs[1, :].tolist()):
    a.axis('off')
fig.suptitle('Eight-image morphology audit (central crop + outline '
             'spectrum) - 2026-10-14', fontsize=14)
fig.tight_layout()
fig.savefig(OUT / '八图形态审计.png', dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'[done] {OUT}')

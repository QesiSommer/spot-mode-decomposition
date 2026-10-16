# -*- coding: utf-8 -*-
"""
stage18_report_overview.py -- 10.16 汇报：八图分解结果总览整合图
====================================================================
每行一张实验图，四列：
  [图名+组合+指标] [实验图] [模型拟合图(σ=2)] [区域相关系数热图(56px)]
数据源 = 终检 v3 交付参数（terminal_inspection_v3.json）。
输出：10.16汇报/八图分解结果总览.png
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
sys.path.insert(0, str(ROOT / 'closed_routes'))  # morph_search_lib 所在目录
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from window_metrics import make_windows, _support_mask
import morph_search_lib as L


def blur2(I):
    """σ=2 高斯模糊（返回 2D 数组）。"""
    from search_two_mode import gaussian_blur
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy()

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = ROOT / '10.16汇报'
OUT.mkdir(exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
mnp = modes.numpy()

ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
         '10奇点', '11奇点']
CN2EN = {'4奇点': '4 Sing.', '5奇点': '5 Sing.', '6奇点': '6 Sing.',
         '8奇点1': '8 Sing.(1)', '8奇点2': '8 Sing.(2)',
         '9奇点': '9 Sing.', '10奇点': '10 Sing.', '11奇点': '11 Sing.'}


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


data = []
for cn in ORDER:
    r = V3[cn]
    NAME = r['name']
    combo = r['combo']
    N = r['N']
    acc = r['accepted']
    d = r.get('delivery', {})
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    Im = L.render_dof(mnp, lab2k, combo,
                      [float(x) for x in acc['theta']],
                      [float(x) for x in acc['sx']],
                      [float(x) for x in acc['sy']],
                      [float(x) for x in acc['shift']],
                      [float(x) for x in acc.get('rot', [0.0] * N)])
    Im_b = blur2(Im)
    Ie_b = blur2(Ie)
    mask = _support_mask(Ie)
    windows, _ = make_windows(224, 56)
    valid = [sl for sl in windows if mask[sl].mean() >= 0.5]
    s2vals = [pcc(Ie_b[sl], Im_b[sl]) for sl in valid]
    hm = np.full((4, 4), np.nan)
    for k in range(len(valid)):
        hm[k // 4, k % 4] = s2vals[k]
    data.append({'cn': cn, 'en': CN2EN[cn], 'combo': combo, 'N': N,
                 'Ie': Ie, 'Im_b': Im_b, 'hm': hm,
                 'C_s02': d.get('C_s02', float('nan')),
                 'W_avg': d.get('W_avg', float('nan')),
                 'cores': d.get('cores'), 'cores_exp': d.get('cores_exp'),
                 'warns': d.get('warns', [])})
    print(f'[render] {cn}: C_s02={data[-1]["C_s02"]:.4f} '
          f'W_avg={data[-1]["W_avg"]:.4f}')

# ---------------- figure ----------------
fig = plt.figure(figsize=(14.5, 22))
gs = fig.add_gridspec(8, 4, width_ratios=[1.25, 1, 1, 1],
                      hspace=0.30, wspace=0.10)
for i, d in enumerate(data):
    # col 0: text
    ax = fig.add_subplot(gs[i, 0])
    ax.axis('off')
    ax.text(0.03, 0.90, d['en'], fontsize=16, weight='bold',
            transform=ax.transAxes)
    ax.text(0.03, 0.70, ' + '.join(d['combo']), fontsize=8.6,
            transform=ax.transAxes, color='#224488')
    ax.text(0.03, 0.40,
            f'C_s02 = {d["C_s02"]:.4f}\n'
            f'W_avg  = {d["W_avg"]:.4f}\n'
            f'暗点：模型 {d["cores"]} / 实验 {d["cores_exp"]}',
            fontsize=10.5, transform=ax.transAxes)
    # col 1: experiment
    ax = fig.add_subplot(gs[i, 1])
    ax.imshow(d['Ie'], cmap='inferno', vmin=0, vmax=1)
    if i == 0:
        ax.set_title('实验图（预处理后）', fontsize=12)
    ax.axis('off')
    # col 2: model
    ax = fig.add_subplot(gs[i, 2])
    ax.imshow(d['Im_b'], cmap='inferno', vmin=0, vmax=1)
    if i == 0:
        ax.set_title('模型拟合图', fontsize=12)
    ax.axis('off')
    # col 3: window heatmap
    ax = fig.add_subplot(gs[i, 3])
    cmap = matplotlib.colormaps['RdYlGn'].copy()
    cmap.set_bad('0.75')
    ax.imshow(d['hm'], cmap=cmap, vmin=0.4, vmax=1.0)
    for r_ in range(4):
        for c_ in range(4):
            if not np.isnan(d['hm'][r_, c_]):
                ax.text(c_, r_, f'{d["hm"][r_, c_]:.2f}', ha='center',
                        va='center', fontsize=8.5)
    if i == 0:
        ax.set_title('区域相关系数（56px 窗口）', fontsize=12)
    ax.axis('off')

cax = fig.add_axes([0.915, 0.11, 0.012, 0.16])
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(0.4, 1.0)),
             cax=cax)
fig.suptitle('八张实验光斑的模式分解总览（终检 v3，全部通过结构验收）'
             '—— 左：图名/模式组合/指标；中：实验图与模型拟合图；'
             '右：4×4 分块区域相关系数', fontsize=14, y=0.995)
fig.savefig(OUT / '八图分解结果总览.png', dpi=110,
            bbox_inches='tight')
plt.close(fig)
print(f'\n[done] {OUT / "八图分解结果总览.png"}')

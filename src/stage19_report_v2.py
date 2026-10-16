# -*- coding: utf-8 -*-
"""
stage19_report_v2.py -- 10.16 汇报 v2
=======================================
1. 总览图：左侧加入每个模式的能量占比与相对相位；
2. Word：客观化措辞、统一字重（正文不加粗）、新增"各模式参数明细"表。
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
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from window_metrics import make_windows, _support_mask
import morph_search_lib as L

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
CN2EN = {'4奇点': '4 奇点', '5奇点': '5 奇点', '6奇点': '6 奇点',
         '8奇点1': '8 奇点(1)', '8奇点2': '8 奇点(2)',
         '9奇点': '9 奇点', '10奇点': '10 奇点', '11奇点': '11 奇点'}


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def blur2(I):
    from search_two_mode import gaussian_blur
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy()


data = []
for cn in ORDER:
    r = V3[cn]
    NAME = r['name']
    combo = r['combo']
    N = r['N']
    acc = r['accepted']
    d = r.get('delivery', {})
    theta = [float(x) for x in acc['theta']]
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    Im = L.render_dof(mnp, lab2k, combo, theta,
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
    # 每模式：能量占比 + 相对相位（度）
    modes_row = []
    for i, m in enumerate(combo):
        energy = theta[i] * 100.0
        if i == 0:
            phase = 0.0
        else:
            ph = np.degrees(theta[N + i - 1]) % 360.0
            phase = ph
        modes_row.append({'name': m, 'energy': energy, 'phase': phase})
    data.append({'cn': cn, 'en': CN2EN[cn], 'N': N,
                 'modes': modes_row,
                 'Ie': Ie, 'Im_b': Im_b, 'hm': hm,
                 'C_s02': d.get('C_s02', float('nan')),
                 'W_avg': d.get('W_avg', float('nan')),
                 'cores': d.get('cores'), 'cores_exp': d.get('cores_exp')})
    print(f'[render] {cn}')

# ================= 总览图 =================
fig = plt.figure(figsize=(15.5, 22))
gs = fig.add_gridspec(8, 4, width_ratios=[1.55, 1, 1, 1],
                      hspace=0.30, wspace=0.10)
for i, d in enumerate(data):
    ax = fig.add_subplot(gs[i, 0])
    ax.axis('off')
    ax.text(0.02, 0.94, d['en'], fontsize=15, weight='bold',
            transform=ax.transAxes)
    y = 0.80
    for mrow in d['modes']:
        ph_txt = '0（参考）' if mrow['phase'] == 0.0 \
            else f'{mrow["phase"]:.1f}°'
        ax.text(0.02, y, mrow['name'], fontsize=9, transform=ax.transAxes,
                color='#1a3a6b')
        ax.text(0.52, y, f'{mrow["energy"]:5.1f}%', fontsize=9,
                transform=ax.transAxes)
        ax.text(0.75, y, f'φ={ph_txt}', fontsize=9,
                transform=ax.transAxes, color='#555555')
        y -= 0.105
    ax.text(0.02, 0.30, f'C_s02 = {d["C_s02"]:.4f}    '
                        f'W_avg = {d["W_avg"]:.4f}',
            fontsize=10, transform=ax.transAxes)
    ax.text(0.02, 0.17, f'暗点：模型 {d["cores"]} / 实验 {d["cores_exp"]}',
            fontsize=10, transform=ax.transAxes)
    # 列标题（首行）
    ax.text(0.52, 0.965, '能量占比', fontsize=8.5, color='#888888',
            transform=ax.transAxes)
    ax.text(0.75, 0.965, '相对相位', fontsize=8.5, color='#888888',
            transform=ax.transAxes)

    ax = fig.add_subplot(gs[i, 1])
    ax.imshow(d['Ie'], cmap='inferno', vmin=0, vmax=1)
    if i == 0:
        ax.set_title('实验图（预处理后）', fontsize=12)
    ax.axis('off')

    ax = fig.add_subplot(gs[i, 2])
    ax.imshow(d['Im_b'], cmap='inferno', vmin=0, vmax=1)
    if i == 0:
        ax.set_title('模型拟合图', fontsize=12)
    ax.axis('off')

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
fig.suptitle('八张实验光斑的模式分解总览（终检 v3，全部通过结构验收）——'
             '左：模式组合与各模式能量占比、相对相位；中：实验图与模型拟合图；'
             '右：4×4 分块区域相关系数', fontsize=13.5, y=0.996)
fig.savefig(OUT / '八图分解结果总览.png', dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'[saved] {OUT / "八图分解结果总览.png"}')

# 保存数据供 docx 使用（去掉大数组字段）
data_lite = [{k: v for k, v in d.items()
              if k not in ('Ie', 'Im_b', 'hm')} for d in data]
(OUT / 'overview_data.json').write_text(
    json.dumps(data_lite, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print('[saved] overview_data.json')

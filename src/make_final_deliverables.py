# -*- coding: utf-8 -*-
"""
make_final_deliverables.py -- 生成"最新结果"交付文件夹
=======================================================

For each of the 8 experiment images, using the NEW-METHOD refined
parameters:
  <n>奇点_拟合对比.png   experiment | model (sigma=2, fair) | |diff|
  <n>奇点_参数与指标.md   modes, energy shares, amplitudes, phases,
                          per-sigma corr, and structural verification
Plus a summary table 汇总表.md.

Output: 归档_八图分解/最新结果/
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
from search_two_mode_lg import rescale_lg
from search_three_mode import eval_c_grid
from refine_all_new_method import (
    stats_for_combo, c_table_N, SIGMAS)

ARCH = ROOT / '归档_八图分解'
DEST = ARCH / '最新结果'
DEST.mkdir(exist_ok=True)
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

with open(ROOT / 'refine_methods_compare' / 'methods_compare.json',
          encoding='utf-8') as f:
    mc = json.load(f)
with open(ROOT / 'refine_all_new_method' / 'all_results.json',
          encoding='utf-8') as f:
    an = json.load(f)

CN = {'4Sing': '4奇点', '8Sing1': '8奇点(1)', '5Sing': '5奇点',
      '6Sing': '6奇点', '8Sing2': '8奇点(2)', '9Sing': '9奇点',
      '11Sing': '11奇点', '10Sing': '10奇点'}

FIGS = []
for tag in ['4Sing', '8Sing1']:
    r = mc[tag]
    b = r['method_B']
    FIGS.append({
        'tag': tag, 'cn': CN[tag], 'img': r.get('image', None),
        'N': None, 'combo': r['combo'], 'params': b['theta'],
        'scale': None, 'corr': b['corr_avg'],
        'cores': b['cores'], 'exp_cores': None, 'mean_nn': b['mean_nn'],
    })
FIGS[0]['img'] = '4Singularities'
FIGS[0]['exp_cores'] = 4
FIGS[1]['img'] = '8Singularities(1)'
FIGS[1]['exp_cores'] = 8
for tag, r in an.items():
    FIGS.append({
        'tag': tag, 'cn': CN[tag], 'img': r['name'], 'N': r['N'],
        'combo': r['combo'], 'params': r['new_params'],
        'scale': r['new_scale'], 'corr': r['new_corr_avg'],
        'cores': r['new_cores'], 'exp_cores': r['exp_cores'],
        'mean_nn': r['new_mean_nn'],
    })


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes_d = modes.to(dev)
    lab2k = {lab: k for k, lab in enumerate(labels)}
    modes_np = modes.numpy()

    summary = []
    for f in FIGS:
        name = f['img']
        cn = f['cn']
        combo = f['combo']
        N = len(combo)
        p = [float(x) for x in f['params']]
        s = f['scale']

        Ie = load_experiment(f'{IMG_DIR}/{name}.png',
                             'nipy_spectral', 'isot')
        raw = plt.imread(f'{IMG_DIR}/{name}.png')
        if raw.ndim == 3 and raw.shape[2] == 4:
            raw = raw[:, :, :3]
        tr = [lab2k[m] for m in combo]

        # model field with the final (new-method) parameters
        mnp = modes_np if s is None else \
            rescale_lg(modes, float(s)).numpy()
        Fs = [np.sqrt(max(p[i], 0)) * mnp[tr[i]] for i in range(N)]
        ph = [0.0] + p[N:]
        E = sum(Fs[i] * np.exp(1j * ph[i]) for i in range(N))
        I = np.abs(E) ** 2
        I = I / max(I.max(), 1e-12)
        I2 = gaussian_blur(
            torch.tensor(I, dtype=torch.float32).unsqueeze(0), 2.0
        )[0].numpy()

        # per-sigma corr with the final parameters
        if s is None:
            stats = stats_for_combo(modes_d, torch.tensor(
                Ie, dtype=torch.float32, device=dev), tr)
        else:
            stats = stats_for_combo(
                rescale_lg(modes_d, float(s)),
                torch.tensor(Ie, dtype=torch.float32, device=dev), tr)
        C = c_table_N(np.array([p], dtype=np.float64), N)
        by_sigma = [round(float(eval_c_grid(C, *st)[0, 0]), 4)
                    for st in stats]
        cavg = float(np.mean(by_sigma))

        # figure
        fig, axs = plt.subplots(1, 4, figsize=(17.5, 4.4))
        axs[0].imshow(raw)
        axs[0].set_title('Experiment (original)')
        axs[1].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
        axs[1].set_title('Experiment (processed)')
        axs[2].imshow(I2, cmap='inferno', vmin=0, vmax=1)
        axs[2].set_title(f'{"+".join(combo)}\ncorr_avg={cavg:.4f}')
        axs[3].imshow(np.abs(Ie - I2), cmap='viridis', vmin=0, vmax=0.6)
        axs[3].set_title('|diff|  (fixed scale)')
        for a in axs:
            a.axis('off')
        fig.suptitle(f'{name}  —  final decomposition '
                     f'({N}-mode, new-method refinement)', fontsize=12)
        fig.tight_layout()
        fig.savefig(DEST / f'{cn}_拟合对比.png', dpi=140,
                    bbox_inches='tight')
        plt.close(fig)

        # text report
        a1 = [float(np.sqrt(max(x, 0))) for x in p[:N]]
        lines = [
            f'# {name} 最终分解结果（新方法微调）',
            '',
            f'- 模式数：{N} 模',
            f'- 拟合对比图：`{cn}_拟合对比.png`'
            f'（原始实验 | 预处理实验 | 模型 σ=2 | 差异）',
            f'- 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0',
            '',
            '## 模式组成',
            '',
            '| 模式 | 能量占比 | 振幅 |',
            '|---|---|---|',
        ]
        for i in range(N):
            lines.append(f'| {combo[i]} | {p[i]*100:.2f}% | {a1[i]:.6f} |')
        lines += [
            '',
            f'归一化检验：Σ aᵢ² = {sum(x*x for x in a1):.6f}',
            '',
            '## 相对相位',
            '',
        ]
        for i in range(1, N):
            lines.append(f'- φ{i+1} = {p[N+i-1]:.6f} rad '
                         f'= {np.degrees(p[N+i-1]):.3f}°')
        lines += [
            '',
            '## 相关系数',
            '',
            f'- **corr_avg（σ=0~6 平均）= {cavg:.4f}**',
            f'- 分尺度：' + '  '.join(
                f'σ={sg}:{v:.4f}' for sg, v in zip(SIGMAS, by_sigma)),
            '',
            '## 结构验收（暗点）',
            '',
            f'- 模型 {f["cores"]} 个 vs 实验 '
            f'{f["exp_cores"] if f["exp_cores"] else "（见归档验收记录）"} 个',
            f'- 平均最近邻位置偏差：'
            f'{round(f["mean_nn"], 2) if f["mean_nn"] is not None else "—"} px',
        ]
        if s is not None:
            lines.insert(4, f'- 缩放参数 s = {s:.4f}')
        (DEST / f'{cn}_参数与指标.md').write_text('\n'.join(lines),
                                                encoding='utf-8')

        summary.append({
            'cn': cn, 'name': name, 'N': N,
            'combo': '+'.join(combo), 'corr_avg': cavg,
            'cores': f['cores'], 'exp_cores': f['exp_cores'],
            'mean_nn': f['mean_nn'],
        })
        print(f'{cn:8s} corr_avg={cavg:.4f}  '
              f'{"+".join(combo):46s} -> saved')

    # summary table
    lines = [
        '# 八图最终分解结果汇总（新方法）',
        '',
        '> 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0；'
        '相关系数为 σ=0~6 多尺度平均。',
        '',
        '| 图 | 模式数 | 模式组合 | 相关系数 | 结构验收（模型 vs 实验暗点） |',
        '|---|---|---|---|---|',
    ]
    order = ['4奇点', '5奇点', '6奇点', '8奇点(1)', '8奇点(2)',
             '9奇点', '10奇点', '11奇点']
    summary.sort(key=lambda x: order.index(x['cn']))
    for r in summary:
        sc = (f'{r["cores"]} vs {r["exp_cores"]}，'
              f'{round(r["mean_nn"], 1)} px'
              if r['mean_nn'] is not None else
              f'{r["cores"]} vs {r["exp_cores"]}')
        lines.append(f'| {r["cn"]} | {r["N"]} | {r["combo"]} | '
                     f'**{r["corr_avg"]:.4f}** | {sc} |')
    lines += [
        '',
        '## 每图详细文件',
        '',
        '| 图 | 拟合对比图 | 参数与指标 |',
        '|---|---|---|',
    ]
    for r in summary:
        lines.append(f'| {r["cn"]} | `{r["cn"]}_拟合对比.png` | '
                     f'`{r["cn"]}_参数与指标.md` |')
    (DEST / '汇总表.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n[done] deliverables in {DEST}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
stage5e_delivery.py -- 最终交付：网格 vs 环形 证据链对比图 + 报告
================================================================
Panels: experiment | 4-mode TOP-1 (incumbent delivery) | pure-HG #1
        HG(2,0)+HG(4,0)+HG(1,4) | LG(2,0)+3HG champion
Metrics annotated per panel: nblk / saddle / squareness A4/A0 /
struct / core-RMS / mesh (step, gap, cv).

Outputs: 归档_八图分解/8奇点1_HG主导搜索/
  - 最终对比_网格证据链.png
  - HG主导搜索报告.md
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model
from stage0_diagnosis import core_depth, saddle_isolation
from stage1_gamma_fix import dark_iou
from stage5a_grid_diag import (grender, gpcc, blocks,
                               squareness_metrics, GAMMA)
from stage5b_hg_search import OUT, IMGDIR, NAME, mesh2x4


def eval_solution(modes_np, lab2k, Ie, pts_e, pts_e_np, name, combo,
                  params):
    I0 = render_model(modes_np, lab2k,
                      {'combo': combo, 'params': params, 'scale': None})
    pts_m0 = find_dark_points(I0)
    if len(pts_m0) >= 2:
        A = np.array([(y, x) for y, x, _ in pts_m0], float)
        C = np.sqrt(((A[:, None, :] - pts_e_np[None, :, :]) ** 2).sum(-1))
        ri, ci = linear_sum_assignment(C)
        dr = C[ri, ci]
        core_rms = float(np.sqrt((dr ** 2).mean()))
        core_max = float(dr.max())
    else:
        core_rms = core_max = 99.0
    mt = match_points(pts_e, pts_m0)
    struct = ('PASS' if (abs(len(pts_m0) - len(pts_e)) <= 1
                         and mt['mean'] is not None
                         and mt['mean'] < 12) else 'fail')
    mesh = mesh2x4(pts_m0)
    Ig = grender(I0, g=0.85)
    blk = blocks(Ig)
    pm = find_dark_points(Ig)
    dep = core_depth(Ig, pm)
    sad = [s for _, s in saddle_isolation(Ig, pm)]
    sm = squareness_metrics(Ig, 0.3)
    d = {'name': name, 'combo': combo, 'params': list(params),
         'nblk': len(blk), 'blk_sizes': blk[:10],
         'depth': float(np.nanmean(dep)) if len(dep) else None,
         'sad_mean': float(np.mean(sad)) if sad else None,
         'sad_min': float(np.min(sad)) if sad else None,
         'gpcc': gpcc(Ie, Ig), 'iou': dark_iou(Ie, Ig),
         'A4_A0': sm['A4_A0'] if sm else None,
         'cores': len(pts_m0), 'struct': struct,
         'core_rms': core_rms, 'core_max': core_max, 'mesh': mesh,
         '_I0': I0, '_Ig': Ig, '_pts': pts_m0}
    return d


def main():
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()
    Ie = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    pts_e_np = np.array([(y, x) for y, x, _ in pts_e], float)
    exp_sm = squareness_metrics(Ie, 0.3)
    exp = {'name': 'experiment', 'nblk': len(blocks(Ie)),
           'depth': float(np.nanmean(core_depth(Ie, pts_e))),
           'sad_mean': float(np.mean(
               [s for _, s in saddle_isolation(Ie, pts_e)])),
           'sad_min': float(np.min(
               [s for _, s in saddle_isolation(Ie, pts_e)])),
           'A4_A0': exp_sm['A4_A0'], 'cores': len(pts_e),
           'mesh': mesh2x4(pts_e)}
    print('experiment:', {k: (round(v, 4) if isinstance(v, float)
                              else v) for k, v in exp.items()
                          if k != 'mesh'})

    sols = []
    # 1. 4-mode TOP-1 (incumbent delivery)
    scan = json.load(open(ROOT / '归档_八图分解' / '8奇点1_四模空间'
                          / 'scan_4mode.json', encoding='utf-8'))
    cur = scan['top3'][0]
    sols.append(eval_solution(modes_np, lab2k, Ie, pts_e, pts_e_np,
                              '4mode TOP-1 (incumbent)', cur['combo'].split('+'),
                              [float(x) for x in cur['params']]))
    # 2. pure-HG #1
    d = json.load(open(OUT / 'pure_hg_deep.json', encoding='utf-8'))
    e1 = d['pure_hg_top'][0]
    sols.append(eval_solution(modes_np, lab2k, Ie, pts_e, pts_e_np,
                              'pure-HG #1', e1['combo'].split('+'),
                              e1['params']))
    # 3. LG+3HG champion
    g = json.load(open(OUT / 'lg3hg_results.json', encoding='utf-8'))
    e2 = g['top15'][0]
    sols.append(eval_solution(modes_np, lab2k, Ie, pts_e, pts_e_np,
                              'LG+3HG #1', e2['combo'].split('+'),
                              e2['params']))

    for s in sols:
        m = s['mesh']
        mm = (f'mesh rms={m.get("rowA_rms",0)+m.get("rowB_rms",0):.2f} '
              f'cv={(m.get("cvA",0)+m.get("cvB",0)):.2f} '
              f'step={m.get("in_row_step",0):.1f} '
              f'gap={m.get("row_gap",0):.1f}'
              if m.get('ok') else 'mesh FAIL')
        print(f'  {s["name"]:24s} nblk={s["nblk"]} coreRMS='
              f'{s["core_rms"]:5.2f} {mm} gpcc={s["gpcc"]:.4f} '
              f'sad={None if s["sad_mean"] is None else round(s["sad_mean"],3)} '
              f'A4={None if s["A4_A0"] is None else round(s["A4_A0"],4)} '
              f'{s["struct"]}')

    # ---------------- figure ----------------
    crop = (slice(45, 185), slice(45, 185))
    fig, axs = plt.subplots(2, 4, figsize=(20, 10.2))
    axs[0, 0].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
    axs[0, 0].set_title(
        f'EXPERIMENT\n{exp["nblk"]} dark cores | saddle '
        f'{exp["sad_mean"]:.3f}/{exp["sad_min"]:.3f}\n'
        f'A4/A0={exp["A4_A0"]:.3f} | mesh: step '
        f'{exp["mesh"]["in_row_step"]:.1f}, gap '
        f'{exp["mesh"]["row_gap"]:.1f}, cv '
        f'{(exp["mesh"]["cvA"]+exp["mesh"]["cvB"])/2:.2f}', fontsize=8.5)
    axs[1, 0].imshow(Ie[crop], cmap='inferno', vmin=0, vmax=1)
    axs[1, 0].set_title('experiment (crop)', fontsize=9)
    for k, s in enumerate(sols):
        ax = axs[0, k + 1]
        ax.imshow(s['_Ig'], cmap='inferno', vmin=0, vmax=1)
        m = s['mesh']
        mm = (f'step {m.get("in_row_step",0):.1f}, gap '
              f'{m.get("row_gap",0):.1f}, cv '
              f'{(m.get("cvA",0)+m.get("cvB",0))/2:.2f}'
              if m.get('ok') else 'mesh FAIL')
        ax.set_title(
            f'{s["name"]}\n{s["combo"]}\n'
            f'nblk={s["nblk"]} depth={round(s["depth"],3)} | saddle '
            f'{round(s["sad_mean"],3)}/{round(s["sad_min"],3)}\n'
            f'coreRMS={s["core_rms"]:.1f}px | A4/A0={round(s["A4_A0"],3)} '
            f'| gpcc={s["gpcc"]:.3f} | {s["struct"]}\nmesh: {mm}',
            fontsize=7.6)
        ax = axs[1, k + 1]
        ax.imshow(s['_Ig'][crop], cmap='inferno', vmin=0, vmax=1)
        y0, x0 = crop[0].start, crop[1].start
        for y, x, _v in s['_pts']:
            ax.plot(x - x0, y - y0, 'x', color='red', ms=7, mew=1.6)
        if len(s['_pts']) >= 2:
            A = np.array([(y, x) for y, x, _ in s['_pts']], float)
            C = np.sqrt(((A[:, None, :] - pts_e_np[None, :, :]) ** 2)
                        .sum(-1))
            ri, ci = linear_sum_assignment(C)
            for a, b in zip(ri, ci):
                ax.plot([A[a, 1] - x0, pts_e_np[b, 1] - x0],
                        [A[a, 0] - y0, pts_e_np[b, 0] - y0], '-',
                        color='cyan', lw=0.7, alpha=0.8)
        for y, x, _v in pts_e:
            ax.plot(x - x0, y - y0, 'o', mfc='none', mec='cyan', ms=10,
                    mew=1.3)
        ax.set_title(f'crop: red x = model cores, cyan o = experiment, '
                     f'lines = assignment', fontsize=9)
    for a in axs.ravel():
        a.axis('off')
    fig.suptitle('8Singularities(1): grid-vs-ring evidence chain — '
                 'experiment vs incumbent vs pure-HG vs LG+3HG '
                 '(display sigma=2.5, gamma=0.85)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / '最终对比_网格证据链.png', dpi=130,
                bbox_inches='tight')
    plt.close(fig)

    # ---------------- report ----------------
    me = exp['mesh']
    m1 = sols[0]['mesh']
    m2 = sols[1]['mesh']
    m3 = sols[2]['mesh']
    rep = [
        '# 8奇点(1) 网格 vs 环形：诊断、搜索与最终判定', '',
        '> 阶段：stage5a（诊断）/ stage5b（全空间搜索）/ stage5b2（纯HG深挖）'
        '/ stage5c（几何专项）/ stage5d（LG+3HG子空间）/ stage5e（交付）', '',
        '## 一、最终判定（对三个假设的回答）', '',
        '| 假设 | 判定 | 证据 |', '|---|---|---|',
        '| “拟合图偏环形、实验图偏网格 → LG 占比过高” | **方向对、因果错** | '
        '实验 8 核确为完美 2×4 网格（两平行行；行内 23.5px 等距 '
        'cv=0.07；行距 36.9px；拟合残差 1.99px，留一验证 2.0px）；'
        '但 LG 含量扫描（全部 366,145 组合分箱）显示拟合质量峰值在 '
        '**LG=36.7%**（0.8826，现任解），降 LG 到 0%/16.7% 时最好解只有 '
        '0.7385/0.7463 —— **LG(2,0) 是实验图环状亮区的必要成分，'
        '不能降**。 |',
        '| “四模与三模视觉无区别 → 实为三模” | **成立** | 第四模 HG(3,3) '
        '仅 2.9% 能量；但它的作用关键：把独立暗块数 6→8（与实验一致）。|',
        '| “暗块连成一片的根源是 LG 环形模” | **不成立** | 暗块连接的'
        '根源是**网格规整度不足**：现任解两行网格 cv=0.25（实验 0.07）、'
        '行距 29.8px（实验 36.9px）、核位置 RMS 6.6px。参数级几何对齐'
        '实验说明这无法靠调参解决（见 stage5c）。|', '',
        '## 二、搜索结论（第 2 步）', '',
        '- **纯 HG 三模全空间**（C(66,3)=45,760）深挖 top-50：最好解 '
        '`HG(2,0)+HG(4,0)+HG(1,4)` 给出**完美网格**（8 核、行残差 0.0、'
        'cv 0.02、step 21.7、gap 41）—— 但网格**轴对齐（0°）**且无环状'
        '亮区（gpcc 0.781、γ 域暗块仅 4）；',
        '- **LG ≤ 10%**：不存在有竞争力的解（能量网格上 {0%, 16.7%} 档的'
        '最优 ≈0.74；16.7% 档无 struct PASS）；',
        '- **LG(2,0)+3HG 四模子空间**（45,760）全扫：无解超越现任'
        '（最好 coreRMS 8.21px、nblk=4、gpcc 0.889）；',
        '- **全空间 366,145 三模**：现任 LG(2,0)+HG(1,1)+HG(2,0) '
        '（+HG(3,3) 2.9%）为唯一结构通过且指标全面领先的解。', '',
        '## 三、指标对照（显示 σ=2.5, γ=0.85）', '',
        '| 指标 | 实验 | 四模 TOP-1（现任） | 纯 HG #1 | LG+3HG #1 |',
        '|---|---|---|---|---|',
        f'| 组合 | — | {"+".join(sols[0]["combo"])} | '
        f'{"+".join(sols[1]["combo"])} | {"+".join(sols[2]["combo"])} |',
        f'| 独立暗块数 | {exp["nblk"]} | {sols[0]["nblk"]} | '
        f'{sols[1]["nblk"]} | {sols[2]["nblk"]} |',
        f'| 核深度 | {exp["depth"]:.3f} | {sols[0]["depth"]:.3f} | '
        f'{sols[1]["depth"]:.3f} | {sols[2]["depth"]:.3f} |',
        f'| 鞍部均值/最小 | {exp["sad_mean"]:.3f}/{exp["sad_min"]:.3f} | '
        f'{sols[0]["sad_mean"]:.3f}/{sols[0]["sad_min"]:.3f} | '
        f'{sols[1]["sad_mean"]:.3f}/{sols[1]["sad_min"]:.3f} | '
        f'{sols[2]["sad_mean"]:.3f}/{sols[2]["sad_min"]:.3f} |',
        f'| 外轮廓方度 A4/A0 | {exp["A4_A0"]:.3f} | '
        f'{sols[0]["A4_A0"]:.3f} | {sols[1]["A4_A0"]:.3f} | '
        f'{sols[2]["A4_A0"]:.3f} |',
        f'| 核位置 RMS (px) | 0 | {sols[0]["core_rms"]:.2f} | '
        f'{sols[1]["core_rms"]:.2f} | {sols[2]["core_rms"]:.2f} |',
        f'| mesh 行残差 (px) | {me["rowA_rms"]+me["rowB_rms"]:.2f} | '
        f'{(m1.get("rowA_rms",9)+m1.get("rowB_rms",9)) if m1.get("ok") else 9:.2f} | '
        f'{(m2.get("rowA_rms",9)+m2.get("rowB_rms",9)) if m2.get("ok") else 9:.2f} | '
        f'{(m3.get("rowA_rms",9)+m3.get("rowB_rms",9)) if m3.get("ok") else 9:.2f} |',
        f'| mesh cv (行内均匀性) | {(me["cvA"]+me["cvB"])/2:.2f} | '
        f'{(m1.get("cvA",9)+m1.get("cvB",9))/2 if m1.get("ok") else 9:.2f} | '
        f'{(m2.get("cvA",9)+m2.get("cvB",9))/2 if m2.get("ok") else 9:.2f} | '
        f'{(m3.get("cvA",9)+m3.get("cvB",9))/2 if m3.get("ok") else 9:.2f} |',
        f'| mesh 行距 (px) | {me["row_gap"]:.1f} | '
        f'{m1.get("row_gap",0) if m1.get("ok") else 0:.1f} | '
        f'{m2.get("row_gap",0) if m2.get("ok") else 0:.1f} | '
        f'{m3.get("row_gap",0) if m3.get("ok") else 0:.1f} |',
        f'| 外观 PCC (γ=0.85) | — | {sols[0]["gpcc"]:.4f} | '
        f'{sols[1]["gpcc"]:.4f} | {sols[2]["gpcc"]:.4f} |',
        f'| 结构验收 | 8 核 | {sols[0]["struct"]} | {sols[1]["struct"]} | '
        f'{sols[2]["struct"]} |', '',
        '## 四、结论', '',
        '1. **现任解（LG 36% + HG 网格）就是全空间最优**：核加权 PCC '
        '0.8826 显著领先全部备选；外观、暗块数、鞍部均最接近实验；',
        '2. **纯 HG 方案不可行**：结构上能造出完美网格，但无法同时提供'
        '实验的环状亮区（这需要 LG(2,0) 36%）；',
        '3. **残余差距 = 网格规整度**：实验核网格 cv=0.07、行距 36.9px；'
        '现任 cv=0.25、行距 29.8px。参数调优与 LG+3HG 子空间搜索均无法'
        '弥合 —— 这是该模式基下的结构性极限（如实披露）；',
        '4. **下一步选项**（供决策）：(a) 接受当前解（已达成 8 暗块 + '
        '鞍部/深度/外观全面接近）；(b) 尝试在基中加入带像散/倾斜的'
        '修正项（实验外轮廓 m=2 分量表明可能存在椭圆像散）；'
        '(c) 更细的（更小步长）相位-能量联合全局优化。', '',
        '## 五、文件', '',
        '| 文件 | 内容 |',
        '|---|---|',
        '| `最终对比_网格证据链.png` | **本图：实验 vs 三代表解（全图+核区）** |',
        '| `诊断_网格vs环形.png` | D1 外轮廓方度 + D2 核排布几何 |',
        '| `D3_LG能量序列.png` / `D3_指标曲线.png` | LG 能量敏感性序列 |',
        '| `纯HG深挖与LG曲线.png` | LG 需求曲线 + 纯 HG top-3 |',
        '| `核几何对齐优化.png` | 核位置定向调优（负结果） |',
        '| `LG+3HG_前三.png` | LG+3HG 子空间前三 |',
        '| `hg_search_results.json` | 全空间三模搜索（分组）|',
        '| `scan3A.pt` | **全 366,145 组合得分全量**（可复查）|',
        '| `pure_hg_deep.json` / `lg3hg_results.json` / `mesh_tune.json` | '
        '各专项明细 |',
    ]
    (OUT / 'HG主导搜索报告.md').write_text('\n'.join(rep),
                                          encoding='utf-8')
    json.dump({'experiment': {k: v for k, v in exp.items()},
               'solutions': [{k: v for k, v in s.items()
                              if not k.startswith('_')}
                             for s in sols]},
              open(OUT / 'final_delivery.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False, default=str)
    print(f'\n[done] deliverables in {OUT}')


if __name__ == '__main__':
    main()

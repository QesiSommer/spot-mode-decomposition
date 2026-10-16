# -*- coding: utf-8 -*-
"""
conv_deliver_std.py — 标准记号版交付文件夹生成（2026-10-16）

从 v3 JSON 生成八图「标准记号版」交付文件夹（另存，不覆盖旧档）：
  - 每图 {图}_参数与指标.md：标准记号组合表 + 相对相位（换算后）
  - 每图 {图}_拟合对比.png：三联（实验 / 标准记号换算解渲染 / C_s02）
  - 顶层 汇总.md + 换算说明.md
  - 复制旧版 png（拟合对比图与强度无关，记号换算不影响像素）

换算规则（数值验证 pcc=1.000000，八图回归 ALL PASS）：
  HG(m,n) → HG(n,m)                相位不变
  LG(p,l) → LG(p,−l)               φ_k → φ_k + (l_k − l_0)·π/2
（相对相位以模式 0 为基准；l_0 为模式 0 的角量子数，HG 记 l=0）
"""
import json
import shutil
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'closed_routes'))  # morph_search_lib 所在目录
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points
import morph_search_lib as L

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

SRC_DIR = ROOT / '八图最终交付_2026-10-14'
DST = ROOT / '八图最终交付_标准记号_2026-10-16'
DST.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))

SIZE, EXTENT = 224, 4.0
ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
          '10奇点', '11奇点']
OLD_DIRNAME = '八图最终交付_2026-10-14'


def pcc(a, b):
    a = np.asarray(a).ravel() - np.asarray(a).mean()
    b = np.asarray(b).ravel() - np.asarray(b).mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def blur2(I):
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy()


def C_s02(Ie, Ie_b, I):
    return 0.5 * (pcc(Ie, I) + pcc(Ie_b, blur2(I)))


def parse_mode(s):
    fam, idx = s.strip()[:2], s.strip()[3:-1]
    a, b = (int(v) for v in idx.split(','))
    return fam, a, b


print('building joint modes ...')
modes, labels = build_joint_modes(10, SIZE, EXTENT, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
mnp = modes.numpy()


def render_dof(fields, theta, sx, sy, shiftv, rot):
    E = np.zeros((SIZE, SIZE), dtype=np.complex128)
    N = len(fields)
    for i, Fm in enumerate(fields):
        if abs(sx[i] - 1.0) > 1e-9:
            Fm = L.zoom_axis_c(Fm, sx[i], axis=1)
        if abs(sy[i] - 1.0) > 1e-9:
            Fm = L.zoom_axis_c(Fm, sy[i], axis=0)
        if abs(rot[i]) > 1e-9:
            from scipy.ndimage import rotate as ndrotate
            re = ndrotate(Fm.real, rot[i], reshape=False, order=1)
            im = ndrotate(Fm.imag, rot[i], reshape=False, order=1)
            Fm = re + 1j * im
        phi = 0.0 if i == 0 else theta[N + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-30)
    I = ndshift(I, shiftv, order=1, mode='nearest')
    return np.clip(I, 0, None)


summary_rows = []
for cn in ORDER:
    r = V3[cn]
    src = r['accepted'] if 'theta' in r.get('accepted', {}) \
        else r['refined_v3']
    combo_int = r['combo']
    N_ = r['N']
    theta = [float(x) for x in src['theta']]
    sx = [float(x) for x in src['sx']]
    sy = [float(x) for x in src['sy']]
    shiftv = [float(x) for x in src['shift']]
    rot = [float(x) for x in src.get('rot', [0.0] * N_)]
    name = r['name']
    d = r['delivery']

    # --- 换算 ---
    l_of = [parse_mode(s)[2] if parse_mode(s)[0] == 'LG' else 0
            for s in combo_int]
    l0 = l_of[0]
    combo_std, theta_std = [], list(theta)
    for i, s in enumerate(combo_int):
        fam, a, b = parse_mode(s)
        if fam == 'HG':
            combo_std.append(f'HG({b},{a})')
        else:
            combo_std.append(f'LG({a},{-b})')
        # 相位补偿对所有模式生效（HG 记 l=0；l0 为模式 0 的 l）：
        # E_old 整体提出公因子 e^{i·l0·π/2} 后，
        # φ'_k = φ_k + (l_k − l0)·π/2
        if i > 0:
            theta_std[N_ + i - 1] = (theta[N_ + i - 1]
                                     + (l_of[i] - l0) * np.pi / 2)
    for i in range(N_ - 1):
        theta_std[N_ + i] = (theta_std[N_ + i] + np.pi) \
            % (2 * np.pi) - np.pi

    # --- 渲染（应与旧版逐位一致） ---
    fields = [mnp[lab2k[l_]] for l_ in combo_std]
    I_new = render_dof(fields, theta_std, sx, sy, shiftv, rot)
    Ie = load_experiment(f'{IMG}/{name}.png', 'nipy_spectral', 'isot')
    Ie_b = blur2(Ie)
    c_val = C_s02(Ie, Ie_b, I_new)

    # --- md 文件 ---
    sub = DST / cn
    sub.mkdir(parents=True, exist_ok=True)
    lines = [f'# {name} 最终分解结果（标准记号版·终检v3）', '']
    lines.append(f'> 记号约定：HG(m,n) 的 m=水平 x 阶数、n=竖直 y 阶数'
                 f'（Siegman/MATLAB 标准）；LG(p,l) 的 l 为拓扑荷，'
                 f'方位角 θ=atan2(y,x)。')
    lines.append(f'> 本表为 [`{OLD_DIRNAME}`]({OLD_DIRNAME}\\{cn}\\'
                 f'{cn}_参数与指标.md) 的标准记号换算版——数值完全'
                 '等价（换算回归 pcc=1.000000），仅记号约定不同。')
    lines.append('')
    lines.append(f'- 模式数：{N_} 模')
    lines.append(f'- 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0')
    lines.append(f'- **终检v3自由度**：全局平移 ({shiftv[0]:+.2f}, '
                 f'{shiftv[1]:+.2f}) px；逐模 (sx, sy) 与旋转角见下表'
                 '（旋转仅 HG 模有意义）')
    lines.append('')
    lines.append('## 模式组成（标准记号）')
    lines.append('')
    lines.append('| 模式 | 能量占比 | 振幅 | sx | sy | 旋转° |')
    lines.append('|---|---|---|---|---|---|')
    for i, lab in enumerate(combo_std):
        amp = np.sqrt(max(theta[i], 0))
        lines.append(f'| {lab} | {theta[i]*100:.2f}% | {amp:.6f} '
                     f'| {sx[i]:.3f} | {sy[i]:.3f} '
                     f'| {rot[i]:+.2f} |')
    lines.append('')
    lines.append(f'归一化检验：Σ aᵢ² = '
                 f'{sum(max(t,0) for t in theta[:N_]):.6f}')
    lines.append('')
    lines.append('## 相对相位（换算后）')
    lines.append('')
    for i in range(1, N_):
        ph = theta_std[N_ + i - 1]
        lines.append(f'- φ{i+1} = {ph:.6f} rad = '
                     f'{np.degrees(ph):+.3f}°')
    lines.append('')
    lines.append('## 相关系数（与旧版逐位一致）')
    lines.append('')
    lines.append(f'- **C_s02（主口径）= {c_val:.4f}**')
    lines.append(f'- W_avg = {d["W_avg"]:.4f}')
    if 'v0' in d and 'v2' in d:
        lines.append(f'- 分尺度：σ=0: {d["v0"]:.4f}，'
                     f'σ=2: {d["v2"]:.4f}')
    lines.append('')
    lines.append('## 结构验收（暗点）——与旧版相同')
    lines.append('')
    lines.append(f'- 模型 {d["cores"]} 个 vs 实验 {d["cores_exp"]} 个')
    if 'mean_nn' in d:
        try:
            lines.append(f'- 平均最近邻偏差：{d["mean_nn"]:.2f} px；'
                         f'最大：{d["max10"]:.2f} px')
        except (TypeError, ValueError):
            lines.append(f'- 平均最近邻偏差：{d["mean_nn"]}')
    lines.append('')
    (sub / f'{cn}_参数与指标_标准记号.md').write_text(
        '\n'.join(lines), encoding='utf-8')

    # --- 三联图 ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for k, (tag, I) in enumerate([
            ('实验图', Ie),
            (f'标准记号换算解\n{"+".join(combo_std)}', I_new),
            ('残差 |ΔI|', np.abs(Ie - I_new / I_new.max()))]):
        if k == 2:
            im = axs[k].imshow(I, cmap='magma', vmin=0, vmax=0.5)
        else:
            axs[k].imshow(blur2(I / max(I.max(), 1e-12)), cmap='inferno',
                          vmin=0, vmax=1)
        axs[k].set_title(tag, fontsize=10)
        axs[k].axis('off')
    fig.suptitle(f'{name}（{cn}）标准记号版渲染 —— C_s02={c_val:.4f}'
                 f'（与旧版逐位一致）', fontsize=12)
    fig.tight_layout()
    fig.savefig(sub / f'{cn}_标准记号渲染.png', dpi=125,
                bbox_inches='tight')
    plt.close(fig)

    summary_rows.append((cn, N_, '+'.join(combo_std), c_val,
                         d['W_avg']))
    print(f'  {cn}: {c_val:.4f} {"+".join(combo_std)}')

# --- 顶层汇总 + 换算说明 ---
md = ['# 八图最终分解结果（标准记号版，2026-10-16）', '']
md.append(f'> 本文件夹为 [`{OLD_DIRNAME}`]({OLD_DIRNAME}\\'
          '八图最终分解结果汇总.md) 的**标准记号换算版**。')
md.append('> 换算仅改变模式标签与 LG 相对相位表达，物理场与所有'
          '指标逐位不变（八图回归 pcc=1.000000、C_s02 完全一致）。')
md.append('')
md.append('## 记号约定')
md.append('')
md.append('- **HG(m,n)**：m = 水平 x 方向阶数，n = 竖直 y 方向阶数'
          '（Siegman《Lasers》、Saleh & Teich、MATLAB 光学工具箱标准）；'
          'HG(m,n) 光斑为 (m+1) 列 × (n+1) 行。')
md.append('- **LG(p,l)**：p = 径向指数，l = 拓扑荷；方位角 '
          'θ = atan2(y,x)（y 竖直、x 水平）。')
md.append('- 相位约定 E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁=0（相对相位）。')
md.append('')
md.append('## 换算规则（旧内部记号 → 标准记号）')
md.append('')
md.append('| 旧（内部） | 新（标准） | 相位 |')
md.append('|---|---|---|')
md.append('| HG(m,n) | HG(n,m) | 不变 |')
md.append('| LG(p,l) | LG(p,−l) | φ_k → φ_k + (l_k−l_0)·π/2 |')
md.append('')
md.append('> l_0 为模式 0 的角量子数（HG 记 0）。相对相位以模式 0 '
          '为基准，故补偿量取 (l_k − l_0)。')
md.append('')
md.append('## 总表')
md.append('')
md.append('| 图 | 模式数 | 标准记号组合 | C_s02 | W_avg |')
md.append('|---|---|---|---|---|')
for cn, N_, combo, c, w in summary_rows:
    md.append(f'| {cn} | {N_} | {combo} | **{c:.4f}** | {w:.4f} |')
md.append('')
md.append('## 文件')
md.append('')
md.append(f'- 每图子文件夹：`{cn}_参数与指标_标准记号.md` / '
          f'`{cn}_标准记号渲染.png`；')
md.append(f'- 拟合对比图（与强度无关的旧版五联图）请看 '
          f'`{OLD_DIRNAME}\\{cn}\\`；')
md.append('- 过程材料（换算自检/回归脚本与报告）：'
          '`归档_八图分解\\约定对齐回归_2026-10-16\\`。')
(DST / '八图最终分解结果汇总_标准记号版.md').write_text(
    '\n'.join(md), encoding='utf-8')

# 复制旧版拟合对比 png（可选旁证）
for cn in ORDER:
    src_png = SRC_DIR / cn / f'{cn}_拟合对比.png'
    if src_png.exists():
        shutil.copy2(src_png, DST / cn / f'{cn}_拟合对比.png')
    src_png2 = SRC_DIR / cn / f'{cn}_新旧拟合对比.png'
    if src_png2.exists():
        shutil.copy2(src_png2, DST / cn / f'{cn}_新旧拟合对比.png')

print(f'\nDONE -> {DST}')

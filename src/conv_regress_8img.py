# -*- coding: utf-8 -*-
"""
conv_regress_8img.py — 轴序修正后的八图回归（2026-10-16）

目的：证明"全部对齐"操作对已交付结果零影响。
方法：
  1. 读 v3 JSON 的八图解（内部记号 + theta/dof）；
  2. 按换算规则转成标准记号：
       HG(m,n) → HG(n,m)             相位不变
       LG(p,l) → LG(p,−l)            φ → φ + l·π/2
  3. 用【修正后的新基】+ render_dof 渲染换算解；
  4. 与【修正前旧实现】渲染的 v3 参考图对比：
       强度 pcc 应 = 1.0000，C_s02 应与 v3 JSON 记录一致；
  5. 附：新基下按内部记号直接渲染（预期显著错位——证明
     修轴序后旧记号确实指向了不同的物理场，不能不换算）。

注意：本脚本的"旧基"用内联函数复刻修正前实现（不依赖文件），
"新基"直接 import 已修正的 build_joint_modes。
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
from scipy.ndimage import shift as ndshift
from scipy.special import hermite, genlaguerre, factorial

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'closed_routes'))  # morph_search_lib 所在目录
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points
import morph_search_lib as L

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = ROOT / '归档_八图分解' / '约定对齐回归_2026-10-16'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))

SIZE, EXTENT = 224, 4.0
ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
          '10奇点', '11奇点']
LABEL_N = {'4奇点': 4, '5奇点': 5, '6奇点': 6, '8奇点1': 8,
           '8奇点2': 8, '9奇点': 9, '10奇点': 10, '11奇点': 11}


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


# ---------------- 修正前旧基（内联复刻） ----------------
def old_hg(m, n):
    x = np.linspace(-EXTENT, EXTENT, SIZE)
    y = np.linspace(-EXTENT, EXTENT, SIZE)
    X, Y = np.meshgrid(x, y, indexing='ij')
    norm = np.sqrt(2.0 / (np.pi * factorial(m, exact=True)
                          * factorial(n, exact=True)
                          * (2.0 ** (m + n))))
    return (norm * hermite(m)(X * np.sqrt(2)) * hermite(n)(Y * np.sqrt(2))
            * np.exp(-(X ** 2 + Y ** 2))).astype(np.complex128)


def old_lg(p, l):
    xs = np.linspace(-EXTENT, EXTENT, SIZE)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    R = np.sqrt(X ** 2 + Y ** 2)
    TH = np.arctan2(Y, X)
    return (genlaguerre(p, abs(l))(2 * R ** 2)
            * (np.sqrt(2) * R) ** abs(l) * np.exp(-R ** 2)
            * np.exp(1j * l * TH)).astype(np.complex128)


def parse_mode(s):
    fam, idx = s.strip()[:2], s.strip()[3:-1]
    a, b = (int(v) for v in idx.split(','))
    return fam, a, b


def convert_mode(s):
    """内部记号 → 标准记号；返回 (label, delta_phi)。"""
    fam, a, b = parse_mode(s)
    if fam == 'HG':
        return f'HG({b},{a})', 0.0
    return f'LG({a},{-b})', b * np.pi / 2.0


# ---------------- 新基（修正后代码） ----------------
print('building new joint modes (post-fix) ...')
modes, labels = build_joint_modes(10, SIZE, EXTENT, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
mnp = modes.numpy()

report = {}
ok_all = True

for cn in ORDER:
    r = V3[cn]
    src = r['accepted'] if 'theta' in r.get('accepted', {}) \
        else r['refined_v3']
    combo_int = r['combo']                    # 内部记号
    N_ = r['N']
    theta = [float(x) for x in src['theta']]  # 前 N 能量 + 后 N-1 相位
    sx = [float(x) for x in src['sx']]
    sy = [float(x) for x in src['sy']]
    shiftv = [float(x) for x in src['shift']]
    rot = [float(x) for x in src.get('rot', [0.0] * N_)]
    v3_C = float(src['C_s02']) if 'C_s02' in src else \
        float(r['delivery']['C_s02'])
    name = r['name']

    # --- 参考图：旧实现按内部记号渲染（= v3 交付管线） ---
    old_fields = []
    for s in combo_int:
        fam, a, b = parse_mode(s)
        old_fields.append(old_hg(a, b) if fam == 'HG' else old_lg(a, b))

    # 手工复刻 render_dof（对齐 morph_search_lib.py:94-113 的精确行为）
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

    I_ref = render_dof(old_fields, theta, sx, sy, shiftv, rot)

    # --- 新基按标准记号渲染换算解 ---
    # 相位补偿必须相对模式 0（render_dof 的 φ_0 ≡ 0）：
    #   B_old(p,l) = e^{+i·l·π/2}·B_new(p,−l)
    #   E_old 整体提出公因子 e^{i·l_0·π/2}（不影响强度）后
    #   相对相位 φ'_k = φ_k + (l_k − l_0)·π/2   (k≥1)
    combo_std = []
    theta_std = list(theta)
    l_of = []
    for s in combo_int:
        fam, a, b = parse_mode(s)
        l_of.append(b if fam == 'LG' else 0)
    l0 = l_of[0]
    for i, s in enumerate(combo_int):
        lab, _dphi = convert_mode(s)      # dphi=l_i·π/2 为绝对补偿，
        combo_std.append(lab)             # 相对补偿在下面统一处理
        if i > 0:
            theta_std[N_ + i - 1] = \
                theta[N_ + i - 1] + (l_of[i] - l0) * np.pi / 2
    # 相位收敛到 (-π, π]
    for i in range(N_ - 1):
        theta_std[N_ + i] = (theta_std[N_ + i] + np.pi) % (2 * np.pi) - np.pi
    new_fields = [mnp[lab2k[l_]] for l_ in combo_std]
    I_new = render_dof(new_fields, theta_std, sx, sy, shiftv, rot)

    # --- 对照：新基按内部记号直渲染（预期错位） ---
    fields_naive = [mnp[lab2k[s]] for s in combo_int]
    I_naive = render_dof(fields_naive, theta, sx, sy, shiftv, rot)

    # --- 打分 ---
    Ie = load_experiment(f'{IMG}/{name}.png', 'nipy_spectral', 'isot')
    Ie_b = blur2(Ie)
    pts_e = find_dark_points(Ie)
    c_ref = C_s02(Ie, Ie_b, I_ref)
    c_new = C_s02(Ie, Ie_b, I_new)
    c_naive = C_s02(Ie, Ie_b, I_naive)
    p_ref_new = pcc(I_ref, I_new)

    row = {
        'name': name, 'N': N_,
        'combo_internal': combo_int, 'combo_std': combo_std,
        'C_json': v3_C, 'C_ref_old': c_ref, 'C_new_std': c_new,
        'C_new_naive': c_naive, 'pcc_ref_new': p_ref_new,
        'detected': len(pts_e), 'I_ref': I_ref, 'I_new': I_new,
    }
    report[cn] = row
    ok = (p_ref_new > 0.99999) and abs(c_new - c_ref) < 5e-4 \
        and c_naive < c_ref - 0.05
    ok_all &= ok
    print(f'{cn:8s} {name:16s} N={N_} '
          f'json={v3_C:.4f} ref={c_ref:.4f} new(std)={c_new:.4f} '
          f'naive={c_naive:.4f} pcc(ref,new)={p_ref_new:.6f} '
          f'{"PASS" if ok else "FAIL"}')

# --- 结果表 + 一张抽核图（6奇点 三联对比） ---
report_plot = {k: {kk: vv for kk, vv in v.items()
                   if kk not in ('I_ref', 'I_new')}
               for k, v in report.items()}
json.dump(report_plot, open(OUT / '回归报告.json', 'w',
                            encoding='utf-8'),
          ensure_ascii=False, indent=2)

cn_show = '6奇点'
r = V3[cn_show]
src = r['accepted'] if 'theta' in r.get('accepted', {}) else r['refined_v3']
Ie = load_experiment(f'{IMG}/{r["name"]}.png', 'nipy_spectral', 'isot')
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
imgs = [('实验图', Ie),
        ('旧实现·内部记号 (v3 交付等价)', report[cn_show].get('I_ref')),
        ('新基·标准记号换算解', report[cn_show].get('I_new'))]
for k, (tag, I) in enumerate(imgs):
    if I is None:
        continue
    axs[k].imshow(blur2(I / max(I.max(), 1e-12)), cmap='inferno',
                  vmin=0, vmax=1)
    axs[k].set_title(tag, fontsize=10)
    axs[k].axis('off')
fig.suptitle(f'{cn_show}：约定对齐回归抽核（应与中间图逐位一致）',
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / '回归抽核_6奇点.png', dpi=125, bbox_inches='tight')
plt.close(fig)

print(f'\n{"=" * 64}\n八图回归总判定: {"ALL PASS" if ok_all else "FAIL"}'
      f'\n输出: {OUT}')
sys.exit(0 if ok_all else 1)

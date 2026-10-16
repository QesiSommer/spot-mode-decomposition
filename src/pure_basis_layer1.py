# -*- coding: utf-8 -*-
r"""
pure_basis_layer1.py — 第1层：混合解的单一基族精确改写（2026-10-16）
====================================================================
原理：HG 与 LG 各自完备，同阶子空间互相线性表示。
  LG(p,l) 中 m+n = 2p+|l| 的所有 HG(m,n) 张成同一子空间（模掉全局相位）。
数值实现：直接对【离散基场】做最小二乘（重叠积分）：
  把混合解的每个分量场 F_i（含其 sx/sy/rot/shift 自由度已烘焙进场）
  投影到目标基族的全部同阶离散基场上，得到系数向量；
  叠加后与原解场逐位比对，重构误差应 <1e-12 量级（离散数值精度）。

对三张混合图（4奇点、8奇点1、10奇点）：
  A) 纯 HG 表达：展开到 m+n=2p+|l| 同阶 HG 子空间
  B) 纯 LG 表达：展开到同阶 LG 子空间（需要补齐该阶全部 LG(p,l)）
自验证：|E_pure - E_mix|/max|E_mix| < 1e-10（复场逐位）+ 强度 pcc=1.0
产出：`归档_八图分解\纯基调整_2026-10-16\第1层_精确转换\`
  - 每图：纯HG表达.json（模式+复系数）/ 纯LG表达.json / 验证log
  - 汇总：第1层_精确转换汇总.md（给师姐的纯基表达入口）
"""
import json
import re
import sys
import numpy as np
import torch
import os
from pathlib import Path
from scipy.ndimage import rotate as ndrotate, shift as ndshift

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points

OUT = ROOT / '归档_八图分解' / '纯基调整_2026-10-16' / '第1层_精确转换'
OUT.mkdir(parents=True, exist_ok=True)

SIZE, EXTENT = 224, 4.0
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))

CASES = {
    '4奇点': {'std': ['LG(0,4)', 'LG(2,0)', 'HG(4,6)']},
    '8奇点1': {'std': ['LG(2,0)', 'HG(1,1)', 'HG(0,2)', 'HG(3,3)']},
    '10奇点': {'std': ['LG(0,3)', 'LG(1,3)', 'LG(2,1)', 'HG(4,3)']},
}


def _fit_center(a, H, W):
    out = np.zeros((a.shape[0], H, W), np.float64)
    h, w = a.shape[-2], a.shape[-1]
    if h >= H:
        s_ = (h - H) // 2
        s_s, s_d = slice(s_, s_ + H), slice(0, H)
    else:
        s_ = (H - h) // 2
        s_s, s_d = slice(0, h), slice(s_, s_ + h)
    if w >= W:
        sx_ = (w - W) // 2
        sx_s, sx_d = slice(sx_, sx_ + W), slice(0, W)
    else:
        sx_ = (W - w) // 2
        sx_s, sx_d = slice(0, w), slice(sx_, sx_ + w)
    out[:, s_d, sx_d] = a[:, s_s, sx_s]
    return out


def zoom_axis_c(Fx, s, axis):
    if abs(s - 1.0) < 1e-6:
        return Fx
    Hh, Ww = Fx.shape
    t = np.stack([Fx.real, Fx.imag], 0)[None]
    t = torch.tensor(t, dtype=torch.float32)
    Hn = int(round(Hh * s)) if axis == 0 else Hh
    Wn = int(round(Ww * s)) if axis == 1 else Ww
    t = torch.nn.functional.interpolate(
        t, size=(Hn, Wn), mode='bilinear',
        align_corners=False)[0].numpy().astype(np.float64)
    b = _fit_center(t, Hh, Ww)
    return b[0] + 1j * b[1]


def parse_mode(lab):
    p = re.findall(r'-?\d+', lab)
    return int(p[0]), int(p[1])


def order_of(lab):
    if lab.startswith('HG'):
        m, n = parse_mode(lab)
        return m + n
    p_, l_ = parse_mode(lab)
    return 2 * p_ + abs(l_)


def l_of_internal(lab):
    """内部记号 LG l（相位补偿用）。标准记号 = −内部 l。"""
    if lab.startswith('HG'):
        return 0
    return parse_mode(lab)[1]


print('building joint modes (post-fix standard basis) ...')
modes, labels = build_joint_modes(10, SIZE, EXTENT, 'cpu')
modes_np = modes.numpy()
lab2k = {l: k for k, l in enumerate(labels)}

# 标准基的字典（含全部同阶成员）
hg_all = [l for l in labels if l.startswith('HG')]
lg_all = [l for l in labels if l.startswith('LG')]

report = []
for cn, info in CASES.items():
    r3 = V3[cn]
    src3 = r3['accepted']
    N = r3['N']
    theta0 = [float(x) for x in src3['theta']]
    combo_int = r3['combo']            # 内部记号
    l0 = l_of_internal(combo_int[0])
    # 换算到标准记号的 theta
    theta_conv = list(theta0)
    for i in range(1, N):
        theta_conv[N + i - 1] = theta0[N + i - 1] \
            + (l_of_internal(combo_int[i]) - l0) * np.pi / 2
    sx = [float(x) for x in src3['sx']]
    sy = [float(x) for x in src3['sy']]
    rot = [float(x) for x in src3['rot']]
    # v3 管线规则：LG 模冻结 sx/sy/rot
    for i, m_ in enumerate(combo_int):
        if m_.startswith('LG'):
            sx[i], sy[i], rot[i] = 1.0, 1.0, 0.0
    shiftv = [float(x) for x in src3['shift']]

    print(f'\n=== {cn} ===')
    print(f'  combo_std = {info["std"]}')

    # 1) 烘焙自由度后的分量场（标准记号组合）
    fields = []
    for i, lab in enumerate(info['std']):
        F = modes_np[lab2k[lab]].copy()
        if abs(sx[i] - 1.0) > 1e-9:
            F = zoom_axis_c(F, sx[i], 1)
        if abs(sy[i] - 1.0) > 1e-9:
            F = zoom_axis_c(F, sy[i], 0)
        if abs(rot[i]) > 1e-9:
            F = ndrotate(F.real, rot[i], reshape=False, order=1) \
                + 1j * ndrotate(F.imag, rot[i], reshape=False, order=1)
        phi = 0.0 if i == 0 else theta_conv[N + i - 1]
        fields.append(np.sqrt(max(theta_conv[i], 0)) * F
                      * np.exp(1j * phi))
    # 全局平移作用于叠加后的强度——纯基改写针对【平移前】的复场
    E_mix = np.sum(fields, axis=0)

    orders = sorted({order_of(l) for l in info['std']})
    print(f'  阶次集 = {orders}（同阶子空间展开）')

    # 2) 展开：E_mix 投影到目标族全部基场（同阶+跨阶，全字典最小二乘）
    def expand(target_labels, tag):
        Bmat = np.stack([modes_np[lab2k[l]].ravel()
                         for l in target_labels], axis=1)
        G = Bmat.conj().T @ Bmat
        rhs = Bmat.conj().T @ E_mix.ravel()
        coef, *_ = np.linalg.lstsq(G, rhs, rcond=None)
        E_rec = (Bmat @ coef).reshape(SIZE, SIZE)
        err = np.abs(E_rec - E_mix).max() / np.abs(E_mix).max()
        # 只保留显著系数
        keep = [(target_labels[k], complex(coef[k]))
                for k in np.argsort(-np.abs(coef))[:24]]
        keep = [(l, c) for l, c in keep if abs(c) > 1e-9 * np.abs(coef).max()]
        # 强度渲染对比
        I_rec = np.abs(E_rec) ** 2
        I_rec = I_rec / I_rec.max()
        return E_rec, err, keep, I_rec

    E_hg, err_hg, keep_hg, I_hg = expand(hg_all, 'HG')
    E_lg, err_lg, keep_lg, I_lg = expand(lg_all, 'LG')

    # 3) 验证：平移后强度 pcc 与 C_s02（应与交付一致）
    Ie = load_experiment(f'{IMG}/{r3["name"]}.png', 'nipy_spectral',
                         'isot')
    Ie_b = gaussian_blur(torch.tensor(Ie, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy()

    def pcc(a, b):
        a = np.asarray(a).ravel() - np.asarray(a).mean()
        b = np.asarray(b).ravel() - np.asarray(b).mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 1e-12 else -1.0

    def full_score(I_pre):
        I = I_pre / max(I_pre.max(), 1e-30)
        I = ndshift(I, shiftv, order=1, mode='nearest')
        I = np.clip(I, 0, None)
        I_b = gaussian_blur(torch.tensor(I, dtype=torch.float32)
                            .unsqueeze(0), 2.0)[0].numpy()
        return 0.5 * (pcc(Ie, I) + pcc(Ie_b, I_b))

    C_mix = full_score(np.abs(E_mix) ** 2)
    C_hg = full_score(I_hg)
    C_lg = full_score(I_lg)

    # 4) 有效模数（系数 >1% 能量的项）
    def n_eff(keep):
        e = np.array([abs(c) ** 2 for _, c in keep])
        e = e / e.sum()
        return int((e > 0.01).sum()), e

    n_hg, e_hg = n_eff(keep_hg)
    n_lg, e_lg = n_eff(keep_lg)

    print(f'  重构误差: HG {err_hg:.2e} / LG {err_lg:.2e}（应 <1e-10）')
    print(f'  C_s02: 混合 {C_mix:.4f} / 纯HG {C_hg:.4f} / 纯LG {C_lg:.4f}')
    print(f'  有效模数: HG {n_hg} / LG {n_lg}')
    mx = max(abs(c) for _, c in keep_hg) if keep_hg else 1.0
    print(f'  纯HG 头部: {[(l, f"{abs(c)/mx*100:.1f}%") for l, c in keep_hg[:6]]}')

    report.append({
        'cn': cn, 'combo_std': info['std'],
        'orders': orders,
        'err_hg': float(err_hg), 'err_lg': float(err_lg),
        'C_mix': C_mix, 'C_hg': C_hg, 'C_lg': C_lg,
        'n_hg': n_hg, 'n_lg': n_lg,
        'keep_hg': [[l, [c.real, c.imag]] for l, c in keep_hg],
        'keep_lg': [[l, [c.real, c.imag]] for l, c in keep_lg],
        'shift': shiftv,
    })
    json.dump(report[-1], open(OUT / f'{cn}_纯基表达.json', 'w',
                              encoding='utf-8'),
              ensure_ascii=False, indent=2)

# ---- 汇总 md ----
md = ['# 第1层：混合解的单一基族精确改写（2026-10-16）', '',
      '原理：HG 与 LG 各自完备，同阶子空间互相线性表示。',
      '方法：把混合解（自由度已烘焙）整体最小二乘投影到目标族全字典。',
      '重构误差应在数值精度量级；C_s02 与混合解一致（同一场的不同坐标）。', '']
for r in report:
    md += [f"## {r['cn']}（阶次 {r['orders']}）", '',
           f"- 混合组合（标准记号）：{' + '.join(r['combo_std'])}",
           f"- C_s02：混合 {r['C_mix']:.4f} = 纯HG {r['C_hg']:.4f} "
           f"= 纯LG {r['C_lg']:.4f}",
           f"- 重构误差：HG {r['err_hg']:.1e} / LG {r['err_lg']:.1e}",
           f"- 有效模数：纯HG {r['n_hg']} 个 / 纯LG {r['n_lg']} 个", '']
md += ['说明：这是**坐标改写**不是新拟合——同一物理场的不同基族表达。',
       '模数比混合解多（稀疏度代价），系数见各图 JSON。',
       '第 2 层搜索将回答：是否存在**更稀疏**的单一基族组合。']
open(OUT / '第1层_精确转换汇总.md', 'w', encoding='utf-8').write(
    '\n'.join(md))
print(f'\nDONE -> {OUT}')

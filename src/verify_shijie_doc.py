# -*- coding: utf-8 -*-
"""
verify_shijie_doc.py — 师姐文档独立复算（终检，2026-10-16）
================================================================
目的：模拟师姐的操作——只拿 `师姐用叠加式_标准约定_完整版.md`
里的手写叠加式（幅度+相位+模式标签），在 MATLAB 标准约定下
从解析式构造每个模式，裸叠加（不含逐模微调），看：
  1. pcc(文档裸叠, 我方换算解裸渲染) —— 应 ≈1.0000（证明文档
     公式与我们换算参数完全等价；本轮此前怀疑的 "π 差异" 若
     真存在，此处会暴露）；
  2. C_s02(实验, 文档裸叠) —— 无 dof 复现值（师姐手叠能到的分）；
  3. C_s02(实验, 我方含 dof 换算解) —— 交付分（应=v3 JSON）。

解析式全部独立实现，不 import 项目基库。
"""
import json
import re
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from scipy.special import hermite, genlaguerre, factorial

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'closed_routes'))  # morph_search_lib 所在目录
from search_two_mode import load_experiment, gaussian_blur
from structural_check import find_dark_points

DOC = ROOT / '归档_八图分解' / '师姐复核检查_2026-10-16' \
    / '师姐用叠加式_标准约定_完整版.md'
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
SIZE, EXTENT = 224, 4.0
ORDER = ['4奇点', '5奇点', '6奇点', '8奇点1', '8奇点2', '9奇点',
         '10奇点', '11奇点']


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


# ---------- MATLAB 标准解析式（完全独立实现） ----------
x = np.linspace(-EXTENT, EXTENT, SIZE)
y = np.linspace(-EXTENT, EXTENT, SIZE)
Y, X = np.meshgrid(y, x, indexing='ij')      # Y 轴0(竖直), X 轴1(水平)
R = np.sqrt(X ** 2 + Y ** 2)
TH = np.arctan2(Y, X)


def std_hg(m, n):
    norm = np.sqrt(2.0 / (np.pi * factorial(m, exact=True)
                          * factorial(n, exact=True) * (2.0 ** (m + n))))
    return (norm * hermite(m)(X * np.sqrt(2)) * hermite(n)(Y * np.sqrt(2))
            * np.exp(-(X ** 2 + Y ** 2))).astype(np.complex128)


def std_lg(p, l):
    return (genlaguerre(p, abs(l))(2 * R ** 2)
            * (np.sqrt(2) * R) ** abs(l) * np.exp(-R ** 2)
            * np.exp(1j * l * TH)).astype(np.complex128)


def field_of(label):
    fam, idx = label[:2], label[3:-1]
    a, b = (int(v) for v in idx.split(','))
    return std_hg(a, b) if fam == 'HG' else std_lg(a, b)


# ---------- 解析文档 ----------
md = DOC.read_text(encoding='utf-8')
term_re = re.compile(
    r'([\d.]+)\*(?:exp\(1i\*([+\-][\d.]+)\)\*)?'
    r'(HG|LG)\((\-?\d+),(\-?\d+)\)\(x,y\)')

# 文档按 ORDER 顺序生成，直接逐节切
sections = re.split(r'\n## ', md)
results = {}
ok_all = True
print(f'{"图":10s} {"pcc(文档,换算)":16s} {"C_文档裸叠":12s} '
      f'{"C_含dof(=v3)":12s} 判定')
for cn in ORDER:
    sec = next(s for s in sections if cn in s.split('，')[0]
               or f'（{cn}，' in s)
    em = re.search(r'- E = (.+)', sec)
    terms = term_re.findall(em.group(1))
    assert len(terms) >= 2, f'{cn}: 解析失败'
    E_doc = np.zeros((SIZE, SIZE), dtype=np.complex128)
    combo_std = []
    for amp_s, phi_s, fam, a_s, b_s in terms:
        amp = float(amp_s)
        phi = float(phi_s) if phi_s else 0.0
        lab = f'{fam}({a_s},{b_s})'
        combo_std.append(lab)
        E_doc += amp * np.exp(1j * phi) * field_of(lab)
    I_doc = np.abs(E_doc) ** 2
    I_doc /= I_doc.max()

    # 我方换算解（含 dof）——从新基渲染
    import morph_search_lib as L
    from scipy.ndimage import shift as ndshift
    from search_two_mode_joint import build_joint_modes
    global _modes_cache
    try:
        modes, labels, lab2k, mnp = _modes_cache
    except NameError:
        modes, labels = build_joint_modes(10, SIZE, EXTENT, 'cpu')
        lab2k = {l: k for k, l in enumerate(labels)}
        mnp = modes.numpy()
        _modes_cache = (modes, labels, lab2k, mnp)

    r = V3[cn]
    src = r['accepted'] if 'theta' in r.get('accepted', {}) \
        else r['refined_v3']
    N = r['N']
    theta = [float(v) for v in src['theta']]

    def parse_mode(s):
        fam2, idx2 = s.strip()[:2], s.strip()[3:-1]
        a2, b2 = (int(v) for v in idx2.split(','))
        return fam2, a2, b2

    l_of = [parse_mode(s)[2] if parse_mode(s)[0] == 'LG' else 0
            for s in r['combo']]
    l0 = l_of[0]
    theta_std = list(theta)
    for i in range(1, N):
        theta_std[N + i - 1] = (theta[N + i - 1]
                               + (l_of[i] - l0) * np.pi / 2)

    # 裸渲染（无 dof；与文档裸叠同口径）
    E_ours = np.zeros((SIZE, SIZE), dtype=np.complex128)
    for i, lab in enumerate(combo_std):
        phi = 0.0 if i == 0 else theta_std[N + i - 1]
        E_ours += np.sqrt(max(theta[i], 0)) * mnp[lab2k[lab]] \
            * np.exp(1j * phi)
    I_ours_bare = np.abs(E_ours) ** 2
    I_ours_bare /= I_ours_bare.max()

    # 含 dof 渲染（=交付口径）
    sx = [float(v) for v in src['sx']]
    sy = [float(v) for v in src['sy']]
    shiftv = [float(v) for v in src['shift']]
    rot = [float(v) for v in src.get('rot', [0.0] * N)]
    E2 = np.zeros((SIZE, SIZE), dtype=np.complex128)
    for i, lab in enumerate(combo_std):
        Fm = mnp[lab2k[lab]]
        if abs(sx[i] - 1.0) > 1e-9:
            Fm = L.zoom_axis_c(Fm, sx[i], axis=1)
        if abs(sy[i] - 1.0) > 1e-9:
            Fm = L.zoom_axis_c(Fm, sy[i], axis=0)
        if abs(rot[i]) > 1e-9:
            from scipy.ndimage import rotate as ndrotate
            re_ = ndrotate(Fm.real, rot[i], reshape=False, order=1)
            im_ = ndrotate(Fm.imag, rot[i], reshape=False, order=1)
            Fm = re_ + 1j * im_
        phi = 0.0 if i == 0 else theta_std[N + i - 1]
        E2 += np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi)
    I_dof = np.abs(E2) ** 2
    I_dof /= I_dof.max()
    I_dof = ndshift(I_dof, shiftv, order=1, mode='nearest')

    name = r['name']
    Ie = load_experiment(f'{IMG}/{name}.png', 'nipy_spectral', 'isot')
    Ie_b = blur2(Ie)
    p_eq = pcc(I_doc, I_ours_bare)
    c_doc = C_s02(Ie, Ie_b, I_doc)
    c_dof = C_s02(Ie, Ie_b, I_dof)
    c_json = float(src.get('C_s02', r['delivery']['C_s02']))
    ok = (p_eq > 0.999) and abs(c_dof - c_json) < 2e-3
    ok_all &= ok
    results[cn] = {'pcc_doc_vs_ours': p_eq, 'C_doc_bare': c_doc,
                   'C_ours_dof': c_dof, 'C_json': c_json}
    print(f'{cn:10s} {p_eq:16.6f} {c_doc:12.4f} {c_dof:12.4f} '
          f'{"PASS" if ok else "FAIL"}'
          + ('' if abs(c_dof - c_json) < 2e-3
             else f'  (json={c_json:.4f}?)'))

json.dump(results, open(ROOT / '归档_八图分解' / '师姐复核检查_2026-10-16'
                       / '文档独立复算_结果.json', 'w',
                       encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'\n{"=" * 60}\n文档独立复算总判定: {"ALL PASS" if ok_all else "FAIL"}')
sys.exit(0 if ok_all else 1)

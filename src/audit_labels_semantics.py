# -*- coding: utf-8 -*-
"""audit_labels_semantics.py — 标签语义复核（换算规则独立验证）"""
import sys
import re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode_joint import build_joint_modes

SIZE, EXTENT = 224, 4.0
modes, labels = build_joint_modes(10, SIZE, EXTENT, 'cpu')
modes = modes.numpy()
lab2k = {l: k for k, l in enumerate(labels)}


def blob_axes(I):
    th = 0.35 * I.max()
    pr = (I > th).sum(axis=1)
    pc = (I > th).sum(axis=0)

    def peaks(p):
        c = 0
        ins = False
        for v in p:
            if v > 0 and not ins:
                c += 1
                ins = True
            elif v == 0:
                ins = False
        return c
    return peaks(pr), peaks(pc)


print('标签语义抽检（新基 labels 应为标准记号语义）:')
fails = 0
for lab in ['HG(2,0)', 'HG(0,2)', 'HG(1,3)', 'HG(3,1)', 'HG(4,6)']:
    m, n = map(int, re.findall(r'-?\d+', lab))
    I = np.abs(modes[lab2k[lab]]) ** 2
    nr, nc = blob_axes(I)
    ok = (nr == n + 1) and (nc == m + 1)
    fails += not ok
    print(f'  {lab}: {nr}行×{nc}列 期望 {n+1}×{m+1}  '
          f'{"PASS" if ok else "FAIL"}')

for lab, expect in [('LG(0,3)', 1080), ('LG(0,-2)', -720)]:
    k = lab2k[lab]
    F = modes[k]
    ph = [np.angle(F[112 + int(60 * np.sin(np.radians(a))),
                   112 + int(60 * np.cos(np.radians(a)))])
          for a in range(0, 361, 30)]
    w = np.degrees(np.sum(np.diff(np.unwrap(np.array(ph)))))
    ok = expect - 40 < w < expect + 40
    fails += not ok
    print(f'  {lab} 绕中心相位累积 {w:.0f}° (期望 {expect}°)  '
          f'{"PASS" if ok else "FAIL"}')

# 抽检内部接口一致性：build_joint_modes 的 LG 与
# search_two_mode_lg.build_lg_modes 是否同一来源（避免双基不同步）
from search_two_mode_lg import build_lg_modes
m2, idx2 = build_lg_modes(max_order=10, size=SIZE, extent=EXTENT,
                          device='cpu')
m2 = m2.numpy()
print('\n双基一致性（joint vs lg 直连，应逐位相同）:')
for (p, l) in [(0, 3), (2, 1), (1, -3)]:
    a = modes[lab2k[f'LG({p},{l})']]
    b = m2[idx2.index((p, l))]
    d = np.abs(a - b).max()
    fails += not (d < 1e-6)
    print(f'  LG({p},{l}): max|Δ|={d:.2e}  '
          f'{"PASS" if d < 1e-6 else "FAIL"}')

print(f'\n总判据: {"ALL PASS" if fails == 0 else f"{fails} FAIL"}')

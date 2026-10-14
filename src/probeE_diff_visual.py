# -*- coding: utf-8 -*-
"""
probeE_diff_visual.py -- 核查：终检新旧图差异是否真实存在、为何肉眼难辨
=====================================================================
1. 数值核查：对八图计算 交付版 vs 终检版 渲染图的逐像素差异
   （max/RMS/相关），并核对差异是否与施加的平移/束腰幅度一致；
2. 可视化：对代表图（8(2)/6奇点）生成
   [旧 | 新 | 差异×10 | 红绿叠加] 四联图 + 闪烁 GIF（交替旧/新）。
Outputs: 归档_八图分解/终检精修_2026-10-10/差异可视化/
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
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from PIL import Image

OUT = ROOT / '归档_八图分解' / '终检精修_2026-10-10'
DIFF = OUT / '差异可视化'
DIFF.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()


def blur_sig(I, sg):
    if sg == 0:
        return I
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), float(sg))[0].numpy()


def render_sol(combo, theta, base_scale, scales, shiftv):
    N = len(combo)
    E = np.zeros(modes_np[0].shape, dtype=np.complex128)
    for i, m in enumerate(combo):
        F = modes_np[lab2k[m]]
        s = (base_scale or 1.0) * (scales[i] if scales else 1.0)
        if abs(s - 1.0) > 1e-6:
            F = rescale_lg(torch.from_numpy(F).unsqueeze(0),
                           float(s))[0].numpy()
        phi = 0.0 if i == 0 else theta[N + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * F * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    if shiftv is not None and (abs(shiftv[0]) > 1e-6
                               or abs(shiftv[1]) > 1e-6):
        from scipy.ndimage import shift as ndshift
        I = ndshift(I, shiftv, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


ti = json.load(open(OUT / 'terminal_inspection.json', encoding='utf-8'))

print(f'{"图":8s} {"平移px":>12s} {"束腰幅度":>10s} '
      f'{"max|ΔI|":>8s} {"RMS(ΔI)":>8s} {"PCC(旧,新)":>9s}')
demo = {}
for CN, rec in ti.items():
    combo, N = rec['combo'], rec['N']
    base = rec['baseline']
    acc = rec['accepted']
    I_old = render_sol(combo, [float(x) for x in base['theta']],
                       rec['base_scale'], None, None)
    I_new = render_sol(combo, [float(x) for x in acc['theta']],
                       rec['base_scale'], acc['scales'], acc['shift'])
    d = np.abs(I_new - I_old)
    a = I_old.ravel() - I_old.mean()
    b = I_new.ravel() - I_new.mean()
    r = float((a * b).sum()
              / np.sqrt((a * a).sum() * (b * b).sum()))
    sh = acc['shift'] if acc['shift'] else [0, 0]
    sc = acc['scales'] if acc['scales'] else [1.0] * N
    sc_span = max(sc) - min(sc)
    print(f'{CN:8s} ({sh[0]:+.2f},{sh[1]:+.2f}) {sc_span:10.2f} '
          f'{d.max():8.4f} {np.sqrt((d**2).mean()):8.4f} {r:9.5f}')
    demo[CN] = (I_old, I_new)

# ---- demo figure for two representative images ----
for CN in ['8奇点2', '6奇点']:
    I_old, I_new = demo[CN]
    rec = ti[CN]
    d = np.abs(I_new - I_old)
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
    axs[0].imshow(blur_sig(I_old, 2), cmap='inferno', vmin=0, vmax=1)
    axs[0].set_title(f'delivered (C_s02='
                     f'{rec["baseline"]["C_s02"]:.4f})')
    axs[1].imshow(blur_sig(I_new, 2), cmap='inferno', vmin=0, vmax=1)
    axs[1].set_title(f'terminal (C_s02='
                     f'{rec["accepted"]["C_s02"]:.4f})')
    axs[2].imshow(d, cmap='hot', vmin=0, vmax=max(d.max(), 1e-6))
    axs[2].set_title(f'|difference| (true scale, max={d.max():.3f})')
    ov = np.zeros((*I_old.shape, 3))
    ov[..., 0] = np.clip(I_old, 0, 1)          # red = delivered
    ov[..., 1] = np.clip(I_new, 0, 1)          # green = terminal
    axs[3].imshow(np.clip(ov, 0, 1))
    axs[3].set_title('overlay: red=delivered, green=terminal\n'
                     '(yellow=identical)')
    for a_ in axs:
        a_.axis('off')
    fig.suptitle(f'{rec["name"]}: delivered vs terminal - where is '
                 f'the difference actually?', fontsize=12)
    fig.tight_layout()
    fig.savefig(DIFF / f'{CN}_差异诊断.png', dpi=140,
                bbox_inches='tight')
    plt.close(fig)

    # blink GIF (old/new alternating, 8 frames)
    frames = []
    for k in range(8):
        I = I_old if k % 2 == 0 else I_new
        arr = (blur_sig(I, 2) * 255).astype(np.uint8)
        rgb = np.stack([arr] * 3, axis=-1)
        frames.append(Image.fromarray(rgb).resize((448, 448),
                                                  Image.NEAREST))
    frames[0].save(DIFF / f'{CN}_闪烁对比.gif', save_all=True,
                   append_images=frames[1:], duration=500, loop=0)
    print(f'  [gif] {CN}_闪烁对比.gif')

# ---- larger-amplified difference for 8(2) shift visualization ----
I_old, I_new = demo['8奇点2']
d = I_new - I_old
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.7))
axs[0].imshow(d, cmap='RdBu_r', vmin=-0.25, vmax=0.25)
axs[0].set_title('signed difference (terminal - delivered)\n'
                 'red=terminal higher, blue=lower')
axs[1].imshow(np.clip(np.abs(d) * 10, 0, 1), cmap='hot')
axs[1].set_title('|difference| x10 (saturation view)')
for a_ in axs:
    a_.axis('off')
fig.suptitle('8Singularities(2): the change is a (+0.5,+3.5)px shift '
             '+ per-mode waist tune - small on a 224px canvas',
             fontsize=11)
fig.tight_layout()
fig.savefig(DIFF / '8奇点2_差异放大.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'[done] {DIFF}')

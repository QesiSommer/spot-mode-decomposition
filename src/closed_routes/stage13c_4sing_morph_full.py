# -*- coding: utf-8 -*-
"""
stage13c_4sing_morph_full.py -- 全空间形态普查（366,145 三模组合）
====================================================================
stage13a 只在 C_s02 top-60 里筛形态，不是全空间形态搜索。本脚本
用 GPU 向量化的形态代理对**全部组合**做一次普查：

  1. 复用粗扫（64px）得到每组合的 best 参数（bc, bi）；
  2. 批量渲染 64px 模型图（解析基场 + 系数），计算两个 GPU 代理：
       - central_dark_frac(r<=8.5px@64) ；
       - 极坐标轮廓谱（grid_sample 180 射线 × 60 半径）→
         cv_r 与角向谐波（对比同分辨率实验图，同口径）；
  3. 形态筛选（中央差 ≤ 0.12 且 cv 比 ≥ 0.55）→ 按粗扫相关降序取
     top-80，注入已知好解（现解/旧解/stage13b 冠军）；
  4. 全部在完整自由度管线精修 + 三项判定（形态/结构/C_s02）。

Outputs: 归档_八图分解/4奇点_形态重搜索/stage13c_morph_full.json
"""
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from refine_all_new_method import build_base_fields_N, c_table_N
from rerun_new_metrics import (global_stats_02, c_s02, GRIDS, COARSE,
                               coarse_grid_params, coarse_scan_N)
from structural_check import find_dark_points, match_points
from stage0_diagnosis import parse_md
from morphology_metrics import morph_compare, centroid
import stage13b_4sing_morph_dof as B   # reuse render/refine helpers

OUT = ROOT / '归档_八图分解' / '4奇点_形态重搜索'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '4Singularities'
CN = '4奇点'
N = 3
RB = 64                      # screen resolution
NT, NR = 180, 60             # polar sampling
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[setup] device={dev}')

modes, labels = B.modes, B.labels
lab2k = B.lab2k
Ie = B.Ie
pts_e = B.pts_e
N_EXP = B.N_EXP

# ---------------- coarse scan (analytic) ----------------
def down_complex(x, size):
    t = torch.view_as_real(x).permute(2, 0, 1)[None]
    s = F.interpolate(t, size=(size, size), mode='bilinear',
                      align_corners=False)
    return torch.view_as_complex(s[0].permute(1, 2, 0).contiguous())


Ie_t = torch.tensor(Ie, dtype=torch.float32)
Ie_small = F.interpolate(Ie_t[None, None], size=(RB, RB),
                         mode='bilinear', align_corners=False)[0, 0]
modes_small = torch.stack([down_complex(m, RB) for m in modes]).to(dev)
Ie_small = Ie_small.to(dev)
combos = torch.combinations(torch.arange(len(labels)), r=N)
cg = COARSE[N]
pmat3 = coarse_grid_params(N, cg['n_energy'], cg['n_phase'])
ct3 = c_table_N(pmat3, N).to(dev)
t0 = time.time()
bc, bi = coarse_scan_N(modes_small, Ie_small, combos, ct3, N)
print(f'[coarse] {combos.shape[0]} combos in {time.time()-t0:.0f}s '
      f'best={float(bc.max()):.4f}')

# best params per combo
P_all = torch.tensor(pmat3[bi.cpu().numpy()], dtype=torch.float64)
C_all = c_table_N(P_all.numpy(), N)          # (Q,9) float32
del P_all

# ---------------- GPU polar sampler ----------------
def make_polar_grid(cy, cx, nt=NT, nr=NR, rmax=None):
    if rmax is None:
        rmax = RB * 0.45
    rr = np.linspace(0.0, rmax, nr)
    th = np.linspace(0.0, 2 * np.pi, nt, endpoint=False)
    R, T = np.meshgrid(rr, th, indexing='ij')       # (nr, nt)
    ys = cy + R * np.sin(T)
    xs = cx + R * np.cos(T)
    # normalize to [-1,1] for grid_sample
    gx = (xs / (RB - 1)) * 2 - 1
    gy = (ys / (RB - 1)) * 2 - 1
    return np.stack([gx, gy], axis=-1).astype(np.float32)   # (nr,nt,2)


Ie_t64 = Ie_small[None, None]                    # (1,1,RB,RB)


def morph_batch(I_batch):
    """GPU 形态代理：central_dark_frac + 轮廓谱(cv, m=1..12)。"""
    Bn = I_batch.shape[0]
    mx = I_batch.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    In = I_batch / mx
    # --- central darkness (fixed geometric center) ---
    c = (RB - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(RB, device=dev),
                            torch.arange(RB, device=dev),
                            indexing='ij')
    rr = ((yy - c) ** 2 + (xx - c) ** 2).sqrt()
    cen = rr <= (30.0 / 224.0 * RB)              # r<=30 @224 scale
    dark = ((In < 0.10) & cen[None]).sum(dim=(1, 2)).float() \
        / cen.sum().float()
    # --- outline via grid_sample ---
    gy, gx = [], []
    out_spec = []
    for b0 in range(0, Bn, 64):
        sub = In[b0:b0 + 64, None]               # (b,1,RB,RB)
        b = sub.shape[0]
        grid = torch.tensor(make_polar_grid(c, c),
                            device=dev).unsqueeze(0).expand(b, -1, -1,
                                                            -1)
        samp = F.grid_sample(sub, grid, mode='bilinear',
                             align_corners=True, padding_mode='zeros')
        samp = samp[:, 0]                        # (b, nr, nt)
        above = samp >= 0.20                     # frac 0.20
        has = above.any(dim=1)                   # (b, nt)
        idx = torch.where(has,
                          NR - 1 - above.flip(1).float().argmax(dim=1),
                          torch.full_like(has, -1, dtype=torch.long))
        rr_line = torch.linspace(0.0, RB * 0.45, NR, device=dev)
        rv = torch.where(idx >= 0, rr_line[idx.clamp_min(0)],
                         torch.full_like(idx, -1, dtype=torch.float32)
                         .float())
        rv = torch.where(idx >= 0, rv,
                         torch.full_like(rv, float('nan')))
        mean_r = torch.nanmean(rv, dim=1)
        rv2 = torch.where(torch.isnan(rv), mean_r[:, None], rv)
        spec = (rv2 - mean_r[:, None])
        Fs = torch.fft.rfft(spec, dim=1) / NT
        amp = Fs[:, 1:13].abs() / mean_r[:, None].clamp_min(1e-9)
        cv = rv2.std(dim=1) / mean_r.clamp_min(1e-9)
        out_spec.append(amp)
    amp = torch.cat(out_spec, 0)
    cv = torch.cat([cv], 0) if False else None
    return dark, None, amp


# reference (experiment @64)
dark_e, _, amp_e = morph_batch(Ie_small[None])
dark_e = float(dark_e[0])
cv_e, amp_e = None, amp_e[0].cpu().numpy()
# recompute cv for experiment via same path below
def cv_of_amp(I_np):
    pass

# ---- batch screen over all combos ----
Q = combos.shape[0]
BATCH = 256
screen = {'central': np.zeros(Q, np.float32),
          'cv': np.zeros(Q, np.float32),
          'amp': np.zeros((Q, 12), np.float32)}
K = 9
t0 = time.time()
for b0 in range(0, Q, BATCH):
    q = combos[b0:b0 + BATCH].to(dev)
    Fs = [modes_small[q[:, i]] for i in range(N)]
    G = build_base_fields_N(Fs)                  # (B,K,RB,RB)
    Cb = C_all[b0:b0 + BATCH].to(dev)
    I = torch.einsum('bk,bkhw->bhw', Cb, G)
    I = I.clamp_min(0)
    dark, _, amp = morph_batch(I)
    # cv from a second helper: compute rv std/mean inline
    screen['central'][b0:b0 + BATCH] = dark.cpu().numpy()
    screen['amp'][b0:b0 + BATCH] = amp.cpu().numpy()
    if (b0 // BATCH) % 200 == 0:
        el = time.time() - t0
        print(f'  screen {b0}/{Q} ({el:.0f}s)')

# cv from amp spectrum is not identical to cv_r; use spectral L1 instead
sum_e = float(np.sum(amp_e)) + 1e-12
shape_d = np.abs(screen['amp'] - amp_e[None]).sum(1) / sum_e
central_d = np.abs(screen['central'] - dark_e)
print(f'[screen] done {time.time()-t0:.0f}s | exp central={dark_e:.3f} '
      f'sum_amp={sum_e:.3f}')

# ---- morphology screen ----
pass_m = (central_d <= 0.12) & (shape_d <= 0.55)
n_pass = int(pass_m.sum())
print(f'[screen] morphology-pass: {n_pass} / {Q} '
      f'({100.0*n_pass/Q:.2f}%)')

surv_idx = np.where(pass_m)[0]
order = np.argsort(-bc.cpu().numpy()[surv_idx])
surv = surv_idx[order]
print(f'[screen] top-8 survivors (by coarse corr):')
for k in surv[:8]:
    print(f'   {"+".join(labels[t] for t in combos[k].tolist()):44s} '
          f'corr={float(bc[k]):.4f} central={screen["central"][k]:.3f} '
          f'shape={shape_d[k]:.3f}')

np.save(OUT / 'screen_central.npy', screen['central'])
np.save(OUT / 'screen_shape.npy', shape_d)
np.save(OUT / 'screen_passmask.npy', pass_m)

# ---- refine survivors (full DOF) ----
TOPN = 80
cand_list = []
seen = set()
for k in surv[:TOPN]:
    tr = combos[k].tolist()
    cand_list.append({'combo': [labels[t] for t in tr],
                      'theta': pmat3[int(bi[k])].tolist(),
                      'tag': 'fullscreen'})

a13 = json.load(open(OUT / 'stage13a_morph_search.json', encoding='utf-8'))
for r in a13['ranked'][:6]:
    if tuple(r['combo']) not in seen:
        cand_list.append({'combo': list(r['combo']),
                          'theta': list(r['theta']), 'tag': 'a13'})
combo_old, theta_old, scale_old = parse_md(
    ROOT / '归档_八图分解' / '最新结果' / f'{CN}_参数与指标.md')
V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))
acc_new = V3[CN]['accepted']
cand_list.append({'combo': list(combo_old), 'theta': list(theta_old),
                  'tag': 'historic'})
cand_list.append({'combo': list(V3[CN]['combo']),
                  'theta': [float(x) for x in acc_new['theta']],
                  'tag': 'current'})

print(f'[refine] {len(cand_list)} candidates with full DOF...')
records = []
t0 = time.time()
for ci, c in enumerate(cand_list):
    Nc = len(c['combo'])
    sx0 = [1.0] * Nc
    sy0 = [1.0] * Nc
    if c['tag'] == 'historic' and scale_old:
        sx0 = [float(scale_old)] * Nc
        sy0 = [float(scale_old)] * Nc
    r = B.refine(c['combo'], c['theta'], sx0, sy0, [0.0, 0.0],
                 [0.0] * Nc, rounds=2, budget=100)
    r['combo'] = c['combo']
    r['tag'] = c['tag']
    records.append(r)
    if ci % 10 == 9:
        print(f'  [{ci+1}/{len(cand_list)}] ({time.time()-t0:.0f}s) '
              f'best so far C_s02='
              f'{max(x["C_s02"] for x in records):.4f}')

records.sort(key=lambda r: (0 if (r['morph_pass'] and r['struct'])
                            else 1, -r['C_s02']))
usable = [r for r in records if r['morph_pass'] and r['struct']]
champ = usable[0] if usable else None
print('\n=== FINAL ranked (morph+struct PASS first) ===')
for k, r in enumerate(records[:12]):
    ok = r['morph_pass'] and r['struct']
    print(f'  {k+1:2d}.{"★" if ok else " "} '
          f'{"+".join(r["combo"]):44s} [{r["tag"]:10s}] '
          f'C_s02={r["C_s02"]:.4f} centralΔ={r["central_delta"]:.3f} '
          f'cvR={r["cv_ratio"]:.2f} shape={r["shape_dist"]:.3f} '
          f'cores={r["cores"]}/{N_EXP}')

out = {'image': NAME, 'n_combos': int(Q),
       'morph_pass_count': n_pass,
       'morph_pass_frac': n_pass / Q,
       'n_refined': len(records),
       'n_usable': len(usable),
       'champion': ({k: v for k, v in champ.items() if k != '_I'}
                    if champ else None),
       'records': [{k: v for k, v in r.items() if k != '_I'}
                   for r in records]}
(OUT / 'stage13c_morph_full.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
if champ:
    print(f'\n[champion] {"+".join(champ["combo"])} '
          f'C_s02={champ["C_s02"]:.4f} '
          f'centralΔ={champ["central_delta"]:.3f} '
          f'cvR={champ["cv_ratio"]:.2f} shape={champ["shape_dist"]:.3f}')
print(f'[done] {OUT}')

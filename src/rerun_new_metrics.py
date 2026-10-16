# -*- coding: utf-8 -*-
"""
rerun_new_metrics.py -- 8 图按"新指标管线"重新分解与拟合
==========================================================

Pipeline per image (N = established mode count):
  1. coarse scan of ALL C(131,N) combos on a 64px downsampled domain
     (N=4 uses the guided pool) -> top-30
  2. refinement with the NEW METHOD (multi-start block refinement,
     OLS projection seed) optimizing the window-averaged objective
     W_avg (56px support windows, sigma in {0,2}); two-mode cases scan
     the scale s on an outer loop
  3. acceptance report: C_s02, W_avg, rho/C_min at 56px & 32px,
     structural (dark-core) verification, angular spectrum, diff
     localization -- GATES ARE ADVISORY ONLY (no auto-rejection)
  4. outputs (same layout as 最新结果):
       <cn>_拟合对比.png       5 panels: original | processed |
                               model(s2) | |diff| | window heatmap
       <cn>_参数与指标.md
       <cn>_指标验收单.md
       <cn>_新旧拟合对比.png    previous model vs new model
       plus 汇总表.md

Output folder: 归档_八图分解/最新结果_新指标/
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from search_two_mode_lg import rescale_lg
from search_three_mode import eval_c_grid
from refine_all_new_method import build_base_fields_N, c_table_N, ols_c, \
    project_c_N
from window_metrics import build_window_stats, w_avg_batch, \
    w_avg_point, window_pccs
from structural_check import find_dark_points, match_points
from calibrate_metrics import angular_distance, render_model
from verify_metrics import diff_localization, pcc, window_stats

ARCH = ROOT / '归档_八图分解'
DEST = ARCH / '最新结果_新指标'
DEST.mkdir(exist_ok=True)
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
SIG02 = (0, 2)

CASES = [
    {'name': '4Singularities', 'cn': '4奇点', 'N': 3},
    {'name': '5Singularities', 'cn': '5奇点', 'N': 2},
    {'name': '6Singularities', 'cn': '6奇点', 'N': 2},
    {'name': '8Singularities(1)', 'cn': '8奇点(1)', 'N': 3},
    {'name': '8Singularities(2)', 'cn': '8奇点(2)', 'N': 2},
    {'name': '9Singularities', 'cn': '9奇点', 'N': 3},
    {'name': '10Singularities', 'cn': '10奇点', 'N': 4},
    {'name': '11Singularities', 'cn': '11奇点', 'N': 3},
]

GRIDS = {2: {'n_ph': 81, 'n_en': 81, 'rounds': 8},
         3: {'n_ph': 61, 'n_en': 41, 'rounds': 6},
         4: {'n_ph': 15, 'n_en': 11, 'rounds': 5}}
COARSE = {2: {'n_energy': 16, 'n_phase': 120},
          3: {'n_energy': 16, 'n_phase': 20},
          4: {'n_energy': 10, 'n_phase': 16}}

# thresholds (calibrated; advisory in this run)
T_C, T0, T2 = 0.83, 0.82, 0.83
RHO_WARN, CMIN_WARN = 0.2, 0.3
RHO56_GATE, CMIN56_GATE = 0.10, 0.40


# ---------------------------------------------------------------------------
# previous solutions (for the fallback seed & old/new comparison)
# ---------------------------------------------------------------------------

def load_prev():
    with open(ROOT / 'refine_methods_compare' / 'methods_compare.json',
              encoding='utf-8') as f:
        mc = json.load(f)
    with open(ROOT / 'refine_all_new_method' / 'all_results.json',
              encoding='utf-8') as f:
        an = json.load(f)
    prev = {}
    for tag, name in [('4Sing', '4Singularities'),
                      ('8Sing1', '8Singularities(1)')]:
        b = mc[tag]['method_B']
        prev[name] = {'combo': list(mc[tag]['combo']),
                      'params': [float(x) for x in b['theta']],
                      'scale': None}
    for tag, r in an.items():
        prev[r['name']] = {'combo': list(r['combo']),
                           'params': [float(x) for x in r['new_params']],
                           'scale': r['new_scale']}
    return prev


# ---------------------------------------------------------------------------
# coarse scan (generic N-mode)
# ---------------------------------------------------------------------------

def energy_simplex_N(N, steps):
    pts = []

    def rec(prefix, remaining, depth):
        if depth == N - 1:
            pts.append(prefix + [remaining])
            return
        for i in range(remaining + 1):
            rec(prefix + [i], remaining - i, depth + 1)
    rec([], steps, 0)
    return np.array(pts, dtype=np.float64) / steps


def coarse_grid_params(N, n_energy, n_phase):
    e_pts = energy_simplex_N(N, n_energy)
    axes = [np.linspace(0, 2 * np.pi, n_phase, endpoint=False)] * (N - 1)
    mesh = np.meshgrid(*axes, indexing='ij')
    ph_pts = np.stack([m.ravel() for m in mesh], axis=1)
    u = np.repeat(e_pts, ph_pts.shape[0], axis=0)
    ph = np.tile(ph_pts, (len(e_pts), 1))
    return np.concatenate([u, ph], axis=1)


def coarse_scan_N(modes_small, Ie_small, combos, c_tab, N, batch=256):
    dev = modes_small.device
    Ie = Ie_small
    mu = Ie.mean()
    var = Ie.var(unbiased=False)
    Q = combos.shape[0]
    P = c_tab.shape[0]
    bc = torch.empty(Q, device=dev)
    bi = torch.empty(Q, dtype=torch.long, device=dev)
    for b0 in range(0, Q, batch):
        q = combos[b0:b0 + batch].to(dev)
        Fs = [modes_small[q[:, i]] for i in range(N)]
        G = build_base_fields_N(Fs)
        B = G.shape[0]
        Gf = G.reshape(B, G.shape[1], -1)
        m = Gf.mean(dim=2)
        E = torch.einsum('bki,i->bk', Gf, Ie.reshape(-1)) / Gf.shape[2]
        Gram = torch.einsum('bki,bli->bkl', Gf, Gf) / Gf.shape[2]
        bmx = torch.full((B,), -2.0, device=dev)
        bmi = torch.zeros(B, dtype=torch.long, device=dev)
        for c0 in range(0, P, 100000):
            cc = c_tab[c0:c0 + 100000]
            corr = eval_c_grid(cc, m, E, Gram, mu, var)
            mx, mi = corr.max(dim=1)
            upd = mx > bmx
            bmx = torch.where(upd, mx, bmx)
            bmi = torch.where(upd, mi + c0, bmi)
        bc[b0:b0 + B] = bmx
        bi[b0:b0 + B] = bmi
    return bc, bi


# ---------------------------------------------------------------------------
# W_avg refinement
# ---------------------------------------------------------------------------

def block_refine_wavg(win_stats, p0, N, n_ph, n_en, rounds):
    cur = [float(x) for x in p0]
    w_ph, w_en = 0.5, 0.08
    for r in range(rounds):
        axes = [np.linspace(cur[N + j] - w_ph, cur[N + j] + w_ph, n_ph)
                for j in range(N - 1)]
        mesh = np.meshgrid(*axes, indexing='ij')
        ph_mat = np.stack([m.ravel() for m in mesh], axis=1)
        u_mat = np.tile(cur[:N], (ph_mat.shape[0], 1))
        C = c_table_N(np.concatenate([u_mat, ph_mat], axis=1).astype(
            np.float64), N)
        corr = w_avg_batch(win_stats, C)
        k = int(torch.argmax(corr))
        for j in range(N - 1):
            cur[N + j] = float(ph_mat[k, j])
        ax_e = [np.linspace(max(0, cur[i] - w_en), min(1, cur[i] + w_en),
                            n_en) for i in range(N - 1)]
        me_ = np.meshgrid(*ax_e, indexing='ij')
        e_mat = np.stack([m.ravel() for m in me_], axis=1)
        e_mat = e_mat[e_mat.sum(axis=1) < 1.0]
        u_mat = np.concatenate(
            [e_mat, (1 - e_mat.sum(axis=1))[:, None]], axis=1)
        ph_rep = np.tile(cur[N:], (len(u_mat), 1))
        C = c_table_N(np.concatenate([u_mat, ph_rep], axis=1).astype(
            np.float64), N)
        corr = w_avg_batch(win_stats, C)
        k = int(torch.argmax(corr))
        cur[:N] = [float(x) for x in u_mat[k]]
        w_ph *= 0.3
        w_en *= 0.3
    return cur, w_avg_point(win_stats, cur, N)


def refine_candidate(modes, Ie, tr, N, p_seed, s_prev, glob_stats):
    """returns (params, wavg, scale, win_stats, win_meta, mask)"""
    c_ols, _ = ols_c(glob_stats, sigma_idx=1)  # index 1 == sigma=2
    seeds = [list(p_seed)]
    p_ols = project_c_N(c_ols, N)
    if p_ols is not None:
        seeds.insert(0, p_ols)
        rng = np.random.default_rng(0)
        for _ in range(2):
            j = [p_ols[i] + rng.normal(0, 0.02 if i < N else 0.2)
                 for i in range(2 * N - 1)]
            j[:N] = [abs(x) for x in j[:N]]
            sj = sum(j[:N])
            j[:N] = [x / sj for x in j[:N]]
            seeds.append(j)
    g = GRIDS[N]

    if N == 2:
        # outer scale scan
        best = None
        for s in np.linspace(s_prev - 0.10, s_prev + 0.10, 7):
            ws, meta, mask = build_window_stats(modes, Ie, tr, s)
            for sd in seeds:
                th, val = block_refine_wavg(ws, sd, N, g['n_ph'],
                                            g['n_en'], rounds=3)
                if best is None or val > best[0]:
                    best = (val, th, float(s))
        ws, meta, mask = build_window_stats(modes, Ie, tr, best[2])
        th, val = block_refine_wavg(ws, best[1], N, g['n_ph'],
                                    g['n_en'], rounds=8)
        return th, val, best[2], ws, meta, mask
    else:
        ws, meta, mask = build_window_stats(modes, Ie, tr, None)
        best = None
        for sd in seeds:
            th, val = block_refine_wavg(ws, sd, N, g['n_ph'],
                                        g['n_en'], rounds=g['rounds'])
            if best is None or val > best[0]:
                best = (val, th)
        return best[1], best[0], None, ws, meta, mask


# ---------------------------------------------------------------------------
# global stats (sigma 0/2) for C_s02
# ---------------------------------------------------------------------------

def global_stats_02(modes, Ie, tr, scale=None):
    dev = modes.device
    mnp = modes if scale is None else rescale_lg(modes, float(scale))
    Fs = [mnp[k].unsqueeze(0) for k in tr]
    G = build_base_fields_N(Fs)[0]
    K = G.shape[0]
    stats = []
    for sg in SIG02:
        Gb = gaussian_blur(G, float(sg))
        Ie_b = gaussian_blur(Ie.unsqueeze(0), float(sg))[0]
        me = Ie_b.mean()
        ve = Ie_b.var(unbiased=False)
        Gf = Gb.reshape(K, -1)
        m = Gf.mean(dim=1).unsqueeze(0)
        E = torch.einsum('ki,i->k', Gf, Ie_b.reshape(-1)
                         ).unsqueeze(0) / Gf.shape[1]
        Gram = torch.einsum('ki,li->kl', Gf, Gf
                            ).unsqueeze(0) / Gf.shape[1]
        stats.append((m, E, Gram, me, ve))
    return stats


def c_s02(stats, params, N):
    C = c_table_N(np.array([params], dtype=np.float64), N)
    vals = [float(eval_c_grid(C, *s)[0, 0]) for s in stats]
    return float(np.mean(vals)), vals


# ---------------------------------------------------------------------------
# main per-image routine
# ---------------------------------------------------------------------------

def main():
    import os
    only = os.environ.get('ONLY_CN')
    cases = [c for c in CASES if (only is None or c['cn'] == only)]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes = modes.to(dev)
    lab2k = {lab: k for k, lab in enumerate(labels)}
    modes_cpu = modes.detach().cpu()
    modes_np = modes_cpu.numpy()
    prev = load_prev()

    def _ds(x, size=64):
        return F.interpolate(x.unsqueeze(1), size=(size, size),
                             mode='bilinear',
                             align_corners=False)[:, 0]
    modes_small = torch.complex(_ds(modes.real), _ds(modes.imag)).to(dev)

    summary = []
    for case in cases:
        name, cn, N = case['name'], case['cn'], case['N']
        t_img = time.time()
        print(f'\n########## {cn} ({name}, N={N}) ##########')
        Ie_np = load_experiment(f'{IMG_DIR}/{name}.png',
                                'nipy_spectral', 'isot')
        Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
        Ie_small = F.interpolate(Ie[None, None], size=(64, 64),
                                 mode='bilinear',
                                 align_corners=False)[0, 0]
        pts_e = find_dark_points(Ie_np)
        pv = prev[name]

        # ---- candidate combos ----
        if N == 4:
            with open(ARCH / '10奇点' / '3_三模搜索与验收'
                      / '10Singularities_three_mode.json',
                      encoding='utf-8') as f:
                tj = json.load(f)
            base_tr = [tuple(sorted(lab2k[m] for m in e['modes']))
                       for e in tj['entries']]
            quads = set()
            for tr in base_tr:
                for x in range(len(labels)):
                    if x in tr:
                        continue
                    quads.add(tuple(sorted(tr + (x,))))
            combos = torch.tensor(sorted(quads), dtype=torch.long)
        else:
            combos = torch.combinations(torch.arange(len(labels)), r=N)
        print(f'  combos: {combos.shape[0]}')

        # ---- coarse scan ----
        p_prev = [float(x) for x in pv['params']]
        cg = COARSE[N]
        params_mat = coarse_grid_params(N, cg['n_energy'],
                                        cg['n_phase'])
        c_tab = c_table_N(params_mat, N)
        t0 = time.time()
        bc, bi = coarse_scan_N(modes_small, Ie_small, combos, c_tab, N)
        print(f'  [coarse] {time.time()-t0:.0f}s, '
              f'best={float(bc.max()):.4f}')
        K_TOP = 30
        vals, idxs = bc.topk(K_TOP)
        top_combos = combos[idxs.cpu()]
        top_p = params_mat[bi[idxs].cpu().numpy()]

        # fallback: ensure previous combo is a candidate
        prev_idx = torch.tensor(sorted(lab2k[m] for m in pv['combo']),
                                dtype=torch.long)
        has_prev = any(set(t.tolist()) == set(prev_idx.tolist())
                       for t in top_combos)
        cand_list = [(top_combos[k].tolist(), top_p[k].tolist())
                     for k in range(top_combos.shape[0])]
        if not has_prev:
            cand_list.append((prev_idx.tolist(), p_prev))
            print('  [fallback] previous combo injected')

        # ---- W_avg refinement of candidates ----
        t0 = time.time()
        results = []
        for cidx, (combo, p0) in enumerate(cand_list):
            tr = list(combo)
            gs = global_stats_02(modes, Ie, tr,
                                 pv['scale'] if N == 2 else None)
            (th, val, s_new, ws, meta, mask) = refine_candidate(
                modes, Ie, tr, N, p0,
                pv['scale'] if N == 2 else 0.0, gs)
            results.append({'combo': tr, 'params': th, 'wavg': val,
                            'scale': s_new, 'ws': ws, 'meta': meta,
                            'mask': mask})
        results.sort(key=lambda r: -r['wavg'])
        print(f'  [refine] {time.time()-t0:.0f}s, '
              f'top-3 wavg: {[round(r["wavg"],4) for r in results[:3]]}')

        # ---- pick: prefer structurally-verified; gates advisory ----
        best = None
        best_any = results[0]
        for r in results[:12]:
            tr = r['combo']
            I = render_model(modes_np, lab2k,
                             {'combo': [labels[t] for t in tr],
                              'params': r['params'],
                              'scale': r['scale']})
            pts_m = find_dark_points(I)
            mt = match_points(pts_e, pts_m)
            n_ok = abs(len(pts_m) - len(pts_e)) <= 1
            d_ok = mt['mean'] is not None and mt['mean'] < 12
            r['cores'] = len(pts_m)
            r['mean_nn'] = mt['mean']
            r['struct'] = 'PASS' if (n_ok and d_ok) else 'fail'
            if best is None and r['struct'] == 'PASS':
                best = r
        if best is None:
            best = best_any
            print('  [warn] no structurally verified solution in top-12; '
                  'using top-1 by W_avg')
        print(f'  [pick] {"+".join(labels[t] for t in best["combo"])} '
              f'wavg={best["wavg"]:.4f} struct={best["struct"]} '
              f'cores={best["cores"]}/{len(pts_e)}')

        # ---- metrics for the chosen solution ----
        tr = best['combo']
        s = best['scale']
        gs = global_stats_02(modes, Ie, tr, s if N == 2 else None)
        C2, (v0, v2) = c_s02(gs, best['params'], N)
        # rebuild 56 / 32 window pccs (56 from refine cache; 32 fresh)
        ws32, meta32, _ = build_window_stats(modes, Ie, tr, s, W=32)
        pccs56 = window_pccs(best['ws'], best['params'], N)
        meta56 = best['meta']
        pccs32 = window_pccs(ws32, best['params'], N)

        def wstats(pccs, meta):
            sg2 = [p for p, m in zip(pccs, meta) if m['sigma'] == 2]
            sg0 = [p for p, m in zip(pccs, meta) if m['sigma'] == 0]
            rho = float(np.mean([p < 0.5 for p in sg2]))
            cmin = float(np.min(sg2))
            return sg0, sg2, rho, cmin
        sg0_56, sg2_56, rho56, cmin56 = wstats(pccs56, meta56)
        sg0_32, sg2_32, rho32, cmin32 = wstats(pccs32, meta32)

        I_model = render_model(modes_np, lab2k,
                               {'combo': [labels[t] for t in tr],
                                'params': best['params'], 'scale': s})
        I2 = gaussian_blur(torch.tensor(I_model, dtype=torch.float32)
                           .unsqueeze(0), 2.0)[0].numpy()
        d_ang, (amp_e, amp_m) = angular_distance(Ie_np, I2)
        conc, loc = diff_localization(Ie_np, I2, best['mask'])

        gate_ok = (C2 >= T_C and v0 >= T0 and v2 >= T2
                   and rho56 <= RHO56_GATE and cmin56 >= CMIN56_GATE)
        warns = []
        if rho32 >= RHO_WARN:
            warns.append(f'rho_32px={rho32:.2f}')
        if cmin32 <= CMIN_WARN:
            warns.append(f'C_min_32px={cmin32:.2f}')

        # ---- outputs ----
        rawname = name
        raw = plt.imread(f'{IMG_DIR}/{name}.png')
        if raw.ndim == 3 and raw.shape[2] == 4:
            raw = raw[:, :, :3]

        # (1) 5-panel fit comparison
        fig, axs = plt.subplots(1, 5, figsize=(21.5, 4.4))
        axs[0].imshow(raw)
        axs[0].set_title('Experiment (original)')
        axs[1].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
        axs[1].set_title('Experiment (processed)')
        axs[2].imshow(I2, cmap='inferno', vmin=0, vmax=1)
        axs[2].set_title(f'{"+".join(labels[t] for t in tr)}\n'
                         f'C_s02={C2:.4f}  W_avg={best["wavg"]:.4f}')
        axs[3].imshow(np.abs(Ie_np - I2), cmap='viridis', vmin=0,
                      vmax=0.6)
        axs[3].set_title('|diff|  (fixed scale)')
        # window heatmap (56px, sigma=2)
        side = 4
        hm = np.full((side, side), np.nan)
        for p, m in zip(pccs56, meta56):
            if m['sigma'] == 2:
                hm[m['row'], m['col']] = p
        cmap = matplotlib.colormaps['RdYlGn'].copy()
        cmap.set_bad('0.7')
        im = axs[4].imshow(hm, cmap=cmap, vmin=0.3, vmax=1.0)
        for i in range(side):
            for j in range(side):
                if not np.isnan(hm[i, j]):
                    axs[4].text(j, i, f'{hm[i, j]:.2f}', ha='center',
                                va='center', fontsize=9)
        axs[4].set_title('window PCC map (56px, sigma=2)')
        for a in axs:
            a.axis('off')
        fig.suptitle(f'{name} - new-metric decomposition (N={N})',
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(DEST / f'{cn}_拟合对比.png', dpi=140,
                    bbox_inches='tight')
        plt.close(fig)

        # (2) old vs new model comparison
        pv_params = [float(x) for x in pv['params']]
        I_old = render_model(modes_np, lab2k,
                             {'combo': pv['combo'],
                              'params': pv_params, 'scale': pv['scale']})
        I_old2 = gaussian_blur(torch.tensor(I_old, dtype=torch.float32)
                               .unsqueeze(0), 2.0)[0].numpy()
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
        axs[0].imshow(I_old2, cmap='inferno', vmin=0, vmax=1)
        axs[0].set_title('previous result\n' + '+'.join(pv['combo']),
                         fontsize=10)
        axs[1].imshow(I2, cmap='inferno', vmin=0, vmax=1)
        axs[1].set_title('new-metric result\n'
                         + '+'.join(labels[t] for t in tr), fontsize=10)
        for a in axs:
            a.axis('off')
        fig.suptitle(f'{name} - old vs new fit (model sigma=2)',
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(DEST / f'{cn}_新旧拟合对比.png', dpi=140,
                    bbox_inches='tight')
        plt.close(fig)

        # (3) parameters & metrics md
        a1 = [float(np.sqrt(max(x, 0))) for x in best['params'][:N]]
        lines = [
            f'# {name} 最终分解结果（新指标管线）',
            '',
            f'- 模式数：{N} 模',
            f'- 拟合对比图：`{cn}_拟合对比.png`（原始 | 预处理 | 模型σ2 | '
            f'差异 | 窗口热力图）',
            f'- 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0',
        ]
        if s is not None:
            lines.append(f'- 缩放参数 s = {s:.4f}')
        lines += ['', '## 模式组成', '',
                  '| 模式 | 能量占比 | 振幅 |', '|---|---|---|']
        for i in range(N):
            lines.append(f'| {labels[tr[i]]} | '
                         f'{best["params"][i]*100:.2f}% | {a1[i]:.6f} |')
        lines += ['', f'归一化检验：Σ aᵢ² = '
                  f'{sum(x*x for x in a1):.6f}', '', '## 相对相位', '']
        for i in range(1, N):
            lines.append(f'- φ{i+1} = {best["params"][N+i-1]:.6f} rad '
                         f'= {np.degrees(best["params"][N+i-1]):.3f}°')
        lines += [
            '', '## 相关系数（新指标口径）', '',
            f'- **C_s02（主口径，σ0/σ2 平均）= {C2:.4f}**',
            f'- **W_avg（56px 窗口等权平均）= {best["wavg"]:.4f}**',
            f'- 分尺度：σ=0: {v0:.4f}，σ=2: {v2:.4f}',
            f'- [对照] 上轮 corr_avg 口径（σ0~6 平均）历史值见归档',
            '',
            '## 结构验收（暗点）', '',
            f'- 模型 {best["cores"]} 个 vs 实验 {len(pts_e)} 个'
            f'（{"通过" if best["struct"] == "PASS" else "未通过"}）',
            f'- 平均最近邻位置偏差：'
            f'{round(best["mean_nn"], 2) if best["mean_nn"] is not None else "—"} px',
        ]
        (DEST / f'{cn}_参数与指标.md').write_text('\n'.join(lines),
                                                 encoding='utf-8')

        # (4) acceptance sheet
        sheet = [
            f'# {name} 指标验收单（新指标管线，门槛仅标注不拦截）', '',
            '## 准入门槛（标注）', '',
            f'- C_s02 = {C2:.4f} （门槛 {T_C}）'
            f' {"✓" if C2 >= T_C else "✗"}',
            f'- PCC(σ=0) = {v0:.4f} （门槛 {T0}）'
            f' {"✓" if v0 >= T0 else "✗"}',
            f'- PCC(σ=2) = {v2:.4f} （门槛 {T2}）'
            f' {"✓" if v2 >= T2 else "✗"}',
            f'- rho_56px = {rho56:.3f} （门槛 ≤{RHO56_GATE}）'
            f' {"✓" if rho56 <= RHO56_GATE else "✗"}',
            f'- C_min_56px = {cmin56:.3f} （门槛 ≥{CMIN56_GATE}）'
            f' {"✓" if cmin56 >= CMIN56_GATE else "✗"}',
            '',
            f'**门槛判定：{"全部满足" if gate_ok else "存在未达标项（仅标注，不拦截）"}**',
            '',
            '## 局部一致性警示（32px 细尺度）', '',
            f'- 32px：最差窗 C_min = {cmin32:.3f}，失败窗占比 '
            f'rho = {rho32:.3f}',
            f'- 警示：{"无" if not warns else "【" + "；".join(warns) + "】建议人工复核"}',
            '',
            '## 角谱诊断（σ=2，m 阶幅值相对差）', '',
            f'- D_ang = {d_ang:.4f}（越低越好）', '',
            '| m 阶 | 实验 | 模型 | 相对差 |', '|---|---|---|---|',
        ]
        for m in range(1, 11):
            ae_, am_ = amp_e[m], amp_m[m]
            sheet.append(f'| m={m} | {ae_:.4f} | {am_:.4f} | '
                         f'{abs(ae_-am_)/(ae_+1e-12):.2f} |')
        sheet += ['', '## 差异定位（56px 块，支持区内）', '']
        if conc is not None:
            sheet.append(f'- 误差集中度（前 3 块占比）= {conc*100:.1f}%')
            for pct, pos in loc:
                sheet.append(f'- {pos} 区域：{pct}% 的误差能量')
        else:
            sheet.append('- （无有效块）')
        sheet += ['', '## 结论', '',
                  f'- 结构验收：{best["struct"]}'
                  f'（{"模型 %d 个暗点" % best["cores"]} vs 实验 {len(pts_e)} 个）',
                  f'- 门槛：{"全部满足" if gate_ok else "存在未达标项"}；'
                  f'警示：{"无" if not warns else "；".join(warns)}']
        (DEST / f'{cn}_指标验收单.md').write_text('\n'.join(sheet),
                                                 encoding='utf-8')

        summary.append({
            'cn': cn, 'name': name, 'N': N,
            'combo_new': '+'.join(labels[t] for t in tr),
            'combo_old': '+'.join(pv['combo']),
            'C_s02': C2, 'W_avg': best['wavg'],
            'rho56': rho56, 'cmin56': cmin56,
            'cores': best['cores'], 'cores_exp': len(pts_e),
            'mean_nn': best['mean_nn'], 'struct': best['struct'],
            'gate': gate_ok, 'warns': warns,
        })
        print(f'  [metrics] C_s02={C2:.4f} W_avg={best["wavg"]:.4f} '
              f'rho56={rho56:.2f} cmin56={cmin56:.2f} gate={gate_ok} '
              f'warns={warns}  ({time.time()-t_img:.0f}s)')

    # ---- summary table ----
    lines = [
        '# 八图最终分解结果汇总（新指标管线）', '',
        '> 主口径 C_s02 = [PCC(σ=0)+PCC(σ=2)]/2；W_avg = 56px 窗口等权平均；',
        '> 门槛仅标注不拦截；结构验收为最终判据；相位约定 φ₁=0。', '',
        '| 图 | 模数 | 本轮组合 | C_s02 | W_avg | 结构验收 | 门槛 | 警示 |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for r in summary:
        lines.append(
            f'| {r["cn"]} | {r["N"]} | {r["combo_new"]} | '
            f'**{r["C_s02"]:.4f}** | **{r["W_avg"]:.4f}** | '
            f'{r["struct"]}（{r["cores"]}vs{r["cores_exp"]}） | '
            f'{"✓" if r["gate"] else "未达标"} | '
            f'{"；".join(r["warns"]) if r["warns"] else "无"} |')
    lines += ['', '## 与上轮对比', '',
              '| 图 | 上轮组合 | 本轮组合 | 组合变化 |', '|---|---|---|---|']
    for r in summary:
        chg = '不变' if r['combo_new'] == r['combo_old'] else '**已变化**'
        lines.append(f'| {r["cn"]} | {r["combo_old"]} | '
                     f'{r["combo_new"]} | {chg} |')
    (DEST / '汇总表.md').write_text('\n'.join(lines), encoding='utf-8')

    (ROOT / 'rerun_new_metrics_results.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False,
                   default=str), encoding='utf-8')
    print(f'\n[done] outputs in {DEST}')


if __name__ == '__main__':
    main()

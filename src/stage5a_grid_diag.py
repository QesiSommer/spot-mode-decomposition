# -*- coding: utf-8 -*-
"""
stage5a_grid_diag.py -- HG 主导搜索 第1步诊断：网格 vs 环形 定量验证
====================================================================

D1 外轮廓方度: 角向 Fourier 谱 (m=4 相对 m=0) — 实验 vs 当前四模 TOP-1
   (sigma=2.5, gamma=0.85)，多个阈值 {0.3, 0.5}
D2 核排布几何: 实验 8 核 -> 平行四边形点阵拟合 (grid RMS) vs
   圆环拟合 (1 环 / 2 环 RMS)，自动判定"网格 or 环形"
D3 LG 能量敏感性: 固定其余三模(按比例重归一化)，LG(2,0) 能量
   35% -> 0% 序列渲染，每档: nblk / depth / 鞍部 / gpcc / 方度

输出: 归档_八图分解/8奇点1_HG主导搜索/
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
from scipy.ndimage import label, map_coordinates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from stage0_diagnosis import core_depth, saddle_isolation
from calibrate_metrics import render_model

OUT = ROOT / '归档_八图分解' / '8奇点1_HG主导搜索'
OUT.mkdir(parents=True, exist_ok=True)
IMGDIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '8Singularities(1)'
SIG_DISP, GAMMA = 2.5, 0.85


# ---------------------------------------------------------------------------
# shared render / metric helpers
# ---------------------------------------------------------------------------

def gpcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def blocks(I, thr=0.10, min_size=20, max_size=3000):
    m = I < thr * I.max()
    lab, n = label(m)
    return sorted((int((lab == k).sum()) for k in range(1, n + 1)
                   if min_size <= int((lab == k).sum()) <= max_size),
                  reverse=True)


def grender(I0, g=GAMMA, sg=SIG_DISP):
    Id = gaussian_blur(torch.tensor(I0, dtype=torch.float32)
                       .unsqueeze(0), float(sg))[0].numpy()
    Ig = np.clip(Id, 0, 1) ** g
    return Ig / max(Ig.max(), 1e-12)


# ---------------------------------------------------------------------------
# D1: outer-contour angular Fourier squareness
# ---------------------------------------------------------------------------

def outer_rcurve(I, thr_rel, n_theta=360, n_r=400):
    """r(theta) of the OUTER boundary of the whole bright region.

    Uses the full mask (no single connected component): for each ray from
    the intensity centroid, the farthest bright sample defines the outer
    contour.  Bright lobes separated by dark cores still contribute."""
    m = I > thr_rel * I.max()
    if not m.any():
        return None
    mm = m.astype(np.float64)
    ys, xs = np.where(m)
    cy, cx = float(ys.mean()), float(xs.mean())
    rmax = 0.55 * np.hypot(*I.shape)
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rs = np.linspace(0, rmax, n_r)
    rr = np.zeros(n_theta)
    for i, t in enumerate(thetas):
        yy = cy + rs * np.sin(t)
        xx = cx + rs * np.cos(t)
        v = map_coordinates(mm, [yy, xx], order=1, mode='constant')
        inside = np.where(v > 0.5)[0]
        rr[i] = rs[inside[-1]] if len(inside) else 0.0
    rr = np.where(rr == 0.0, np.nan, rr)
    if np.isnan(rr).all():
        return None
    rr = np.where(np.isnan(rr), np.nanmean(rr), rr)
    f = np.fft.rfft(rr - rr.mean())
    amp = np.abs(f) * 2.0 / n_theta            # amp[m] for m>=0
    return {'theta': thetas, 'r': rr, 'amp': amp, 'a0': float(rr.mean()),
            'center': (cy, cx), 'mask': m}


def squareness_metrics(I, thr_rel):
    out = outer_rcurve(I, thr_rel)
    if out is None:
        return None
    a0 = out['a0']
    amp = out['amp']
    return {'thr': thr_rel,
            'A2_A0': float(amp[2] / a0), 'A4_A0': float(amp[4] / a0),
            'A6_A0': float(amp[6] / a0) if len(amp) > 6 else None,
            'S4_ratio': float(amp[4] / (amp[2] + amp[6] + 1e-12)),
            'r_mean': a0, 'r_std': float(out['r'].std()),
            '_curve': out}


def draw_contour(ax, I, out):
    ax.imshow(I, cmap='inferno', vmin=0, vmax=1)
    yy = out['center'][0] + out['r'] * np.sin(out['theta'])
    xx = out['center'][1] + out['r'] * np.cos(out['theta'])
    ax.plot(xx, yy, '-', color='cyan', lw=1.2)
    ax.plot(out['center'][1], out['center'][0], '+', color='cyan', ms=10,
            mew=2)


# ---------------------------------------------------------------------------
# D2: core-arrangement geometry -- grid (2D lattice) vs rings
# ---------------------------------------------------------------------------

def kmeans1d(vals, k, iters=100):
    cs = np.quantile(vals, np.linspace(0.3, 0.7, k)).astype(float)
    a = np.zeros(len(vals), dtype=int)
    for _ in range(iters):
        a = np.argmin(np.abs(vals[:, None] - cs[None, :]), axis=1)
        ncs = np.array([vals[a == j].mean() if (a == j).any() else cs[j]
                        for j in range(k)])
        if np.allclose(ncs, cs):
            break
        cs = ncs
    return a, cs


def circle_fit2d(P):
    """Kasa algebraic circle fit. P: (N,2) as (y,x). Returns (cy,cx,R), rms."""
    y, x = P[:, 0], P[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    R = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    r = np.hypot(x - cx, y - cy)
    return (cy, cx, R), float(np.sqrt(((r - R) ** 2).mean()))


def ring_fit(P):
    """1-ring and 2-ring (radial kmeans) models. P: (N,2) (y,x)."""
    c0 = P.mean(0)
    r = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
    res = {}
    # single ring, fitted center
    (cy, cx, R), rms1 = circle_fit2d(P)
    res['ring1'] = {'center': [cy, cx], 'R': R, 'rms': rms1,
                    'radii_to_fit_center':
                    np.hypot(P[:, 0] - cy, P[:, 1] - cx).tolist()}
    # two rings: radial kmeans assignment, per-ring circle fit
    a, cs = kmeans1d(r, 2)
    rings = []
    dev = []
    for j in range(2):
        sel = a == j
        if sel.sum() >= 3:
            (cy2, cx2, R2), rmsj = circle_fit2d(P[sel])
            rings.append({'center': [cy2, cx2], 'R': R2,
                          'n': int(sel.sum()),
                          'idx': np.where(sel)[0].tolist(),
                          'rms': rmsj})
            cj = np.array([cy2, cx2])
            for i in np.where(sel)[0]:
                dev.append(np.hypot(*(P[i] - cj)) - R2)
    rms2 = float(np.sqrt(np.mean(np.square(dev)))) if dev else None
    # also center-fixed variants (image center = optical axis guess)
    ctr = np.array([(224 - 1) / 2.0, (224 - 1) / 2.0])
    r_ctr = np.hypot(P[:, 0] - ctr[0], P[:, 1] - ctr[1])
    rms_ctr1 = float(np.sqrt(np.var(r_ctr)))
    res['ring2'] = {'rings': rings, 'rms': rms2,
                    'assign': a.tolist(), 'kmeans_centers': cs.tolist()}
    res['center_fixed'] = {'rms1': rms_ctr1,
                           'radii': r_ctr.tolist()}
    # concentric two-ring model: shared center, two radii (4 params)
    from scipy.optimize import minimize

    def conc_obj(v):
        cy0, cx0, R1, R2 = v
        rr = np.hypot(P[:, 0] - cy0, P[:, 1] - cx0)
        d = np.minimum(np.abs(rr - R1), np.abs(rr - R2))
        return float((d ** 2).mean())

    x0 = np.array([P[:, 0].mean(), P[:, 1].mean(),
                   np.quantile(r, 0.3), np.quantile(r, 0.7)])
    r_conc = minimize(conc_obj, x0, method='Nelder-Mead',
                      options={'xatol': 1e-4, 'fatol': 1e-8,
                               'maxiter': 4000})
    res['ring2_concentric'] = {
        'center': [float(r_conc.x[0]), float(r_conc.x[1])],
        'R': sorted([float(r_conc.x[2]), float(r_conc.x[3])]),
        'rms': float(np.sqrt(r_conc.fun))}
    return res


def grid_fit(P):
    """Parallelogram-lattice fit: p = O + m*b1 + n*b2, m,n integers.

    Global search: enumerate ALL pairwise difference vectors as candidate
    basis pairs (non-parallel), run integer-assignment / least-squares
    iterations for each, keep the best final RMS.  This avoids the local
    optimum of a naive bootstrapped basis."""
    N = len(P)
    diffs = []
    for i in range(N):
        for j in range(N):
            if i != j:
                diffs.append(P[j] - P[i])
    best = None
    L2 = [(a, b, diffs[a], diffs[b])
          for a in range(len(diffs)) for b in range(a + 1, len(diffs))]
    for (ia, ib, d1, d2) in L2:
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        if n1 < 4 or n2 < 4:
            continue
        cosang = abs(d1 @ d2) / (n1 * n2 + 1e-12)
        if cosang > 0.97 or cosang < 1e-9:
            continue
        B = np.stack([d1, d2], axis=1)
        O = P.mean(0)
        MNi = None
        ok = True
        for it in range(8):
            try:
                Binv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                ok = False
                break
            MN = (P - O) @ Binv.T
            MNi_new = np.round(MN)
            if MNi is not None and np.array_equal(MNi_new, MNi):
                break
            MNi = MNi_new
            dP = P - P.mean(0)
            dM = MNi - MNi.mean(0)
            Bt, res_, rank_, sv_ = np.linalg.lstsq(dM, dP, rcond=None)
            if rank_ < 2:
                ok = False
                break
            B = Bt.T
            O = P.mean(0) - MNi.mean(0) @ B.T
        if not ok:
            continue
        resid = P - (O + MNi @ B.T)
        rms = float(np.sqrt((resid ** 2).sum(1).mean()))
        span = (int(np.ptp(MNi[:, 0])) + 1) * (int(np.ptp(MNi[:, 1])) + 1)
        if span < 6:      # require a genuinely 2D region (not a line)
            continue
        if best is None or rms < best['rms']:
            best = {'O': O.tolist(), 'b1': B[:, 0].tolist(),
                    'b2': B[:, 1].tolist(), 'MN': MNi.tolist(),
                    'rms': rms,
                    'resid_px': np.hypot(resid[:, 0],
                                         resid[:, 1]).tolist(),
                    'frac_dev': (MN - MNi).tolist(),
                    'span': span,
                    'n_evals': len(L2)}
    return best


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()

    with open(ROOT / '归档_八图分解' / '8奇点1_四模空间'
              / 'scan_4mode.json', encoding='utf-8') as f:
        scan = json.load(f)
    cur = scan['top3'][0]
    combo = cur['combo'].split('+')
    params = [float(x) for x in cur['params']]
    print('[setup] current 4-mode TOP-1:', '+'.join(combo))
    print('        energies:', [round(x, 4) for x in params[:4]],
          'phases:', [round(x, 4) for x in params[4:]])

    Ie = load_experiment(f'{IMGDIR}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    I0_cur = render_model(modes_np, lab2k,
                          {'combo': combo, 'params': params, 'scale': None})
    Ig_cur = grender(I0_cur)
    print(f'[setup] experiment cores = {len(pts_e)}, '
          f'model cores(linear) = {len(find_dark_points(I0_cur))}')

    report = {'current': {'combo': combo, 'params': params,
                          'sigma': SIG_DISP, 'gamma': GAMMA}}

    # ---------- D1: outline squareness ----------
    print('\n[D1] outer-contour angular Fourier')
    d1 = {}
    for tag, I in [('experiment', Ie), ('model_g085', Ig_cur)]:
        row = {}
        for thr in (0.3, 0.5):
            sm = squareness_metrics(I, thr)
            row[str(thr)] = {k: v for k, v in sm.items() if k != '_curve'}
            if thr == 0.5:
                row['_curve'] = sm['_curve']
            print(f'  {tag:12s} thr={thr}: A2/A0={sm["A2_A0"]:.4f} '
                  f'A4/A0={sm["A4_A0"]:.4f} A6/A0={sm["A6_A0"]:.4f}')
        d1[tag] = row
    report['D1'] = {k: {kk: vv for kk, vv in v.items() if kk != '_curve'}
                    for k, v in d1.items()}

    # ---------- D2: core arrangement ----------
    print('\n[D2] core geometry: grid vs rings')
    P = np.array([(y, x) for y, x, _ in pts_e], dtype=np.float64)
    gf = grid_fit(P)
    rf = ring_fit(P)
    print(f'  grid fit : b1={np.round(gf["b1"], 2)} '
          f'b2={np.round(gf["b2"], 2)}  RMS={gf["rms"]:.3f}px '
          f'span={gf["span"]} (per-point {np.round(gf["resid_px"], 2)})')
    print(f'  ring1    : R={rf["ring1"]["R"]:.2f}  '
          f'RMS={rf["ring1"]["rms"]:.3f}px')
    print(f'  ring2    : R={[round(r["R"],2) for r in rf["ring2"]["rings"]]} '
          f'RMS={rf["ring2"]["rms"]:.3f}px  '
          f'assign={rf["ring2"]["assign"]}')
    rc = rf['ring2_concentric']
    print(f'  ring2-concentric: center={[round(x,1) for x in rc["center"]]} '
          f'R={[round(x,2) for x in rc["R"]]}  RMS={rc["rms"]:.3f}px')
    print(f'  center-fixed rings RMS={rf["center_fixed"]["rms1"]:.3f}px')
    print(f'  >> VERDICT: grid RMS={gf["rms"]:.2f}px vs '
          f'concentric-ring RMS={rc["rms"]:.2f}px')
    report['D2'] = {'grid': gf, 'ring1': rf['ring1'],
                    'ring2': rf['ring2'],
                    'ring2_concentric': rc,
                    'center_fixed': rf['center_fixed'],
                    'cores_yx': P.tolist()}

    # ---------- D3: LG energy sensitivity ----------
    print('\n[D3] LG(2,0) energy sensitivity sequence')
    u_other = np.array(params[1:4], dtype=np.float64)
    ph_other = list(params[4:])
    W_SEQ = [0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]
    seq = []
    for w_lg in W_SEQ:
        sc = (1.0 - w_lg) / u_other.sum()
        p_w = [w_lg] + list(u_other * sc) + ph_other
        I0 = render_model(modes_np, lab2k,
                          {'combo': combo, 'params': p_w, 'scale': None})
        Ig = grender(I0)
        pts_m0 = find_dark_points(I0)
        mt = match_points(pts_e, pts_m0)
        pts_m = find_dark_points(Ig)
        dep = core_depth(Ig, pts_m)
        sad = [s for _, s in saddle_isolation(Ig, pts_m)]
        blk = blocks(Ig)
        sm = squareness_metrics(Ig, 0.3)
        entry = {
            'w_lg': w_lg,
            'energies': [round(float(x), 5) for x in p_w[:4]],
            'nblk': len(blk), 'blk_sizes': blk[:10],
            'depth': float(np.nanmean(dep)) if len(dep) else None,
            'sad_mean': float(np.mean(sad)) if sad else None,
            'sad_min': float(np.min(sad)) if sad else None,
            'gpcc': gpcc(Ie, Ig),
            'A4_A0': sm['A4_A0'] if sm else None,
            'cores0': len(pts_m0), 'cores_g': len(pts_m),
            'mean_nn': mt['mean'],
            'struct': 'PASS' if (abs(len(pts_m0) - len(pts_e)) <= 1
                                 and mt['mean'] is not None
                                 and mt['mean'] < 12) else 'fail',
        }
        seq.append(entry)
        print(f'  w_LG={w_lg:.2f}  nblk={entry["nblk"]} '
              f'depth={entry["depth"] if entry["depth"] is None else round(entry["depth"],3)} '
              f'sad={entry["sad_mean"] if entry["sad_mean"] is None else round(entry["sad_mean"],3)} '
              f'gpcc={entry["gpcc"]:.4f} A4/A0={entry["A4_A0"]:.4f} '
              f'cores={entry["cores0"]}/{entry["cores_g"]} '
              f'NN={None if mt["mean"] is None else round(mt["mean"],1)} '
              f'{entry["struct"]}')
        entry['_I0'] = I0
        entry['_Ig'] = Ig
    report['D3'] = [{k: v for k, v in e.items() if not k.startswith('_')}
                    for e in seq]

    # ================= figures =================
    # ---- fig 1: D1 + D2 ----
    fig, axs = plt.subplots(2, 4, figsize=(21, 10.4))
    draw_contour(axs[0, 0], Ie, d1['experiment']['_curve'])
    axs[0, 0].set_title('D1 experiment: outer contour (thr=0.5)',
                        fontsize=10)
    draw_contour(axs[0, 1], Ig_cur, d1['model_g085']['_curve'])
    axs[0, 1].set_title('D1 model (4mode TOP-1, g=0.85)', fontsize=10)
    ce = d1['experiment']['_curve']
    cm = d1['model_g085']['_curve']
    ax = axs[0, 2]
    ax.plot(np.degrees(ce['theta']), ce['r'], '-', lw=1.6,
            label=f'experiment (r={ce["a0"]:.1f})')
    ax.plot(np.degrees(cm['theta']), cm['r'], '-', lw=1.6,
            label=f'model (r={cm["a0"]:.1f})')
    ax.set_xlabel('theta (deg)')
    ax.set_ylabel('outer radius (px)')
    ax.set_title('D1 r(theta) profile')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ms = np.arange(0, 9)
    ae = ce['amp'][:9] / ce['a0']
    am = cm['amp'][:9] / cm['a0']
    ax = axs[0, 3]
    ax.bar(ms - 0.18, ae, 0.36, label='experiment')
    ax.bar(ms + 0.18, am, 0.36, label='model')
    ax.set_xlabel('angular order m')
    ax.set_ylabel('A_m / A_0')
    ax.set_title('D1 angular spectrum of outline')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # D2: grid overlay
    ax = axs[1, 0]
    ax.imshow(Ie, cmap='inferno', vmin=0, vmax=1)
    O, b1, b2 = np.array(gf['O']), np.array(gf['b1']), np.array(gf['b2'])
    MNi = np.array(gf['MN'])
    gridP = O + MNi @ np.stack([b1, b2], 1).T
    for i in range(len(P)):
        ax.plot([P[i, 1], gridP[i, 1]], [P[i, 0], gridP[i, 0]], '-',
                color='lime', lw=0.8)
        ax.plot(gridP[i, 1], gridP[i, 0], '+', color='lime', ms=11,
                mew=1.6)
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if np.abs(MNi[i] - MNi[j]).sum() == 1:
                ax.plot([P[i, 1], P[j, 1]], [P[i, 0], P[j, 0]], '-',
                        color='cyan', lw=0.9, alpha=0.8)
    ax.set_title(f'D2 grid fit (RMS={gf["rms"]:.2f}px): '
                 f'same row/col pairs linked', fontsize=10)
    ax = axs[1, 1]
    ax.imshow(Ie, cmap='inferno', vmin=0, vmax=1)
    for rg in rf['ring2']['rings']:
        cy, cx = rg['center']
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cx + rg['R'] * np.cos(th), cy + rg['R'] * np.sin(th),
                '--', color='cyan', lw=1.1)
        ax.plot(cx, cy, '+', color='cyan', ms=10, mew=2)
    cyc, cxc = rc['center']
    th = np.linspace(0, 2 * np.pi, 200)
    for R in rc['R']:
        ax.plot(cxc + R * np.cos(th), cyc + R * np.sin(th), '-',
                color='lime', lw=1.0, alpha=0.9)
    ax.plot(cxc, cyc, 'x', color='lime', ms=11, mew=2)
    ax.set_title(f'D2 ring fits: free(cyan) RMS={rf["ring2"]["rms"]:.2f}px '
                 f'| concentric(lime) RMS={rc["rms"]:.2f}px', fontsize=10)
    ax = axs[1, 2]
    for i in range(len(P)):
        ax.plot(MNi[i, 1], MNi[i, 0], 'o', ms=9, mfc='none', mec='tab:blue')
        ax.annotate(f'{i}', (MNi[i, 1], MNi[i, 0]), fontsize=9,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('n (lattice coord, b2 axis)')
    ax.set_ylabel('m (lattice coord, b1 axis)')
    ax.set_xticks(range(int(np.floor(MNi[:, 1].min())) - 1,
                        int(np.ceil(MNi[:, 1].max())) + 2))
    ax.set_yticks(range(int(np.floor(MNi[:, 0].min())) - 1,
                        int(np.ceil(MNi[:, 0].max())) + 2))
    ax.set_title('D2 integer coords (perfect lattice => integer)',
                 fontsize=10)
    ax.grid(alpha=0.4)
    ax = axs[1, 3]
    r_fit = np.hypot(P[:, 0] - rc['center'][0],
                     P[:, 1] - rc['center'][1])
    ax.plot(np.sort(r_fit), 'o-', label='radius (conc. center)')
    for R in rc['R']:
        ax.axhline(R, color='g', ls=':', lw=1, label=f'conc R={R:.1f}')
    ax.axhline(rf['ring1']['R'], color='r', ls='--', lw=1,
               label=f'ring1 R={rf["ring1"]["R"]:.1f}')
    ax.set_xlabel('core index (sorted)')
    ax.set_ylabel('radius (px)')
    ax.set_title('D2 radial distribution', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for a in axs.ravel():
        a.axis('off') if a.get_images() else None
    fig.suptitle('8Singularities(1) grid-vs-ring diagnosis (D1/D2)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / '诊断_网格vs环形.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- fig 2: D3 sequence ----
    fig, axs = plt.subplots(2, 4, figsize=(22, 10.8))
    for k, e in enumerate(seq):
        ax = axs[k // 4, k % 4]
        ax.imshow(e['_Ig'], cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f'LG energy {e["w_lg"]*100:.0f}%  '
                     f'(HG trio {(1-e["w_lg"])*100:.0f}%)\n'
                     f'nblk={e["nblk"]}  cores={e["cores0"]}  '
                     f'NN={None if e["mean_nn"] is None else round(e["mean_nn"],1)}  '
                     f'gpcc={e["gpcc"]:.3f}\n'
                     f'A4/A0={e["A4_A0"]:.4f}  sad={None if e["sad_mean"] is None else round(e["sad_mean"],3)}',
                     fontsize=8.5)
    for a in axs.ravel():
        a.axis('off')
    fig.suptitle('D3 LG(2,0) energy ramp 35% -> 0% (other trio scaled) '
                 'sigma=2.5 gamma=0.85', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'D3_LG能量序列.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- fig 3: D3 curves ----
    w = [e['w_lg'] for e in seq]
    fig, axs = plt.subplots(1, 5, figsize=(22, 4.2))
    axes_def = [('nblk', 'dark blocks (thr 10%)', None),
                ('A4_A0', 'outline A4/A0 (squareness)', None),
                ('depth', 'mean core depth', None),
                ('sad_mean', 'saddle mean', None),
                ('gpcc', 'appearance PCC', None)]
    for ax, (key, title, _) in zip(axs, axes_def):
        vals = [e[key] for e in seq]
        ax.plot(w, vals, 'o-')
        ax.set_xlabel('LG(2,0) energy')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(0, color='gray', lw=0.5)
    if True:
        # experiment references
        axs[0].axhline(8, color='r', ls='--', lw=1, label='exp=8')
        axs[0].legend(fontsize=8)
        axs[3].axhline(0.4568, color='r', ls='--', lw=1, label='exp=0.457')
        axs[3].legend(fontsize=8)
        axs[2].axhline(0.8724, color='r', ls='--', lw=1, label='exp=0.872')
        axs[2].legend(fontsize=8)
        sm_e = squareness_metrics(Ie, 0.3)
        axs[1].axhline(sm_e['A4_A0'], color='r', ls='--', lw=1,
                       label=f'exp={sm_e["A4_A0"]:.3f}')
        axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'D3_指标曲线.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- fig 4: hypothesis preview: exp vs LG-0 (pure-HG-trio) ----
    e0 = seq[-1]
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.0))
    axs[0].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
    axs[0].set_title('experiment')
    axs[1].imshow(seq[0]['_Ig'], cmap='inferno', vmin=0, vmax=1)
    axs[1].set_title('current (LG 35%)')
    axs[2].imshow(e0['_Ig'], cmap='inferno', vmin=0, vmax=1)
    axs[2].set_title('LG -> 0 (preview)')
    axs[3].imshow(np.abs(Ie - e0['_Ig']), cmap='viridis', vmin=0, vmax=0.6)
    axs[3].set_title('|exp - LG0| (fixed scale)')
    for a in axs:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / 'D3_极限预览.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    (OUT / 'diag_grid_ring.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

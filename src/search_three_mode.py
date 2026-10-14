# -*- coding: utf-8 -*-
"""
search_three_mode.py -- 三模解析搜索 (Three-mode analytical search)
====================================================================

E = a1*F1 + a2*F2*exp(i*phi2) + a3*F3*exp(i*phi3),  a1^2+a2^2+a3^2=1

9 real base fields: B_k=|F_k|^2, C_jk=Re[F_j F_k*], S_jk=Im[F_j F_k*]
I = c . G with 9 coefficient components derived from
(a1^2, a2^2, a3^2, phi2, phi3).  Pearson corr is a quadratic form in c:

  corr = [c.E - (c.m) * mu_e] / sqrt([c^T Gram c - (c.m)^2] * var_e)

Layers:
  L1  full-space coarse scan (366,145 triples, 64x64 downsampled,
      simplex-energy grid x phase grid), top-200 kept;
  L2  refinement of top-200 on the full 224 image (alternating
      energy/phase 2D grids, then small joint 4D grid);
  L3  structural verification (dark-core detection) of top-30.

Usage:
  python search_three_mode.py --synth          # synthetic self-test
  python search_three_mode.py --image <path>   # single image
  python search_three_mode.py                  # all configured images
"""

import argparse
import json
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points

IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
FAILED_IMAGES = ['10Singularities', '11Singularities', '4Singularities',
                 '8Singularities(1)', '9Singularities']
OUT_DEFAULT = 'three_mode_search'


def parse_args():
    p = argparse.ArgumentParser(
        description='Three-mode analytical search',
        epilog='任何具有多个亮斑/暗核结构的相干光斑图都可作为输入；'
               '用 --img_dir 指向你的图像目录，用 --image 指定单张图。')
    p.add_argument('--image', type=str, default=None,
                   help='单张图像的路径或文件名（不含或含扩展名均可）')
    p.add_argument('--images', type=str, default=None,
                   help='逗号分隔的图像名列表（不含扩展名），'
                        '默认使用项目自带的演示图集')
    p.add_argument('--img_dir', type=str, default=None,
                   help='图像所在目录；默认指向本项目演示数据目录')
    p.add_argument('--out_dir', type=str, default=OUT_DEFAULT)
    p.add_argument('--max_order', type=int, default=10)
    p.add_argument('--small', type=int, default=64,
                   help='downsampled size for layer-1 coarse scan')
    p.add_argument('--n_energy', type=int, default=16,
                   help='energy simplex steps (N -> (N+1)(N+2)/2 pts)')
    p.add_argument('--n_phase', type=int, default=20,
                   help='phase grid points per axis (deg = 360/n)')
    p.add_argument('--top_coarse', type=int, default=200)
    p.add_argument('--top_struct', type=int, default=30)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--cmap', type=str, default='nipy_spectral',
                   help='伪彩配色；灰度图请传 none')
    p.add_argument('--preprocess', type=str, default='isot',
                   help='预处理模式（isot = 无量纲拓扑对齐）')
    p.add_argument('--synth', action='store_true')
    p.add_argument('--quick', action='store_true',
                   help='limit layer-1 to first 2000 triples (debug)')
    return p.parse_args()


def resolve_image_path(name_or_path, img_dir=None):
    """把 --image / --images 的取值解析成实际图像路径。

    接受：绝对路径 / 相对路径 / 纯文件名（含或不含扩展名）。
    未指定 img_dir 时回落到本项目的演示数据目录。
    """
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    p = Path(name_or_path)
    if p.is_file():
        return p
    base = Path(img_dir) if img_dir else Path(IMG_DIR)
    direct = base / p.name
    if direct.is_file():
        return direct
    if direct.suffix.lower() not in IMG_EXTS:
        for ext in IMG_EXTS:
            cand = direct.with_suffix(ext)
            if cand.is_file():
                return cand
    return None


# ---------------------------------------------------------------------------
# grids and coefficient tables
# ---------------------------------------------------------------------------

def energy_simplex(N):
    """Simplex grid for (a1^2, a2^2, a3^2), step 1/N. (P,3) tensor."""
    pts = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            pts.append((i / N, j / N, k / N))
    return torch.tensor(pts, dtype=torch.float32)


def build_c_table(e_pts, n_phase):
    """Coefficient table c (P,9) + parameter table (P,5).

    c layout: [B1,B2,B3, C12,S12, C13,S13, C23,S23]
    params:   [u,v,w, phi2, phi3]  (u=a1^2, v=a2^2, w=a3^2)
    """
    phis = torch.linspace(0, 2 * np.pi, n_phase + 1)[:-1]
    P2 = phis.numel()
    rows_c = []
    rows_p = []
    for (u, v, w) in e_pts.tolist():
        a1, a2, a3 = np.sqrt(u), np.sqrt(v), np.sqrt(w)
        for p2 in phis.tolist():
            for p3 in phis.tolist():
                rows_c.append([
                    u, v, w,
                    2 * a1 * a2 * np.cos(p2), 2 * a1 * a2 * np.sin(p2),
                    2 * a1 * a3 * np.cos(p3), 2 * a1 * a3 * np.sin(p3),
                    2 * a2 * a3 * np.cos(p2 - p3),
                    -2 * a2 * a3 * np.sin(p2 - p3),
                ])
                rows_p.append([u, v, w, p2, p3])
    return (torch.tensor(rows_c, dtype=torch.float32),
            torch.tensor(rows_p, dtype=torch.float32))


def build_base_fields(F1, F2, F3):
    """9 base fields from three complex mode fields. (B,9,H,W)."""
    B1 = (F1.real ** 2 + F1.imag ** 2)
    B2 = (F2.real ** 2 + F2.imag ** 2)
    B3 = (F3.real ** 2 + F3.imag ** 2)
    X12 = F1 * F2.conj()
    X13 = F1 * F3.conj()
    X23 = F2 * F3.conj()
    fields = torch.stack([B1, B2, B3,
                          X12.real, X12.imag,
                          X13.real, X13.imag,
                          X23.real, X23.imag], dim=1)
    return fields


def eval_c_grid(c_tab, m, E, Gram, mu_e, var_e):
    """corr for coefficient rows against batch statistics.

    c_tab: (P,9); m/E: (B,9); Gram: (B,9,9); returns (B,P).
    """
    c_tab = c_tab.to(m.device)
    cm = torch.einsum('pk,bk->bp', c_tab, m)
    cE = torch.einsum('pk,bk->bp', c_tab, E)
    cG = torch.einsum('pk,bkl,pl->bp', c_tab, Gram, c_tab)
    cov = cE - cm * mu_e
    vart = cG - cm * cm
    # validity mask: same relative-variance guard as the two-mode engine
    valid = (var_e > 1e-12) & (vart > 1e-6 * (cG.abs() + cm * cm + 1e-12))
    corr = torch.where(
        valid,
        cov / torch.sqrt(vart.clamp_min(1e-12) * var_e),
        torch.full_like(cov, -1.0))
    return corr.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Layer 1: coarse scan
# ---------------------------------------------------------------------------

def coarse_scan(np_modes_small, Ie_small, triples, c_tab, batch, dev):
    """Returns best_corr (T,), best_idx (T,) over the coarse grid."""
    Ie = Ie_small.to(dev)
    mu_e = Ie.mean()
    var_e = Ie.var(unbiased=False)
    T = triples.shape[0]
    best_corr = torch.empty(T, device=dev)
    best_idx = torch.empty(T, dtype=torch.long, device=dev)

    for b0 in range(0, T, batch):
        tr = triples[b0:b0 + batch].to(dev)
        F1 = np_modes_small[tr[:, 0]]
        F2 = np_modes_small[tr[:, 1]]
        F3 = np_modes_small[tr[:, 2]]
        G9 = build_base_fields(F1, F2, F3)  # (B,9,H,W)
        B = G9.shape[0]
        Gf = G9.reshape(B, 9, -1)
        m = Gf.mean(dim=2)                                # (B,9)
        E = torch.einsum('bki,i->bk', Gf, Ie.reshape(-1)) / Gf.shape[2]
        Gram = torch.einsum('bki,bli->bkl', Gf, Gf) / Gf.shape[2]
        corr = eval_c_grid(c_tab.to(dev), m, E, Gram, mu_e, var_e)
        mx, mi = corr.max(dim=1)
        best_corr[b0:b0 + B] = mx
        best_idx[b0:b0 + B] = mi
    return best_corr, best_idx


# ---------------------------------------------------------------------------
# Layer 2: refinement on full resolution
# ---------------------------------------------------------------------------

def stats_for_triples(modes, Ie, triples):
    """Full-resolution statistics for a small set of triples."""
    dev = modes.device
    tr = triples.to(dev)
    F1 = modes[tr[:, 0]]
    F2 = modes[tr[:, 1]]
    F3 = modes[tr[:, 2]]
    G9 = build_base_fields(F1, F2, F3)
    B = G9.shape[0]
    Gf = G9.reshape(B, 9, -1)
    m = Gf.mean(dim=2)
    E = torch.einsum('bki,i->bk', Gf, Ie.reshape(-1)) / Gf.shape[2]
    Gram = torch.einsum('bki,bli->bkl', Gf, Gf) / Gf.shape[2]
    return m, E, Gram


def refine_triples(modes, Ie, triples, params0, n_energy=20, n_phase=21):
    """Alternating energy/phase 2D grids + small joint 4D grid."""
    dev = modes.device
    Ie = Ie.to(dev)
    mu_e = Ie.mean()
    var_e = Ie.var(unbiased=False)
    m, E, Gram = stats_for_triples(modes, Ie, triples)

    T = triples.shape[0]
    out = []
    for t in range(T):
        u0, v0, w0, p20, p30 = [float(x) for x in params0[t]]
        mt, Et, Gt = m[t:t + 1], E[t:t + 1], Gram[t:t + 1]

        def best_over(u_pts, v_pts, ph2_pts, ph3_pts):
            e_np = np.array([(u, v) for u in u_pts for v in v_pts
                             if u + v < 1.0], dtype=np.float64)
            ph_np = np.array([(a, b) for a in ph2_pts
                              for b in ph3_pts], dtype=np.float64)
            a1 = np.sqrt(e_np[:, 0])
            a2 = np.sqrt(e_np[:, 1])
            a3 = np.sqrt(np.clip(1 - e_np[:, 0] - e_np[:, 1], 0, None))
            rows = []
            for i in range(e_np.shape[0]):
                for j in range(ph_np.shape[0]):
                    p2, p3 = ph_np[j, 0], ph_np[j, 1]
                    rows.append([
                        e_np[i, 0], e_np[i, 1], 1 - e_np[i, 0] - e_np[i, 1],
                        2 * a1[i] * a2[i] * np.cos(p2),
                        2 * a1[i] * a2[i] * np.sin(p2),
                        2 * a1[i] * a3[i] * np.cos(p3),
                        2 * a1[i] * a3[i] * np.sin(p3),
                        2 * a2[i] * a3[i] * np.cos(p2 - p3),
                        -2 * a2[i] * a3[i] * np.sin(p2 - p3)])
            c = torch.tensor(np.array(rows, dtype=np.float32))
            corr = eval_c_grid(c, mt, Et, Gt, mu_e, var_e)[0]
            k = int(torch.argmax(corr))
            uv = k // ph_np.shape[0]
            pk = k % ph_np.shape[0]
            return (float(corr[k]), float(e_np[uv, 0]), float(e_np[uv, 1]),
                    float(1 - e_np[uv, 0] - e_np[uv, 1]),
                    float(ph_np[pk, 0]), float(ph_np[pk, 1]))

        # alternating 2D refinements (energy, then phase)
        du = dv = 0.12
        dp = 0.40
        for it in range(2):
            u_pts = np.linspace(max(0.0, u0 - du), min(1.0, u0 + du),
                                n_energy)
            v_pts = np.linspace(max(0.0, v0 - dv), min(1.0, v0 + dv),
                                n_energy)
            r_u = best_over(u_pts, v_pts, [p20], [p30])
            _, u0, v0, w0, p20, p30 = r_u
            ph2 = np.linspace(p20 - dp, p20 + dp, n_phase)
            ph3 = np.linspace(p30 - dp, p30 + dp, n_phase)
            r_p = best_over([u0], [v0], ph2, ph3)
            _, u0, v0, w0, p20, p30 = r_p
            du *= 0.4
            dv *= 0.4
            dp *= 0.4
        # small joint 4D grid
        eu = np.linspace(max(0, u0 - 0.02), min(1, u0 + 0.02), 5)
        ev = np.linspace(max(0, v0 - 0.02), min(1, v0 + 0.02), 5)
        p2g = np.linspace(p20 - 0.16, p20 + 0.16, 7)
        p3g = np.linspace(p30 - 0.16, p30 + 0.16, 7)
        e_all = [(uu, vv) for uu in eu for vv in ev if uu + vv < 1]
        ph_rows = [(a, b) for a in p2g for b in p3g]
        rows = []
        elist = []
        for (uu, vv) in e_all:
            ww = 1 - uu - vv
            a1, a2, a3 = np.sqrt(uu), np.sqrt(vv), np.sqrt(max(ww, 0))
            for (p2, p3) in ph_rows:
                rows.append([
                    uu, vv, ww,
                    2 * a1 * a2 * np.cos(p2), 2 * a1 * a2 * np.sin(p2),
                    2 * a1 * a3 * np.cos(p3), 2 * a1 * a3 * np.sin(p3),
                    2 * a2 * a3 * np.cos(p2 - p3),
                    -2 * a2 * a3 * np.sin(p2 - p3)])
                elist.append((uu, vv, ww, p2, p3))
        c = torch.tensor(rows, dtype=torch.float32)
        corr = eval_c_grid(c, mt, Et, Gt, mu_e, var_e)[0]
        k = int(torch.argmax(corr))
        final = (float(corr[k]),) + elist[k]
        out.append(final)
    return out


# ---------------------------------------------------------------------------
# Layer 3 + rendering
# ---------------------------------------------------------------------------

def render_three_mode_np(modes_np, i, j, k, params):
    u, v, w, p2, p3 = params
    a1, a2, a3 = np.sqrt(u), np.sqrt(v), np.sqrt(max(w, 0))
    E = a1 * modes_np[i] + a2 * modes_np[j] * np.exp(1j * p2) \
        + a3 * modes_np[k] * np.exp(1j * p3)
    I = np.abs(E) ** 2
    return I / max(I.max(), 1e-12)


def corr_avg_full(modes, Ie, tr, params, sigmas=(0, 1, 2, 3, 4, 5, 6)):
    """Multi-sigma averaged corr for one triple (two-mode-equivalent
    protocol: both sides low-passed by the same sigma)."""
    dev = modes.device
    Ie = Ie.to(dev)
    mu_e = Ie.mean()
    var_e = Ie.var(unbiased=False)
    F1 = modes[tr[0]].unsqueeze(0)
    F2 = modes[tr[1]].unsqueeze(0)
    F3 = modes[tr[2]].unsqueeze(0)
    G9 = build_base_fields(F1, F2, F3)[0]  # (9,224,224)
    u, v, w, p2, p3 = params
    a1, a2, a3 = np.sqrt(max(u, 0)), np.sqrt(max(v, 0)), \
        np.sqrt(max(w, 0))
    c = torch.tensor([[
        u, v, w,
        2 * a1 * a2 * np.cos(p2), 2 * a1 * a2 * np.sin(p2),
        2 * a1 * a3 * np.cos(p3), 2 * a1 * a3 * np.sin(p3),
        2 * a2 * a3 * np.cos(p2 - p3),
        -2 * a2 * a3 * np.sin(p2 - p3)]], dtype=torch.float32)
    vals = []
    for sg in sigmas:
        Gb = gaussian_blur(G9, float(sg))          # (9,H,W)
        Ie_b = gaussian_blur(Ie.unsqueeze(0), float(sg))[0]
        me = Ie_b.mean()
        ve = Ie_b.var(unbiased=False)
        Gf = Gb.reshape(9, -1)
        m = Gf.mean(dim=1).unsqueeze(0)
        E = torch.einsum('ki,i->k', Gf, Ie_b.reshape(-1)
                         ).unsqueeze(0) / Gf.shape[1]
        Gram = torch.einsum('ki,li->kl', Gf, Gf
                            ).unsqueeze(0) / Gf.shape[1]
        cv = eval_c_grid(c, m, E, Gram, me, ve)[0, 0]
        vals.append(float(cv))
    return float(np.mean(vals)), vals


def main():
    args = parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(args.max_order, 224, 4.0, 'cpu')
    modes = modes.to(dev)
    N = modes.shape[0]
    modes_np = modes.detach().cpu().numpy()

    # downsampled fields for layer 1
    def _ds(x):
        return F.interpolate(x.unsqueeze(1),
                             size=(args.small, args.small),
                             mode='bilinear',
                             align_corners=False)[:, 0]

    modes_small = torch.complex(_ds(modes.real), _ds(modes.imag)).to(dev)

    triples = torch.combinations(torch.arange(N), r=3)
    if args.quick:
        triples = triples[:2000]
    print(f'[setup] {N} modes, {triples.shape[0]} triples, '
          f'small grid {args.small}')

    e_pts = energy_simplex(args.n_energy)
    c_tab, p_tab = build_c_table(e_pts, args.n_phase)
    print(f'[setup] coarse grid: {c_tab.shape[0]} points/'
          f'triple ({e_pts.shape[0]} energy x '
          f'{c_tab.shape[0]//e_pts.shape[0]} phase)')

    if args.synth:
        synth_test(args, modes, modes_np, triples, c_tab, p_tab, dev,
                   modes_small, labels)
        return

    if args.image:
        images = [args.image]
    elif args.images:
        images = [s.strip() for s in args.images.split(',') if s.strip()]
    else:
        images = FAILED_IMAGES

    out_root = Path(args.out_dir)
    out_root.mkdir(exist_ok=True)
    for name in images:
        img_path = resolve_image_path(name, args.img_dir)
        if img_path is None:
            print(f'[skip] 找不到图像: {name} '
                  f'(img_dir={args.img_dir or IMG_DIR})')
            continue
        name = img_path.stem
        print(f'\n########## {name} ##########')
        Ie_np = load_experiment(str(img_path), args.cmap, args.preprocess)
        Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
        Ie_small = F.interpolate(Ie[None, None], size=(
            args.small, args.small), mode='bilinear',
            align_corners=False)[0, 0]

        t0 = time.time()
        bc, bi = coarse_scan(modes_small, Ie_small, triples, c_tab,
                             args.batch, dev)
        print(f'  [L1] coarse scan done in {time.time()-t0:.0f}s, '
              f'best corr={float(bc.max()):.4f}')
        topk = min(args.top_coarse, bc.numel())
        vals, idxs = bc.topk(topk)
        top_triples = triples[idxs.cpu()]
        top_params = p_tab[bi[idxs].cpu()]  # (K,5)

        t0 = time.time()
        refined = refine_triples(modes, Ie, top_triples,
                                 top_params.tolist())
        print(f'  [L2] refinement done in {time.time()-t0:.0f}s')

        order = np.argsort([-r[0] for r in refined])
        n_struct = min(args.top_struct, len(order))
        entries = []
        for oi in order[:n_struct]:
            corr, u, v, w, p2, p3 = refined[oi]
            tr = top_triples[oi].tolist()
            cavg, _ = corr_avg_full(modes, Ie, tr, (u, v, w, p2, p3))
            entries.append({
                'rank': 0,
                'triple_idx': int(oi),
                'modes': [labels[tr[0]], labels[tr[1]], labels[tr[2]]],
                'energy': [u, v, w],
                'amps': [float(np.sqrt(max(u, 0))),
                         float(np.sqrt(max(v, 0))),
                         float(np.sqrt(max(w, 0)))],
                'phi2': p2, 'phi3': p3,
                'corr': corr, 'corr_avg': cavg,
            })
        entries.sort(key=lambda e: -e['corr_avg'])
        for i, e in enumerate(entries, start=1):
            e['rank'] = i
        # structural verification against the experiment
        pts_e = find_dark_points(Ie_np)
        print(f'  [L3] experiment cores = {len(pts_e)}')
        best_pass = None
        for e in entries:
            tr = top_triples[e['triple_idx']].tolist()
            I = render_three_mode_np(modes_np, tr[0], tr[1], tr[2],
                                     (e['energy'][0], e['energy'][1],
                                      e['energy'][2], e['phi2'],
                                      e['phi3']))
            pts_m = find_dark_points(I)
            match = match_points(pts_e, pts_m)
            e['mean_nn_px'] = match['mean']
            n_ok = abs(len(pts_m) - len(pts_e)) <= 1
            d_ok = match['mean'] is not None and match['mean'] < 12
            e['verdict'] = 'PASS' if (n_ok and d_ok) else 'fail'
            print(f'    #{e["rank"]:2d} {"+".join(e["modes"]):38s} '
                  f'cavg={e["corr_avg"]:.4f} c0={e["corr"]:.4f} '
                  f'cores={len(pts_m)} (exp {len(pts_e)}) meanNN='
                  f'{e["mean_nn_px"] if e["mean_nn_px"] is None else round(e["mean_nn_px"], 1)} '
                  f'{e["verdict"]}')
            if e['verdict'] == 'PASS' and best_pass is None:
                best_pass = e

        payload = {
            'image': name,
            'n_core_experiment': len(pts_e),
            'n_triples_scanned': int(triples.shape[0]),
            'entries': entries,
            'best_pass': best_pass,
        }
        (out_root / f'{name}_three_mode.json').write_text(
            json.dumps(payload, indent=2), encoding='utf-8')

        # verification figure for rank-1 (and best pass if different)
        show = best_pass if best_pass is not None else entries[0]
        tr = top_triples[show['triple_idx']].tolist()
        I = render_three_mode_np(modes_np, tr[0], tr[1], tr[2],
                                 (show['energy'][0], show['energy'][1],
                                  show['energy'][2], show['phi2'],
                                  show['phi3']))
        pts_m = find_dark_points(I)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6))
        axs[0].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
        for (x, y, _v) in pts_e:
            axs[0].plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.4)
        axs[0].set_title(f'{name} experiment: {len(pts_e)} cores',
                         fontsize=10)
        axs[1].imshow(I, cmap='inferno', vmin=0, vmax=1)
        for (x, y, _v) in pts_m:
            axs[1].plot(x, y, 'o', mfc='none', mec='red', ms=11, mew=1.4)
        axs[1].set_title(f'{"+".join(show["modes"])}\n'
                         f'corr={show["corr"]:.4f} ({len(pts_m)} cores)',
                         fontsize=9)
        axs[2].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
        for (x, y, _v) in pts_e:
            axs[2].plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.4)
        for (x, y, _v) in pts_m:
            axs[2].plot(x, y, 'x', color='red', ms=9, mew=1.8)
        axs[2].set_title(f'overlay [{show["verdict"].upper()}]', fontsize=10)
        for a in axs:
            a.axis('off')
        fig.tight_layout()
        fig.savefig(out_root / f'{name}_three_mode_structural.png', dpi=130)
        plt.close(fig)

    print(f'\n[done] outputs in {out_root}')


def synth_test(args, modes, modes_np, triples, c_tab, p_tab, dev,
               modes_small, labels):
    """Synthetic three-mode recovery test."""
    print('=== THREE-MODE SYNTHETIC SELF-TEST ===')
    rng = np.random.default_rng(3)
    for trial in range(3):
        ti = int(rng.integers(0, triples.shape[0]))
        tr = triples[ti].tolist()
        u, v = rng.uniform(0.15, 0.5), rng.uniform(0.1, 0.4)
        if u + v > 0.8:
            u, v = u * 0.5, v * 0.5
        w = 1 - u - v
        p2, p3 = rng.uniform(0, 2 * np.pi, 2)
        It = render_three_mode_np(modes_np, tr[0], tr[1], tr[2],
                                  (u, v, w, p2, p3))
        It = np.clip(It + rng.normal(0, 0.01, It.shape), 0, 1)
        Ie = torch.tensor(It, dtype=torch.float32, device=dev)
        Ie_small = F.interpolate(Ie[None, None],
                                 size=(args.small, args.small),
                                 mode='bilinear',
                                 align_corners=False)[0, 0]
        bc, bi = coarse_scan(modes_small, Ie_small, triples, c_tab,
                             args.batch, dev)
        top2 = bc.topk(2)
        hit = False
        got_names = []
        for idx in top2.indices.tolist():
            got = sorted(triples[idx].tolist())
            got_names.append([labels[t] for t in got])
            if got == tr:
                hit = True
        print(f'  [{trial+1}] truth {[labels[t] for t in tr]} '
              f'u={u:.2f}/v={v:.2f}/w={w:.2f} | top1 corr='
              f'{float(top2.values[0]):.4f} top1={got_names[0]} '
              f'{"HIT" if hit else "MISS"}')
    print('[synth] done')


if __name__ == '__main__':
    main()

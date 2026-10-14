# -*- coding: utf-8 -*-
"""
search_four_mode.py -- 四模解析搜索 (Four-mode analytical search)
=================================================================

E = a1F1 + a2F2 e^{i phi2} + a3F3 e^{i phi3} + a4F4 e^{i phi4}
    a1^2+a2^2+a3^2+a4^2 = 1

16 real base fields: 4 self intensities B_k and 6 complex cross terms
(C_ij, S_ij) -> 16 coefficient components.

Two candidate-pool modes:
  --mode guided : triples from the three-mode top list + one extra mode
                  (fast, targeted)
  --mode full   : all C(131,4) quadruples on a coarse downsampled grid
                  (insurance sweep)

Layers: coarse grid scan -> per-quadruple statistics (full res) ->
coordinate-descent refinement -> dark-core structural verification.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from search_three_mode import eval_c_grid, resolve_image_path, IMG_DIR
from structural_check import find_dark_points, match_points

OUT = Path(__file__).resolve().parent / 'four_mode_search'


def parse_args():
    p = argparse.ArgumentParser(
        description='Four-mode analytical search',
        epilog='--mode guided 需要先用 search_three_mode.py 产出同名 '
               '*_three_mode.json；--mode full 不依赖三模结果。')
    p.add_argument('--image', type=str, default='10Singularities',
                   help='图像路径或文件名（可用 --img_dir 指定目录）')
    p.add_argument('--img_dir', type=str, default=None,
                   help='图像所在目录；默认指向本项目演示数据目录')
    p.add_argument('--three_dir', type=str, default=None,
                   help='guided 模式读取三模结果的目录；默认 out_dir/../three_mode_search')
    p.add_argument('--out_dir', type=str, default=None,
                   help='输出目录；默认 ./four_mode_search')
    p.add_argument('--mode', type=str, default='guided',
                   choices=['guided', 'full'])
    p.add_argument('--n_energy', type=int, default=10,
                   help='energy simplex steps (N -> C(N+3,3) pts)')
    p.add_argument('--n_phase', type=int, default=16,
                   help='phase grid per axis')
    p.add_argument('--top_coarse', type=int, default=120)
    p.add_argument('--top_struct', type=int, default=30)
    p.add_argument('--batch', type=int, default=192)
    p.add_argument('--small', type=int, default=48,
                   help='downsample size for coarse scan')
    p.add_argument('--cmap', type=str, default='nipy_spectral',
                   help='伪彩配色；灰度图请传 none')
    p.add_argument('--preprocess', type=str, default='isot',
                   help='预处理模式（isot = 无量纲拓扑对齐）')
    p.add_argument('--synth', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# base fields / coefficients / grids
# ---------------------------------------------------------------------------

def build_base_fields4(F1, F2, F3, F4):
    """16 base fields from four complex mode fields. (B,16,H,W)."""
    B1 = F1.real ** 2 + F1.imag ** 2
    B2 = F2.real ** 2 + F2.imag ** 2
    B3 = F3.real ** 2 + F3.imag ** 2
    B4 = F4.real ** 2 + F4.imag ** 2
    X12 = F1 * F2.conj()
    X13 = F1 * F3.conj()
    X14 = F1 * F4.conj()
    X23 = F2 * F3.conj()
    X24 = F2 * F4.conj()
    X34 = F3 * F4.conj()
    return torch.stack([B1, B2, B3, B4,
                        X12.real, X12.imag, X13.real, X13.imag,
                        X14.real, X14.imag, X23.real, X23.imag,
                        X24.real, X24.imag, X34.real, X34.imag], dim=1)


def energy_simplex4(N):
    pts = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            for k in range(N + 1 - i - j):
                l = N - i - j - k
                pts.append((i / N, j / N, k / N, l / N))
    return np.array(pts, dtype=np.float64)


def build_c_table4(e_pts, n_phase):
    """Vectorized coefficient table for the 4-mode model.

    Returns c (P,16), params (P,7): [u1..u4, phi2, phi3, phi4].
    """
    ph1 = np.linspace(0, 2 * np.pi, n_phase + 1)[:-1]
    g2, g3, g4 = np.meshgrid(ph1, ph1, ph1, indexing='ij')
    ph_pts = np.stack([g2.ravel(), g3.ravel(), g4.ravel()], axis=1)
    E, P = len(e_pts), len(ph_pts)
    u = np.repeat(e_pts, P, axis=0)          # (EP,4)
    ph = np.tile(ph_pts, (E, 1))             # (EP,3)
    a = np.sqrt(np.clip(u, 0, None))
    p2, p3, p4 = ph[:, 0], ph[:, 1], ph[:, 2]
    cols = [
        u[:, 0], u[:, 1], u[:, 2], u[:, 3],
        2 * a[:, 0] * a[:, 1] * np.cos(p2), 2 * a[:, 0] * a[:, 1] * np.sin(p2),
        2 * a[:, 0] * a[:, 2] * np.cos(p3), 2 * a[:, 0] * a[:, 2] * np.sin(p3),
        2 * a[:, 0] * a[:, 3] * np.cos(p4), 2 * a[:, 0] * a[:, 3] * np.sin(p4),
        2 * a[:, 1] * a[:, 2] * np.cos(p2 - p3),
        -2 * a[:, 1] * a[:, 2] * np.sin(p2 - p3),
        2 * a[:, 1] * a[:, 3] * np.cos(p2 - p4),
        -2 * a[:, 1] * a[:, 3] * np.sin(p2 - p4),
        2 * a[:, 2] * a[:, 3] * np.cos(p3 - p4),
        -2 * a[:, 2] * a[:, 3] * np.sin(p3 - p4),
    ]
    c = np.stack(cols, axis=1).astype(np.float32)
    params = np.concatenate([u, ph], axis=1).astype(np.float32)
    return torch.tensor(c), torch.tensor(params)


def c_row4(params):
    """Single-row c coefficients from (u1..u4, p2, p3, p4)."""
    u1, u2, u3, u4, p2, p3, p4 = params
    a1, a2, a3, a4 = (np.sqrt(max(u1, 0)), np.sqrt(max(u2, 0)),
                      np.sqrt(max(u3, 0)), np.sqrt(max(u4, 0)))
    return torch.tensor([[
        u1, u2, u3, u4,
        2 * a1 * a2 * np.cos(p2), 2 * a1 * a2 * np.sin(p2),
        2 * a1 * a3 * np.cos(p3), 2 * a1 * a3 * np.sin(p3),
        2 * a1 * a4 * np.cos(p4), 2 * a1 * a4 * np.sin(p4),
        2 * a2 * a3 * np.cos(p2 - p3), -2 * a2 * a3 * np.sin(p2 - p3),
        2 * a2 * a4 * np.cos(p2 - p4), -2 * a2 * a4 * np.sin(p2 - p4),
        2 * a3 * a4 * np.cos(p3 - p4), -2 * a3 * a4 * np.sin(p3 - p4),
    ]], dtype=torch.float32)


def render_four_mode_np(modes_np, quad, params):
    u1, u2, u3, u4, p2, p3, p4 = params
    a = [np.sqrt(max(x, 0)) for x in (u1, u2, u3, u4)]
    E = (a[0] * modes_np[quad[0]]
         + a[1] * modes_np[quad[1]] * np.exp(1j * p2)
         + a[2] * modes_np[quad[2]] * np.exp(1j * p3)
         + a[3] * modes_np[quad[3]] * np.exp(1j * p4))
    I = np.abs(E) ** 2
    return I / max(I.max(), 1e-12)


# ---------------------------------------------------------------------------
# coarse scan
# ---------------------------------------------------------------------------

def coarse_scan4(modes_small, Ie_small, quads, c_tab, batch,
                 c_chunk=100000):
    dev = modes_small.device
    Ie = Ie_small.to(dev)
    mu_e = Ie.mean()
    var_e = Ie.var(unbiased=False)
    Q = quads.shape[0]
    P = c_tab.shape[0]
    best_corr = torch.empty(Q, device=dev)
    best_idx = torch.empty(Q, dtype=torch.long, device=dev)
    for b0 in range(0, Q, batch):
        q = quads[b0:b0 + batch].to(dev)
        G16 = build_base_fields4(
            modes_small[q[:, 0]], modes_small[q[:, 1]],
            modes_small[q[:, 2]], modes_small[q[:, 3]])
        B = G16.shape[0]
        Gf = G16.reshape(B, 16, -1)
        m = Gf.mean(dim=2)
        E = torch.einsum('bki,i->bk', Gf, Ie.reshape(-1)) / Gf.shape[2]
        Gram = torch.einsum('bki,bli->bkl', Gf, Gf) / Gf.shape[2]
        # evaluate the coefficient grid in chunks (memory guard)
        bmx = torch.full((B,), -2.0, device=dev)
        bmi = torch.zeros(B, dtype=torch.long, device=dev)
        for c0 in range(0, P, c_chunk):
            cc = c_tab[c0:c0 + c_chunk]
            corr = eval_c_grid(cc, m, E, Gram, mu_e, var_e)
            mx, mi = corr.max(dim=1)
            upd = mx > bmx
            bmx = torch.where(upd, mx, bmx)
            bmi = torch.where(upd, mi + c0, bmi)
        best_corr[b0:b0 + B] = bmx
        best_idx[b0:b0 + B] = bmi
    return best_corr, best_idx


# ---------------------------------------------------------------------------
# refinement (per-quadruple, precomputed per-sigma statistics)
# ---------------------------------------------------------------------------

SIGMAS = [0, 1, 2, 3, 4, 5, 6]


def refine_quad(modes, Ie, quad, p0, rounds=3):
    dev = modes.device
    F1, F2, F3, F4 = (modes[quad[0]].unsqueeze(0),
                      modes[quad[1]].unsqueeze(0),
                      modes[quad[2]].unsqueeze(0),
                      modes[quad[3]].unsqueeze(0))
    G16 = build_base_fields4(F1, F2, F3, F4)[0]  # (16,H,W)
    stats = []
    for sg in SIGMAS:
        Gb = gaussian_blur(G16, float(sg))
        Ie_b = gaussian_blur(Ie.unsqueeze(0), float(sg))[0]
        me = Ie_b.mean()
        ve = Ie_b.var(unbiased=False)
        Gf = Gb.reshape(16, -1)
        m = Gf.mean(dim=1).unsqueeze(0)
        E = torch.einsum('ki,i->k', Gf, Ie_b.reshape(-1)
                         ).unsqueeze(0) / Gf.shape[1]
        Gram = torch.einsum('ki,li->kl', Gf, Gf
                            ).unsqueeze(0) / Gf.shape[1]
        stats.append((m, E, Gram, me, ve))

    def ev(params):
        c = c_row4(params).to(dev)
        vals = [float(eval_c_grid(c, *s)[0, 0]) for s in stats]
        return float(np.mean(vals))

    cur = [float(x) for x in p0]
    h = [0.08, 0.08, 0.08, 0.08, 0.35, 0.35, 0.35]
    for rd in range(rounds):
        for k in range(7):
            pts = np.linspace(cur[k] - h[k], cur[k] + h[k], 41)
            best_v, best_x = -2.0, cur[k]
            for x in pts:
                trial = list(cur)
                trial[k] = float(x)
                if sum(trial[:4]) > 1.0 or min(trial[:4]) < 0:
                    continue
                cv = ev(trial)
                if cv > best_v:
                    best_v, best_x = cv, float(x)
            cur[k] = best_x
        h = [x * 0.3 for x in h]
    s = sum(cur[:4])
    cur[:4] = [x / s for x in cur[:4]]
    return ev(cur), cur


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    global OUT
    OUT = Path(args.out_dir) if args.out_dir else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes = modes.to(dev)
    modes_np = modes.detach().cpu().numpy()
    lab2k = {lab: k for k, lab in enumerate(labels)}
    N = modes.shape[0]

    def _ds(x, size):
        return F.interpolate(x.unsqueeze(1), size=(size, size),
                             mode='bilinear',
                             align_corners=False)[:, 0]

    modes_small = torch.complex(_ds(modes.real, args.small),
                                _ds(modes.imag, args.small)).to(dev)

    img_path = resolve_image_path(args.image, args.img_dir)
    if img_path is None:
        print(f'[error] 找不到图像: {args.image} '
              f'(img_dir={args.img_dir or IMG_DIR})')
        return
    args.image = img_path.stem
    Ie_np = load_experiment(str(img_path), args.cmap, args.preprocess)
    Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
    Ie_small = F.interpolate(Ie[None, None],
                             size=(args.small, args.small),
                             mode='bilinear', align_corners=False)[0, 0]
    pts_e = find_dark_points(Ie_np)
    print(f'[setup] image={args.image}, experiment cores={len(pts_e)}, '
          f'mode={args.mode}')

    # ---- candidate quadruples ----
    if args.mode == 'guided':
        three_dir = Path(args.three_dir) if args.three_dir else \
            (Path(args.out_dir) if args.out_dir else Path.cwd()) / '..' / 'three_mode_search'
        src_file = Path(three_dir) / f'{args.image}_three_mode.json'
        if not src_file.is_file():
            print(f'[error] guided 模式需要三模结果: {src_file}\n'
                  f'        请先用 search_three_mode.py 跑同一张图，'
                  f'或用 --three_dir 指定其输出目录；'
                  f'也可改用 --mode full。')
            return
        src = json.load(open(src_file, encoding='utf-8'))
        base_triples = []
        for e in src['entries']:
            tr = tuple(sorted(lab2k[m] for m in e['modes']))
            if tr not in base_triples:
                base_triples.append(tr)
        quads = set()
        for tr in base_triples:
            for x in range(N):
                if x in tr:
                    continue
                quads.add(tuple(sorted(tr + (x,))))
        quads = torch.tensor(sorted(quads), dtype=torch.long)
        print(f'[setup] guided pool: {len(base_triples)} triples '
              f'-> {quads.shape[0]} quadruples')
    else:
        quads = torch.combinations(torch.arange(N), r=4)
        print(f'[setup] full pool: {quads.shape[0]} quadruples')

    # ---- coarse scan ----
    e_pts = energy_simplex4(args.n_energy)
    c_tab, p_tab = build_c_table4(e_pts, args.n_phase)
    print(f'[setup] coarse grid: {c_tab.shape[0]} points/quadruple '
          f'({len(e_pts)} energy x {args.n_phase ** 3} phase)')

    if args.synth:
        rng = np.random.default_rng(5)
        qi = int(rng.integers(0, quads.shape[0]))
        quad = quads[qi].tolist()
        u = rng.dirichlet([2.0, 1.2, 1.0, 0.8])
        p2, p3, p4 = rng.uniform(0, 2 * np.pi, 3)
        It = render_four_mode_np(modes_np, quad, (*u, p2, p3, p4))
        It = np.clip(It + rng.normal(0, 0.01, It.shape), 0, 1)
        Ie_s = torch.tensor(It, dtype=torch.float32, device=dev)
        Ie_s = F.interpolate(Ie_s[None, None],
                             size=(args.small, args.small),
                             mode='bilinear', align_corners=False)[0, 0]
        bc, bi = coarse_scan4(modes_small, Ie_s, quads, c_tab.to(dev),
                              args.batch)
        top1 = quads[int(bc.argmax())].tolist()
        truth_names = sorted(labels[q] for q in quad)
        found_names = sorted(labels[q] for q in top1)

        def conj_set(names):
            out = []
            for nm in names:
                kind, rest = nm.split('(')
                pp, ll = rest[:-1].split(',')
                out.append(f'{kind}({pp},{-int(ll)})')
            return sorted(out)

        hit = found_names == truth_names or \
            found_names == conj_set(truth_names)
        print(f'[synth] truth {[labels[q] for q in quad]} '
              f'u={np.round(u, 3).tolist()}')
        print(f'[synth] found {[labels[q] for q in top1]} '
              f'corr={float(bc.max()):.4f} '
              f'{"HIT" if hit else "MISS"}'
              f'{" (conjugate-equivalent)" if found_names == conj_set(truth_names) else ""}')
        return

    t0 = time.time()
    bc, bi = coarse_scan4(modes_small, Ie_small, quads, c_tab.to(dev),
                          args.batch)
    print(f'[L1] coarse scan done in {time.time() - t0:.0f}s, '
          f'best corr={float(bc.max()):.4f}')

    topk = min(args.top_coarse, bc.numel())
    vals, idxs = bc.topk(topk)
    top_quads = quads[idxs.cpu()]
    top_params = p_tab[bi[idxs].cpu()]

    # ---- refinement ----
    t0 = time.time()
    refined = []
    for k in range(top_quads.shape[0]):
        quad = top_quads[k].tolist()
        cavg, params = refine_quad(modes, Ie, quad,
                                   top_params[k].tolist())
        refined.append((cavg, params, quad))
    refined.sort(key=lambda r: -r[0])
    print(f'[L2] refinement done in {time.time() - t0:.0f}s')

    # ---- structural verification ----
    entries = []
    for cavg, params, quad in refined[:args.top_struct]:
        I = render_four_mode_np(modes_np, quad, params)
        pts_m = find_dark_points(I)
        match = match_points(pts_e, pts_m)
        n_ok = abs(len(pts_m) - len(pts_e)) <= 1
        d_ok = match['mean'] is not None and match['mean'] < 12
        entries.append({
            'modes': [labels[q] for q in quad],
            'energy': params[:4],
            'amps': [float(np.sqrt(max(x, 0))) for x in params[:4]],
            'phis': params[4:],
            'corr_avg': cavg,
            'n_core': len(pts_m),
            'mean_nn_px': match['mean'],
            'verdict': 'PASS' if (n_ok and d_ok) else 'fail',
        })
    entries.sort(key=lambda e: -e['corr_avg'])
    for r, e in enumerate(entries, start=1):
        e['rank'] = r
        print(f'    #{r:2d} {"+".join(e["modes"]):46s} '
              f'cavg={e["corr_avg"]:.4f} cores={e["n_core"]} '
              f'(exp {len(pts_e)}) meanNN='
              f'{e["mean_nn_px"] if e["mean_nn_px"] is None else round(e["mean_nn_px"], 1)} '
              f'{e["verdict"]}')

    best_pass = next((e for e in entries if e['verdict'] == 'PASS'), None)
    payload = {'image': args.image, 'mode': args.mode,
               'n_quadruples': int(quads.shape[0]),
               'n_core_experiment': len(pts_e),
               'entries': entries, 'best_pass': best_pass}
    (OUT / f'{args.image}_four_mode_{args.mode}.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')

    # ---- figure for rank-1 / best pass ----
    show = best_pass if best_pass is not None else entries[0]
    qidx = next(i for i, (c, p, q) in enumerate(refined)
                if [labels[x] for x in q] == show['modes'])
    quad = refined[qidx][2]
    params = refined[qidx][1]
    I = render_four_mode_np(modes_np, quad, params)
    pts_m = find_dark_points(I)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6))
    axs[0].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
    for (x, y, _v) in pts_e:
        axs[0].plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.4)
    axs[0].set_title(f'{args.image} experiment: {len(pts_e)} cores',
                     fontsize=10)
    axs[1].imshow(I, cmap='inferno', vmin=0, vmax=1)
    for (x, y, _v) in pts_m:
        axs[1].plot(x, y, 'o', mfc='none', mec='red', ms=11, mew=1.4)
    axs[1].set_title(f'{"+".join(show["modes"])}\n'
                     f'corr_avg={show["corr_avg"]:.4f} '
                     f'({len(pts_m)} cores)', fontsize=8)
    axs[2].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
    for (x, y, _v) in pts_e:
        axs[2].plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.4)
    for (x, y, _v) in pts_m:
        axs[2].plot(x, y, 'x', color='red', ms=9, mew=1.8)
    axs[2].set_title(f'overlay [{show["verdict"].upper()}]', fontsize=10)
    for a in axs:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / f'{args.image}_four_mode_structural.png', dpi=140)
    plt.close(fig)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

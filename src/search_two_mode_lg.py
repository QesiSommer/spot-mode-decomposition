# -*- coding: utf-8 -*-
"""
search_two_mode_lg.py -- Two-mode LG (Laguerre-Gaussian) analytical
parameter-scan decomposition
====================================================================

Physical model (two coherent LG modes at the waist):

    E = a1*L1 + a2*L2*exp(i*phi),   a1=sqrt(r), a2=sqrt(1-r)
    I_t = |E|^2
        = r*A + (1-r)*B + 2*s*(cos(phi)*C1 - sin(phi)*C2)

    A=|L1|^2, B=|L2|^2, C1=Re[L1*conj(L2)], C2=Im[L1*conj(L2)],
    s = sqrt(r*(1-r))

Search space (GPU analytical, 18 precomputed moments per pair):
    LG pair (2p+|l| <= 10, 66 modes -> 2145 pairs)
    x r in (0,1) x phi in [0, 2pi)   (rotation == phase shift for LG)
    x zoom scale s, x blur sigma

Output: top-K table + three-panel comparison PNG per image.
Run --synth for a self-test.
"""

import argparse
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import genlaguerre

from search_two_mode import (load_experiment, rescale_modes, gaussian_blur,
                             invert_colormap)

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def parse_args():
    p = argparse.ArgumentParser(description='Two-mode LG analytical search')
    p.add_argument('--image', type=str, default=None)
    p.add_argument('--images_dir', type=str,
                   default=os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images')))
    p.add_argument('--out_dir', type=str, default='two_mode_search_lg')
    p.add_argument('--max_order', type=int, default=10)
    p.add_argument('--grid_r', type=int, default=101)
    p.add_argument('--grid_phi', type=int, default=360)
    p.add_argument('--refine_k', type=int, default=12)
    p.add_argument('--refine_grid', type=int, default=101)
    p.add_argument('--scales', type=str, default='1.0,1.25,1.5,1.75,2.0')
    p.add_argument('--sigmas', type=str, default='0,1,2,3,4')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--cmap', type=str, default='nipy_spectral')
    p.add_argument('--preprocess', type=str, default='isot',
                   choices=['isot', 'crop'])
    p.add_argument('--synth', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# LG basis
# ---------------------------------------------------------------------------

def build_lg_modes(max_order=10, size=224, extent=4.0, device='cuda'):
    xs = np.linspace(-extent, extent, size)
    # 标准光学约定：方位角 TH = atan2(y, x)，y=竖直(轴0)、x=水平(轴1)。
    # 'ij' 索引下 axis0=竖直 y、axis1=水平 x，故先 Y 后 X。
    # (2026-10-16 修正：此前 X 沿轴0，TH 与标准方位角互为转置，
    #  导致 our LG(p,l) = e^{+i·l·π/2}·std LG(p,-l)——强度相同、
    #  螺旋相位手性相反。修正后 LG(p,l) 与学术/MATLAB 记号一致。)
    Y, X = np.meshgrid(xs, xs, indexing='ij')
    R = np.sqrt(X ** 2 + Y ** 2)
    TH = np.arctan2(Y, X)
    idx = []
    for N in range(max_order + 1):
        for l in range(-N, N + 1):
            pp = (N - abs(l)) // 2
            if 2 * pp + abs(l) == N:
                idx.append((pp, l))
    modes = np.zeros((len(idx), size, size), dtype=np.complex64)
    for k, (pp, l) in enumerate(idx):
        L = genlaguerre(pp, abs(l))(2 * R ** 2)
        amp = (np.sqrt(2) * R) ** abs(l) * L * np.exp(-R ** 2)
        modes[k] = (amp * np.exp(1j * l * TH)).astype(np.complex64)
    return torch.tensor(modes, device=device), idx


def rescale_lg(modes, scale):
    """Zoom complex mode fields (real/imag interpolated separately)."""
    if abs(scale - 1.0) < 1e-6:
        return modes

    def _zoom(x):  # (K,H,W) -> zoomed (K,H,W)
        G = x.unsqueeze(0)
        if scale > 1.0:
            big = F.interpolate(G, scale_factor=scale, mode='bilinear',
                                align_corners=False)[0]
            H = big.shape[-1]
            c0 = (H - x.shape[-1]) // 2
            return big[:, c0:c0 + x.shape[-1], c0:c0 + x.shape[-1]]
        small = F.interpolate(G, scale_factor=scale, mode='bilinear',
                              align_corners=False)[0]
        H = small.shape[-1]
        out = torch.zeros(x.shape[0], x.shape[-1], x.shape[-1],
                          device=x.device, dtype=x.dtype)
        c0 = (x.shape[-1] - H) // 2
        out[:, c0:c0 + H, c0:c0 + H] = small
        return out

    return torch.complex(_zoom(modes.real), _zoom(modes.imag))


# ---------------------------------------------------------------------------
# Analytical scan (LG / complex version)
# ---------------------------------------------------------------------------

def scan_pairs_lg(modes, Ie_b, pairs_ij, r_grid, phi_grid, sigma,
                  batch=128):
    """Returns corr (P,), r_best (P,), phi_best (P,)."""
    dev = modes.device
    Ie_b = Ie_b.to(dev)
    me = Ie_b.mean()
    ve = Ie_b.var(unbiased=False).clamp_min(1e-12)

    R = r_grid.numel()
    Pp = phi_grid.numel()
    r = r_grid.view(1, -1, 1)
    rp = 1.0 - r
    s = torch.sqrt((r * rp).clamp_min(0))
    cosf = torch.cos(phi_grid).view(1, 1, -1)
    sinf = torch.sin(phi_grid).view(1, 1, -1)

    P = pairs_ij.shape[0]
    corr_out = torch.empty(P, device=dev)
    r_out = torch.empty(P, device=dev)
    phi_out = torch.empty(P, device=dev)

    for b0 in range(0, P, batch):
        ij = pairs_ij[b0:b0 + batch]
        L1 = modes[ij[:, 0]]
        L2 = modes[ij[:, 1]]
        A = (L1.real ** 2 + L1.imag ** 2)
        B = (L2.real ** 2 + L2.imag ** 2)
        cr = L1.real * L2.real + L1.imag * L2.imag
        ci = L1.imag * L2.real - L1.real * L2.imag
        A = gaussian_blur(A, sigma)
        B = gaussian_blur(B, sigma)
        cr = gaussian_blur(cr, sigma)
        ci = gaussian_blur(ci, sigma)

        nA = A.mean(dim=(1, 2)).view(-1, 1, 1)
        nB = B.mean(dim=(1, 2)).view(-1, 1, 1)
        nC1 = cr.mean(dim=(1, 2)).view(-1, 1, 1)
        nC2 = ci.mean(dim=(1, 2)).view(-1, 1, 1)
        qAA = (A * A).mean(dim=(1, 2)).view(-1, 1, 1)
        qBB = (B * B).mean(dim=(1, 2)).view(-1, 1, 1)
        qAB = (A * B).mean(dim=(1, 2)).view(-1, 1, 1)
        qAC1 = (A * cr).mean(dim=(1, 2)).view(-1, 1, 1)
        qAC2 = (A * ci).mean(dim=(1, 2)).view(-1, 1, 1)
        qBC1 = (B * cr).mean(dim=(1, 2)).view(-1, 1, 1)
        qBC2 = (B * ci).mean(dim=(1, 2)).view(-1, 1, 1)
        qC11 = (cr * cr).mean(dim=(1, 2)).view(-1, 1, 1)
        qC22 = (ci * ci).mean(dim=(1, 2)).view(-1, 1, 1)
        qC12 = (cr * ci).mean(dim=(1, 2)).view(-1, 1, 1)
        eA = (A * Ie_b).mean(dim=(1, 2)).view(-1, 1, 1)
        eB = (B * Ie_b).mean(dim=(1, 2)).view(-1, 1, 1)
        eC1 = (cr * Ie_b).mean(dim=(1, 2)).view(-1, 1, 1)
        eC2 = (ci * Ie_b).mean(dim=(1, 2)).view(-1, 1, 1)

        mt = (r * nA + rp * nB
              + 2 * s * (cosf * nC1 - sinf * nC2))
        Ete = (r * eA + rp * eB
               + 2 * s * (cosf * eC1 - sinf * eC2))
        Et2 = ((r * r) * qAA + (rp * rp) * qBB + 2 * r * rp * qAB
               + 4 * s * cosf * (r * qAC1 + rp * qBC1)
               - 4 * s * sinf * (r * qAC2 + rp * qBC2)
               + 4 * s * s * (cosf * cosf * qC11
                              - 2 * cosf * sinf * qC12
                              + sinf * sinf * qC22))

        cov = Ete - mt * me
        vart_raw = Et2 - mt * mt
        # reject numerically degenerate points (variance collapses to
        # zero, e.g. nearly constant I_t) that would blow up corr
        vart_scale = Et2.abs() + mt * mt + 1e-12
        valid = vart_raw > 1e-6 * vart_scale
        corr = torch.where(
            valid,
            cov / torch.sqrt(vart_raw.clamp_min(1e-12) * ve),
            torch.full_like(cov, -1.0))
        corr = corr.clamp(-1.0, 1.0)  # (B, R, Pp)

        Bb = corr.shape[0]
        flat = corr.reshape(Bb, -1)
        mx, amax = flat.max(dim=1)
        ri = amax // Pp
        pi = amax % Pp
        corr_out[b0:b0 + Bb] = mx
        r_out[b0:b0 + Bb] = r_grid[ri]
        phi_out[b0:b0 + Bb] = phi_grid[pi]

    return corr_out, r_out, phi_out


def render_lg(modes_np, i, j, r, phi, sigma=0.0):
    L1 = modes_np[i]
    L2 = modes_np[j]
    E = np.sqrt(r) * L1 + np.sqrt(1 - r) * L2 * np.exp(1j * phi)
    It = (np.abs(E) ** 2).astype(np.float64)
    if sigma > 0:
        dev_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = torch.tensor(It, dtype=torch.float32, device=dev_)
        It = gaussian_blur(t.unsqueeze(0), sigma)[0].cpu().numpy()
    It = It - It.min()
    m = It.max()
    return It / m if m > 1e-12 else It


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def decompose_image(img_path, idx, modes, args, out_root):
    stem = Path(img_path).stem
    m_tv = re.search(r'(\d+)\s*Singularit', stem)
    target_vor = int(m_tv.group(1)) if m_tv else None
    Ie_np = load_experiment(img_path, args.cmap, args.preprocess)
    dev = modes.device
    Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)

    N = modes.shape[0]
    pairs = torch.tensor([(i, j) for i in range(N) for j in range(i + 1, N)],
                         dtype=torch.long, device=dev)
    P = pairs.shape[0]
    scales = [float(x) for x in args.scales.split(',')]
    sigmas = [float(x) for x in args.sigmas.split(',')]
    S, G = len(scales), len(sigmas)

    r_grid = torch.linspace(0.01, 0.99, args.grid_r, device=dev)
    phi_grid = torch.linspace(0.0, 2 * np.pi, args.grid_phi + 1, device=dev)
    phi_grid = phi_grid[:-1]  # [0, 2pi)

    corr_cube = torch.full((S, G, P), -2.0, device=dev)
    r_cube = torch.zeros((S, G, P), device=dev)
    p_cube = torch.zeros((S, G, P), device=dev)
    for si, sc in enumerate(scales):
        ms = rescale_lg(modes, sc)
        for gi, sig in enumerate(sigmas):
            Ie_b = gaussian_blur(Ie.unsqueeze(0), sig)[0]
            corr, rb, pb = scan_pairs_lg(ms, Ie_b, pairs, r_grid, phi_grid,
                                         sig)
            corr_cube[si, gi] = corr
            r_cube[si, gi] = rb
            p_cube[si, gi] = pb

    score = corr_cube.mean(dim=1)  # (S, P) mean over sigma
    top_vals, top_idx = score.reshape(-1).topk(args.refine_k)

    dr = (r_grid[-1] - r_grid[0]).item() / (args.grid_r - 1)
    dphi = 2 * np.pi / args.grid_phi
    ref_corr_avg, ref_corr_best, ref_r, ref_phi, ref_scale, ref_sig, \
        ref_pair = [], [], [], [], [], [], []
    for idxc in top_idx.tolist():
        si = idxc // P
        pair_id = idxc % P
        sc = scales[si]
        gi = int(corr_cube[si, :, pair_id].argmax())
        r0 = float(r_cube[si, gi, pair_id])
        p0 = float(p_cube[si, gi, pair_id])
        rg = torch.linspace(max(0.005, r0 - 2 * dr),
                            min(0.995, r0 + 2 * dr),
                            args.refine_grid, device=dev)
        pg = torch.linspace(p0 - 2 * dphi, p0 + 2 * dphi,
                            args.refine_grid, device=dev)
        ms = rescale_lg(modes, sc)
        sig = sigmas[gi]
        Ie_b = gaussian_blur(Ie.unsqueeze(0), sig)[0]
        kc, kr, kp = scan_pairs_lg(ms, Ie_b,
                                   pairs[pair_id:pair_id + 1], rg, pg, sig)
        rr = float(kr[0])
        pp = float(kp[0] % (2 * np.pi))
        # recompute across sigmas at refined (r, phi)
        corrs = []
        rt = torch.tensor([rr], device=dev)
        pt = torch.tensor([pp], device=dev)
        for sig2 in sigmas:
            Ie_b2 = gaussian_blur(Ie.unsqueeze(0), sig2)[0]
            cv, _, _ = scan_pairs_lg(ms, Ie_b2,
                                     pairs[pair_id:pair_id + 1], rt, pt,
                                     sig2)
            corrs.append(float(cv[0]))
        corrs = torch.tensor(corrs)
        ref_corr_avg.append(float(corrs.mean()))
        ref_corr_best.append(float(corrs.max()))
        ref_sig.append(sigmas[int(corrs.argmax())])
        ref_r.append(rr)
        ref_phi.append(pp)
        ref_scale.append(sc)
        ref_pair.append(pair_id)

    ref_order = np.argsort(ref_corr_avg)[::-1]

    best_pair_id = ref_pair[ref_order[0]]
    best_scale = ref_scale[ref_order[0]]
    best_sig = ref_sig[ref_order[0]]
    best_r = ref_r[ref_order[0]]
    best_phi = ref_phi[ref_order[0]]

    m1, n1 = idx[pairs[best_pair_id][0].item()]
    m2, n2 = idx[pairs[best_pair_id][1].item()]

    rows = []
    for rank, oi in enumerate(ref_order, start=1):
        pid = ref_pair[oi]
        i, j = pairs[pid].tolist()
        p1, l1 = idx[i]
        p2, l2 = idx[j]
        r = ref_r[oi]
        rows.append({
            'rank': rank,
            'mode1': f'LG({p1},{l1})',
            'mode2': f'LG({p2},{l2})',
            'r_energy1': r,
            'r_energy2': 1 - r,
            'amp1': float(np.sqrt(r)),
            'amp2': float(np.sqrt(1 - r)),
            'phi_rad': ref_phi[oi],
            'scale': ref_scale[oi],
            'sigma_best': ref_sig[oi],
            'corr_avg': ref_corr_avg[oi],
            'corr_best': ref_corr_best[oi],
        })
    df = pd.DataFrame(rows)

    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / f'{stem}_top.csv', index=False, encoding='utf-8-sig')

    ms_np = rescale_lg(modes, best_scale).detach().cpu().numpy()
    It = render_lg(ms_np,
                   pairs[best_pair_id][0].item(),
                   pairs[best_pair_id][1].item(),
                   best_r, best_phi, sigma=best_sig)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    axs[0].imshow(Ie_np, cmap='inferno'); axs[0].set_title('Experiment')
    axs[1].imshow(It, cmap='inferno')
    axs[1].set_title(f'LG({p1},{l1})+LG({p2},{l2})  '
                     f's={best_scale:.2f}, sig={best_sig:.0f}\n'
                     f'r={best_r:.3f}, phi={best_phi:.3f}, '
                     f'corr_avg={ref_corr_avg[ref_order[0]]:.4f}')
    axs[2].imshow(np.abs(Ie_np - It), cmap='viridis')
    axs[2].set_title('|diff|')
    for a in axs:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(out_root / f'{stem}_viz.png', dpi=130)
    plt.close(fig)

    summary = {
        'image': Path(img_path).name,
        'basis': 'LG',
        'target_vortex': target_vor,
        'best_pair': f'LG({p1},{l1}) + LG({p2},{l2})',
        'energy_ratio': [best_r, 1 - best_r],
        'amplitudes': [float(np.sqrt(best_r)), float(np.sqrt(1 - best_r))],
        'phi_rad': best_phi,
        'scale': best_scale,
        'sigma': best_sig,
        'corr_avg': ref_corr_avg[ref_order[0]],
        'corr_best': ref_corr_best[ref_order[0]],
    }
    return summary, df


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------

def synth_test(idx, modes, args):
    dev = modes.device
    rng = np.random.default_rng(7)
    N = modes.shape[0]
    pairs = torch.tensor([(a, b) for a in range(N) for b in range(a + 1, N)],
                         dtype=torch.long, device=dev)
    rg = torch.linspace(0.01, 0.99, args.grid_r, device=dev)
    pg = torch.linspace(0.0, 2 * np.pi, args.grid_phi + 1, device=dev)[:-1]
    ms_np = modes.detach().cpu().numpy()

    print('=== LG SYNTHETIC SELF-TEST (s=1, no blur) ===')
    hits = 0
    cases = [(0, 4, 0, 0, 0.5, 0.7)]
    for _ in range(3):
        i, j = sorted(rng.choice(N, size=2, replace=False).tolist())
        cases.append((i, j, None, None, float(rng.uniform(0.2, 0.8)),
                      float(rng.uniform(0, 2 * np.pi))))
    for (i, j, _a, _b, r, phi) in cases:
        It = render_lg(ms_np, i, j, r, phi)
        noisy = np.clip(It + rng.normal(0, 0.01, It.shape), 0, 1)
        Ie = torch.tensor(noisy, dtype=torch.float32, device=dev)
        corr, rb, pb = scan_pairs_lg(modes, Ie, pairs, rg, pg, 0.0)
        k = int(torch.argmax(corr))
        bi, bj = pairs[k].tolist()
        # (l1,l2) and (-l1,-l2) are complex-conjugate twins: identical I
        same = (bi, bj) == (i, j)
        mirror = (idx[bi] == (idx[i][0], -idx[i][1])
                  and idx[bj] == (idx[j][0], -idx[j][1]))
        ok = same or mirror
        hits += int(ok)
        tag = 'OK' if same else ('OK(mirror-equiv)' if mirror else 'MISS')
        print(f'  true LG{idx[i]}+LG{idx[j]} r={r:.3f} phi={phi:.3f} | '
              f'found LG{idx[bi]}+LG{idx[bj]} r={float(rb[k]):.3f} '
              f'phi={float(pb[k]):.3f} corr={float(corr[k]):.4f} {tag}')
    print(f'  pair recovery: {hits}/{len(cases)}')
    return hits == len(cases)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device == 'cuda':
        print('[warn] CUDA unavailable, fallback CPU')
        args.device = 'cpu'

    modes, idx = build_lg_modes(args.max_order, 224, 4.0, args.device)
    print(f'[engine] LG basis: {len(idx)} modes, '
          f'pairs={len(idx) * (len(idx) - 1) // 2}, device={args.device}')

    if args.synth:
        ok = synth_test(idx, modes, args)
        print('[synth] all passed' if ok else '[synth] SOME FAILED')
        return

    if args.image:
        images = [Path(args.image)]
    else:
        d = Path(args.images_dir)
        images = sorted(p for p in d.iterdir()
                        if p.suffix.lower() in IMG_EXTS)
    out_root = Path(args.out_dir)
    print(f'[input] {len(images)} image(s)')

    lines = ['# Two-mode LG analytical search summary\n']
    for img in images:
        print(f'\n===== {img.name} =====')
        summary, df = decompose_image(img, idx, modes, args, out_root)
        print(f"  best: {summary['best_pair']} | s={summary['scale']:.2f} "
              f"sig={summary['sigma']:.0f} | "
              f"r={summary['energy_ratio'][0]:.3f}/"
              f"{summary['energy_ratio'][1]:.3f} | "
              f"phi={summary['phi_rad']:.4f} | "
              f"corr_avg={summary['corr_avg']:.4f} "
              f"(best {summary['corr_best']:.4f}) | "
              f"target_vortex={summary['target_vortex']}")
        for _, row in df.head(3).iterrows():
            print(f"    {row['mode1']:>10s} + {row['mode2']:<10s} "
                  f"r={row['r_energy1']:.3f} phi={row['phi_rad']:.3f} "
                  f"s={row['scale']:.2f} sig={row['sigma_best']:.0f} "
                  f"cavg={row['corr_avg']:.4f} cbest={row['corr_best']:.4f}")
        lines.append(f"\n## {summary['image']}\n")
        lines.append(f"- best pair: **{summary['best_pair']}**, "
                     f"r={summary['energy_ratio'][0]:.4f}/"
                     f"{summary['energy_ratio'][1]:.4f}, "
                     f"phi={summary['phi_rad']:.4f}, "
                     f"scale={summary['scale']:.2f}, "
                     f"sigma={summary['sigma']:.0f}, "
                     f"corr_avg=**{summary['corr_avg']:.4f}**, "
                     f"corr_best={summary['corr_best']:.4f}")
        lines.append('\n| rank | pair | r1 | r2 | amp1 | amp2 | '
                     'phi(rad) | s | sig | corr_avg | corr_best |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for _, row in df.iterrows():
            lines.append(
                f"| {row['rank']} | {row['mode1']}+{row['mode2']} | "
                f"{row['r_energy1']:.4f} | {row['r_energy2']:.4f} | "
                f"{row['amp1']:.4f} | {row['amp2']:.4f} | "
                f"{row['phi_rad']:.4f} | {row['scale']:.2f} | "
                f"{row['sigma_best']:.0f} | {row['corr_avg']:.4f} | "
                f"{row['corr_best']:.4f} |")

    # strip emoji-free plain md
    (out_root / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n[done] outputs in {out_root}')


if __name__ == '__main__':
    main()

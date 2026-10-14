# -*- coding: utf-8 -*-
"""
search_two_mode.py -- Two-mode analytical parameter-scan decomposition
=======================================================================

Physical model (two coherent HG modes, waist plane):

    I_t = |a1*G1 + a2*G2*exp(i*phi)|^2
        = r*G1^2 + (1-r)*G2^2 + 2*sqrt(r*(1-r))*c*G1*G2

    a1 = sqrt(r), a2 = sqrt(1-r)      (a1^2 + a2^2 = 1, auto-normalized)
    c  = cos(dphi)                    (intensity depends only on cos)

Search space (all on GPU, analytical, no per-sample generation):
    mode pair (i<j, m+n<=10, K=66 -> 2145 pairs)
    x energy ratio r in (0,1)  x cos(phi) in [-1,1]
    x zoom scale s (fill-ratio mismatch correction)
    x blur sigma (resolution mismatch correction)

Key trick: both experimental and theoretical images are low-passed by
the SAME gaussian sigma before comparing (fair comparison at their
common resolvable scale).  With B1=blur(G1^2), B2=blur(G2^2),
Bc=blur(G1*G2), and Ie_b = blur(Ie, sigma), Pearson corr between
I_t = r*B1+(1-r)*B2+2*s*c*Bc and Ie_b reduces to precomputed moments:

    mt   = r*n1 + (1-r)*n2 + 2*s*c*nc
    Ete  = r*e1 + (1-r)*e2 + 2*s*c*ec
    Et2  = r^2*q1 + (1-r)^2*q2 + 2*r*(1-r)*q12 + 4*r*(1-r)*c^2*qc
           + 4*r*s*c*q1c + 4*(1-r)*s*c*q2c
    corr = (Ete - mt*<Ie>) / sqrt((Et2 - mt^2) * var(Ie))

Outputs (per image): top-K table CSV, three-panel comparison PNG.
Run `--synth` for a self-test on synthetic two-mode images.
"""

import argparse
import json
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from preprocess import ExperimentalImageProcessor
from hg_torch import TorchHGEngine

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def parse_args():
    p = argparse.ArgumentParser(
        description='Two-mode (m+n<=10) analytical parameter search')
    p.add_argument('--image', type=str, default=None,
                   help='single experimental image path')
    p.add_argument('--images_dir', type=str,
                   default=os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images')),
                   help='directory containing experimental images')
    p.add_argument('--out_dir', type=str, default='two_mode_search')
    p.add_argument('--max_order', type=int, default=10)
    p.add_argument('--grid', type=int, default=201,
                   help='coarse grid points per axis (r and cos_phi)')
    p.add_argument('--refine_k', type=int, default=12,
                   help='number of top candidates to refine locally')
    p.add_argument('--refine_grid', type=int, default=101)
    p.add_argument('--scales', type=str, default='0.75,1.0,1.25,1.5,1.75,2.0',
                   help='mode-field zoom factors (fill-ratio correction)')
    p.add_argument('--sigmas', type=str, default='0,1,2,3,4,5,6',
                   help='common low-pass gaussian sigma (px) to scan')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--cmap', type=str, default='nipy_spectral',
                   help="pseudocolor to intensity inversion "
                        "('nipy_spectral' | 'viridis' | 'none'=luma)")
    p.add_argument('--preprocess', type=str, default='isot',
                   choices=['isot', 'crop'],
                   help="experimental image preprocessing: 'isot' "
                        "(isotropic, no hard edges) | 'crop' (legacy)")
    p.add_argument('--synth', action='store_true',
                   help='self-test on synthetic two-mode images')
    p.add_argument('--synth_rounds', type=int, default=5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def invert_colormap(rgb_img, cmap_name='nipy_spectral', n=256):
    """Invert a pseudocolor image to intensity via nearest-color lookup."""
    import matplotlib
    cmap = matplotlib.colormaps[cmap_name]
    lut = (np.array([cmap(i / (n - 1))[:3] for i in range(n)]) * 255.0)
    H, W = rgb_img.shape[:2]
    flat = rgb_img.reshape(-1, 3).astype(np.float32)
    idx = np.empty(flat.shape[0], dtype=np.int32)
    chunk = 200000
    for b0 in range(0, flat.shape[0], chunk):
        f = flat[b0:b0 + chunk]
        d2 = ((f[:, None, :] - lut[None, :, :]) ** 2).sum(-1)
        idx[b0:b0 + chunk] = d2.argmin(axis=1)
    return (idx.reshape(H, W) / (n - 1)).astype(np.float64)


def isot_process(I, target_r=78.0):
    """Baseline-subtract + centroid + ISOTROPIC scale to fixed R95 radius.

    Replaces the legacy crop pipeline (bbox + anisotropic stretch +
    hard threshold edges), which distorted aspect ratio and created
    hard boundaries the theoretical images do not have -- the main
    cause of the correlation ceiling (~0.87-0.90) and the sigma->4
    boundary runaway.
    """
    I = I - np.percentile(I, 5)
    I = np.clip(I, 0, None)
    tot = I.sum()
    if tot < 1e-12:
        return np.zeros((224, 224), dtype=np.float64)
    H, W = I.shape
    ys, xs = np.mgrid[0:H, 0:W]
    cx = (xs * I).sum() / tot
    cy = (ys * I).sum() / tot
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    order = np.argsort(r.ravel())
    w = I.ravel()[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    r_sorted = r.ravel()[order]
    R95 = float(np.interp(0.95, cw, r_sorted))
    scale = target_r / max(R95, 1e-6)
    tt = torch.tensor(I, dtype=torch.float32)[None, None]
    new_size = max(8, int(round(H * scale)))
    big = F.interpolate(tt, size=(new_size, new_size), mode='bilinear',
                        align_corners=False)[0, 0].numpy()
    out = np.zeros((224, 224), dtype=np.float64)
    cy2 = new_size / 2.0 + (cy - H / 2.0) * scale
    cx2 = new_size / 2.0 + (cx - W / 2.0) * scale
    oy = int(round(112 - cy2))
    ox = int(round(112 - cx2))
    ys0, xs0 = max(0, oy), max(0, ox)
    ye, xe = min(224, oy + new_size), min(224, ox + new_size)
    by0, bx0 = ys0 - oy, xs0 - ox
    out[ys0:ye, xs0:xe] = big[by0:by0 + (ye - ys0), bx0:bx0 + (xe - xs0)]
    m = out.max()
    return out / m if m > 1e-12 else out


def load_experiment(img_path, cmap_mode='nipy_spectral',
                    preprocess='isot') -> np.ndarray:
    if cmap_mode and cmap_mode.lower() not in ('none', 'luma'):
        rgb = np.array(Image.open(str(img_path)).convert('RGB'))
        inten = invert_colormap(rgb, cmap_mode)
    else:
        inten = np.array(Image.open(str(img_path)).convert('L'),
                         dtype=np.float64)
    if preprocess == 'crop':
        proc = ExperimentalImageProcessor()
        return proc.process(inten)['image']  # (224,224) float64 [0,1]
    return isot_process(inten)


def rescale_modes(modes, scale):
    """Zoom mode fields by `scale` (center crop / zero-pad back to 224)."""
    if abs(scale - 1.0) < 1e-6:
        return modes
    G = modes.unsqueeze(0)
    if scale > 1.0:
        big = F.interpolate(G, scale_factor=scale, mode='bilinear',
                            align_corners=False)[0]
        H = big.shape[-1]
        c0 = (H - modes.shape[-1]) // 2
        return big[:, c0:c0 + modes.shape[-1], c0:c0 + modes.shape[-1]]
    small = F.interpolate(G, scale_factor=scale, mode='bilinear',
                          align_corners=False)[0]
    H = small.shape[-1]
    out = torch.zeros(modes.shape[0], modes.shape[-1], modes.shape[-1],
                      device=modes.device, dtype=modes.dtype)
    c0 = (modes.shape[-1] - H) // 2
    out[:, c0:c0 + H, c0:c0 + H] = small
    return out


def _gaussian_kernel1d(sigma, device):
    radius = int(3.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-x * x / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(x, sigma):
    """Separable gaussian blur. x: (B,H,W). sigma<=0 -> identity."""
    if sigma <= 0:
        return x
    k = _gaussian_kernel1d(sigma, x.device)
    r = (k.numel() - 1) // 2
    x4 = x.unsqueeze(1)
    w_h = k.view(1, 1, 1, -1)
    w_v = k.view(1, 1, -1, 1)
    x4 = F.conv2d(x4, w_h, padding=(0, r))
    x4 = F.conv2d(x4, w_v, padding=(r, 0))
    return x4.squeeze(1)


# ---------------------------------------------------------------------------
# Core: analytical scan
# ---------------------------------------------------------------------------

def scan_pairs(modes, bg2, Ie_b, pairs_ij, r_grid, c_grid, sigma,
               batch=256):
    """Analytical Pearson-corr scan over (r, cos_phi) for each pair.

    Args:
        modes:  (K,224,224) sharp mode fields (for Bc = blur(G1*G2))
        bg2:    (K,224,224) precomputed blur(G_i^2, sigma)
        Ie_b:   (224,224) experimental image, blurred by the same sigma
    Returns corr (P,), r_best (P,), c_best (P,).
    """
    dev = modes.device
    Ie_b = Ie_b.to(dev)
    me = Ie_b.mean()
    ve = Ie_b.var(unbiased=False).clamp_min(1e-12)

    R = r_grid.numel()
    C = c_grid.numel()
    r = r_grid.view(1, -1, 1)
    c = c_grid.view(1, 1, -1)
    s = torch.sqrt((r * (1 - r)).clamp_min(0))

    P = pairs_ij.shape[0]
    corr_out = torch.empty(P, device=dev)
    r_out = torch.empty(P, device=dev)
    c_out = torch.empty(P, device=dev)

    for b0 in range(0, P, batch):
        ij = pairs_ij[b0:b0 + batch]
        G1 = modes[ij[:, 0]]
        G2 = modes[ij[:, 1]]
        B1 = bg2[ij[:, 0]]
        B2 = bg2[ij[:, 1]]
        Bc = gaussian_blur(G1 * G2, sigma)

        n1 = B1.mean(dim=(1, 2)).view(-1, 1, 1)
        n2 = B2.mean(dim=(1, 2)).view(-1, 1, 1)
        nc = Bc.mean(dim=(1, 2)).view(-1, 1, 1)
        q1 = (B1 * B1).mean(dim=(1, 2)).view(-1, 1, 1)
        q2 = (B2 * B2).mean(dim=(1, 2)).view(-1, 1, 1)
        qc = (Bc * Bc).mean(dim=(1, 2)).view(-1, 1, 1)
        q12 = (B1 * B2).mean(dim=(1, 2)).view(-1, 1, 1)
        q1c = (B1 * Bc).mean(dim=(1, 2)).view(-1, 1, 1)
        q2c = (B2 * Bc).mean(dim=(1, 2)).view(-1, 1, 1)
        e1 = (B1 * Ie_b).mean(dim=(1, 2)).view(-1, 1, 1)
        e2 = (B2 * Ie_b).mean(dim=(1, 2)).view(-1, 1, 1)
        ec = (Bc * Ie_b).mean(dim=(1, 2)).view(-1, 1, 1)

        mt = r * n1 + (1 - r) * n2 + 2 * s * c * nc
        Ete = r * e1 + (1 - r) * e2 + 2 * s * c * ec
        Et2 = ((r * r) * q1 + ((1 - r) ** 2) * q2
               + 2 * r * (1 - r) * q12
               + 4 * r * (1 - r) * c * c * qc
               + 4 * r * s * c * q1c
               + 4 * (1 - r) * s * c * q2c)

        cov = Ete - mt * me
        vart = (Et2 - mt * mt).clamp_min(1e-12)
        corr = cov / torch.sqrt(vart * ve)  # (B, R, C)

        B = corr.shape[0]
        flat = corr.reshape(B, -1)
        mx, amax = flat.max(dim=1)
        ri = amax // C
        ci = amax % C

        corr_out[b0:b0 + B] = mx
        r_out[b0:b0 + B] = r_grid[ri]
        c_out[b0:b0 + B] = c_grid[ci]

    return corr_out, r_out, c_out


def single_mode_check(bg2, Ie_b):
    """Correlation of each blurred single-mode intensity with the image."""
    dev = bg2.device
    Ie_b = Ie_b.to(dev)
    me = Ie_b.mean()
    ve = Ie_b.var(unbiased=False).clamp_min(1e-12)
    best = []
    for i in range(bg2.shape[0]):
        B = bg2[i]
        m = B.mean()
        v = (B * B).mean() - m * m
        cov = (B * Ie_b).mean() - m * me
        best.append(float(cov / torch.sqrt(v.clamp_min(1e-12) * ve)))
    return np.array(best)


def render_two_mode(modes, i, j, r, c, sigma=0.0):
    G1 = modes[i]
    G2 = modes[j]
    It = r * G1 * G1 + (1 - r) * G2 * G2 \
        + 2 * np.sqrt(max(r * (1 - r), 0)) * c * G1 * G2
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

def prepare_scenario(modes, scale, sigma):
    """Rescale + per-mode blurred-square fields for one (s, sigma)."""
    ms = rescale_modes(modes, scale)
    bg2 = gaussian_blur(ms * ms, sigma)  # (K,224,224)
    return ms, bg2


def count_vortices(E, charge_thresh=0.5, amp_floor_ratio=5e-3):
    """Count phase vortices (singularities) via loop charge on the
    complex field E.  Plaquettes touching near-zero amplitude (outside
    the beam) are excluded to avoid phase-noise false positives.
    Returns (n_positive, n_negative)."""
    A = np.abs(E)
    ph = np.angle(E)
    ok = A > amp_floor_ratio * A.max()

    def wrap(d):
        return (d + np.pi) % (2 * np.pi) - np.pi

    d1 = wrap(ph[:-1, 1:] - ph[:-1, :-1])
    d2 = wrap(ph[1:, 1:] - ph[:-1, 1:])
    d3 = wrap(ph[1:, :-1] - ph[1:, 1:])
    d4 = wrap(ph[:-1, :-1] - ph[1:, :-1])
    charge = (d1 + d2 + d3 + d4) / (2 * np.pi)
    plaquette_ok = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    n_pos = int(((charge > charge_thresh) & plaquette_ok).sum())
    n_neg = int(((charge < -charge_thresh) & plaquette_ok).sum())
    return n_pos, n_neg


def pair_vortex_count(modes_np, i, j, r, c):
    """Total vortex count of the two-mode complex field at (r, cos_phi)."""
    G1 = modes_np[i]
    G2 = modes_np[j]
    phi = np.arccos(np.clip(c, -1, 1))
    E = np.sqrt(r) * G1 + np.sqrt(1 - r) * G2 * np.exp(1j * phi)
    n_pos, n_neg = count_vortices(E)
    return n_pos + n_neg


def decompose_image(img_path, engine, modes, modes_np, args, out_root):
    stem = Path(img_path).stem
    m_tv = re.search(r'(\d+)\s*Singularit', stem)
    target_vor = int(m_tv.group(1)) if m_tv else None
    Ie_np = load_experiment(img_path, args.cmap, args.preprocess)
    Ie = torch.tensor(Ie_np, dtype=torch.float32)
    dev = modes.device
    Ie = Ie.to(dev)

    N = modes.shape[0]
    pairs = torch.tensor([(i, j) for i in range(N) for j in range(i + 1, N)],
                         dtype=torch.long, device=dev)
    P = pairs.shape[0]
    scales = [float(s) for s in args.scales.split(',')]
    sigmas = [float(s) for s in args.sigmas.split(',')]
    S, G = len(scales), len(sigmas)

    r_grid = torch.linspace(0.005, 0.995, args.grid, device=dev)
    c_grid = torch.linspace(-1.0, 1.0, args.grid, device=dev)

    # ---- coarse scan over the full (scale, sigma, pair) grid ----
    corr_cube = torch.full((S, G, P), -2.0, device=dev)
    r_cube = torch.zeros((S, G, P), device=dev)
    c_cube = torch.zeros((S, G, P), device=dev)
    for si, s in enumerate(scales):
        ms = rescale_modes(modes, s)
        for gi, sig in enumerate(sigmas):
            bg2 = gaussian_blur(ms * ms, sig)
            Ie_b = gaussian_blur(Ie.unsqueeze(0), sig)[0]
            corr, rb, cb = scan_pairs(ms, bg2, Ie_b, pairs, r_grid, c_grid,
                                      sig)
            corr_cube[si, gi] = corr
            r_cube[si, gi] = rb
            c_cube[si, gi] = cb

    # ranking score: MEAN corr across blur levels. A genuine match stays
    # high at every sigma; a lucky smooth false-positive only peaks at
    # one sigma (e.g. a donut-shaped low-order blob under heavy blur).
    score = corr_cube.mean(dim=1)  # (S, P)
    top_vals, top_idx = score.reshape(-1).topk(args.refine_k)

    dr = 2.0 / (args.grid - 1)
    dc = 2.0 / (args.grid - 1)
    ref_corr_avg, ref_corr_best, ref_r, ref_c, ref_scale, ref_sig, \
        ref_pair, ref_nvor = [], [], [], [], [], [], [], []
    for idx in top_idx.tolist():
        si = idx // P
        pair_id = idx % P
        s = scales[si]
        gi = int(corr_cube[si, :, pair_id].argmax())
        r0 = float(r_cube[si, gi, pair_id])
        c0 = float(c_cube[si, gi, pair_id])
        rg2 = torch.linspace(max(0.001, r0 - dr), min(0.999, r0 + dr),
                             args.refine_grid, device=dev)
        cg2 = torch.linspace(max(-1.0, c0 - dc), min(1.0, c0 + dc),
                             args.refine_grid, device=dev)
        ms = rescale_modes(modes, s)
        sig = sigmas[gi]
        bg2 = gaussian_blur(ms * ms, sig)
        Ie_b = gaussian_blur(Ie.unsqueeze(0), sig)[0]
        k_corr, k_r, k_c = scan_pairs(ms, bg2, Ie_b,
                                      pairs[pair_id:pair_id + 1], rg2, cg2,
                                      sig)
        rr = float(k_r[0])
        cc = float(k_c[0].clamp(-1.0, 1.0))
        # vortex count of the candidate field (topology, scale-invariant)
        ms_np_c = ms.detach().cpu().numpy()
        n_vor = pair_vortex_count(ms_np_c, pairs[pair_id][0].item(),
                                  pairs[pair_id][1].item(), rr, cc)
        # recompute at refined (r, c) across all sigmas
        corrs = []
        rt = torch.tensor([rr], device=dev)
        ct = torch.tensor([cc], device=dev)
        for sig2 in sigmas:
            bg2b = gaussian_blur(ms * ms, sig2)
            Ie_b2 = gaussian_blur(Ie.unsqueeze(0), sig2)[0]
            cv, _, _ = scan_pairs(ms, bg2b, Ie_b2,
                                  pairs[pair_id:pair_id + 1], rt, ct, sig2)
            corrs.append(float(cv[0]))
        corrs = torch.tensor(corrs)
        ref_corr_avg.append(float(corrs.mean()))
        ref_corr_best.append(float(corrs.max()))
        ref_sig.append(sigmas[int(corrs.argmax())])
        ref_r.append(rr)
        ref_c.append(cc)
        ref_scale.append(s)
        ref_pair.append(pair_id)
        ref_nvor.append(n_vor)

    ref_order = np.argsort(ref_corr_avg)[::-1]

    best_pair_id = ref_pair[ref_order[0]]
    best_scale = ref_scale[ref_order[0]]
    best_sig = ref_sig[ref_order[0]]
    best_r = ref_r[ref_order[0]]
    best_c = float(np.clip(ref_c[ref_order[0]], -1, 1))

    ms_best, bg2_best = prepare_scenario(modes, best_scale, best_sig)
    Ie_best = gaussian_blur(Ie.unsqueeze(0), best_sig)[0]
    sm_corr = single_mode_check(bg2_best, Ie_best)
    sm_best = int(np.argmax(sm_corr))

    rows = []
    for rank, oi in enumerate(ref_order, start=1):
        pid = ref_pair[oi]
        i, j = pairs[pid].tolist()
        m1, n1 = engine.mode_indices[i]
        m2, n2 = engine.mode_indices[j]
        r = ref_r[oi]
        c = float(np.clip(ref_c[oi], -1, 1))
        phi = float(np.arccos(c))
        rows.append({
            'rank': rank,
            'mode1': f'HG({m1},{n1})',
            'mode2': f'HG({m2},{n2})',
            'r_energy1': r,
            'r_energy2': 1 - r,
            'amp1': float(np.sqrt(r)),
            'amp2': float(np.sqrt(1 - r)),
            'cos_phi': c,
            'phi_rad': phi,
            'phi_deg': float(np.degrees(phi)),
            'scale': ref_scale[oi],
            'sigma_best': ref_sig[oi],
            'n_vortex': ref_nvor[oi],
            'corr_avg': ref_corr_avg[oi],
            'corr_best': ref_corr_best[oi],
        })
    df = pd.DataFrame(rows)

    m1, n1 = engine.mode_indices[pairs[best_pair_id][0].item()]
    m2, n2 = engine.mode_indices[pairs[best_pair_id][1].item()]

    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / f'{stem}_top.csv', index=False, encoding='utf-8-sig')

    # visualization: experiment | model (at best sigma) | pixel diff
    ms_np = ms_best.detach().cpu().numpy()
    It = render_two_mode(ms_np,
                         pairs[best_pair_id][0].item(),
                         pairs[best_pair_id][1].item(),
                         best_r, best_c, sigma=best_sig)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    axs[0].imshow(Ie_np, cmap='inferno'); axs[0].set_title('Experiment')
    axs[1].imshow(It, cmap='inferno')
    axs[1].set_title(f'HG({m1},{n1})+HG({m2},{n2})  '
                     f's={best_scale:.2f}, sig={best_sig:.0f}\n'
                     f'r={best_r:.3f}, cos={best_c:.3f}, '
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
        'cmap': args.cmap,
        'preprocess': args.preprocess,
        'target_vortex': target_vor,
        'best_pair': f'HG({m1},{n1}) + HG({m2},{n2})',
        'energy_ratio': [best_r, 1 - best_r],
        'amplitudes': [float(np.sqrt(best_r)), float(np.sqrt(1 - best_r))],
        'cos_phi': best_c,
        'phi_rad': float(np.arccos(best_c)),
        'scale': best_scale,
        'sigma': best_sig,
        'corr_avg': ref_corr_avg[ref_order[0]],
        'corr_best': ref_corr_best[ref_order[0]],
        'corr_single_mode_best': float(sm_corr[sm_best]),
        'single_mode_best': str(engine.mode_indices[sm_best]),
    }
    return summary, df


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------

def synth_test(engine, modes, modes_np, args):
    rng = np.random.default_rng(42)
    N = modes.shape[0]
    dev = modes.device
    print('=== SYNTHETIC SELF-TEST (s=1, no blur) ===')
    hits = 0
    pairs = torch.tensor(
        [(a, b) for a in range(N) for b in range(a + 1, N)],
        dtype=torch.long, device=dev)
    bg2 = modes * modes
    rg = torch.linspace(0.005, 0.995, args.grid, device=dev)
    cg = torch.linspace(-1.0, 1.0, args.grid, device=dev)

    for k in range(args.synth_rounds):
        i, j = sorted(rng.choice(N, size=2, replace=False).tolist())
        r = float(rng.uniform(0.15, 0.85))
        c = float(rng.uniform(-1.0, 1.0))
        It = render_two_mode(modes_np, i, j, r, c)
        noisy = np.clip(It + rng.normal(0, 0.01, It.shape), 0, 1)

        Ie = torch.tensor(noisy, dtype=torch.float32, device=dev)
        corr, rb, cb = scan_pairs(modes, bg2, Ie, pairs, rg, cg, 0.0)
        idx = int(torch.argmax(corr))
        bi, bj = pairs[idx].tolist()
        ok = (bi, bj) == (i, j)
        hits += int(ok)
        m1, n1 = engine.mode_indices[i]
        m2, n2 = engine.mode_indices[j]
        bm1, bn1 = engine.mode_indices[bi]
        bm2, bn2 = engine.mode_indices[bj]
        print(f'  [{k + 1}] truth HG({m1},{n1})+HG({m2},{n2}) '
              f'r={r:.3f} c={c:+.3f}  |  found '
              f'HG({bm1},{bn1})+HG({bm2},{bn2}) '
              f'r={float(rb[idx]):.3f} c={float(cb[idx]):+.3f} '
              f'corr={float(corr[idx]):.4f}  {"OK" if ok else "MISS"}')
    print(f'  pair recovery: {hits}/{args.synth_rounds}')
    return hits == args.synth_rounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device == 'cuda':
        print('[warn] CUDA not available, falling back to CPU')
        args.device = 'cpu'

    engine = TorchHGEngine(max_order_N=args.max_order, grid_size=224,
                           device=args.device)
    modes = engine.hg_real  # (K, 224, 224) float32
    modes_np = modes.detach().cpu().numpy()
    print(f'[engine] {engine.num_modes} modes, max order N={args.max_order}, '
          f'pairs={engine.num_modes * (engine.num_modes - 1) // 2}, '
          f'device={args.device}')

    if args.synth:
        ok = synth_test(engine, modes, modes_np, args)
        print('[synth] all passed' if ok else '[synth] SOME FAILED')
        return

    if args.image:
        images = [Path(args.image)]
    else:
        d = Path(args.images_dir)
        images = sorted(p for p in d.iterdir()
                        if p.suffix.lower() in IMG_EXTS)
    if not images:
        print('[error] no images found')
        return

    out_root = Path(args.out_dir)
    print(f'[input] {len(images)} image(s)')

    lines = ['# Two-mode analytical search summary\n']
    for img in images:
        print(f'\n===== {img.name} =====')
        summary, df = decompose_image(img, engine, modes, modes_np,
                                      args, out_root)
        print(f"  best: {summary['best_pair']} | s={summary['scale']:.2f} "
              f"sig={summary['sigma']:.0f} | "
              f"r={summary['energy_ratio'][0]:.3f}/"
              f"{summary['energy_ratio'][1]:.3f} | "
              f"phi={summary['phi_rad']:.4f} rad | "
              f"corr_avg={summary['corr_avg']:.4f} "
              f"(best {summary['corr_best']:.4f}) | "
              f"target_vortex={summary['target_vortex']}")
        print(f"  top-3:")
        for _, row in df.head(3).iterrows():
            print(f"    {row['mode1']:>11s} + {row['mode2']:<11s} "
                  f"r={row['r_energy1']:.3f} "
                  f"phi={row['phi_rad']:.3f} "
                  f"s={row['scale']:.2f} sig={row['sigma_best']:.0f} "
                  f"nvor={row['n_vortex']:>3d} "
                  f"cavg={row['corr_avg']:.4f} cbest={row['corr_best']:.4f}")
        lines.append(f"\n## {summary['image']}\n")
        lines.append(f"- best pair: **{summary['best_pair']}**, "
                     f"r={summary['energy_ratio'][0]:.4f}/"
                     f"{summary['energy_ratio'][1]:.4f}, "
                     f"cos_phi={summary['cos_phi']:.4f}, "
                     f"scale={summary['scale']:.2f}, "
                     f"sigma={summary['sigma']:.0f}, "
                     f"corr_avg=**{summary['corr_avg']:.4f}**, "
                     f"corr_best={summary['corr_best']:.4f}")
        lines.append(f"- best single mode: "
                     f"{summary['single_mode_best']} "
                     f"(corr={summary['corr_single_mode_best']:.4f})")
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

    (out_root / 'summary.md').write_text('\n'.join(lines),
                                         encoding='utf-8')
    print(f'\n[done] outputs in {out_root}')


if __name__ == '__main__':
    main()

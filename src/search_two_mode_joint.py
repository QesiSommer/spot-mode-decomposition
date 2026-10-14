# -*- coding: utf-8 -*-
"""
search_two_mode_joint.py -- JOINT HG+LG two-mode analytical search
===================================================================

Basis: 66 LG modes (2p+|l|<=10) + 66 HG modes (m+n<=10) = 132 modes.
All C(132,2) = 8646 mode pairs are scanned (including mixed HG-LG
pairs), so the model automatically covers pure-HG, pure-LG and
intermediate configurations.

All fields are treated as complex (HG fields have zero imaginary
part).  For a pair:

    E  = sqrt(r)*F1 + sqrt(1-r)*F2*exp(i*phi)
    I  = r*A + (1-r)*B + 2*s*(cos(phi)*C1 - sin(phi)*C2)
    A=|F1|^2, B=|F2|^2, C1=Re[F1*conj(F2)], C2=Im[F1*conj(F2)]

Search space: (pair, r, phi, zoom scale, blur sigma), all analytical
and vectorized on GPU.  Ranking score = mean Pearson corr across blur
levels (robust against smooth false positives).

NOTE: (l1,l2) and (-l1,-l2) are complex-conjugate twins producing
identical intensities -- both may rank; they are physically the same.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import (build_lg_modes, rescale_lg,
                                scan_pairs_lg, render_lg)
from hg_torch import TorchHGEngine

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def parse_args():
    p = argparse.ArgumentParser(description='Joint HG+LG two-mode search')
    p.add_argument('--image', type=str, default=None)
    p.add_argument('--images_dir', type=str,
                   default=os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images')))
    p.add_argument('--out_dir', type=str, default='two_mode_search_joint')
    p.add_argument('--max_order', type=int, default=10)
    p.add_argument('--grid_r', type=int, default=81)
    p.add_argument('--grid_phi', type=int, default=180)
    p.add_argument('--refine_k', type=int, default=12)
    p.add_argument('--refine_grid', type=int, default=101)
    p.add_argument('--scales', type=str, default='1.0,1.25,1.5,1.75,2.0')
    p.add_argument('--sigmas', type=str, default='0,2,4')
    p.add_argument('--batch', type=int, default=384)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--cmap', type=str, default='nipy_spectral')
    p.add_argument('--preprocess', type=str, default='isot',
                   choices=['isot', 'crop'])
    p.add_argument('--synth', action='store_true')
    return p.parse_args()


def build_joint_modes(max_order=10, size=224, extent=4.0, device='cuda'):
    """LG(65)+HG(66) complex fields with string labels.

    LG(0,0) is dropped: it is the same physical field as HG(0,0)
    (fundamental gaussian), and a self-pair degenerates the
    correlation.
    """
    lg_modes, lg_idx = build_lg_modes(max_order, size, extent, device)
    keep = [k for k, (p, l) in enumerate(lg_idx) if not (p == 0 and l == 0)]
    lg_modes = lg_modes[keep]
    lg_idx = [lg_idx[k] for k in keep]
    eng = TorchHGEngine(max_order_N=max_order, grid_size=size,
                        device='cpu')
    hg = eng.hg_real.to(device).float()
    hg_c = torch.complex(hg, torch.zeros_like(hg))
    modes = torch.cat([lg_modes, hg_c], 0)
    labels = [f'LG({p},{l})' for (p, l) in lg_idx] \
           + [f'HG({m},{n})' for (m, n) in eng.mode_indices]
    return modes, labels


def decompose_image(img_path, labels, modes, args, out_root):
    stem = Path(img_path).stem
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
    phi_grid = torch.linspace(0.0, 2 * np.pi, args.grid_phi + 1,
                              device=dev)[:-1]

    corr_cube = torch.full((S, G, P), -2.0, device=dev)
    r_cube = torch.zeros((S, G, P), device=dev)
    p_cube = torch.zeros((S, G, P), device=dev)
    for si, sc in enumerate(scales):
        ms = rescale_lg(modes, sc)
        for gi, sig in enumerate(sigmas):
            Ie_b = gaussian_blur(Ie.unsqueeze(0), sig)[0]
            corr, rb, pb = scan_pairs_lg(ms, Ie_b, pairs, r_grid, phi_grid,
                                         sig, batch=args.batch)
            corr_cube[si, gi] = corr
            r_cube[si, gi] = rb
            p_cube[si, gi] = pb

    score = corr_cube.mean(dim=1)  # (S, P)
    top_vals, top_idx = score.reshape(-1).topk(args.refine_k)

    dr = 2.0 / (args.grid_r - 1)
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
        kc, kr, kp = scan_pairs_lg(ms, Ie_b, pairs[pair_id:pair_id + 1],
                                   rg, pg, sig, batch=args.batch)
        rr = float(kr[0])
        pp = float(kp[0] % (2 * np.pi))
        corrs = []
        rt = torch.tensor([rr], device=dev)
        pt = torch.tensor([pp], device=dev)
        for sig2 in sigmas:
            Ie_b2 = gaussian_blur(Ie.unsqueeze(0), sig2)[0]
            cv, _, _ = scan_pairs_lg(ms, Ie_b2,
                                     pairs[pair_id:pair_id + 1], rt, pt,
                                     sig2, batch=args.batch)
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

    i_best, j_best = pairs[best_pair_id].tolist()
    lab1, lab2 = labels[i_best], labels[j_best]

    rows = []
    for rank, oi in enumerate(ref_order, start=1):
        pid = ref_pair[oi]
        i, j = pairs[pid].tolist()
        r = ref_r[oi]
        rows.append({
            'rank': rank,
            'mode1': labels[i],
            'mode2': labels[j],
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
    It = render_lg(ms_np, i_best, j_best, best_r, best_phi, sigma=best_sig)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    axs[0].imshow(Ie_np, cmap='inferno'); axs[0].set_title('Experiment')
    axs[1].imshow(It, cmap='inferno')
    axs[1].set_title(f'{lab1}+{lab2}  s={best_scale:.2f}, sig={best_sig:.0f}\n'
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
        'basis': 'joint HG+LG',
        'best_pair': f'{lab1} + {lab2}',
        'energy_ratio': [best_r, 1 - best_r],
        'amplitudes': [float(np.sqrt(best_r)), float(np.sqrt(1 - best_r))],
        'phi_rad': best_phi,
        'scale': best_scale,
        'sigma': best_sig,
        'corr_avg': ref_corr_avg[ref_order[0]],
        'corr_best': ref_corr_best[ref_order[0]],
    }
    return summary, df


def synth_test(labels, modes, args):
    dev = modes.device
    rng = np.random.default_rng(11)
    N = modes.shape[0]
    pairs = torch.tensor([(a, b) for a in range(N) for b in range(a + 1, N)],
                         dtype=torch.long, device=dev)
    rg = torch.linspace(0.01, 0.99, args.grid_r, device=dev)
    pg = torch.linspace(0.0, 2 * np.pi, args.grid_phi + 1, device=dev)[:-1]
    ms_np = modes.detach().cpu().numpy()

    # (HG(0,1), LG(0,2)); (LG(1,-1), LG(0,3)) 5Sing-candidate;
    # (HG(0,3), HG(3,1)) 6Sing-candidate
    lkp = {lab: k for k, lab in enumerate(labels)}
    cases = [
        (lkp['HG(0,1)'], lkp['LG(0,2)'], 0.4, 1.2),
        (lkp['LG(1,-1)'], lkp['LG(0,3)'], 0.33, 0.93),
        (lkp['HG(0,3)'], lkp['HG(3,1)'], 0.46, 1.54),
    ]
    print('=== JOINT SYNTHETIC SELF-TEST (s=1, no blur) ===')
    hits = 0
    for (i, j, r, phi) in cases:
        It = render_lg(ms_np, i, j, r, phi)
        noisy = np.clip(It + rng.normal(0, 0.01, It.shape), 0, 1)
        Ie = torch.tensor(noisy, dtype=torch.float32, device=dev)
        corr, rb, pb = scan_pairs_lg(modes, Ie, pairs, rg, pg, 0.0,
                                     batch=args.batch)
        k = int(torch.argmax(corr))
        bi, bj = pairs[k].tolist()
        same = (bi, bj) == (i, j)
        hits += int(same)
        print(f'  true {labels[i]}+{labels[j]} r={r:.3f} phi={phi:.3f} | '
              f'found {labels[bi]}+{labels[bj]} r={float(rb[k]):.3f} '
              f'phi={float(pb[k]):.3f} corr={float(corr[k]):.4f} '
              f'{"OK" if same else "MISS"}')
    print(f'  recovery: {hits}/{len(cases)}')
    return hits == len(cases)


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device == 'cuda':
        print('[warn] CUDA unavailable, fallback CPU')
        args.device = 'cpu'

    modes, labels = build_joint_modes(args.max_order, 224, 4.0, args.device)
    print(f'[engine] joint basis: {len(labels)} modes '
          f'({sum(1 for x in labels if x.startswith("LG"))} LG + '
          f'{sum(1 for x in labels if x.startswith("HG"))} HG), '
          f'pairs={len(labels) * (len(labels) - 1) // 2}')

    if args.synth:
        ok = synth_test(labels, modes, args)
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

    lines = ['# Joint HG+LG two-mode search summary\n']
    for img in images:
        print(f'\n===== {img.name} =====')
        summary, df = decompose_image(img, labels, modes, args, out_root)
        print(f"  best: {summary['best_pair']} | s={summary['scale']:.2f} "
              f"sig={summary['sigma']:.0f} | "
              f"r={summary['energy_ratio'][0]:.3f}/"
              f"{summary['energy_ratio'][1]:.3f} | "
              f"phi={summary['phi_rad']:.4f} | "
              f"corr_avg={summary['corr_avg']:.4f} "
              f"(best {summary['corr_best']:.4f})")
        for _, row in df.head(5).iterrows():
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

    (out_root / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n[done] outputs in {out_root}')


if __name__ == '__main__':
    main()

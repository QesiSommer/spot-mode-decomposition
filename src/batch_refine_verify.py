# -*- coding: utf-8 -*-
"""
batch_refine_verify.py -- 批量微调 + 结构验收 (remaining 6 images)
==================================================================

For each image:
  1. take top-3 candidates from the joint search;
  2. refine each candidate (coarse (s,sigma) scan -> local r/phi grid
     -> coordinate descent), reporting corr averaged over sigma=0..6;
  3. structural verification: dark-core detection on the experiment
     (benchmark) and on each refined candidate, count + NN distance
     match, verdict table, per-image verification figure.

Outputs: batch_refine_verify/<img>_refine.json, _structural.png,
_results.csv; console summary.
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import rescale_lg, scan_pairs_lg
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, render_complex, match_points

IMG_DIR = Path(os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images')))
OUT = Path(__file__).resolve().parent / 'batch_refine_verify'
OUT.mkdir(exist_ok=True)
DEV = 'cuda'
SIGMA_REPORT = [0, 1, 2, 3, 4, 5, 6]

TARGETS = ['10Singularities', '11Singularities', '4Singularities',
           '8Singularities(1)', '8Singularities(2)', '9Singularities']


def main():
    modes, labels = build_joint_modes(10, 224, 4.0, DEV)
    lab2k = {lab: k for k, lab in enumerate(labels)}
    dev = modes.device

    summary_rows = []
    for name in TARGETS:
        print(f'\n########## {name} ##########')
        img = IMG_DIR / f'{name}.png'
        Ie_np = load_experiment(str(img), 'nipy_spectral', 'isot')
        Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)

        # structural benchmark from the experiment image
        pts_e = find_dark_points(Ie_np)
        print(f'  experiment dark cores: {len(pts_e)}')

        # top candidates from the joint search
        df = pd.read_csv(f'two_mode_search_joint/{name}_top.csv')
        df = df.head(3)

        def eval_point(pair, r, phi, s, sigma, cache={}):
            key = round(float(s), 4)
            if key not in cache:
                cache[key] = rescale_lg(modes, float(s))
            ms = cache[key]
            Ie_b = gaussian_blur(Ie.unsqueeze(0), sigma)[0]
            cv, _, _ = scan_pairs_lg(
                ms, Ie_b, pair,
                torch.tensor([r], device=dev),
                torch.tensor([phi], device=dev), sigma)
            return float(cv[0])

        def avg_over_sigmas(pair, r, phi, s):
            ms = rescale_lg(modes, float(s))
            vals = []
            for sg in SIGMA_REPORT:
                Ie_b = gaussian_blur(Ie.unsqueeze(0), sg)[0]
                cv, _, _ = scan_pairs_lg(
                    ms, Ie_b, pair,
                    torch.tensor([r], device=dev),
                    torch.tensor([phi], device=dev), sg)
                vals.append(float(cv[0]))
            return float(np.mean(vals)), vals

        def grid_scan(pair, s, sigma, nr=81, nphi=120):
            ms = rescale_lg(modes, float(s))
            Ie_b = gaussian_blur(Ie.unsqueeze(0), sigma)[0]
            rg = torch.linspace(0.01, 0.99, nr, device=dev)
            pg = torch.linspace(0.0, 2 * np.pi, nphi, device=dev)
            cv, rb, pb = scan_pairs_lg(ms, Ie_b, pair, rg, pg, sigma)
            k = int(torch.argmax(cv))
            return float(cv[k]), float(rb[k]), float(pb[k])

        results = []
        for _, row in df.iterrows():
            m1, m2 = row['mode1'], row['mode2']
            pair = torch.tensor([[lab2k[m1], lab2k[m2]]],
                                dtype=torch.long, device=dev)
            # ---- Round 1: coarse (s, sigma) ----
            best = (-2, None)
            for s in [1.0, 1.25, 1.5, 1.75, 2.0]:
                for sigma in [0, 2, 4, 6]:
                    c, r, p = grid_scan(pair, s, float(sigma))
                    if c > best[0]:
                        best = (c, (r, p, s, float(sigma)))
            c1, (r_c, p_c, s_c, sg_c) = best
            # ---- Round 2: local (r, phi) grid x fine s ----
            s_fine = np.linspace(max(0.7, s_c - 0.1), s_c + 0.1, 5)
            sig_list = [max(0.0, sg_c - 1), sg_c, min(8.0, sg_c + 1)]
            r_lo = max(0.005, r_c - 0.04)
            r_hi = min(0.995, r_c + 0.04)
            p_lo, p_hi = p_c - 0.12, p_c + 0.12
            best2 = (-2, None)
            for s in s_fine:
                ms = rescale_lg(modes, float(s))
                rg = torch.linspace(r_lo, r_hi, 81, device=dev)
                pg = torch.linspace(p_lo, p_hi, 81, device=dev)
                acc = None
                for sg in sig_list:
                    Ie_b = gaussian_blur(Ie.unsqueeze(0), sg)[0]
                    cv, rb, pb = scan_pairs_lg(ms, Ie_b, pair, rg, pg, sg)
                    acc = cv if acc is None else acc + cv
                acc = acc / len(sig_list)
                k = int(torch.argmax(acc))
                if float(acc.flatten()[k]) > best2[0]:
                    best2 = (float(acc.flatten()[k]),
                             (float(rg[k // 81]), float(pg[k % 81]),
                              float(s)))
            _, (r_c, p_c, s_c) = best2
            # ---- Round 3: coordinate descent (2 iterations) ----
            for it in range(2):
                rs = np.linspace(max(0.005, r_c - 0.03),
                                 min(0.995, r_c + 0.03), 121)
                vals = [np.mean([eval_point(pair, float(x), p_c, s_c, sg)
                                 for sg in sig_list]) for x in rs]
                r_c = float(rs[int(np.argmax(vals))])
                ps = np.linspace(p_c - 0.08, p_c + 0.08, 121)
                vals = [np.mean([eval_point(pair, r_c, float(x), s_c, sg)
                                 for sg in sig_list]) for x in ps]
                p_c = float(ps[int(np.argmax(vals))])
                ss = np.linspace(max(0.6, s_c - 0.06), s_c + 0.06, 31)
                vals = [np.mean([eval_point(pair, r_c, p_c, float(x), sg)
                                 for sg in sig_list]) for x in ss]
                s_c = float(ss[int(np.argmax(vals))])
            cavg, cvals = avg_over_sigmas(pair, r_c, p_c, s_c)

            # ---- structural check ----
            ms = rescale_lg(modes, s_c).detach().cpu().numpy()
            E = render_complex(ms, lab2k[m1], lab2k[m2], r_c, p_c)
            Im = np.abs(E) ** 2
            Im = Im / Im.max()
            pts_m = find_dark_points(Im)
            match = match_points(pts_e, pts_m)
            d = match['mean']
            n_ok = abs(len(pts_m) - len(pts_e)) <= 1
            d_ok = d is not None and d < 12
            verdict = 'PASS' if (n_ok and d_ok) else 'fail'
            print(f'  {m1}+{m2}: corr_avg={cavg:.4f}  cores={len(pts_m)} '
                  f'(exp {len(pts_e)})  meanNN='
                  f'{d if d is None else round(d, 1)}px  {verdict}')
            results.append({
                'pair': [m1, m2], 'r': r_c,
                'amp1': float(np.sqrt(r_c)),
                'amp2': float(np.sqrt(1 - r_c)),
                'phi_rad': p_c, 'scale': s_c,
                'corr_avg': cavg,
                'corr_by_sigma': [round(v, 4) for v in cvals],
                'n_core_model': len(pts_m),
                'mean_nn_px': d,
                'verdict': verdict,
            })

        # champion = best corr_avg among PASS, else best corr_avg
        passing = [x for x in results if x['verdict'] == 'PASS']
        pool = passing if passing else results
        champ = max(pool, key=lambda x: x['corr_avg'])
        print(f'  >>> CHAMPION: {champ["pair"][0]}+{champ["pair"][1]} '
              f'corr={champ["corr_avg"]:.4f} verdict={champ["verdict"]}')

        payload = {'image': name, 'n_core_experiment': len(pts_e),
                   'core_exp_positions': [(int(x), int(y))
                                          for x, y, _ in pts_e],
                   'candidates': results, 'champion': champ['pair']}
        (OUT / f'{name}_refine.json').write_text(
            json.dumps(payload, indent=2), encoding='utf-8')

        # ---- per-image verification figure ----
        ms = rescale_lg(modes, champ['scale']).detach().cpu().numpy()
        E = render_complex(ms, lab2k[champ['pair'][0]],
                           lab2k[champ['pair'][1]], champ['r'],
                           champ['phi_rad'])
        Im = np.abs(E) ** 2
        Im = Im / Im.max()
        pts_m = find_dark_points(Im)

        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6))
        axs[0].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
        for (x, y, _v) in pts_e:
            axs[0].plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.4)
        axs[0].set_title(f'{name} experiment: {len(pts_e)} cores',
                         fontsize=10)
        axs[1].imshow(Im, cmap='inferno', vmin=0, vmax=1)
        for (x, y, _v) in pts_m:
            axs[1].plot(x, y, 'o', mfc='none', mec='red', ms=11, mew=1.4)
        axs[1].set_title(f'{champ["pair"][0]}+{champ["pair"][1]} '
                         f'({len(pts_m)} cores)\n'
                         f'r={champ["r"]:.3f}, phi={champ["phi_rad"]:.3f}, '
                         f'corr_avg={champ["corr_avg"]:.3f}', fontsize=10)
        axs[2].imshow(Ie_np, cmap='inferno', vmin=0, vmax=1)
        for (x, y, _v) in pts_e:
            axs[2].plot(x, y, 'o', mfc='none', mec='cyan', ms=11, mew=1.4)
        for (x, y, _v) in pts_m:
            axs[2].plot(x, y, 'x', color='red', ms=9, mew=1.8)
        axs[2].set_title(f'overlay  meanNN='
                         f'{champ["mean_nn_px"] if champ["mean_nn_px"] is None else round(champ["mean_nn_px"], 1)}px'
                         f'  [{champ["verdict"].upper()}]', fontsize=10)
        for a in axs:
            a.axis('off')
        fig.tight_layout()
        fig.savefig(OUT / f'{name}_structural.png', dpi=130)
        plt.close(fig)

        summary_rows.append({
            'image': name,
            'pair': f'{champ["pair"][0]}+{champ["pair"][1]}',
            'r1': round(champ['r'], 4), 'amp1': round(champ['amp1'], 4),
            'amp2': round(champ['amp2'], 4),
            'phi_rad': round(champ['phi_rad'], 4),
            'corr_avg': round(champ['corr_avg'], 4),
            'cores_exp': len(pts_e), 'cores_model': champ['n_core_model'],
            'mean_nn_px': None if champ['mean_nn_px'] is None
            else round(champ['mean_nn_px'], 1),
            'verdict': champ['verdict'],
        })

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT / 'batch_summary.csv', index=False, encoding='utf-8-sig')
    print('\n===== BATCH SUMMARY =====')
    print(sdf.to_string(index=False))
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

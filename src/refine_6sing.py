# -*- coding: utf-8 -*-
"""
refine_6sing.py -- 6Singularities 单图精细微调
================================================

对 6Singularities.png 的 top 候选做连续参数微调:
  (r, phi, scale s, blur sigma) 四参数, 三轮细化:
    Round 1  粗:  s 网格 x sigma 网格, 每个组合全 (r, phi) 网格
    Round 2  中:  局部 (r, phi) 网格 x s 细网格 (sigma 邻域三点平均)
    Round 3  细:  坐标下降逐参数收敛
质量评估: PCC(多 sigma), SSIM, 差异图;
稳健性: 4 种预处理组合交叉验证;
敏感性: 每参数 1D 扰动曲线。

输出: refine_6sing/ 目录 (数据 + 图件)
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_lg import build_lg_modes, rescale_lg, scan_pairs_lg, \
    render_lg
from search_two_mode_joint import build_joint_modes

IMG = os.path.join(os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images')), '6Singularities.png')
OUT = Path('refine_6sing')
OUT.mkdir(parents=True, exist_ok=True)
DEV = 'cuda'
SIGMA_REPORT = [0, 1, 2, 3, 4, 5, 6]

CANDIDATES = [
    ('HG(0,3)', 'HG(3,1)'),
    ('HG(0,4)', 'HG(4,1)'),
    ('HG(0,2)', 'HG(3,1)'),
    ('LG(1,-2)', 'HG(0,2)'),
]


def ssim_simple(a, b, sig=1.5):
    mu_a = gaussian_filter(a, sig)
    mu_b = gaussian_filter(b, sig)
    va = gaussian_filter(a * a, sig) - mu_a ** 2
    vb = gaussian_filter(b * b, sig) - mu_b ** 2
    cab = gaussian_filter(a * b, sig) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cab + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def main():
    torch.manual_seed(0)
    modes, labels = build_joint_modes(10, 224, 4.0, DEV)
    lab2k = {lab: k for k, lab in enumerate(labels)}
    dev = modes.device
    Ie_np = load_experiment(IMG, 'nipy_spectral', 'isot')
    Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
    me = Ie.mean()
    ve = Ie.var(unbiased=False)

    def eval_point(pair, r, phi, s, sigma, ms_cache={}):
        """instantaneous corr at one parameter point."""
        if s not in ms_cache:
            ms_cache[s] = rescale_lg(modes, s)
        ms = ms_cache[s]
        Ie_b = gaussian_blur(Ie.unsqueeze(0), sigma)[0]
        cv, _, _ = scan_pairs_lg(
            ms, Ie_b, pair,
            torch.tensor([r], device=dev),
            torch.tensor([phi], device=dev), sigma)
        return float(cv[0])

    def avg_over_sigmas(pair, r, phi, s, sigmas=SIGMA_REPORT):
        ms = rescale_lg(modes, s)
        vals = []
        for sg in sigmas:
            Ie_b = gaussian_blur(Ie.unsqueeze(0), sg)[0]
            cv, _, _ = scan_pairs_lg(
                ms, Ie_b, pair,
                torch.tensor([r], device=dev),
                torch.tensor([phi], device=dev), sg)
            vals.append(float(cv[0]))
        return float(np.mean(vals)), vals

    def grid_scan(pair, s, sigma, r0=0.01, r1=0.99, nr=81,
                  p0=0.0, p1=2 * np.pi, nphi=120):
        ms = rescale_lg(modes, s)
        Ie_b = gaussian_blur(Ie.unsqueeze(0), sigma)[0]
        rg = torch.linspace(r0, r1, nr, device=dev)
        pg = torch.linspace(p0, p1, nphi, device=dev)
        cv, rb, pb = scan_pairs_lg(ms, Ie_b, pair, rg, pg, sigma)
        k = int(torch.argmax(cv))
        return float(cv[k]), float(rb[k]), float(pb[k])

    results = {}
    for tag in CANDIDATES:
        pair = torch.tensor([[lab2k[tag[0]], lab2k[tag[1]]]],
                            dtype=torch.long, device=dev)
        print(f'\n===== candidate {tag[0]} + {tag[1]} =====')

        # Round 1: coarse (s, sigma) grid with full (r, phi) scan
        best = (-2, None)
        for s in np.linspace(1.0, 1.5, 11):
            for sigma in np.linspace(0, 6, 13):
                c, r, p = grid_scan(pair, float(s), float(sigma))
                if c > best[0]:
                    best = (c, (r, p, float(s), float(sigma)))
        c1, (r1c, p1c, s1c, sig1c) = best
        print(f'  R1 coarse : corr={c1:.4f} r={r1c:.4f} '
              f'phi={p1c:.4f} s={s1c:.2f} sig={sig1c:.1f}')

        # Round 2: local (r, phi) grid x fine s, sigma = mean(3)
        s_fine = np.linspace(max(0.8, s1c - 0.075), s1c + 0.075, 7)
        sig_list = [max(0.0, sig1c - 1), sig1c, sig1c + 1]
        r_lo, r_hi = max(0.005, r1c - 0.03), min(0.995, r1c + 0.03)
        p_lo, p_hi = p1c - 0.10, p1c + 0.10
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
            ri = k // 81
            pi = k % 81
            if float(acc.flatten()[k]) > best2[0]:
                best2 = (float(acc.flatten()[k]),
                         (float(rg[ri]), float(pg[pi]), float(s)))
        c2, (r2c, p2c, s2c) = best2
        print(f'  R2 local  : corr3avg={c2:.4f} r={r2c:.4f} '
              f'phi={p2c:.4f} s={s2c:.3f}')

        # Round 3: coordinate descent (r, phi, s, sigma)
        r_c, p_c, s_c, sg_c = r2c, p2c, s2c, sig1c
        for it in range(3):
            # r
            rs = np.linspace(max(0.005, r_c - 0.02), min(0.995, r_c + 0.02),
                             161)
            vals = [np.mean([eval_point(pair, float(x), p_c, s_c, sg)
                             for sg in sig_list]) for x in rs]
            r_c = float(rs[int(np.argmax(vals))])
            # phi
            ps = np.linspace(p_c - 0.06, p_c + 0.06, 161)
            vals = [np.mean([eval_point(pair, r_c, float(x), s_c, sg)
                             for sg in sig_list]) for x in ps]
            p_c = float(ps[int(np.argmax(vals))])
            # s
            ss = np.linspace(max(0.5, s_c - 0.05), s_c + 0.05, 41)
            vals = [np.mean([eval_point(pair, r_c, p_c, float(x), sg)
                             for sg in sig_list]) for x in ss]
            s_c = float(ss[int(np.argmax(vals))])
            # sigma (single sigma, instantaneous)
            sgs = np.linspace(0, 8, 33)
            vals = [eval_point(pair, r_c, p_c, s_c, float(x)) for x in sgs]
            sg_c = float(sgs[int(np.argmax(vals))])
            print(f'  R3 iter{it + 1}: r={r_c:.4f} phi={p_c:.4f} '
                  f's={s_c:.3f} sig={sg_c:.2f} '
                  f'(point corr={max(vals):.4f})')

        cavg, cvals = avg_over_sigmas(pair, r_c, p_c, s_c)
        cbest = float(np.max(cvals))
        sig_at_best = SIGMA_REPORT[int(np.argmax(cvals))]
        print(f'  FINAL     : corr_avg={cavg:.4f} '
              f'(best {cbest:.4f} @ sig={sig_at_best})')
        results[f'{tag[0]}+{tag[1]}'] = {
            'pair': list(tag),
            'r': r_c, 'amp1': float(np.sqrt(r_c)),
            'amp2': float(np.sqrt(1 - r_c)),
            'phi_rad': p_c,
            'scale': s_c, 'sigma': sg_c,
            'corr_avg': cavg, 'corr_best': cbest,
            'corr_by_sigma': dict(zip(SIGMA_REPORT,
                                      [round(v, 4) for v in cvals])),
        }

    # ---- champion ----
    champ = max(results.items(), key=lambda kv: kv[1]['corr_avg'])
    cname, cinfo = champ
    print(f'\n=== CHAMPION: {cname} corr_avg={cinfo["corr_avg"]:.4f} ===')
    pair = torch.tensor([[lab2k[cinfo['pair'][0]], lab2k[cinfo['pair'][1]]]],
                        dtype=torch.long, device=dev)
    r_c, p_c, s_c = cinfo['r'], cinfo['phi_rad'], cinfo['scale']

    # ---- sensitivity curves ----
    print('\n=== sensitivity analysis ===')
    sig_list = [max(0.0, cinfo['sigma'] - 1), cinfo['sigma'],
                min(8.0, cinfo['sigma'] + 1)]
    sens = {}
    rs = np.linspace(0.30, 0.62, 81)
    sens['r'] = [[float(x) for x in rs],
                 [float(np.mean([eval_point(pair, float(x), p_c, s_c, sg)
                                 for sg in sig_list])) for x in rs]]
    ps = np.linspace(p_c - 0.35, p_c + 0.35, 81)
    sens['phi'] = [[float(x) for x in ps],
                   [float(np.mean([eval_point(pair, r_c, float(x), s_c, sg)
                                   for sg in sig_list])) for x in ps]]
    ss = np.linspace(1.0, 1.5, 81)
    sens['s'] = [[float(x) for x in ss],
                 [float(np.mean([eval_point(pair, r_c, p_c, float(x), sg)
                                 for sg in sig_list])) for x in ss]]
    sgs = np.linspace(0, 8, 81)
    sens['sigma'] = [[float(x) for x in sgs],
                     [eval_point(pair, r_c, p_c, s_c, float(x))
                      for x in sgs]]
    for k, (xs, vs) in sens.items():
        i0 = int(np.argmax(vs))
        print(f'  {k}: peak at {xs[i0]:.4f} (corr={vs[i0]:.4f})')

    # ---- preprocessing robustness ----
    print('\n=== robustness across preprocess/cmap ===')
    rob = {}
    for cmap in ['nipy_spectral', 'none']:
        for prep in ['isot', 'crop']:
            Ie_v = load_experiment(IMG, cmap, prep)
            Ie_vt = torch.tensor(Ie_v, dtype=torch.float32, device=dev)
            ms = rescale_lg(modes, s_c)
            vals = []
            for sg in SIGMA_REPORT:
                Ie_b = gaussian_blur(Ie_vt.unsqueeze(0), sg)[0]
                cv, _, _ = scan_pairs_lg(
                    ms, Ie_b, pair,
                    torch.tensor([r_c], device=dev),
                    torch.tensor([p_c], device=dev), sg)
                vals.append(float(cv[0]))
            key = f'{cmap}|{prep}'
            rob[key] = {'corr_avg': float(np.mean(vals)),
                        'corr_best': float(np.max(vals)),
                        'by_sigma': [round(v, 4) for v in vals]}
            print(f'  {key:24s} corr_avg={np.mean(vals):.4f} '
                  f'best={np.max(vals):.4f}')

    # ---- final render + figures ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ms_np = rescale_lg(modes, s_c).detach().cpu().numpy()
    It_raw = render_lg(ms_np, lab2k[cinfo['pair'][0]], lab2k[cinfo['pair'][1]],
                       r_c, p_c, sigma=0.0)
    It_blur = render_lg(ms_np, lab2k[cinfo['pair'][0]],
                        lab2k[cinfo['pair'][1]], r_c, p_c,
                        sigma=cinfo['sigma'])
    diff = np.abs(Ie_np - It_blur)
    ssim_val = ssim_simple(Ie_np / Ie_np.max(), It_blur / It_blur.max())

    fig, axs = plt.subplots(1, 4, figsize=(17.5, 4.4))
    axs[0].imshow(Ie_np, cmap='inferno'); axs[0].set_title('Experiment')
    axs[1].imshow(It_blur, cmap='inferno')
    axs[1].set_title(f'{cinfo["pair"][0]}+{cinfo["pair"][1]}\n'
                     f'r={r_c:.3f}, phi={p_c:.3f}, s={s_c:.2f}, '
                     f'sig={cinfo["sigma"]:.1f}')
    axs[2].imshow(diff, cmap='viridis')
    axs[2].set_title(f'|diff|  corr_avg={cinfo["corr_avg"]:.4f}')
    axs[3].contour(Ie_np, levels=6, colors='w', linewidths=0.8)
    axs[3].contour(It_blur, levels=6, colors='r', linewidths=0.8,
                   linestyles='--')
    axs[3].set_title(f'overlay (white=exp, red=model)  SSIM={ssim_val:.4f}')
    for a in axs:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / 'final_compare.png', dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key in zip(axs.ravel(), ['r', 'phi', 's', 'sigma']):
        xs, vs = sens[key]
        ax.plot(xs, vs, lw=1.6)
        i0 = int(np.argmax(vs))
        ax.axvline(xs[i0], color='r', ls='--', lw=0.9)
        ax.set_title(f'corr vs {key}  (peak {xs[i0]:.4f})')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'sensitivity.png', dpi=150)
    plt.close(fig)

    payload = {
        'image': '6Singularities.png',
        'candidates': results,
        'champion': cname,
        'champion_info': cinfo,
        'ssim': ssim_val,
        'sensitivity_peaks': {k: float(v[0][int(np.argmax(v[1]))])
                              for k, v in sens.items()},
        'sensitivity': sens,
        'robustness': rob,
    }
    (OUT / 'refine_6sing_results.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    np.save(OUT / 'Ie_processed.npy', Ie_np)
    np.save(OUT / 'I_model_final.npy', It_blur)
    print(f'\n[ssim] {ssim_val:.4f}')
    print(f'[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

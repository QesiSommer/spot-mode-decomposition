# -*- coding: utf-8 -*-
"""
stage0_diagnosis.py -- 阶段0：诊断（不改任何解）
===============================================

D1  核深度剖面：实验 vs 模型（σ=0/1/2/3）每个暗点的
    深度（核底/局部峰值）与孤立度（相邻核鞍部亮度）
D2  gamma 检测：支持区内按模型强度分箱的 exp/model 比值
    + 全局 log-log 拟合
D3  模糊度复核：模型核深度随 σ 的变化，找与实验匹配的 σ*

输入：最新结果_新指标/<cn>_参数与指标.md（解析参数）
输出：局部优化_5及8奇点/阶段0_诊断/
"""

import re
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from calibrate_metrics import render_model

SRC = ROOT / '归档_八图分解' / '最新结果_新指标'
OUT = ROOT / '归档_八图分解' / '局部优化_5及8奇点' / '阶段0_诊断'
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

CASES = [
    {'name': '5Singularities', 'cn': '5奇点'},
    {'name': '8Singularities(1)', 'cn': '8奇点(1)'},
]


def parse_md(md_path):
    """Parse combo / amps / phases / scale from the deliverable md."""
    txt = md_path.read_text(encoding='utf-8')
    combo, amps = [], []
    for m in re.finditer(r'^\| (\S+) \| [\d.]+% \| ([\d.]+) \|$',
                         txt, re.M):
        combo.append(m.group(1))
        amps.append(float(m.group(2)))
    phis = [float(m.group(1)) for m in
            re.finditer(r'^- φ\d+ = (-?[\d.]+) rad', txt, re.M)]
    ms = re.search(r'^- 缩放参数 s = ([\d.]+)$', txt, re.M)
    scale = float(ms.group(1)) if ms else None
    N = len(combo)
    u = [a * a for a in amps]
    s = sum(u)
    u = [x / s for x in u]
    params = list(u) + phis
    assert len(params) == 2 * N - 1, (md_path, combo, phis)
    return combo, params, scale


def core_depth(I, pts, win=5, ann=(8, 16)):
    """depth = 1 - min(I near core)/max(I in annulus)."""
    H, W = I.shape
    depths = []
    for (y, x, _v) in pts:
        y0, y1 = max(0, y - win), min(H, y + win + 1)
        x0, x1 = max(0, x - win), min(W, x + win + 1)
        Imin = I[y0:y1, x0:x1].min()
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
        ring = (r >= ann[0]) & (r <= ann[1])
        Ipk = I[ring].max() if ring.any() else np.nan
        depths.append(1 - Imin / Ipk if Ipk > 1e-9 else np.nan)
    return np.array(depths)


def saddle_isolation(I, pts, max_dist=40, n_samp=40):
    """for each close core pair: saddle/peak (lower = more isolated)."""
    vals = []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            (y1, x1, _), (y2, x2, _) = pts[i], pts[j]
            d = np.hypot(x2 - x1, y2 - y1)
            if d > max_dist or d < 4:
                continue
            ts = np.linspace(0, 1, n_samp)
            ys = np.clip((y1 + ts * (y2 - y1)).round().astype(int),
                         0, I.shape[0] - 1)
            xs = np.clip((x1 + ts * (x2 - x1)).round().astype(int),
                         0, I.shape[1] - 1)
            saddle = I[ys, xs].max()
            pk = max(I[max(0, y1 - 12):y1 + 13, max(0, x1 - 12):x1 + 13]
                     .max(),
                     I[max(0, y2 - 12):y2 + 13, max(0, x2 - 12):x2 + 13]
                     .max())
            vals.append((d, saddle / pk if pk > 1e-9 else np.nan))
    return vals


def gamma_bins(Ie, Im, mask, n_bins=10):
    """median exp/model ratio per model-intensity bin (support only)."""
    e = Ie[mask]
    m = Im[mask]
    qs = np.quantile(m, np.linspace(0, 1, n_bins + 1))
    rows = []
    for k in range(n_bins):
        lo, hi = qs[k], qs[k + 1]
        sel = (m >= lo) & (m <= hi if k == n_bins - 1 else m < hi)
        if sel.sum() < 20:
            rows.append((float((lo + hi) / 2), np.nan, int(sel.sum())))
            continue
        ratio = float(np.median(e[sel]) / max(np.median(m[sel]), 1e-9))
        rows.append((float(np.median(m[sel])), ratio, int(sel.sum())))
    return rows


def main():
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {l: k for k, l in enumerate(labels)}
    modes_np = modes.numpy()

    report = {}
    for case in CASES:
        name, cn = case['name'], case['cn']
        print(f'\n########## {cn} 阶段0诊断 ##########')
        combo, params, scale = parse_md(
            SRC / f'{cn}_参数与指标.md')
        print(f'parsed: {combo} scale={scale}')
        Ie = load_experiment(f'{IMG_DIR}/{name}.png', 'nipy_spectral',
                             'isot')
        I0 = render_model(modes_np, lab2k,
                          {'combo': combo, 'params': params,
                           'scale': scale})
        pts_e = find_dark_points(Ie)

        # ---- D1: core depth & isolation ----
        print('\n[D1] core depth / isolation')
        dep_e = core_depth(Ie, pts_e)
        print(f'  experiment: {len(pts_e)} cores, depth mean='
              f'{np.nanmean(dep_e):.3f} min={np.nanmin(dep_e):.3f}')
        sad_e = saddle_isolation(Ie, pts_e)
        print(f'  experiment saddles: '
              f'{[round(s, 2) for _, s in sad_e]}')
        mod_by_sigma = {}
        for sg in [0, 1, 2, 3]:
            if sg == 0:
                Im = I0
            else:
                Im = gaussian_blur(
                    torch.tensor(I0, dtype=torch.float32)
                    .unsqueeze(0), float(sg))[0].numpy()
            mod_by_sigma[sg] = Im
            pts_m = find_dark_points(Im)
            dep_m = core_depth(Im, pts_m)
            mt = match_points(pts_e, pts_m)
            print(f'  model s={sg}: {len(pts_m)} cores, depth mean='
                  f'{np.nanmean(dep_m):.3f}, meanNN='
                  f'{mt["mean"] if mt["mean"] is None else round(mt["mean"], 1)}')
        # matched-pair depth comparison at sigma=2 (display scale)
        Im2 = mod_by_sigma[2]
        pts_m2 = find_dark_points(Im2)
        dep_m2 = core_depth(Im2, pts_m2)
        sad_m = saddle_isolation(Im2, pts_m2)
        print(f'  model s=2 saddles: '
              f'{[round(float(s), 2) for _, s in sad_m]}')
        # saddle at sigma* too (blur-match scale)
        Im3 = mod_by_sigma[3]
        pts_m3 = find_dark_points(Im3)
        sad_m3 = saddle_isolation(Im3, pts_m3)
        print(f'  model s=3 saddles: '
              f'{[round(float(s), 2) for _, s in sad_m3]}')

        # ---- D2: gamma ----
        print('\n[D2] gamma check (binned exp/model ratio)')
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(Ie > 0.05 * Ie.max(), iterations=8)
        rows = gamma_bins(Ie, Im2, mask)
        for mid, ratio, cnt in rows[:4]:
            print(f'  model bin ~{mid:.3f}: exp/model = '
                  f'{ratio:.3f}  (n={cnt})')
        dark_ratio = rows[0][1]
        bright_ratio = rows[-1][1]
        # global log-log fit
        e = Ie[mask]
        m = Im2[mask]
        sel = (m > 0.02) & (e > 0.02)
        if sel.sum() > 100:
            A = np.vstack([np.log(m[sel]), np.ones(sel.sum())]).T
            gamma_fit = float(np.linalg.lstsq(
                A, np.log(e[sel]), rcond=None)[0][0])
        else:
            gamma_fit = np.nan
        print(f'  darkest-bin ratio={dark_ratio:.3f} vs '
              f'brightest-bin ratio={bright_ratio:.3f}')
        print(f'  global log-log gamma fit = {gamma_fit:.3f} '
              f'(1.0 = linear)')

        # ---- D3: sigma sweep of mean depth ----
        print('\n[D3] model mean core depth vs sigma:')
        sweep = {}
        for sg in [0, 1, 2, 3]:
            Im = mod_by_sigma[sg]
            pts_m = find_dark_points(Im)
            dep_m = core_depth(Im, pts_m)
            sweep[sg] = float(np.nanmean(dep_m))
            print(f'  sigma={sg}: depth={sweep[sg]:.3f}')
        target = float(np.nanmean(dep_e))
        sigma_star = min(sweep, key=lambda k: abs(sweep[k] - target))
        print(f'  experiment mean depth = {target:.3f} '
              f'-> sigma* = {sigma_star}')

        # ---- figure ----
        fig, axs = plt.subplots(2, 4, figsize=(18, 8.6))
        # row 1: center zooms
        c0, c1 = 52, 172
        axs[0, 0].imshow(Ie[c0:c1, c0:c1], cmap='inferno', vmin=0,
                         vmax=1)
        axs[0, 0].set_title('experiment (center zoom)')
        for k, sg in enumerate([0, 1, 2]):
            axs[0, k + 1].imshow(mod_by_sigma[sg][c0:c1, c0:c1],
                                 cmap='inferno', vmin=0, vmax=1)
            axs[0, k + 1].set_title(f'model sigma={sg}')
        # row 2: diagnostics
        ax = axs[1, 0]
        ax.bar(np.arange(len(dep_e)) - 0.2, dep_e, 0.4,
               label='experiment')
        ax.bar(np.arange(len(dep_m2)) + 0.2, dep_m2, 0.4,
               label='model (s=2)')
        ax.set_title('core depth per point')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axs[1, 1]
        mids = [r[0] for r in rows]
        ratios = [r[1] for r in rows]
        ax.plot(mids, ratios, 'o-')
        ax.axhline(1.0, color='gray', ls='--', lw=1)
        ax.set_xlabel('model intensity (bin median)')
        ax.set_ylabel('exp/model ratio')
        ax.set_title(f'D2 gamma check (global fit g='
                     f'{gamma_fit:.2f})')
        ax.grid(alpha=0.3)
        ax = axs[1, 2]
        ax.plot(list(sweep.keys()), list(sweep.values()), 'o-',
                label='model depth')
        ax.axhline(target, color='r', ls='--', label='experiment')
        ax.set_xlabel('sigma')
        ax.set_ylabel('mean core depth')
        ax.set_title(f'D3 blur check (sigma*={sigma_star})')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axs[1, 3]
        if sad_e:
            ax.hist([s for _, s in sad_e], bins=8, alpha=0.6,
                    label='experiment')
        if sad_m:
            ax.hist([s for _, s in sad_m], bins=8, alpha=0.6,
                    label='model (s=2)')
        ax.set_xlabel('saddle / local peak (lower=isolated)')
        ax.set_title('core isolation')
        ax.legend(fontsize=8)
        for a in axs.ravel():
            if a.get_images():
                a.axis('off')
        fig.suptitle(f'{name} Stage-0 diagnosis', fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f'{cn}_诊断.png', dpi=140,
                    bbox_inches='tight')
        plt.close(fig)

        report[cn] = {
            'combo': combo, 'scale': scale,
            'n_cores_exp': len(pts_e),
            'depth_exp_mean': target,
            'depth_exp_all': [round(float(x), 3) for x in dep_e],
            'depth_model_by_sigma': {str(k): round(v, 3)
                                      for k, v in sweep.items()},
            'sigma_star': sigma_star,
            'saddles_exp': [(round(float(d), 1), round(float(s), 3))
                            for d, s in sad_e],
            'saddles_model_s2': [(round(float(d), 1), round(float(s), 3))
                                 for d, s in sad_m],
            'saddles_model_s3': [(round(float(d), 1), round(float(s), 3))
                                 for d, s in sad_m3],
            'gamma_bins': [(round(a, 3), None if b != b else round(b, 3),
                            c) for a, b, c in rows],
            'gamma_global_fit': None if gamma_fit != gamma_fit
            else round(gamma_fit, 3),
            'dark_ratio': None if dark_ratio != dark_ratio
            else round(dark_ratio, 3),
            'bright_ratio': None if bright_ratio != bright_ratio
            else round(bright_ratio, 3),
        }

    import json
    (OUT / '诊断数据.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

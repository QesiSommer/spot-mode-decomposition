# -*- coding: utf-8 -*-
"""
refine_methods_compare.py -- 精调方法对比实验
=============================================

Method A (current): coordinate descent with interval bisection
    (h -> h/4 each round, 41-point line scans) -- replica of refine_8sing1.

Method B (new combo):
    (3) OLS closed-form upper bound + analytic manifold projection
        -> initialization
    (2) alternating block refinement: dense phase grid (61x61) and
        energy simplex grid (41x41) with geometric shrinkage
    (5) multi-start (OLS seed, coarse seed, jittered seeds) + best pick

Fair comparison: same start point, same objective (corr_avg over
sigma=0..6), same hardware; metrics: final corr_avg, wall time,
number of evaluated points, structural verification of the solution.

Cases: 4Singularities (best-pass triple) and 8Singularities(1)
(champion triple). Read-only on archived data; outputs to
refine_methods_compare/.
"""

import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from search_three_mode import build_base_fields, eval_c_grid, \
    render_three_mode_np
from structural_check import find_dark_points, match_points

ARCH = ROOT / '归档_八图分解'
OUT = ROOT / 'refine_methods_compare'
OUT.mkdir(exist_ok=True)
SIGMAS = [0, 1, 2, 3, 4, 5, 6]
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

CASES = [
    {
        'tag': '4Sing',
        'name': '4Singularities',
        'three_json': ARCH / '4奇点' / '3_三模搜索与验收'
                      / '4Singularities_three_mode.json',
        'combo': ('LG(0,1)', 'LG(1,1)', 'LG(0,5)'),
    },
    {
        'tag': '8Sing1',
        'name': '8Singularities(1)',
        'three_json': ARCH / '8奇点(1)' / '3_三模搜索与验收'
                      / '8Singularities(1)_three_mode.json',
        'combo': ('LG(2,0)', 'HG(1,1)', 'HG(2,0)'),
    },
    {
        'tag': '4Sing_alt',
        'name': '4Singularities',
        'three_json': ARCH / '4奇点' / '3_三模搜索与验收'
                      / '4Singularities_three_mode.json',
        'combo': ('LG(0,-1)', 'LG(2,1)', 'LG(0,5)'),
        'coarsen': True,
    },
]


# ---------------------------------------------------------------------------
# statistics and point evaluation
# ---------------------------------------------------------------------------

def precompute_stats(modes, Ie, tr):
    dev = modes.device
    F = [modes[tr[i]].unsqueeze(0) for i in range(3)]
    G9 = build_base_fields(F[0], F[1], F[2])[0]  # (9,H,W)
    stats = []
    for sg in SIGMAS:
        Gb = gaussian_blur(G9, float(sg))
        Ie_b = gaussian_blur(Ie.unsqueeze(0), float(sg))[0]
        me = Ie_b.mean()
        ve = Ie_b.var(unbiased=False)
        Gf = Gb.reshape(9, -1)
        m = Gf.mean(dim=1).unsqueeze(0)
        E = torch.einsum('ki,i->k', Gf, Ie_b.reshape(-1)
                         ).unsqueeze(0) / Gf.shape[1]
        Gram = torch.einsum('ki,li->kl', Gf, Gf
                            ).unsqueeze(0) / Gf.shape[1]
        stats.append((m, E, Gram, me, ve))
    return stats


def c_of(theta):
    u, v, w, p2, p3 = theta
    a1, a2, a3 = np.sqrt(max(u, 0)), np.sqrt(max(v, 0)), \
        np.sqrt(max(w, 0))
    return torch.tensor([[
        u, v, w,
        2 * a1 * a2 * np.cos(p2), 2 * a1 * a2 * np.sin(p2),
        2 * a1 * a3 * np.cos(p3), 2 * a1 * a3 * np.sin(p3),
        2 * a2 * a3 * np.cos(p2 - p3), -2 * a2 * a3 * np.sin(p2 - p3),
    ]], dtype=torch.float32)


def corr_avg_point(stats, theta):
    c = c_of(theta)
    vals = [float(eval_c_grid(c, *s)[0, 0]) for s in stats]
    return float(np.mean(vals))


def corr_avg_batch(stats, C):
    """C: (N,9) -> (N,) corr_avg."""
    total = None
    for s in stats:
        v = eval_c_grid(C, *s).reshape(-1)  # (N,)
        total = v if total is None else total + v
    return total / len(stats)


# ---------------------------------------------------------------------------
# Method A: bisection coordinate descent (current approach)
# ---------------------------------------------------------------------------

def method_bisection(stats, theta0, max_iters=12, eval_counter=None):
    cur = [float(x) for x in theta0]
    h = [0.02, 0.02, 0.02, 0.10, 0.10]
    hist = []
    n_ev = 0
    for it in range(max_iters):
        improved = False
        for k in range(5):
            pts = np.linspace(cur[k] - h[k], cur[k] + h[k], 41)
            best_v, best_x = -2.0, cur[k]
            for x in pts:
                trial = list(cur)
                trial[k] = float(x)
                if sum(trial[:3]) > 1.0 or min(trial[:3]) < 0:
                    continue
                cv = corr_avg_point(stats, trial)
                n_ev += 1
                if cv > best_v:
                    best_v, best_x = cv, float(x)
            if abs(best_x - cur[k]) > 1e-9:
                improved = True
            cur[k] = best_x
        h = [x * 0.25 for x in h]
        s = sum(cur[:3])
        cur[:3] = [x / s for x in cur[:3]]
        cavg = corr_avg_point(stats, cur)
        n_ev += 1
        hist.append((n_ev, cavg))
        if not improved and all(x <= 1e-6 for x in h):
            break
    return cur, hist, n_ev


# ---------------------------------------------------------------------------
# Method B: OLS init (3) + block refinement (2) + multi-start (5)
# ---------------------------------------------------------------------------

def ols_c_init(stats, sigmas_use=(0, 2, 4)):
    """Unconstrained optimal c per sigma -> averaged, normalized direction."""
    cs = []
    for si in sigmas_use:
        m, E, Gram, me, ve = stats[si]
        m9 = m[0]
        Gc = (Gram[0] - m9[:, None] * m9[None, :]).double()
        Ec = (E[0] - m9 * me).double()
        Gc = Gc + 1e-8 * torch.eye(9, dtype=Gc.dtype, device=Gc.device)
        c = torch.linalg.solve(Gc, Ec)
        bsum = c[:3].clamp_min(0).sum()
        if bsum > 1e-9:
            cs.append((c / bsum).cpu().numpy())
    if not cs:
        return None, None
    c_avg = np.mean(cs, axis=0)
    # reference upper bound at sigma=2
    m, E, Gram, me, ve = stats[2]
    m9, G9m = m[0].double(), Gram[0].double()
    Ec = (E[0] - m[0] * me).double()
    Gc = G9m - m9[:, None] * m9[None, :]
    c2 = torch.linalg.solve(
        Gc + 1e-8 * torch.eye(9, dtype=Gc.dtype, device=Gc.device), Ec)
    ub = float(torch.sqrt((Ec @ c2) / ve))
    return c_avg, ub


def project_c_to_params(c):
    """Analytic projection of the c vector onto the physical manifold."""
    b = np.clip(np.asarray(c[:3], dtype=np.float64), 0, None)
    if b.sum() < 1e-9:
        return None
    u, v, w = b / b.sum()
    z12 = complex(c[3], c[4])       # ~ e^{i phi2}
    z13 = complex(c[5], c[6])       # ~ e^{i phi3}
    z23 = complex(c[7], -c[8])      # ~ e^{i(phi2-phi3)}
    if abs(z12) < 1e-9 or abs(z13) < 1e-9:
        return None
    p2 = float(np.angle(z12))
    p3 = float(np.angle(z13))
    # consistency correction using the (2,3) cross term
    if abs(z23) > 1e-9:
        delta = float(np.angle(z23 / (z12 * np.conj(z13))))
        p2 += delta / 2.0
        p3 -= delta / 2.0
    return [u, v, w, p2, p3]


def _c_phase_block(u, v, w, p2s, p3s):
    P2, P3 = np.meshgrid(p2s, p3s, indexing='ij')
    a1, a2, a3 = np.sqrt(u), np.sqrt(v), np.sqrt(w)
    n = P2.size
    cols = [np.full(n, u), np.full(n, v), np.full(n, w),
            2 * a1 * a2 * np.cos(P2.ravel()),
            2 * a1 * a2 * np.sin(P2.ravel()),
            2 * a1 * a3 * np.cos(P3.ravel()),
            2 * a1 * a3 * np.sin(P3.ravel()),
            2 * a2 * a3 * np.cos((P2 - P3).ravel()),
            -2 * a2 * a3 * np.sin((P2 - P3).ravel())]
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32), n


def _c_energy_block(us, vs, p2, p3):
    U, V = np.meshgrid(us, vs, indexing='ij')
    mask = (U + V) < 1.0
    U, V = U[mask], V[mask]
    W = 1.0 - U - V
    a1, a2, a3 = np.sqrt(U), np.sqrt(V), np.sqrt(np.clip(W, 0, None))
    cols = [U, V, W,
            2 * a1 * a2 * np.cos(p2), 2 * a1 * a2 * np.sin(p2),
            2 * a1 * a3 * np.cos(p3), 2 * a1 * a3 * np.sin(p3),
            2 * a2 * a3 * np.cos(p2 - p3), -2 * a2 * a3 * np.sin(p2 - p3)]
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32), U.size


def block_refine(stats, theta0, rounds=7, n_ph=61, n_en=41):
    cur = [float(x) for x in theta0]
    w_ph, w_en = 0.5, 0.08
    hist = []
    n_ev = 0
    for r in range(rounds):
        # phase block
        p2s = np.linspace(cur[3] - w_ph, cur[3] + w_ph, n_ph)
        p3s = np.linspace(cur[4] - w_ph, cur[4] + w_ph, n_ph)
        C, n = _c_phase_block(cur[0], cur[1], cur[2], p2s, p3s)
        corr = corr_avg_batch(stats, C)
        k = int(torch.argmax(corr))
        n_ev += n
        cur[3] = float(p2s[k // n_ph])
        cur[4] = float(p3s[k % n_ph])
        # energy block
        us = np.linspace(max(0, cur[0] - w_en), cur[0] + w_en, n_en)
        vs = np.linspace(max(0, cur[1] - w_en), cur[1] + w_en, n_en)
        C, n = _c_energy_block(us, vs, cur[3], cur[4])
        corr = corr_avg_batch(stats, C)
        k = int(torch.argmax(corr))
        U, V = np.meshgrid(us, vs, indexing='ij')
        mask = (U + V) < 1.0
        uu, vv = U[mask][k], V[mask][k]
        n_ev += n
        cur[0], cur[1] = float(uu), float(vv)
        cur[2] = 1.0 - cur[0] - cur[1]
        w_ph *= 0.3
        w_en *= 0.3
        hist.append((n_ev, float(torch.max(corr))))
    return cur, hist, n_ev


def method_new(stats, theta_seed, n_jitter=2, seed=0):
    c_avg, ub = ols_c_init(stats)
    seeds = [list(theta_seed)]
    if c_avg is not None:
        th_ols = project_c_to_params(c_avg)
        if th_ols is not None:
            seeds.insert(0, th_ols)
            rng = np.random.default_rng(seed)
            for _ in range(n_jitter):
                j = [th_ols[0] + rng.normal(0, 0.02),
                     th_ols[1] + rng.normal(0, 0.02),
                     th_ols[2] + rng.normal(0, 0.02),
                     th_ols[3] + rng.normal(0, 0.20),
                     th_ols[4] + rng.normal(0, 0.20)]
                j[0], j[1] = abs(j[0]), abs(j[1])
                if j[0] + j[1] > 0.98:
                    j[0], j[1] = j[0] * 0.5, j[1] * 0.5
                j[2] = 1 - j[0] - j[1]
                seeds.append(j)
    best = None
    total_hist = []
    n_ev_total = 0
    for s in seeds:
        th, hist, n_ev = block_refine(stats, s)
        n_ev_total += n_ev
        for (ne, cv) in hist:
            total_hist.append((n_ev_total - n_ev + ne, cv))
        cavg = corr_avg_point(stats, th)
        if best is None or cavg > best[0]:
            best = (cavg, th)
    return best[1], best[0], ub, total_hist, n_ev_total, len(seeds)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes = modes.to(dev)
    lab2k = {lab: k for k, lab in enumerate(labels)}
    modes_np = modes.detach().cpu().numpy()

    results = {}
    for case in CASES:
        name = case['name']
        tag = case['tag']
        print(f'\n########## {tag}  ({name}) ##########')
        Ie_np = load_experiment(f'{IMG_DIR}/{name}.png',
                                'nipy_spectral', 'isot')
        Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
        pts_e = find_dark_points(Ie_np)

        with open(case['three_json'], encoding='utf-8') as f:
            tj = json.load(f)
        entry = next(e for e in tj['entries']
                     if set(e['modes']) == set(case['combo']))
        theta_start = [entry['energy'][0], entry['energy'][1],
                       entry['energy'][2], entry['phi2'], entry['phi3']]
        if case.get('coarsen'):
            theta_start = [round(theta_start[0] / 0.05) * 0.05,
                           round(theta_start[1] / 0.05) * 0.05,
                           round(theta_start[2] / 0.05) * 0.05,
                           round(theta_start[3] / 0.3) * 0.3,
                           round(theta_start[4] / 0.3) * 0.3]
            s0 = sum(theta_start[:3])
            theta_start[:3] = [x / s0 for x in theta_start[:3]]
        print(f'combo: {case["combo"]}  start corr_avg='
              f'{entry["corr_avg"]:.6f}'
              f'{" (coarsened start)" if case.get("coarsen") else ""}')
        tr = [lab2k[m] for m in case['combo']]
        stats = precompute_stats(modes, Ie, tr)
        c_start = corr_avg_point(stats, theta_start)
        print(f'  start corr_avg (actual) = {c_start:.6f}')

        # ---- Method A ----
        t0 = time.time()
        thA, histA, nA = method_bisection(stats, theta_start)
        tA = time.time() - t0
        cA = corr_avg_point(stats, thA)

        # ---- Method B ----
        t0 = time.time()
        thB, cB, ub, histB, nB, nseeds = method_new(stats, theta_start)
        tB = time.time() - t0

        # ---- structural verification ----
        def struct_check(theta):
            I = render_three_mode_np(modes_np, tr[0], tr[1], tr[2], theta)
            pts_m = find_dark_points(I)
            mt = match_points(pts_e, pts_m)
            return len(pts_m), mt['mean']

        nA_core, dA = struct_check(thA)
        nB_core, dB = struct_check(thB)

        print(f'  Method A (bisection) : corr_avg={cA:.6f}  '
              f'evals={nA}  time={tA:.1f}s  cores={nA_core} '
              f'(exp {len(pts_e)}) meanNN={dA if dA is None else round(dA, 2)}')
        print(f'  Method B (new combo) : corr_avg={cB:.6f}  '
              f'evals={nB}  time={tB:.1f}s  cores={nB_core} '
              f'(exp {len(pts_e)}) meanNN={dB if dB is None else round(dB, 2)}')
        print(f'    OLS upper bound (sigma=2): {ub:.6f} | '
              f'seeds used: {nseeds}')

        results[tag] = {
            'combo': list(case['combo']),
            'start': {'theta': theta_start, 'corr_avg': c_start,
                      'archived_corr_avg': entry['corr_avg']},
            'method_A': {'theta': thA, 'corr_avg': cA, 'evals': nA,
                         'time_s': tA, 'cores': nA_core, 'mean_nn': dA,
                         'hist': histA},
            'method_B': {'theta': thB, 'corr_avg': cB, 'evals': nB,
                         'time_s': tB, 'cores': nB_core, 'mean_nn': dB,
                         'hist': histB, 'ols_ub_sigma2': ub,
                         'n_seeds': nseeds},
        }

    # ---- OLS upper-bound ranking experiment (4Sing top combos) ----
    print('\n########## OLS 上界 vs 现有排序（4奇点 top 组合）##########')
    name = '4Singularities'
    Ie_np = load_experiment(f'{IMG_DIR}/{name}.png',
                            'nipy_spectral', 'isot')
    Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
    with open(CASES[0]['three_json'], encoding='utf-8') as f:
        tj = json.load(f)
    seen, combos = set(), []
    for e in tj['entries']:
        canon = tuple(sorted(m.split('(')[0] + '(' + m.split('(')[1].split(',')[0]
                             + ',' + str(abs(int(m.split(',')[1][:-1]))) + ')'
                             for m in e['modes']))
        if canon not in seen:
            seen.add(canon)
            combos.append(e)
        if len(combos) >= 5:
            break
    print(f'{"combo":44s} {"corr_avg":>9s} {"OLS_ub(s2)":>10s}')
    ub_rows = []
    for e in combos:
        tr = [lab2k[m] for m in e['modes']]
        stats = precompute_stats(modes, Ie, tr)
        _, ub = ols_c_init(stats)
        ub_rows.append((e['modes'], e['corr_avg'], ub))
        print(f'{"+".join(e["modes"]):44s} {e["corr_avg"]:9.4f} {ub:10.4f}')
    results['_ols_ranking_4Sing'] = [
        {'modes': m, 'corr_avg': c, 'ols_ub': u} for (m, c, u) in ub_rows]

    (OUT / 'methods_compare.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding='utf-8')

    # ---- convergence figure ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, tag in zip(axs, ['4Sing', '8Sing1', '4Sing_alt']):
        r = results[tag]
        for meth, col in [('method_A', 'tab:blue'),
                          ('method_B', 'tab:red')]:
            xs = [h[0] for h in r[meth]['hist']]
            ys = [h[1] for h in r[meth]['hist']]
            rb, cur_best = [], -2
            for y in ys:
                cur_best = max(cur_best, y)
                rb.append(cur_best)
            ax.plot(xs, rb, marker='o', ms=3, color=col,
                    label=f'{meth} (final {r[meth]["corr_avg"]:.4f}, '
                          f'{r[meth]["time_s"]:.1f}s)')
        ax.axhline(r['start']['corr_avg'], color='gray', ls='--', lw=1,
                   label=f'start {r["start"]["corr_avg"]:.4f}')
        ax.set_xscale('log')
        ax.set_xlabel('evaluated points (cumulative)')
        ax.set_ylabel('best corr_avg so far')
        ax.set_title(f'{tag}  {"+".join(r["combo"])}', fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'convergence_compare.png', dpi=140)
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

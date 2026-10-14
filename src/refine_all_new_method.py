# -*- coding: utf-8 -*-
"""
refine_all_new_method.py -- 新方法(方案三+二+五)推广到全部模式数
=================================================================

Generic N-mode (N=2,3,4) version of the new refinement pipeline:
  (3) OLS closed-form upper bound + analytic manifold projection
  (2) alternating block refinement (dense phase grid + energy simplex)
  (5) multi-start best pick

For each of the remaining 6 images:
  A. upper-bound re-check: OLS bound vs current ranking for top-5 combos
  B. refinement: new method from (archived solution, OLS projection,
     jittered) seeds; two-mode cases also re-scan the scale s
  C. structural verification + comparison table
"""

import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from search_two_mode_lg import rescale_lg
from search_three_mode import eval_c_grid
from structural_check import find_dark_points, match_points

ARCH = ROOT / '归档_八图分解'
OUT = ROOT / 'refine_all_new_method'
OUT.mkdir(exist_ok=True)
SIGMAS = list(range(7))
IMG_DIR = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))

GRIDS = {2: {'n_ph': 81, 'n_en': 81, 'rounds': 8},
         3: {'n_ph': 61, 'n_en': 41, 'rounds': 7},
         4: {'n_ph': 15, 'n_en': 11, 'rounds': 6}}


# ---------------------------------------------------------------------------
# generic N-mode core
# ---------------------------------------------------------------------------

def build_base_fields_N(Fs):
    """Fs: list of (B,H,W) complex -> (B,K,H,W) real; K=N+2*C(N,2)."""
    N = len(Fs)
    fields = [F.real ** 2 + F.imag ** 2 for F in Fs]
    for i in range(N):
        for j in range(i + 1, N):
            X = Fs[i] * Fs[j].conj()
            fields.append(X.real)
            fields.append(X.imag)
    return torch.stack(fields, dim=1)


def c_table_N(params_mat, N):
    """params (P, 2N-1) = [u1..uN, phi2..phiN] -> c (P, K)."""
    P = params_mat.shape[0]
    u = params_mat[:, :N]
    ph = np.concatenate([np.zeros((P, 1)), params_mat[:, N:]], axis=1)
    a = np.sqrt(np.clip(u, 0, None))
    cols = [u[:, i] for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = ph[:, i] - ph[:, j]
            cols.append(2 * a[:, i] * a[:, j] * np.cos(d))
            cols.append(-2 * a[:, i] * a[:, j] * np.sin(d))
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)


def stats_for_combo(modes, Ie, tr):
    dev = modes.device
    Fs = [modes[k].unsqueeze(0) for k in tr]
    G = build_base_fields_N(Fs)[0]  # (K,H,W)
    K = G.shape[0]
    stats = []
    for sg in SIGMAS:
        Gb = gaussian_blur(G, float(sg))
        Ie_b = gaussian_blur(Ie.unsqueeze(0), float(sg))[0]
        me = Ie_b.mean()
        ve = Ie_b.var(unbiased=False)
        Gf = Gb.reshape(K, -1)
        m = Gf.mean(dim=1).unsqueeze(0)
        E = torch.einsum('ki,i->k', Gf, Ie_b.reshape(-1)
                         ).unsqueeze(0) / Gf.shape[1]
        Gram = torch.einsum('ki,li->kl', Gf, Gf
                            ).unsqueeze(0) / Gf.shape[1]
        stats.append((m, E, Gram, me, ve))
    return stats


def corr_avg_batch(stats, C):
    tot = None
    for s in stats:
        v = eval_c_grid(C, *s).reshape(-1)
        tot = v if tot is None else tot + v
    return tot / len(stats)


def corr_avg_point(stats, p, N):
    C = c_table_N(np.array([p], dtype=np.float64), N)
    return float(corr_avg_batch(stats, C)[0])


def ols_c(stats, sigma_idx=2):
    m, E, Gram, me, ve = stats[sigma_idx]
    m9 = m[0].double()
    Gc = (Gram[0] - m9[:, None] * m9[None, :]).double()
    Ec = (E[0] - m[0] * me).double()
    Gc = Gc + 1e-8 * torch.eye(Gc.shape[0], dtype=Gc.dtype,
                               device=Gc.device)
    c = torch.linalg.solve(Gc, Ec)
    ub = float(torch.sqrt((Ec @ c) / ve))
    return c.cpu().numpy(), ub


def project_c_N(c, N):
    b = np.clip(c[:N], 0, None)
    if b.sum() < 1e-9:
        return None
    u = b / b.sum()
    phis = [0.0]
    for j in range(1, N):
        z = complex(c[N + 2 * (j - 1)], c[N + 2 * (j - 1) + 1])
        if abs(z) < 1e-9:
            return None
        phis.append(float(np.angle(z)))
    return list(u) + phis[1:]


def block_refine(stats, p0, N, n_ph, n_en, rounds):
    cur = [float(x) for x in p0]
    w_ph, w_en = 0.5, 0.08
    hist = []
    n_ev = 0
    for r in range(rounds):
        # phase block: (N-1)-dim grid
        axes = [np.linspace(cur[N + j] - w_ph, cur[N + j] + w_ph, n_ph)
                for j in range(N - 1)]
        mesh = np.meshgrid(*axes, indexing='ij')
        ph_mat = np.stack([m.ravel() for m in mesh], axis=1)
        P = ph_mat.shape[0]
        u_mat = np.tile(cur[:N], (P, 1))
        C = c_table_N(np.concatenate([u_mat, ph_mat], axis=1), N)
        corr = corr_avg_batch(stats, C)
        n_ev += P
        k = int(torch.argmax(corr))
        for j in range(N - 1):
            cur[N + j] = float(ph_mat[k, j])
        # energy block: (N-1)-dim simplex grid
        ax_e = [np.linspace(max(0, cur[i] - w_en), min(1, cur[i] + w_en),
                            n_en) for i in range(N - 1)]
        me_ = np.meshgrid(*ax_e, indexing='ij')
        e_mat = np.stack([m.ravel() for m in me_], axis=1)
        e_mat = e_mat[e_mat.sum(axis=1) < 1.0]
        u_mat = np.concatenate(
            [e_mat, (1 - e_mat.sum(axis=1))[:, None]], axis=1)
        ph_rep = np.tile(cur[N:], (len(u_mat), 1))
        C = c_table_N(np.concatenate([u_mat, ph_rep], axis=1), N)
        corr = corr_avg_batch(stats, C)
        n_ev += len(u_mat)
        k = int(torch.argmax(corr))
        cur[:N] = [float(x) for x in u_mat[k]]
        w_ph *= 0.3
        w_en *= 0.3
        hist.append((n_ev, float(torch.max(corr))))
    return cur, hist, n_ev


def method_new(stats, p_seed, N, n_jitter=2, seed=0):
    """multi-start block refinement; returns (best_params, corr, ub)."""
    c_ols, ub = ols_c(stats)
    seeds = [list(p_seed)]
    p_ols = project_c_N(c_ols, N)
    if p_ols is not None:
        seeds.insert(0, p_ols)
        rng = np.random.default_rng(seed)
        for _ in range(n_jitter):
            j = [p_ols[i] + rng.normal(0, 0.02 if i < N else 0.20)
                 for i in range(2 * N - 1)]
            j[:N] = [abs(x) for x in j[:N]]
            sj = sum(j[:N])
            j[:N] = [x / sj for x in j[:N]]
            seeds.append(j)
    g = GRIDS[N]
    best = None
    for s in seeds:
        th, hist, n_ev = block_refine(stats, s, N, g['n_ph'], g['n_en'],
                                      g['rounds'])
        cavg = corr_avg_point(stats, th, N)
        if best is None or cavg > best[0]:
            best = (cavg, th)
    return best[1], best[0], ub


def refine_two_with_s(modes, Ie, tr, r0, phi0, s0):
    """two-mode case: outer scan over the scale s (statistics rebuilt)."""
    best = None
    for s in np.linspace(s0 - 0.10, s0 + 0.10, 7):
        ms = rescale_lg(modes, float(s))
        stats = stats_for_combo(ms, Ie, tr)
        th, cavg, ub = method_new(stats, [r0, 1 - r0, phi0], 2)
        if best is None or cavg > best[0]:
            best = (cavg, th, float(s), ub)
    return best


# ---------------------------------------------------------------------------
# per-image data extraction
# ---------------------------------------------------------------------------

def canon(modes):
    out = []
    for m in modes:
        kind, rest = m.split('(')
        p, l = rest[:-1].split(',')
        out.append(f'{kind}({p},{abs(int(l))})')
    return tuple(sorted(out))


def extract_case(case):
    """returns (final_sol, top_combos) where sol = dict(N, combo, params)"""
    N = case['N']
    if case['kind'] == 'two_csv':
        df = pd.read_csv(case['path'])
        rows = []
        seen = set()
        for _, row in df.iterrows():
            combo = (row['mode1'], row['mode2'])
            if canon(combo) in seen:
                continue
            seen.add(canon(combo))
            # NOTE: two-mode engine uses the e^{+i phi} convention while
            # the generic N-mode evaluator uses e^{-i phi}; mirror the
            # phase on import so parameters are comparable.
            rows.append({'combo': combo, 'r': float(row['r_energy1']),
                         'phi': -float(row['phi_rad']),
                         'scale': float(row['scale']),
                         'corr_avg_arch': float(row['corr_avg'])})
            if len(rows) >= 5:
                break
        fin = rows[0]
        return ({'N': 2, 'combo': fin['combo'],
                 'params': [fin['r'], 1 - fin['r'], fin['phi']],
                 'scale': fin['scale'],
                 'corr_avg_arch': fin['corr_avg_arch']}, rows)
    if case['kind'] == 'two_refine':
        d = json.load(open(case['path'], encoding='utf-8'))
        cands = d['candidates']
        if isinstance(cands, dict):
            items = list(cands.values())
            champ_pair = d['champion'].split('+')
        else:
            items = cands
            champ_pair = d['champion']
        champ = next(c for c in items
                     if set(c['pair']) == set(champ_pair))
        rows = []
        seen = set()
        for c in items:
            if canon(c['pair']) in seen:
                continue
            seen.add(canon(c['pair']))
            rows.append({'combo': tuple(c['pair']), 'r': c['r'],
                         'phi': -c['phi_rad'], 'scale': c['scale'],
                         'corr_avg_arch': c['corr_avg']})
        return ({'N': 2, 'combo': tuple(champ['pair']),
                 'params': [champ['r'], 1 - champ['r'],
                            -champ['phi_rad']],
                 'scale': champ['scale'],
                 'corr_avg_arch': champ['corr_avg']}, rows)
    if case['kind'] == 'three':
        d = json.load(open(case['path'], encoding='utf-8'))
        rows = []
        seen = set()
        for e in d['entries']:
            if canon(e['modes']) in seen:
                continue
            seen.add(canon(e['modes']))
            rows.append({'combo': tuple(e['modes']),
                         'params': list(e['energy']) + [e['phi2'],
                                                        e['phi3']],
                         'corr_avg_arch': e['corr_avg'],
                         'verdict': e.get('verdict')})
            if len(rows) >= 5:
                break
        bp = d.get('best_pass')
        fin_src = bp if bp is not None else d['entries'][0]
        fin = {'N': 3, 'combo': tuple(fin_src['modes']),
               'params': list(fin_src['energy']) + [fin_src['phi2'],
                                                    fin_src['phi3']],
               'corr_avg_arch': fin_src['corr_avg']}
        return fin, rows
    if case['kind'] == 'four':
        d = json.load(open(case['path'], encoding='utf-8'))
        rows = []
        seen = set()
        for e in d['entries']:
            if canon(e['modes']) in seen:
                continue
            seen.add(canon(e['modes']))
            rows.append({'combo': tuple(e['modes']),
                         'params': list(e['energy']) + list(e['phis']),
                         'corr_avg_arch': e['corr_avg'],
                         'verdict': e.get('verdict')})
            if len(rows) >= 5:
                break
        bp = d.get('best_pass')
        fin_src = bp if bp is not None else d['entries'][0]
        fin = {'N': 4, 'combo': tuple(fin_src['modes']),
               'params': list(fin_src['energy']) + list(fin_src['phis']),
               'corr_avg_arch': fin_src['corr_avg']}
        return fin, rows
    raise ValueError(case['kind'])


CASES = [
    {'tag': '5Sing', 'name': '5Singularities', 'N': 2,
     'kind': 'two_csv',
     'path': ARCH / '5奇点' / '1_两模搜索' / '5Singularities_top.csv'},
    {'tag': '6Sing', 'name': '6Singularities', 'N': 2,
     'kind': 'two_refine',
     'path': ARCH / '6奇点' / '3_精密微调' / 'refine_6sing_results.json'},
    {'tag': '8Sing2', 'name': '8Singularities(2)', 'N': 2,
     'kind': 'two_refine',
     'path': ARCH / '8奇点(2)' / '2_两模微调与验收'
             / '8Singularities(2)_refine.json'},
    {'tag': '9Sing', 'name': '9Singularities', 'N': 3,
     'kind': 'three',
     'path': ARCH / '9奇点' / '3_三模搜索与验收'
             / '9Singularities_three_mode.json'},
    {'tag': '11Sing', 'name': '11Singularities', 'N': 3,
     'kind': 'three',
     'path': ARCH / '11奇点' / '3_三模搜索与验收'
             / '11Singularities_three_mode.json'},
    {'tag': '10Sing', 'name': '10Singularities', 'N': 4,
     'kind': 'four',
     'path': ARCH / '10奇点' / '4_四模搜索与验收'
             / '10Singularities_four_mode_guided.json'},
]


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    modes = modes.to(dev)
    lab2k = {lab: k for k, lab in enumerate(labels)}
    modes_np = modes.detach().cpu().numpy()

    results = {}
    for case in CASES:
        tag = case['tag']
        name = case['name']
        print(f'\n########## {tag} ({name}) ##########')
        Ie_np = load_experiment(f'{IMG_DIR}/{name}.png',
                                'nipy_spectral', 'isot')
        Ie = torch.tensor(Ie_np, dtype=torch.float32, device=dev)
        pts_e = find_dark_points(Ie_np)
        fin, top_rows = extract_case(case)
        N = fin['N']

        # ---- A. upper-bound re-check over top combos ----
        print(f'  [A] upper-bound re-check (top {len(top_rows)} combos):')
        ub_table = []
        for row in top_rows:
            tr = [lab2k[m] for m in row['combo']]
            stats = stats_for_combo(modes, Ie, tr)
            _, ub = ols_c(stats)
            ub_table.append({'combo': list(row['combo']),
                             'corr_avg_arch': row['corr_avg_arch'],
                             'ols_ub_s2': ub})
            print(f'      {"+".join(row["combo"]):46s} '
                  f'arch={row["corr_avg_arch"]:.4f}  ub={ub:.4f}')

        # ---- B. refinement of the final solution ----
        print(f'  [B] refinement of final solution: '
              f'{"+".join(fin["combo"])}')
        tr = [lab2k[m] for m in fin['combo']]
        t0 = time.time()
        if N == 2:
            cavg_new, p_new, s_new, ub = refine_two_with_s(
                modes, Ie, tr, fin['params'][0], fin['params'][2],
                fin['scale'])
        else:
            stats = stats_for_combo(modes, Ie, tr)
            p_new, cavg_new, ub = method_new(stats, fin['params'], N)
            s_new = None
        t_new = time.time() - t0

        # recompute archived start value with the same 7-sigma protocol
        if N == 2:
            ms0 = rescale_lg(modes, fin['scale'])
            stats0 = stats_for_combo(ms0, Ie, tr)
            c_arch = corr_avg_point(stats0, fin['params'], 2)
        else:
            c_arch = corr_avg_point(stats, fin['params'], N)

        # ---- C. structural verification ----
        def render_N(params, s):
            mnp = modes_np if s is None else \
                rescale_lg(modes, s).detach().cpu().numpy()
            Fs = [np.sqrt(max(params[i], 0)) * mnp[tr[i]]
                  for i in range(N)]
            ph = [0.0] + list(params[N:])
            E = sum(Fs[i] * np.exp(1j * ph[i]) for i in range(N))
            I = np.abs(E) ** 2
            return I / max(I.max(), 1e-12)

        I_new = render_N(p_new, s_new)
        pts_m = find_dark_points(I_new)
        mt = match_points(pts_e, pts_m)
        n_ok = abs(len(pts_m) - len(pts_e)) <= 1
        d_ok = mt['mean'] is not None and mt['mean'] < 12
        verdict = 'PASS' if (n_ok and d_ok) else 'fail'

        print(f'      archived start (7sig) : {c_arch:.6f}')
        print(f'      new method            : {cavg_new:.6f}  '
              f'({t_new:.1f}s)  cores={len(pts_m)} (exp {len(pts_e)}) '
              f'meanNN={mt["mean"] if mt["mean"] is None else round(mt["mean"], 2)}'
              f'  {verdict}')
        print(f'      OLS upper bound (s2)  : {ub:.6f}')
        if s_new is not None:
            print(f'      optimal scale s = {s_new:.4f} '
                  f'(was {fin["scale"]:.4f})')

        results[tag] = {
            'name': name, 'N': N, 'combo': list(fin['combo']),
            'ub_table': ub_table,
            'arch_corr_avg_7sig': c_arch,
            'arch_corr_avg_original': fin['corr_avg_arch'],
            'new_corr_avg': cavg_new,
            'new_params': list(p_new),
            'new_scale': s_new,
            'ols_ub_s2': ub,
            'new_cores': len(pts_m), 'exp_cores': len(pts_e),
            'new_mean_nn': mt['mean'], 'new_verdict': verdict,
            'time_s': t_new,
        }

    (OUT / 'all_results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding='utf-8')

    print('\n===== SUMMARY =====')
    print(f'{"img":8s} {"N":>2s} {"archived":>10s} {"new":>10s} '
          f'{"delta":>8s} {"verdict":>8s}')
    for tag, r in results.items():
        print(f'{tag:8s} {r["N"]:2d} {r["arch_corr_avg_7sig"]:10.6f} '
              f'{r["new_corr_avg"]:10.6f} '
              f'{r["new_corr_avg"] - r["arch_corr_avg_7sig"]:+8.6f} '
              f'{r["new_verdict"]:>8s}')
    print(f'\n[done] outputs in {OUT}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
stage13b_4sing_morph_dof.py -- 形态候选的完整自由度精修与判定
================================================================
stage13a 结论：朴素管线（无平移/各向异性/旋转）下，三模空间里
"形态通过"的组合结构不过（7 核），"结构通过"的组合形态不过
（轮廓过圆滑 cvR=0.21）。但已知额外自由度会显著改变判定
（现解：朴素 0.8892/6 核 → v3 管线 0.9163/5 核）。

本脚本取 stage13a 的 top 候选（形态通过 + 形态近通过 + 两个基线），
在**完整自由度管线**（全局平移 + 逐模各向异性 sx/sy + 逐模旋转）
下重做坐标下降精修（目标 C_s02，约束结构 PASS），随后统一用
形态守门员 + C_s02 + 结构三项评估，给出可替换候选。

判定协议：
  可替换 = 形态 PASS 且 结构 PASS 且 C_s02 ≥ 0.90
  最优 = 形态 PASS 且 结构 PASS 中 C_s02 最高者
Outputs: 归档_八图分解/4奇点_形态重搜索/stage13b_morph_dof.json
"""
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import shift as ndshift
from scipy.ndimage import rotate as ndrotate

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from structural_check import find_dark_points, match_points
from stage0_diagnosis import parse_md
from morphology_metrics import morph_compare

OUT = ROOT / '归档_八图分解' / '4奇点_形态重搜索'
OUT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
NAME = '4Singularities'
CN = '4奇点'

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()
Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
Ie_b2 = gaussian_blur(torch.tensor(Ie, dtype=torch.float32)
                      .unsqueeze(0), 2.0)[0].numpy()
pts_e = find_dark_points(Ie)
N_EXP = len(pts_e)


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def blur2(I):
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy()


def _fit_center(a, H, W):
    out = np.zeros((a.shape[0], H, W), np.float64)
    h, w = a.shape[-2], a.shape[-1]
    if h >= H:
        s_ = (h - H) // 2
        s_s, s_d = slice(s_, s_ + H), slice(0, H)
    else:
        s_ = (H - h) // 2
        s_s, s_d = slice(0, h), slice(s_, s_ + h)
    if w >= W:
        sx_ = (w - W) // 2
        sx_s, sx_d = slice(sx_, sx_ + W), slice(0, W)
    else:
        sx_ = (W - w) // 2
        sx_s, sx_d = slice(0, w), slice(sx_, sx_ + w)
    out[:, s_d, sx_d] = a[:, s_s, sx_s]
    return out


def zoom_axis_c(Fx, s, axis):
    if abs(s - 1.0) < 1e-6:
        return Fx
    Hh, Ww = Fx.shape
    t = np.stack([Fx.real, Fx.imag], 0)[None]
    t = torch.tensor(t, dtype=torch.float32)
    Hn = int(round(Hh * s)) if axis == 0 else Hh
    Wn = int(round(Ww * s)) if axis == 1 else Ww
    t = torch.nn.functional.interpolate(
        t, size=(Hn, Wn), mode='bilinear',
        align_corners=False)[0].numpy().astype(np.float64)
    b = _fit_center(t, Hh, Ww)
    return b[0] + 1j * b[1]


def render_dof(combo, theta, sx, sy, shiftv, rot):
    N = len(combo)
    E = np.zeros(modes_np[0].shape, dtype=np.complex128)
    for i, m in enumerate(combo):
        Fm = modes_np[lab2k[m]]
        if abs(sx[i] - 1) > 1e-6 or abs(sy[i] - 1) > 1e-6:
            Fm = zoom_axis_c(Fm, sx[i], 1)
            Fm = zoom_axis_c(Fm, sy[i], 0)
        if abs(rot[i]) > 1e-9:
            Fm = ndrotate(Fm.real, rot[i], reshape=False, order=1) + \
                1j * ndrotate(Fm.imag, rot[i], reshape=False, order=1)
        phi = 0.0 if i == 0 else theta[N + i - 1]
        E = E + np.sqrt(max(theta[i], 0)) * Fm * np.exp(1j * phi)
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    if abs(shiftv[0]) > 1e-6 or abs(shiftv[1]) > 1e-6:
        I = ndshift(I, shiftv, order=1, mode='nearest')
        I = np.clip(I, 0, None)
    return I


def ev_full(combo, theta, sx, sy, shiftv, rot, want_morph=False):
    N = len(combo)
    Im = render_dof(combo, theta, sx, sy, shiftv, rot)
    v0 = pcc(Ie, Im)
    v2 = pcc(Ie_b2, blur2(Im))
    C2 = 0.5 * (v0 + v2)
    pts_m = find_dark_points(Im)
    mt = match_points(pts_e, pts_m)
    dists = np.array(mt['dists']) if mt['dists'] else np.array([99.])
    struct = (abs(len(pts_m) - N_EXP) <= 1
              and mt['mean'] is not None and mt['mean'] < 12)
    max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
    r = {'C_s02': float(C2), 'v0': float(v0), 'v2': float(v2),
         'struct': bool(struct), 'cores': len(pts_m),
         'mean_nn': float(mt['mean']) if mt['mean'] is not None else 99.0,
         'max10': max10, 'theta': list(theta), 'sx': list(sx),
         'sy': list(sy), 'shift': list(shiftv), 'rot': list(rot),
         '_I': Im}
    if want_morph:
        rep = morph_compare(Ie, Im)
        r.update({'morph': rep, 'morph_pass': rep['morph_pass'],
                  'central_delta': rep['central_delta'],
                  'cv_ratio': rep['cv_ratio'],
                  'shape_dist': rep['shape_dist']})
    return r


def jitter(theta, rng, N, s_en=0.010, s_ph=0.035):
    u = np.array(theta[:N]) + rng.normal(0, s_en, N)
    u = np.clip(u, 0.005, None)
    u = u / u.sum()
    ph = np.array(theta[N:]) + rng.normal(0, s_ph, N - 1)
    return list(u) + list(ph)


def refine(combo, theta0, sx0, sy0, shift0, rot0, rounds=2,
           budget=120):
    N = len(combo)
    rot_idx = [i for i, m in enumerate(combo) if m.startswith('HG')]
    rng = np.random.default_rng(1300 + N + len(combo))
    base = ev_full(combo, theta0, sx0, sy0, shift0, rot0, True)
    margin = max(14.0, base['max10'])
    best = base
    cur = base
    for rnd in range(rounds):
        for k in range(budget):
            th = jitter(cur['theta'], rng, N)
            r = ev_full(combo, th, cur['sx'], cur['sy'], cur['shift'],
                        cur['rot'])
            if r['struct'] and r['max10'] <= margin \
                    and r['C_s02'] > best['C_s02']:
                best = r
                cur = r
        for step, rad in [(1.0, 2.0), (0.25, 0.5)]:
            for dy in np.arange(-rad, rad + 1e-9, step):
                for dx in np.arange(-rad, rad + 1e-9, step):
                    sh = [cur['shift'][0] + dy, cur['shift'][1] + dx]
                    r = ev_full(combo, cur['theta'], cur['sx'],
                                cur['sy'], sh, cur['rot'])
                    if r['struct'] and r['max10'] <= margin \
                            and r['C_s02'] > best['C_s02']:
                        best = r
                        cur = r
        for step, rad in [(0.02, 0.06), (0.01, 0.02)]:
            for i in range(N):
                for ax in (0, 1):
                    for d in np.arange(-rad, rad + 1e-9, step):
                        sxv = list(cur['sx'])
                        syv = list(cur['sy'])
                        tgt = sxv if ax == 0 else syv
                        tgt[i] = float(np.clip(tgt[i] + d, 0.7, 1.4))
                        r = ev_full(combo, cur['theta'], sxv, syv,
                                    cur['shift'], cur['rot'])
                        if r['struct'] and r['max10'] <= margin \
                                and r['C_s02'] > best['C_s02']:
                            best = r
                            cur = r
        for step, rad in [(1.0, 5.0), (0.25, 1.0)]:
            for i in rot_idx:
                for d in np.arange(-rad, rad + 1e-9, step):
                    rv = list(cur['rot'])
                    rv[i] = float(np.clip(rv[i] + d, -20.0, 20.0))
                    r = ev_full(combo, cur['theta'], cur['sx'],
                                cur['sy'], cur['shift'], rv)
                    if r['struct'] and r['max10'] <= margin \
                            and r['C_s02'] > best['C_s02']:
                        best = r
                        cur = r
    final = ev_full(combo, best['theta'], best['sx'], best['sy'],
                    best['shift'], best['rot'], True)
    final['improved_over_start'] = bool(
        final['C_s02'] >= base['C_s02'] + 1e-9)
    final['start'] = {k: v for k, v in base.items() if k != '_I'}
    return final


# ---------------- candidates ----------------
def main():
    a = json.load(open(OUT / 'stage13a_morph_search.json',
                       encoding='utf-8'))
    V3 = json.load(open(ROOT / '归档_八图分解' / '终检精修v3_2026-10-12'
                        / 'terminal_inspection_v3.json',
                        encoding='utf-8'))
    acc_new = V3[CN]['accepted']

    cands = []
    seen = set()
    for r in a['ranked'][:14]:
        k = tuple(r['combo'])
        if k in seen:
            continue
        seen.add(k)
        N = len(r['combo'])
        cands.append({'combo': list(r['combo']),
                      'theta': list(r['theta']),
                      'sx': [1.0] * N, 'sy': [1.0] * N,
                      'shift': [0.0, 0.0], 'rot': [0.0] * N,
                      'tag': 'morph' if r['morph_pass'] else 'near'})

    combo_old, theta_old, scale_old = parse_md(
        ROOT / '归档_八图分解' / '最新结果' / f'{CN}_参数与指标.md')
    if tuple(combo_old) not in seen:
        cands.append({'combo': list(combo_old),
                      'theta': list(theta_old),
                      'sx': [scale_old] * len(combo_old),
                      'sy': [scale_old] * len(combo_old),
                      'shift': [0.0, 0.0],
                      'rot': [0.0] * len(combo_old),
                      'tag': 'historic'})
    if tuple(V3[CN]['combo']) not in seen:
        cands.append({'combo': list(V3[CN]['combo']),
                      'theta': [float(x) for x in acc_new['theta']],
                      'sx': [float(x) for x in acc_new['sx']],
                      'sy': [float(x) for x in acc_new['sy']],
                      'shift': [float(x) for x in acc_new['shift']],
                      'rot': [float(x) for x in acc_new.get('rot',
                                                            [0.0] * 3)],
                      'tag': 'current'})

    print(f'[setup] {len(cands)} candidates to refine with full DOF')
    records = []
    for ci, c in enumerate(cands):
        t0 = time.time()
        r = refine(c['combo'], c['theta'], c['sx'], c['sy'],
                   c['shift'], c['rot'])
        r['combo'] = c['combo']
        r['tag'] = c['tag']
        records.append(r)
        print(f'  [{ci+1}/{len(cands)}] {"+".join(c["combo"]):44s} '
              f'C_s02={r["C_s02"]:.4f} morph='
              f'{"PASS" if r["morph_pass"] else "fail"} '
              f'centralΔ={r["central_delta"]:.3f} '
              f'cvR={r["cv_ratio"]:.2f} shape={r["shape_dist"]:.3f} '
              f'cores={r["cores"]}/{N_EXP} struct={r["struct"]} '
              f'({time.time()-t0:.0f}s)')

    records.sort(key=lambda r: (0 if (r['morph_pass'] and r['struct'])
                                else 1, -r['C_s02']))
    usable = [r for r in records if r['morph_pass'] and r['struct']]
    champion = usable[0] if usable else None
    cur = next(r for r in records if r['tag'] == 'current')

    print('\n=== ranked (morph+struct PASS first, then C_s02) ===')
    for k, r in enumerate(records[:12]):
        ok = r['morph_pass'] and r['struct']
        print(f'  {k+1:2d}.{"★" if ok else " "} '
              f'{"+".join(r["combo"]):44s} C_s02={r["C_s02"]:.4f} '
              f'centralΔ={r["central_delta"]:.3f} '
              f'cvR={r["cv_ratio"]:.2f} shape={r["shape_dist"]:.3f} '
              f'cores={r["cores"]}/{N_EXP}')

    verdict = {
        'champion': ({k: v for k, v in champion.items() if k != '_I'}
                     if champion else None),
        'current_v3': {k: v for k, v in cur.items() if k != '_I'},
        'replace': bool(champion is not None
                        and champion['C_s02'] >= cur['C_s02'] - 0.01),
        'n_morph_struct_pass': len(usable),
        'records': [{k: v for k, v in r.items() if k != '_I'}
                    for r in records]}
    (OUT / 'stage13b_morph_dof.json').write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8')
    print(f'\n[current] C_s02={cur["C_s02"]:.4f} morph='
          f'{"PASS" if cur["morph_pass"] else "fail"} '
          f'struct={cur["struct"]}')
    if champion:
        print(f'[champion] {"+".join(champion["combo"])} '
              f'C_s02={champion["C_s02"]:.4f} '
              f'(Δ vs current '
              f'{champion["C_s02"]-cur["C_s02"]:+.4f}) '
              f'centralΔ={champion["central_delta"]:.3f} '
              f'cvR={champion["cv_ratio"]:.2f} '
              f'cores={champion["cores"]}/{N_EXP}')
        print(f'[replace] {verdict["replace"]}')
    else:
        print('[champion] NONE — 完整自由度下仍无"形态+结构"双通过解')
    print(f'[done] {OUT}')


if __name__ == '__main__':
    main()

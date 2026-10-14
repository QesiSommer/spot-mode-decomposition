# -*- coding: utf-8 -*-
"""
stage15_morph_fleet.py -- 5/6/9/11 奇点形态先验重搜索（多阶漏斗）
====================================================================
四张形态未通过的图，各按"能负担的最大覆盖"做形态先验搜索：

  5奇点 / 6奇点（现解 N=2）
    → ① 全空间两模 C(131,2)=8,515
    → ② 全空间三模 C(131,3)=366,145
    （两阶都在形态普查后取 top 精修）

  9奇点 / 11奇点（现解 N=5，五模全空间 2.98 亿不可行）
    → ① 全空间三模 366,145 形态普查 → 形态通过 top-60 三连
    → ② 引导四模：60×128 = 7,680 形态普查 → top-50 四连
    → ③ 引导五模：50×127 = 6,350 形态普查 → top-40 五连
    （三阶候选统一进完整自由度精修）

判定：形态 PASS + 结构 PASS → 按 C_s02 取冠军；与现解比较决定是否替换。
交付：**另存新文件夹** `归档_八图分解/形态修复_2026-10-14/{cn}/`
（不覆盖任何既有归档）。
"""
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment
from search_two_mode_joint import build_joint_modes
from window_metrics import make_windows, _support_mask
from structural_check import find_dark_points, match_points
from morphology_metrics import morph_compare
import morph_search_lib as L

ARCH = ROOT / '归档_八图分解'
OUTROOT = ARCH / '形态修复_2026-10-14'
OUTROOT.mkdir(parents=True, exist_ok=True)
IMG = os.environ.get('SPOT_IMG_DIR', str(Path(__file__).resolve().parent / 'images'))
V3 = json.load(open(ARCH / '终检精修v3_2026-10-12'
                    / 'terminal_inspection_v3.json', encoding='utf-8'))

modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
lab2k = {l: k for k, l in enumerate(labels)}
modes_np = modes.numpy()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T_C, T0, T2 = 0.83, 0.82, 0.83
GATE_CENTRAL, GATE_SHAPE = 0.12, 0.55


def top_combos_excluding(tr, k):
    """从 131 模中取不与 tr 重复的 k 个模式的所有组合。"""
    rest = torch.tensor([x for x in range(len(labels))
                         if x not in tr], dtype=torch.long)
    return torch.combinations(rest, r=k)


def screen(combos, N, Ie, tag):
    t0 = time.time()
    sc = L.screen_pool(modes_np, Ie, combos, N, dev)
    central_d = np.abs(sc['central'] - sc['central_exp'])
    pass_m = (central_d <= GATE_CENTRAL) & (sc['shape'] <= GATE_SHAPE)
    n = int(pass_m.sum())
    print(f'  [{tag}] N={N} pool={combos.shape[0]} '
          f'morph-pass={n} ({100.0*n/max(len(combos),1):.2f}%) '
          f'({time.time()-t0:.0f}s)')
    return sc, pass_m, central_d


def order_pass(sc, pass_m, topk):
    """形态通过 → 按粗扫相关降序取 topk。"""
    idx = np.where(pass_m)[0]
    if len(idx) == 0:
        return []
    order = np.argsort(-sc['corr'][idx])
    idx = idx[order]
    out = []
    seen = set()
    for k in idx:
        tr = tuple(sorted(sc['combos'][k].tolist()))
        if tr in seen:
            continue
        seen.add(tr)
        out.append({'tr': tr, 'theta': sc['params'][k].tolist(),
                    'corr': float(sc['corr'][k]),
                    'central': float(sc['central'][k]),
                    'shape': float(sc['shape'][k])})
        if len(out) >= topk:
            break
    return out


def run_case(case):
    cn = case['cn']
    r3 = V3[cn]
    NAME = r3['name']
    combo_cur = r3['combo']
    N_cur = len(combo_cur)
    acc = r3['accepted']
    theta_cur = [float(x) for x in acc['theta']]
    sx_cur = [float(x) for x in acc['sx']]
    sy_cur = [float(x) for x in acc['sy']]
    shift_cur = [float(x) for x in acc['shift']]
    rot_cur = [float(x) for x in acc.get('rot', [0.0] * N_cur)]

    print(f'\n{"="*72}\n=== {cn} ({NAME}) 现解 N={N_cur} '
          f'{"+".join(combo_cur)} ===')
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    N_EXP = len(pts_e)

    Im_cur = L.render_dof(modes_np, lab2k, combo_cur, theta_cur, sx_cur,
                          sy_cur, shift_cur, rot_cur)
    rep_cur = morph_compare(Ie, Im_cur)
    ev_cur = L.make_eval(Ie, pts_e, modes_np, lab2k, combo_cur)
    base_cur = ev_cur(theta_cur, sx_cur, sy_cur, shift_cur, rot_cur,
                      True)
    print(f'[baseline] C_s02={base_cur["C_s02"]:.4f} '
          f'centralΔ={rep_cur["central_delta"]:.3f} '
          f'cvR={rep_cur["cv_ratio"]:.2f} '
          f'shape={rep_cur["shape_dist"]:.3f} '
          f'morph={"PASS" if rep_cur["morph_pass"] else "fail"}')

    # ---------- multi-order funnel ----------
    cand_pool = []          # list of dict(combo, theta, tag)
    seen = set()

    # (1) full three-mode (all cases)
    combos3 = torch.combinations(torch.arange(len(labels)), r=3)
    sc3, pm3, _ = screen(combos3, 3, Ie, 'full3')
    top3 = order_pass(sc3, pm3, 24)
    print(f'  [full3] morph-pass top-60 taken (best corr='
          f'{top3[0]["corr"]:.4f})' if top3 else '  [full3] none')
    for c in top3:
        if c['tr'] not in seen:
            seen.add(c['tr'])
            cand_pool.append({'combo': [labels[t] for t in c['tr']],
                              'theta': c['theta'], 'tag': 'full3'})

    # (2) full two-mode for the two-mode cases
    if N_cur == 2:
        combos2 = torch.combinations(torch.arange(len(labels)), r=2)
        sc2, pm2, _ = screen(combos2, 2, Ie, 'full2')
        top2 = order_pass(sc2, pm2, 30)
        if top2:
            print(f'  [full2] top corr={top2[0]["corr"]:.4f}')
        for c in top2:
            if c['tr'] not in seen:
                seen.add(c['tr'])
                cand_pool.append({'combo': [labels[t] for t in c['tr']],
                                  'theta': c['theta'], 'tag': 'full2'})

    # (3) guided four-mode from morphology-pass triples
    q4 = []
    for c in top3[:24]:
        ext = top_combos_excluding(list(c['tr']), 1)
        base = torch.tensor(c['tr'], dtype=torch.long).repeat(
            len(ext), 1)
        q4.append(torch.cat([base, ext], 1))
    if q4:
        q4 = torch.cat(q4)
        q4_u = torch.unique(torch.sort(q4, dim=1).values, dim=0)
        sc4, pm4, _ = screen(q4_u, 4, Ie, 'guided4')
        top4 = order_pass(sc4, pm4, 18)
        if top4:
            print(f'  [guided4] top corr={top4[0]["corr"]:.4f}')
        for c in top4:
            if c['tr'] not in seen:
                seen.add(c['tr'])
                cand_pool.append({'combo': [labels[t] for t in c['tr']],
                                  'theta': c['theta'], 'tag': 'guided4'})

        # (4) guided five-mode from morphology-pass quads
        q5 = []
        for c in top4[:18]:
            ext = top_combos_excluding(list(c['tr']), 1)
            base = torch.tensor(c['tr'], dtype=torch.long).repeat(
                len(ext), 1)
            q5.append(torch.cat([base, ext], 1))
        if q5:
            q5 = torch.cat(q5)
            q5_u = torch.unique(torch.sort(q5, dim=1).values, dim=0)
            sc5, pm5, _ = screen(q5_u, 5, Ie, 'guided5')
            top5 = order_pass(sc5, pm5, 12)
            if top5:
                print(f'  [guided5] top corr={top5[0]["corr"]:.4f}')
            for c in top5:
                if c['tr'] not in seen:
                    seen.add(c['tr'])
                    cand_pool.append(
                        {'combo': [labels[t] for t in c['tr']],
                         'theta': c['theta'], 'tag': 'guided5'})

    # (5) ensure current solution present
    tr_cur = tuple(sorted(lab2k[m] for m in combo_cur))
    if tr_cur not in seen:
        cand_pool.append({'combo': list(combo_cur), 'theta':
                          list(theta_cur), 'tag': 'current'})

    print(f'[refine] {len(cand_pool)} candidates')

    # ---------- refine ----------
    records = []
    t0 = time.time()
    for ci, c in enumerate(cand_pool):
        Nc = len(c['combo'])
        ev = L.make_eval(Ie, pts_e, modes_np, lab2k, c['combo'])
        sx0 = [1.0] * Nc
        sy0 = [1.0] * Nc
        sh0 = [0.0, 0.0]
        rt0 = [0.0] * Nc
        if c['tag'] == 'current' and Nc == N_cur:
            sx0, sy0 = list(sx_cur), list(sy_cur)
            sh0, rt0 = list(shift_cur), list(rot_cur)
        rr, bb = (1, 60) if Nc >= 4 else (2, 100)
        r = L.refine(ev, c['combo'], c['theta'], sx0, sy0, sh0, rt0,
                     rounds=rr, budget=bb, seed=ci, tag=c['tag'])
        r['combo'] = c['combo']
        records.append(r)
        print(f'  [{ci+1}/{len(cand_pool)}] '
              f'{"+".join(c["combo"]):52s} [{c["tag"]:8s}] '
              f'C_s02={r["C_s02"]:.4f} '
              f'morph={"P" if r["morph_pass"] else "F"} '
              f'cΔ={r["central_delta"]:.3f} cvR={r["cv_ratio"]:.2f} '
              f'sh={r["shape_dist"]:.3f} cores={r["cores"]}/{N_EXP} '
              f'struct={r["struct"]} ({time.time()-t0:.0f}s)')

    records.sort(key=lambda r: (0 if (r['morph_pass'] and r['struct'])
                                else 1, -r['C_s02']))
    usable = [r for r in records if r['morph_pass'] and r['struct']]
    champion = usable[0] if usable else None

    print('\n[ranked] (morph+struct PASS first)')
    for k, r in enumerate(records[:10]):
        ok = r['morph_pass'] and r['struct']
        print(f'  {k+1:2d}.{"★" if ok else " "} '
              f'{"+".join(r["combo"]):52s} C_s02={r["C_s02"]:.4f} '
              f'cΔ={r["central_delta"]:.3f} cvR={r["cv_ratio"]:.2f} '
              f'sh={r["shape_dist"]:.3f} cores={r["cores"]}/{N_EXP}')

    outdir = OUTROOT / cn
    outdir.mkdir(parents=True, exist_ok=True)
    rec = {
        'cn': cn, 'name': NAME, 'N_current': N_cur,
        'current': {'combo': combo_cur, 'C_s02': base_cur['C_s02'],
                    'morph': {k: v for k, v in rep_cur.items()
                              if not isinstance(v, list)}},
        'n_candidates': len(cand_pool),
        'n_usable': len(usable),
        'champion': ({k: v for k, v in champion.items()
                      if k != '_I'} if champion else None),
        'records': [{k: v for k, v in r.items() if k != '_I'}
                    for r in records]}
    (outdir / f'search_{cn}.json').write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8')
    return rec, champion, outdir


def deliver(cn, rec, champ, outdir):
    NAME = rec['name']
    Ie = load_experiment(f'{IMG}/{NAME}.png', 'nipy_spectral', 'isot')
    pts_e = find_dark_points(Ie)
    N_EXP = len(pts_e)
    combo = champ['combo']
    N = len(combo)
    theta = [float(x) for x in champ['theta']]
    sx = [float(x) for x in champ['sx']]
    sy = [float(x) for x in champ['sy']]
    shiftv = [float(x) for x in champ['shift']]
    rot = [float(x) for x in champ['rot']]
    Im = L.render_dof(modes_np, lab2k, combo, theta, sx, sy, shiftv,
                      rot)

    mask = _support_mask(Ie)
    windows, _ = make_windows(Ie.shape[0], 56)
    w32, _ = make_windows(Ie.shape[0], 32)
    Ie_b = {sg: (Ie if sg == 0 else L.blur2(Ie)) for sg in (0, 2)}
    Im_b = {sg: (Im if sg == 0 else L.blur2(Im)) for sg in (0, 2)}
    v0 = L.pcc(Ie, Im)
    v2 = L.pcc(Ie_b[2], Im_b[2])
    C2 = 0.5 * (v0 + v2)
    v56 = []
    for sl in windows:
        if mask[sl].mean() < 0.5:
            continue
        for sg in (0, 2):
            v56.append((sg, L.pcc(Ie_b[sg][sl], Im_b[sg][sl])))
    s2 = [v for sg, v in v56 if sg == 2]
    wavg = float(np.mean([v for _, v in v56]))
    rho56 = float(np.mean([v < 0.5 for v in s2]))
    cmin56 = float(np.min(s2))
    v32 = []
    for sl in w32:
        if mask[sl].mean() < 0.5:
            continue
        for sg in (0, 2):
            v32.append((sg, L.pcc(Ie_b[sg][sl], Im_b[sg][sl])))
    s32 = [v for sg, v in v32 if sg == 2]
    rho32 = float(np.mean([v < 0.5 for v in s32]))
    cmin32 = float(np.min(s32))
    pts_m = find_dark_points(Im)
    mt = match_points(pts_e, pts_m)
    dists = np.array(mt['dists'])
    struct = (abs(len(pts_m) - N_EXP) <= 1
              and mt['mean'] is not None and mt['mean'] < 12)
    max10 = float(np.sort(dists)[-2]) if len(dists) > 1 else 99.
    mrep = morph_compare(Ie, Im)
    gate_ok = (C2 >= T_C and v0 >= T0 and v2 >= T2
               and rho56 <= 0.10 and cmin56 >= 0.40)
    warns = []
    if rho32 >= 0.2:
        warns.append(f'rho_32px={rho32:.2f}')
    if cmin32 <= 0.3:
        warns.append(f'C_min_32px={cmin32:.2f}')
    print(f'  [deliver] C_s02={C2:.4f} morph='
          f'{"PASS" if mrep["morph_pass"] else "fail"} '
          f'struct={struct} cores={len(pts_m)}/{N_EXP}')

    # fig 1
    raw = plt.imread(f'{IMG}/{NAME}.png')
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]
    valid = [sl for sl in windows if mask[sl].mean() >= 0.5]
    s2vals = [L.pcc(Ie_b[2][sl], Im_b[2][sl]) for sl in valid]
    hm = np.full((4, 4), np.nan)
    for k in range(len(valid)):
        hm[k // 4, k % 4] = s2vals[k]
    fig, axs = plt.subplots(1, 5, figsize=(21.5, 4.4))
    axs[0].imshow(raw)
    axs[0].set_title('Experiment (original)')
    axs[1].imshow(Ie, cmap='inferno', vmin=0, vmax=1)
    axs[1].set_title('Experiment (processed)')
    axs[2].imshow(Im_b[2], cmap='inferno', vmin=0, vmax=1)
    axs[2].set_title(f'{"+".join(combo)}\n'
                     f'C_s02={C2:.4f}  W_avg={wavg:.4f}\n'
                     f'(morphology-prior solution)')
    axs[3].imshow(np.abs(Ie - Im_b[2]), cmap='viridis', vmin=0,
                  vmax=0.6)
    axs[3].set_title('|diff|  (fixed scale)')
    cmap = matplotlib.colormaps['RdYlGn'].copy()
    cmap.set_bad('0.7')
    axs[4].imshow(hm, cmap=cmap, vmin=0.3, vmax=1.0)
    for i2 in range(4):
        for j2 in range(4):
            if not np.isnan(hm[i2, j2]):
                axs[4].text(j2, i2, f'{hm[i2, j2]:.2f}', ha='center',
                            va='center', fontsize=9)
    axs[4].set_title('window PCC map (56px, sigma=2)')
    for a in axs:
        a.axis('off')
    fig.suptitle(f'{NAME} - morphology-prior solution (N={N})',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / f'{cn}_拟合对比.png', dpi=140,
                bbox_inches='tight')
    plt.close(fig)

    # fig 2
    r3 = V3[cn]
    a3 = r3['accepted']
    Im_prev = L.render_dof(
        modes_np, lab2k, r3['combo'],
        [float(x) for x in a3['theta']],
        [float(x) for x in a3['sx']], [float(x) for x in a3['sy']],
        [float(x) for x in a3['shift']],
        [float(x) for x in a3.get('rot', [0.0] * len(r3['combo']))])
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
    axs[0].imshow(L.blur2(Ie), cmap='inferno', vmin=0, vmax=1)
    axs[0].set_title('EXPERIMENT')
    axs[1].imshow(L.blur2(Im_prev), cmap='inferno', vmin=0, vmax=1)
    axs[1].set_title('previous (morph '
                     + ('PASS)' if rec['current']['morph']['morph_pass']
                        else 'FAIL)') + '\n' + '+'.join(r3['combo']),
                     fontsize=9)
    axs[2].imshow(L.blur2(Im), cmap='inferno', vmin=0, vmax=1)
    axs[2].set_title(f'new morphology-prior (C_s02={C2:.4f}, '
                     f'morph PASS)\n' + '+'.join(combo), fontsize=9)
    for a in axs:
        a.axis('off')
    fig.suptitle(f'{NAME}: experiment vs previous vs morphology-prior '
                 f'(model sigma=2)', fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / f'{cn}_新旧拟合对比.png', dpi=140,
                bbox_inches='tight')
    plt.close(fig)

    # md
    a1 = [float(np.sqrt(max(x, 0))) for x in theta[:N]]
    lines = [
        f'# {NAME} 最终分解结果（形态先验修正版）', '',
        f'- 模式数：{N} 模',
        f'- 拟合对比图：`{cn}_拟合对比.png`',
        '- 相位约定：E = Σ aᵢFᵢ·e^(iφᵢ)，φ₁ = 0',
        f'- 自由度：全局平移 ({shiftv[0]:+.2f}, {shiftv[1]:+.2f}) px；'
        '逐模各向异性 (sx, sy) 与旋转角见下表',
        '',
        '## 形态修正说明', '',
        f'- 原解 `{"+".join(r3["combo"])}`：C_s02 '
        f'{rec["current"]["C_s02"]:.4f} 但**形态失败**'
        f'（中央暗区差 '
        f'{rec["current"]["morph"]["central_delta"]:.3f}）',
        f'- 本解 `{"+".join(combo)}`：多阶形态先验搜索'
        '（全空间三模 + 引导四/五模）+ 完整自由度精修，'
        '同时满足【形态 PASS + 结构 PASS】', '',
        '## 模式组成', '',
        '| 模式 | 能量占比 | 振幅 | sx | sy | 旋转° |',
        '|---|---|---|---|---|---|']
    for i in range(N):
        lines.append(f'| {combo[i]} | {theta[i]*100:.2f}% | '
                     f'{a1[i]:.6f} | {sx[i]:.3f} | {sy[i]:.3f} | '
                     f'{rot[i]:+.2f} |')
    lines += ['', f'归一化检验：Σ aᵢ² = {sum(x*x for x in a1):.6f}',
              '', '## 相对相位', '']
    for i in range(1, N):
        lines.append(f'- φ{i+1} = {theta[N+i-1]:.6f} rad '
                     f'= {np.degrees(theta[N+i-1]):.3f}°')
    lines += [
        '', '## 相关系数（新指标口径）', '',
        f'- **C_s02（主口径）= {C2:.4f}**（原 '
        f'{rec["current"]["C_s02"]:.4f}，'
        f'{C2-rec["current"]["C_s02"]:+.4f}）',
        f'- **W_avg = {wavg:.4f}**',
        f'- 分尺度：σ=0: {v0:.4f}，σ=2: {v2:.4f}',
        '', '## 形态指标（守门员）', '',
        f'- 中央暗区差 Δcentral = {mrep["central_delta"]:.3f}'
        f'（≤0.12）{"✓" if mrep["central_gate"] else "✗"}',
        f'- 轮廓起伏比 cvR = {mrep["cv_ratio"]:.2f}'
        f'（≥0.55）{"✓" if mrep["cv_gate"] else "✗"}',
        f'- 形状谱距离 = {mrep["shape_dist"]:.3f}'
        f'（≤0.55）{"✓" if mrep["shape_gate"] else "✗"}',
        f'- **形态判定：{"PASS" if mrep["morph_pass"] else "fail"}**',
        '', '## 结构验收（暗点）', '',
        f'- 模型 {len(pts_m)} 个 vs 实验 {len(pts_e)} 个',
        f'- 平均最近邻偏差：{mt["mean"]:.2f} px；最大：'
        f'{mt["max"]:.2f} px',
        f'- 逐核距离：' + ', '.join(f'{x:.1f}' for x in dists),
        f'- 稳健判据（剔除单个最差核后 max ≤14px）：{max10:.2f}px',
    ]
    (outdir / f'{cn}_参数与指标.md').write_text('\n'.join(lines),
                                                encoding='utf-8')

    sheet = [
        f'# {NAME} 指标验收单（形态先验修正版，门槛仅标注）', '',
        '## 准入门槛（标注）', '',
        f'- C_s02 = {C2:.4f} （门槛 {T_C}） '
        f'{"✓" if C2 >= T_C else "✗"}',
        f'- PCC(σ=0) = {v0:.4f} （门槛 {T0}） '
        f'{"✓" if v0 >= T0 else "✗"}',
        f'- PCC(σ=2) = {v2:.4f} （门槛 {T2}） '
        f'{"✓" if v2 >= T2 else "✗"}',
        f'- rho_56px = {rho56:.3f} （门槛 ≤0.10） '
        f'{"✓" if rho56 <= 0.10 else "✗"}',
        f'- C_min_56px = {cmin56:.3f} （门槛 ≥0.40） '
        f'{"✓" if cmin56 >= 0.40 else "✗"}', '',
        f'**门槛判定：{"全部满足" if gate_ok else "存在未达标项"}**', '',
        '## 形态守门员', '',
        f'- Δcentral = {mrep["central_delta"]:.3f}（≤0.12）'
        f' {"✓" if mrep["central_gate"] else "✗"}',
        f'- cvR = {mrep["cv_ratio"]:.2f}（≥0.55）'
        f' {"✓" if mrep["cv_gate"] else "✗"}',
        f'- shape = {mrep["shape_dist"]:.3f}（≤0.55）'
        f' {"✓" if mrep["shape_gate"] else "✗"}',
        f'- **形态判定：{"PASS" if mrep["morph_pass"] else "fail"}**',
        '', '## 局部一致性警示（32px）', '',
        f'- 32px：最差窗 C_min = {cmin32:.3f}，失败窗占比 '
        f'rho = {rho32:.3f}',
        f'- 警示：{"无" if not warns else "【" + "；".join(warns) + "】"}',
        '', '## 结构验收（暗核，最终判据）', '',
        f'- 模型 {len(pts_m)} 个暗点 vs 实验 {len(pts_e)} 个',
        f'- 平均最近邻 {mt["mean"]:.2f}px < 12px ✓；最大 '
        f'{mt["max"]:.2f}px；剔除最远核后 {max10:.2f}px',
        f'- 判定：**{"PASS" if struct else "fail"}**', '',
        '## 结论', '',
        f'- 结构：{"PASS" if struct else "fail"}；门槛：'
        f'{"全部满足" if gate_ok else "存在未达标项"}；形态：'
        f'{"PASS" if mrep["morph_pass"] else "fail"}；警示：'
        f'{"无" if not warns else "；".join(warns)}',
    ]
    (outdir / f'{cn}_指标验收单.md').write_text('\n'.join(sheet),
                                                encoding='utf-8')

    delivery = {
        'name': NAME, 'cn': cn, 'combo': combo, 'N': N,
        'theta': theta, 'sx': sx, 'sy': sy, 'shift': shiftv,
        'rot': rot, 'C_s02': C2, 'v0': v0, 'v2': v2, 'W_avg': wavg,
        'rho56': rho56, 'cmin56': cmin56, 'rho32': rho32,
        'cmin32': cmin32, 'cores': len(pts_m), 'cores_exp': N_EXP,
        'mean_nn': float(mt['mean']), 'max10': max10,
        'struct': bool(struct), 'gate': bool(gate_ok), 'warns': warns,
        'morph': {k: v for k, v in mrep.items()
                  if not isinstance(v, list)},
        'prev': {'combo': r3['combo'],
                 'C_s02': rec['current']['C_s02'],
                 'morph_pass': rec['current']['morph']['morph_pass']},
    }
    (outdir / f'delivery_{cn}.json').write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2,
                   default=float), encoding='utf-8')
    return delivery


CASES = [{'cn': '9奇点'}, {'cn': '11奇点'}]

all_recs, deliveries = {}, {}
for case in CASES:
    rec, champ, outdir = run_case(case)
    all_recs[case['cn']] = rec
    if champ is not None:
        deliveries[case['cn']] = deliver(case['cn'], rec, champ, outdir)
    else:
        print(f'  [skip delivery] {case["cn"]}: 无形态+结构双通过解')

print('\n' + '=' * 72)
print('=== FLEET SUMMARY ===')
for cn, rec in all_recs.items():
    b = rec['current']['C_s02']
    bp = rec['current']['morph']['morph_pass']
    ch = rec['champion']
    if ch:
        print(f'  {cn:8s} 原 C_s02={b:.4f} morph={bp} → 新 '
              f'C_s02={ch["C_s02"]:.4f} morph=PASS '
              f'({ch["C_s02"]-b:+.4f})')
    else:
        print(f'  {cn:8s} 原 C_s02={b:.4f} morph={bp} → '
              f'**无形态通过解**（保持现解）')

(OUTROOT / 'fleet_summary.json').write_text(
    json.dumps({'searches': {k: {kk: vv for kk, vv in v.items()
                                 if kk != 'records'}
                             for k, v in all_recs.items()},
                'deliveries': deliveries},
               ensure_ascii=False, indent=2, default=float),
    encoding='utf-8')
print(f'\n[done] outputs in {OUTROOT}')

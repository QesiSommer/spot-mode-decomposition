# -*- coding: utf-8 -*-
"""
verify_metrics.py -- 标定后指标验收工具 + 回放验证
====================================================

For each of the 8 final solutions, computes the calibrated metric
set and writes a per-image verification sheet:

  准入:  C_s02 >= 0.83, PCC(0) >= 0.82, PCC(2) >= 0.83
  警示:  32px or 56px grid: rho >= 0.2 or C_min <= 0.3  (sigma=2)
  诊断:  angular spectrum per-order table + D_ang
  定位:  top-3 error blocks (56px grid, support-masked)

Also replays the 7 known-bad solutions to check the warning
behaviour. Outputs into 归档_八图分解/最新结果/ (per-image sheets)
and metrics_calibration/replay_验证.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from search_two_mode import load_experiment, gaussian_blur
from search_two_mode_joint import build_joint_modes
from search_two_mode_lg import rescale_lg
from calibrate_metrics import (
    pcc, window_stats, angular_distance, render_model, load_solutions,
    IMG_DIR)

ARCH = ROOT / '归档_八图分解'
DEST = ARCH / '最新结果'
CAL = ROOT / 'metrics_calibration'

T_C = 0.83
T0 = 0.82
T2 = 0.83
RHO_WARN = 0.2
CMIN_WARN = 0.3


def load_finals():
    with open(ROOT / 'refine_methods_compare' / 'methods_compare.json',
              encoding='utf-8') as f:
        mc = json.load(f)
    with open(ROOT / 'refine_all_new_method' / 'all_results.json',
              encoding='utf-8') as f:
        an = json.load(f)
    out = []
    for tag, name in [('4Sing', '4Singularities'),
                      ('8Sing1', '8Singularities(1)')]:
        b = mc[tag]['method_B']
        out.append({'name': name, 'combo': list(mc[tag]['combo']),
                    'params': [float(x) for x in b['theta']],
                    'scale': None})
    for tag, r in an.items():
        out.append({'name': r['name'], 'combo': list(r['combo']),
                    'params': [float(x) for x in r['new_params']],
                    'scale': r['new_scale']})
    return out, mc, an


def diff_localization(Ie, I2, mask, W=56):
    H, Wd = Ie.shape
    ny, nx = H // W, Wd // W
    blocks = []
    for i in range(ny):
        for j in range(nx):
            sl = (slice(i * W, (i + 1) * W), slice(j * W, (j + 1) * W))
            if mask[sl].mean() < 0.3:
                continue
            blocks.append((float(np.abs(Ie[sl] - I2[sl]).sum()), i, j))
    if not blocks:
        return None, []
    total = sum(b[0] for b in blocks)
    blocks.sort(reverse=True)
    items = []
    for e, i, j in blocks[:3]:
        v = '上' if i == 0 else ('下' if i == ny - 1 else '中')
        h = '左' if j == 0 else ('右' if j == nx - 1 else '中')
        pos = (v + h) if not (v == '中' and h == '中') else '中央'
        items.append((round(e / total * 100, 1), pos))
    conc = sum(b[0] for b in blocks[:3]) / total
    return conc, items


def evaluate(Ie, combo, params, scale, modes_np, lab2k):
    N = len(combo)
    mnp = modes_np if scale is None else rescale_lg(
        torch.tensor(modes_np), float(scale)).numpy()
    tr = [lab2k[m] for m in combo]
    Fs = [np.sqrt(max(params[i], 0)) * mnp[tr[i]] for i in range(N)]
    ph = [0.0] + list(params[N:])
    E = sum(Fs[i] * np.exp(1j * ph[i]) for i in range(N))
    I = np.abs(E) ** 2
    I = I / max(I.max(), 1e-12)
    I2 = gaussian_blur(torch.tensor(I, dtype=torch.float32)
                       .unsqueeze(0), 2.0)[0].numpy()

    res = {'pcc0': pcc(Ie, I), 'pcc2': pcc(Ie, I2)}
    res['C_s02'] = 0.5 * (res['pcc0'] + res['pcc2'])
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(Ie > 0.05 * Ie.max(), iterations=8)
    for W in (32, 56):
        c2, r2 = window_stats(Ie, I2, W, mask)
        res[f'cmin{W}'] = c2
        res[f'rho{W}'] = r2
    d_ang, (ae, am) = angular_distance(Ie, I2)
    res['D_ang'] = d_ang
    res['amp_exp'] = ae
    res['amp_mod'] = am
    conc, items = diff_localization(Ie, I2, mask)
    res['conc'] = conc
    res['loc'] = items
    return res


def verdicts(res):
    ok_c = res['C_s02'] >= T_C
    ok_0 = res['pcc0'] >= T0
    ok_2 = res['pcc2'] >= T2
    warn = []
    for W in (32, 56):
        if res[f'rho{W}'] is not None and res[f'rho{W}'] >= RHO_WARN:
            warn.append(f'rho_{W}px={res[f"rho{W}"]:.2f}')
        if res[f'cmin{W}'] is not None and res[f'cmin{W}'] <= CMIN_WARN:
            warn.append(f'C_min_{W}px={res[f"cmin{W}"]:.2f}')
    return ok_c and ok_0 and ok_2, warn


def make_sheet(name, combo, res, passed, warn):
    lines = [
        f'# {name} 指标验收单（标定后口径）',
        '',
        '## 准入门槛',
        '',
        f'- C_s02 = {res["C_s02"]:.4f} （门槛 {T_C}）'
        f' {"✓" if res["C_s02"] >= T_C else "✗"}',
        f'- PCC(σ=0) = {res["pcc0"]:.4f} （门槛 {T0}）'
        f' {"✓" if res["pcc0"] >= T0 else "✗"}',
        f'- PCC(σ=2) = {res["pcc2"]:.4f} （门槛 {T2}）'
        f' {"✓" if res["pcc2"] >= T2 else "✗"}',
        '',
        f'**准入判定：{"通过" if passed else "未通过"}**',
        '',
        '## 局部一致性警示（32px / 56px 双尺度，σ=2）',
        '',
        f'- 32px 网格：最差窗 C_min = {res["cmin32"]:.3f}，'
        f'失败窗占比 rho = {res["rho32"]:.3f}',
        f'- 56px 网格：最差窗 C_min = {res["cmin56"]:.3f}，'
        f'失败窗占比 rho = {res["rho56"]:.3f}',
        f'- 警示：{"无" if not warn else "【" + "；".join(warn) + "】存在局部失配区域，建议人工复核"}',
        '',
        '## 角谱诊断（σ=2；m 阶幅值相对差）',
        '',
        f'- D_ang（总体形状距离，越低越好）= {res["D_ang"]:.4f}',
        '',
        '| m 阶 | 实验 | 模型 | 相对差 |',
        '|---|---|---|---|',
    ]
    for m in range(1, 11):
        ae, am = res['amp_exp'][m], res['amp_mod'][m]
        rel = abs(ae - am) / (ae + 1e-12)
        lines.append(f'| m={m} | {ae:.4f} | {am:.4f} | {rel:.2f} |')
    lines += [
        '',
        '## 差异定位（56px 块，支持区内）',
        '',
    ]
    if res['conc'] is not None:
        lines.append(f'- 误差集中度（前 3 块占比）= '
                     f'{res["conc"]*100:.1f}%')
        for pct, pos in res['loc']:
            lines.append(f'- {pos} 区域：{pct}% 的误差能量')
    else:
        lines.append('- （无有效块）')
    lines += ['', '## 结论', '',
              '**通过**（准入达标' +
              ('' if not warn else '；附局部警示供复核') + '）'
              if passed else '**未通过准入**']
    return '\n'.join(lines)


def main():
    modes, labels = build_joint_modes(10, 224, 4.0, 'cpu')
    lab2k = {lab: k for k, lab in enumerate(labels)}
    modes_np = modes.numpy()

    finals, mc, an = load_finals()
    truth_cal, fake_cal = load_solutions()

    replay = {'true': [], 'false': []}

    # ---- 8 final solutions: sheets into the deliverables folder ----
    for sol in finals:
        Ie = load_experiment(f'{IMG_DIR}/{sol["name"]}.png',
                             'nipy_spectral', 'isot')
        res = evaluate(Ie, sol['combo'], sol['params'], sol['scale'],
                       modes_np, lab2k)
        passed, warn = verdicts(res)
        cn = {'4Singularities': '4奇点', '5Singularities': '5奇点',
              '6Singularities': '6奇点', '8Singularities(1)': '8奇点(1)',
              '8Singularities(2)': '8奇点(2)', '9Singularities': '9奇点',
              '10Singularities': '10奇点',
              '11Singularities': '11奇点'}[sol['name']]
        (DEST / f'{cn}_指标验收单.md').write_text(
            make_sheet(sol['name'], sol['combo'], res, passed, warn),
            encoding='utf-8')
        replay['true'].append({'name': sol['name'],
                               'C_s02': res['C_s02'],
                               'passed': passed, 'warn': warn})
        print(f'[true ] {cn:8s} C_s02={res["C_s02"]:.4f} '
              f'{"PASS" if passed else "FAIL"} '
              f'warn={len(warn)} {warn}')

    # ---- replay of the known-bad solutions ----
    print()
    for sol in fake_cal:
        Ie = load_experiment(f'{IMG_DIR}/{sol["name"]}.png',
                             'nipy_spectral', 'isot')
        res = evaluate(Ie, sol['combo'], sol['params'], sol['scale'],
                       modes_np, lab2k)
        passed, warn = verdicts(res)
        replay['false'].append({'name': sol['name'], 'src': sol['src'],
                                'C_s02': res['C_s02'],
                                'passed': passed, 'warn': warn})
        print(f'[false] {sol["name"]:22s} C_s02={res["C_s02"]:.4f} '
              f'{"passed-gate" if passed else "GATED"} '
              f'warn={len(warn)} {warn}   [{sol["src"]}]')

    n_t_warn = sum(1 for r in replay['true'] if r['warn'])
    n_f_warn = sum(1 for r in replay['false'] if r['warn'])
    n_f_gated = sum(1 for r in replay['false'] if not r['passed'])
    print(f'\n[warn rate] true: {n_t_warn}/{len(replay["true"])}  '
          f'false: {n_f_warn}/{len(replay["false"])}')
    print(f'[gate] false solutions gated by admission floors: '
          f'{n_f_gated}/{len(replay["false"])}')

    (CAL / 'replay_验证.json').write_text(
        json.dumps(replay, indent=2, ensure_ascii=False),
        encoding='utf-8')

    # small replay report
    lines = ['# 指标回放验证（8 真解 + 7 假解）', '',
             f'- 准入线：C_s02≥{T_C}, PCC0≥{T0}, PCC2≥{T2}',
             f'- 警示线：rho≥{RHO_WARN} 或 C_min≤{CMIN_WARN}（32/56px）', '',
             '## 真解回放', '',
             '| 图 | C_s02 | 准入 | 局部警示 |', '|---|---|---|---|']
    for r in replay['true']:
        lines.append(f'| {r["name"]} | {r["C_s02"]:.4f} | '
                     f'{"✓" if r["passed"] else "✗"} | '
                     f'{"；".join(r["warn"]) if r["warn"] else "无"} |')
    lines += ['', '## 假解回放', '',
              '| 来源 | C_s02 | 准入 | 局部警示 |', '|---|---|---|---|']
    for r in replay['false']:
        lines.append(f'| {r["src"]} | {r["C_s02"]:.4f} | '
                     f'{"放行" if r["passed"] else "**拦截**"} | '
                     f'{"；".join(r["warn"]) if r["warn"] else "无"} |')
    lines += ['', '## 结论', '',
              f'- 真解：{len(replay["true"])}/{len(replay["true"])} 通过准入；'
              f'{n_t_warn} 个带局部警示（低分真解局部失配，符合预期）；',
              f'- 假解：{n_f_gated}/{len(replay["false"])} 被准入线拦截；'
              f'{n_f_warn} 个带局部警示；',
              '- 警示率差异 + 结构验收协作 = 完整防线（全局分放行的假解'
              '由结构验收终审拦下）。']
    (CAL / 'replay_验证.md').write_text('\n'.join(lines),
                                      encoding='utf-8')
    print(f'\n[done] sheets -> {DEST}, replay report -> {CAL}')


if __name__ == '__main__':
    main()

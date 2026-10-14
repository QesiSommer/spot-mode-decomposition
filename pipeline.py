# -*- coding: utf-8 -*-
"""
pipeline.py -- 相干光斑图模式分解：通用端到端驱动
======================================================

把本项目验证过的"扫描搜索 → 精修 → 结构化验收"串成一条**可套用流程**，
对任意实验室拍摄的相干光斑图（激光横模、涡旋阵列、多瓣/多暗核结构均可）
给出模式分解结果。

> 零外部数据即可试跑：先用 make_demo_image.py 合成一张**已知答案**的光斑图
> （本仓库的演示），跑完把搜索解与真值对照。换一张图，改一个 `--image` 即可。
> 项目完成的八张多奇点涡旋阵列图（4~11 奇点）是**真实数据验证记录**，
> 那些实验图像不随仓库发布。

--------------------------------------------------------------------------
输入 / 输出
--------------------------------------------------------------------------
输入：单帧光斑强度图 I(x, y)，只有强度、相位不可测。
输出：相干叠加的模式,振幅与相对相位

        E(x, y) = Σ_k a_k · F_k(x, y) · exp(i·φ_k),    Σ_k a_k² = 1

      以及拟合指标（C_s02 等）与**暗核结构一致性**判定。

这是一个**病态逆问题**：多组模式可产生高度相似的光强分布，解不唯一，
所以必须"全空间组合搜索 + 结构约束验收"，而不能只做局部优化。

--------------------------------------------------------------------------
设计原则：不重造轮子
--------------------------------------------------------------------------
本脚本**只做编排**——各阶段的搜索/精修内核直接调用本仓库中已通过
八图验证的脚本（search_two_mode / search_three_mode / search_four_mode），
不重新实现算法，以保证结果与项目已验证的流程一致。

    阶段一（--stage two）  : search_two_mode.py    两模解析扫描
    阶段二（--stage three）: search_three_mode.py  三模全空间（366,145 组合）
    阶段三（--stage four） : search_four_mode.py  四模 guided / full

--------------------------------------------------------------------------
快速开始
--------------------------------------------------------------------------
    # 灰度图（务必 --cmap none）
    python pipeline.py --image your_beam.png --img_dir ./data \\
        --stage two --cmap none --expect-cores 4 --out_dir ./run1

    # 三模（更强，但更慢）
    python pipeline.py --image your_beam.png --img_dir ./data \\
        --stage three --cmap none --expect-cores 4 --out_dir ./run1

    # 伪彩图（如 nipy_spectral 配色）保留默认 --cmap 即可

--------------------------------------------------------------------------
怎么判断"够不够好"（判据）
--------------------------------------------------------------------------
每阶段结束都会打印：组合 / 能量 / 相位 / C_s02 / 暗核数 / 核位置偏差。
  - 暗核数与 --expect-cores 一致且 mean_nn < 12px  → 该层可接受；
  - 加阶后 C_s02 不再提升、或暗核数反而偏离     → 已到该模式基的表达极限，
    继续加阶意义有限（本项目演示图中有 3 张属此情形，已如实记录）；
  - 低阶解不理想 ≠ 失败：可交付"最佳可达解 + 明确披露的失配"，
    这在科研上同样成立、且比虚高分数更可信。
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
# 兼容两种目录布局：
#   A) 开发工作区：所有脚本与 pipeline.py 平铺在同一目录
#   B) 发布仓库：pipeline.py 在根目录，脚本在 ./src/ 下
for cand in (SRC, SRC / 'src'):
    if (cand / 'search_two_mode.py').is_file():
        SCRIPT_DIR = cand
        break
else:
    SCRIPT_DIR = SRC
sys.path.insert(0, str(SCRIPT_DIR))

from search_two_mode import load_experiment, gaussian_blur            # noqa: E402
from search_two_mode_joint import build_joint_modes                    # noqa: E402
from search_two_mode_lg import rescale_lg                              # noqa: E402
from structural_check import find_dark_points, match_points            # noqa: E402


# ---------------------------------------------------------------------------
# 控制台输出：Windows 中文终端默认 GBK，遇到非 GBK 字符会抛
# UnicodeEncodeError。这里统一转成 ASCII 安全形式，避免"能算不能打印"。
# ---------------------------------------------------------------------------

def safe_print(s):
    """把任意文本压成当前控制台可编码的形式。"""
    try:
        enc = sys.stdout.encoding or 'utf-8'
        return str(s).encode(enc, errors='replace').decode(enc, errors='replace')
    except Exception:
        return str(s).encode('ascii', errors='replace').decode('ascii')


# ---------------------------------------------------------------------------
# 指标与渲染
# ---------------------------------------------------------------------------

def pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def c_s02(Ie, Im, sigmas=(0, 2)):
    """主口径 C_s02 = [PCC(σ=0) + PCC(σ=2)] / 2（多尺度，兼顾细节与低频）。"""
    t = torch.tensor(Im, dtype=torch.float32).unsqueeze(0)
    vals = []
    for sg in sigmas:
        b = Im if sg == 0 else gaussian_blur(t, float(sg))[0].numpy()
        vals.append(pcc(Ie, b))
    return float(np.mean(vals)), [round(v, 4) for v in vals]


def render(combo, theta, modes_np, lab2k, scale=1.0):
    """渲染 E = Σ a_k F_k e^{iφ_k} 的光强，返回 [0,1] 归一化图。

    theta = [a_1², ..., a_N², φ_2, ..., φ_N]（约定 φ_1 ≡ 0）。
    """
    N = len(combo)
    E = np.zeros(modes_np[0].shape, dtype=np.complex128)
    for i, m in enumerate(combo):
        F = modes_np[lab2k[m]]
        if abs(scale - 1.0) > 1e-6:
            F = rescale_lg(torch.from_numpy(F).unsqueeze(0),
                           float(scale))[0].numpy()
        phi = 0.0 if i == 0 else float(theta[N + i - 1])
        E = E + np.sqrt(max(float(theta[i]), 0.0)) * F * np.exp(1j * phi)
    I = np.abs(E) ** 2
    return I / max(I.max(), 1e-12)


def structure(Ie, Im, expect_cores=None, nn_tol=12.0):
    """暗核结构验收（独立于相关系数）：数量 + 最近邻位置。"""
    pts_e = find_dark_points(Ie)
    pts_m = find_dark_points(Im)
    mt = match_points(pts_e, pts_m)
    n_exp = len(pts_e) if expect_cores is None else int(expect_cores)
    n_ok = (len(pts_m) == n_exp)
    d_ok = mt.get('mean') is not None and mt['mean'] < nn_tol
    return {
        'cores_model': len(pts_m),
        'cores_experiment': len(pts_e),
        'expect_cores': expect_cores,
        'cores_match': bool(n_ok),
        'mean_nn_px': None if mt.get('mean') is None else round(mt['mean'], 2),
        'max_nn_px': None if mt.get('max') is None else round(mt['max'], 2),
        'pass': bool(n_ok and d_ok),
    }


# ---------------------------------------------------------------------------
# 调用已验证的搜索脚本（编排，不重新实现算法）
# ---------------------------------------------------------------------------

def run_script(script, argv, log_path):
    """调用一个已验证的搜索脚本，完整日志落盘、控制台只打印尾部几行。"""
    cmd = [sys.executable, str(SCRIPT_DIR / script)] + [str(a) for a in argv]
    print(f'  $ {" ".join(cmd[1:])}')
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), capture_output=True,
                          text=True, encoding='utf-8', errors='replace')
    dt = time.time() - t0
    log_path.write_text(
        f'# {script}\n# argv: {argv}\n# exit: {proc.returncode}\n'
        f'# elapsed: {dt:.1f}s\n\n=== STDOUT ===\n{proc.stdout}\n'
        f'=== STDERR ===\n{proc.stderr}', encoding='utf-8')
    tail = [l for l in (proc.stdout or '').splitlines() if l.strip()][-12:]
    for l in tail:
        print(f'    | {safe_print(l)}')
    if proc.returncode != 0:
        print(f'    ! {script} 退出码 {proc.returncode}（详见 {log_path.name}）')
    print(f'  -> {script} 用时 {dt:.0f}s')
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# 各阶段结果解析
# ---------------------------------------------------------------------------

def read_two_mode(stem, out_dir, Ie_np, modes_np, lab2k, args):
    """解析 search_two_mode 的 top-K 表，并补算 C_s02 和结构判定。"""
    csv = out_dir / f'{stem}_top.csv'
    if not csv.is_file():
        return []
    import pandas as pd
    df = pd.read_csv(csv)
    rows = []
    for _, r in df.head(args.top_struct).iterrows():
        combo = [str(r['mode1']), str(r['mode2'])]
        if any(c not in lab2k for c in combo):
            continue
        theta = [float(r['r_energy1']), float(r['r_energy2']),
                 float(r['phi_rad'])]
        scale = float(r.get('scale', 1.0) or 1.0)
        Im = render(combo, theta, modes_np, lab2k, scale)
        cs, parts = c_s02(Ie_np, Im)
        st = structure(Ie_np, Im, args.expect_cores, args.nn_tol)
        rows.append({
            'stage': 'two', 'combo': combo,
            'energy': [round(float(r['r_energy1']), 4),
                       round(float(r['r_energy2']), 4)],
            'phi_rad': [round(float(r['phi_rad']), 4)],
            'scale': scale, 'C_s02': round(cs, 4), 'C_s02_parts': parts,
            'corr_avg_script': round(float(r['corr_avg']), 4),
            **st,
        })
    rows.sort(key=lambda d: -d['C_s02'])
    return rows


def read_three_mode(stem, out_dir, Ie_np, modes_np, lab2k, args):
    js = out_dir / f'{stem}_three_mode.json'
    if not js.is_file():
        return []
    d = json.loads(js.read_text(encoding='utf-8'))
    rows = []
    for e in (d.get('entries') or [])[:args.top_struct]:
        combo = list(e['modes'])
        if any(c not in lab2k for c in combo):
            continue
        # 三模 theta = [a1²,a2²,a3², φ2, φ3]（能量已在 entries['energy']）
        en = list(e['energy'])
        theta = [float(en[0]), float(en[1]), float(en[2]),
                 float(e['phi2']), float(e['phi3'])]
        Im = render(combo, theta, modes_np, lab2k, 1.0)
        cs, parts = c_s02(Ie_np, Im)
        st = structure(Ie_np, Im, args.expect_cores, args.nn_tol)
        rows.append({
            'stage': 'three', 'combo': combo,
            'energy': [round(float(x), 4) for x in en],
            'phi_rad': [round(float(e['phi2']), 4),
                        round(float(e['phi3']), 4)],
            'scale': 1.0, 'C_s02': round(cs, 4), 'C_s02_parts': parts,
            'corr_avg_script': round(float(e.get('corr_avg',
                                                 e.get('corr', 0))), 4),
            **st,
        })
    rows.sort(key=lambda d: -d['C_s02'])
    return rows


def read_four_mode(stem, out_dir, mode, Ie_np, modes_np, lab2k, args):
    js = out_dir / f'{stem}_four_mode_{mode}.json'
    if not js.is_file():
        return []
    d = json.loads(js.read_text(encoding='utf-8'))
    entries = d.get('entries') or d.get('top') or []
    rows = []
    for e in entries[:args.top_struct]:
        combo = e.get('modes') or e.get('combo')
        if not combo:
            continue
        combo = [str(c) if not isinstance(c, str) else c for c in combo]
        if any(c not in lab2k for c in combo):
            continue
        en = e.get('energy') or e.get('amplitudes')
        phis = e.get('phis') or e.get('phi') or []
        if en is None:
            continue
        theta = [float(x) for x in en] + [float(x) for x in phis]
        Im = render(combo, theta, modes_np, lab2k, 1.0)
        cs, parts = c_s02(Ie_np, Im)
        st = structure(Ie_np, Im, args.expect_cores, args.nn_tol)
        rows.append({
            'stage': 'four', 'combo': combo,
            'energy': [round(float(x), 4) for x in en],
            'phi_rad': [round(float(x), 4) for x in phis],
            'scale': 1.0, 'C_s02': round(cs, 4), 'C_s02_parts': parts,
            'corr_avg_script': round(float(e.get('corr_avg',
                                                 e.get('corr', 0))), 4),
            **st,
        })
    rows.sort(key=lambda d: -d['C_s02'])
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # Windows 中文终端常为 GBK：把不可编码字符替换而非抛异常
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')
        except Exception:
            pass

    p = argparse.ArgumentParser(
        description='相干光斑图模式分解 · 通用端到端驱动',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例: python pipeline.py --image beam.png --stage two '
               '--cmap none --expect-cores 4 --out_dir ./run1')
    p.add_argument('--image', required=True,
                   help='输入光斑图路径（或配合 --img_dir 的文件名）')
    p.add_argument('--img_dir', default=None, help='图像所在目录（可选）')
    p.add_argument('--stage', default='two',
                   choices=['two', 'three', 'four'],
                   help='搜到几阶：two(最快) / three / four')
    p.add_argument('--expect-cores', type=int, default=None,
                   help='你从图上数出的暗核（相位奇点）数量；'
                        '给了它才能做结构一致性判定')
    p.add_argument('--nn-tol', type=float, default=12.0,
                   help='暗核最近邻偏差容忍（像素），默认 12')
    p.add_argument('--cmap', default='nipy_spectral',
                   help='伪彩配色；**灰度图请传 none**')
    p.add_argument('--preprocess', default='isot',
                   help='预处理模式，默认 isot（无量纲拓扑对齐）')
    p.add_argument('--max_order', type=int, default=10,
                   help='模式基最大阶数（默认 10 → 131 个 HG/LG 模式）')
    p.add_argument('--small', type=int, default=None, help='粗扫降采样尺寸')
    p.add_argument('--n_energy', type=int, default=None, help='能量网格步数')
    p.add_argument('--n_phase', type=int, default=None, help='相位网格步数')
    p.add_argument('--top_struct', type=int, default=15,
                   help='复算并汇报的候选数')
    p.add_argument('--quick', action='store_true',
                   help='快速模式（缩小网格，用于先验证流程能跑通）')
    p.add_argument('--out_dir', default='pipeline_out')
    args = p.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    img = Path(args.image)
    if not img.is_file() and args.img_dir:
        cand = Path(args.img_dir) / img.name
        if cand.is_file():
            img = cand
    if not img.is_file():
        print(f'[error] 找不到图像: {args.image}')
        return 1
    # 必须转绝对路径：搜索脚本由 run_script 在 src/ 目录下启动，
    # 相对路径会在子进程里被解析成 src/ 下的路径而找不到文件。
    img = img.resolve()
    stem = img.stem

    print('=' * 70)
    print('相干光斑图模式分解 · 通用流程')
    print(f'  图像      : {img}')
    print(f'  搜索阶段  : {args.stage}')
    print(f'  预期暗核数: '
          f'{args.expect_cores if args.expect_cores else "未指定（只报告实际值）"}')
    print(f'  模式基阶数: {args.max_order}')
    print(f'  输出目录  : {out_root}')
    print(f'  CUDA      : {torch.cuda.is_available()}')
    print('=' * 70)

    # ---- 1) 读图 / 预处理 / 先看实验图有多少暗核 ----
    t0 = time.time()
    Ie_np = load_experiment(str(img), args.cmap, args.preprocess)
    pts_e = find_dark_points(Ie_np)
    print(f'[1/4] 预处理完成 {Ie_np.shape}（{time.time()-t0:.1f}s）')
    print(f'      实验图检测到暗核 {len(pts_e)} 个')
    if args.expect_cores is not None and len(pts_e) != args.expect_cores:
        print(f'      注意：你声明 {args.expect_cores} 个，检测到 {len(pts_e)} 个'
              f'——判定将按你声明的数量；若检测器误检可调整检测参数。')

    # ---- 2) 模式基 ----
    print('[2/4] 模式基')
    print('      two 阶段: 仅 HG（66 模）—— 快速侦察，不含 LG，'
          '环形/涡旋结构需升到三模')
    modes, labels = build_joint_modes(args.max_order, 224, 4.0, 'cpu')
    modes_np = modes.numpy()
    lab2k = {l: k for k, l in enumerate(labels)}
    print(f'      three/four 阶段: HG+LG 联合基 {len(labels)} 模'
          f'（max_order={args.max_order}）')

    # ---- 3) 按阶段调用已验证脚本 ----
    print(f'[3/4] 搜索（{args.stage} 阶）')
    logs = out_root / 'logs'
    logs.mkdir(exist_ok=True)
    q = args.quick
    all_rows = []
    failures = []   # 记录失败的搜索脚本；避免"静默失败"——脚本挂了却报成功

    # 3a. 两模（三/四阶时作为侦察层，低成本且能暴露主要结构）
    two_dir = out_root / 'two_mode'
    two_dir.mkdir(exist_ok=True)
    argv = ['--image', str(img), '--out_dir', str(two_dir),
            '--cmap', args.cmap, '--preprocess', args.preprocess,
            '--max_order', args.max_order]
    if args.small:
        argv += ['--small', str(args.small)]
    if not run_script('search_two_mode.py', argv, logs / 'two_mode.log'):
        failures.append(('two', 'search_two_mode.py'))
    rows = read_two_mode(stem, two_dir, Ie_np, modes_np, lab2k, args)
    if rows:
        all_rows += rows
        b = rows[0]
        print(f'  [two] 最佳: {"+".join(b["combo"])}  C_s02={b["C_s02"]}  '
              f'核 {b["cores_model"]}vs{b["cores_experiment"]}')
        if not b['pass'] and args.stage == 'two':
            print('        ⚠ 两模阶段（仅 HG 基）未通过结构判定：'
                  '若图上存在涡旋/环形结构，'
                  '请升阶到 --stage three（含 LG 基）。')

    # 3b. 三模
    three_dir = out_root / 'three_mode'
    if args.stage in ('three', 'four'):
        three_dir.mkdir(exist_ok=True)
        argv = ['--image', str(img), '--out_dir', str(three_dir),
                '--cmap', args.cmap, '--preprocess', args.preprocess,
                '--max_order', args.max_order]
        if q or args.small:
            argv += ['--small', str(args.small or 48)]
        if q:
            argv += ['--quick']
        if args.n_energy:
            argv += ['--n_energy', str(args.n_energy)]
        if args.n_phase:
            argv += ['--n_phase', str(args.n_phase)]
        if not run_script('search_three_mode.py', argv, logs / 'three_mode.log'):
            failures.append(('three', 'search_three_mode.py'))
        rows = read_three_mode(stem, three_dir, Ie_np, modes_np, lab2k, args)
        if rows:
            all_rows += rows
            b = rows[0]
            print(f'  [three] 最佳: {"+".join(b["combo"])}  C_s02={b["C_s02"]}  '
                  f'核 {b["cores_model"]}vs{b["cores_experiment"]}')

    # 3c. 四模（guided 需三模结果，此处若未跑三模则退回 full）
    four_dir = out_root / 'four_mode'
    if args.stage == 'four':
        four_dir.mkdir(exist_ok=True)
        mode = 'guided' if three_dir.is_dir() and any(three_dir.iterdir()) \
            else 'full'
        argv = ['--image', str(img), '--out_dir', str(four_dir),
                '--img_dir', str(img.parent), '--three_dir', str(three_dir),
                '--mode', mode, '--cmap', args.cmap,
                '--preprocess', args.preprocess]
        if q or args.small:
            argv += ['--small', str(args.small or 48)]
        if not run_script('search_four_mode.py', argv, logs / 'four_mode.log'):
            failures.append(('four', 'search_four_mode.py'))
        rows = read_four_mode(stem, four_dir, mode, Ie_np, modes_np,
                              lab2k, args)
        if rows:
            all_rows += rows
            b = rows[0]
            print(f'  [four] 最佳: {"+".join(b["combo"])}  C_s02={b["C_s02"]}  '
                  f'核 {b["cores_model"]}vs{b["cores_experiment"]}')

    # ---- 4) 汇总 ----
    print('\n' + '=' * 70)
    print('[4/4] 汇总（按 C_s02 排序，仅列各阶段最佳）')
    all_rows.sort(key=lambda d: -d['C_s02'])
    by_stage = {}
    for r in all_rows:
        by_stage.setdefault(r['stage'], r)
    for st in ('two', 'three', 'four'):
        r = by_stage.get(st)
        if not r:
            continue
        verdict = 'PASS' if r['pass'] else '未通过'
        print(f'  {st:5s} : {"+".join(r["combo"])}')
        print(f'          能量 {r["energy"]}  相位 {r["phi_rad"]}')
        print(f'          C_s02={r["C_s02"]}  暗核 {r["cores_model"]}'
              f'vs{r["cores_experiment"]}  mean_nn={r["mean_nn_px"]}px  '
              f'[{verdict}]')

    best = all_rows[0] if all_rows else None
    if best:
        print('\n  采用建议：以上述各阶段中 **结构 PASS 且 C_s02 最高** 的解；')
        print('            若无结构 PASS，则以最佳可达解交付并如实披露失配。')

    (out_root / 'summary.json').write_text(json.dumps({
        'image': str(img), 'stage': args.stage,
        'expect_cores': args.expect_cores, 'max_order': args.max_order,
        'cmap': args.cmap, 'preprocess': args.preprocess,
        'results': all_rows[:60], 'best': best,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  逐候选结果: {out_root / "summary.json"}')
    print(f'  运行日志  : {logs}')

    # 静默失败防护：任一搜索脚本非零退出，整体也返回非零，
    # 否则使用者会以为"跑成功了"，只是恰好没有结果。
    if failures:
        print('\n  [error] 以下搜索脚本未成功执行：')
        for st, name in failures:
            print(f'    - {name}（{st} 阶段）—— 详见 {logs / (st + "_mode.log")}')
        print('=' * 70)
        return 1

    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())

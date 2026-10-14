# -*- coding: utf-8 -*-
"""
make_demo_image.py -- 生成一张合成光斑图，用于试跑流程
========================================================

本仓库不发布任何实验图像（课题组数据），所以这里用仓库自带的物理引擎
合成一张"已知答案"的光斑图，让你在不接触任何外部数据的情况下：

    1. 造出一张输入图；
    2. 跑通 `pipeline.py`；
    3. 把搜索得到的模式组合与**已知真值**对照——直接看到方法准不准。

合成原理就是本方法要反解的那个前向模型：

    E(x, y) = Σ_k a_k · F_k(x, y) · exp(i·φ_k)

    I(x, y) = |E(x, y)|²         （只有强度，相位被丢弃——这正是问题的难点）

其中 F_k 直接取自仓库的模式基（`src/search_two_mode_joint.py::build_joint_modes`，
LG 65 模 + HG 66 模，共 131 模）。

--------------------------------------------------------------------------
用法
--------------------------------------------------------------------------
    # 1) 生成一张两模灰度图（默认 HG(0,3) + HG(3,1)），并写下真值
    python make_demo_image.py --out ./data/demo.png --truth ./data/demo_truth.json

    # 2) 按它打印的 --expect-cores 跑流程（灰度图务必 --cmap none）
    python pipeline.py --image demo.png --img_dir ./data \
        --stage two --cmap none --expect-cores 7 --out_dir ./run1

    # 3) 把 run1/summary.json 里的模式组合/能量/相位与 demo_truth.json 对照
    #    实测（默认 --noise 0）：组合完全一致，能量 [0.5069, 0.4931] vs 真值
    #    [0.5, 0.5]，相位 1.552 vs 真值 1.55，C_s02 = 0.9863，暗核 7 vs 7 → PASS。
    #    这条"合成图 → 搜索 → 与真值对照"的闭环就是本方法的精度自检。

--------------------------------------------------------------------------
常用参数
--------------------------------------------------------------------------
    --combo    模式组合，用 '+' 连接，如 "HG(0,3)+HG(3,1)"
               也可用 LG 模，如 "LG(1,-1)+LG(0,3)"（环形/涡旋结构要用 LG）
    --energy   各模能量占比（Σa²=1），逗号分隔，如 "0.47,0.53"
    --phi      各模相对相位（rad），逗号分隔；第一个恒为 0（全局相位无意义）
    --noise    加性高斯噪声标准差（默认 0，即不加）
    --cmap     存图配色：gray（默认，配 --cmap none）或 nipy_spectral 等伪彩
    --truth    把真值写入 JSON，便于逐项对照搜索结果
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# 兼容两种目录布局：
#   A) 开发工作区：所有脚本与 pipeline.py 平铺在同一目录
#   B) 发布仓库：pipeline.py 在根目录，脚本在 ./src/ 下
for cand in (ROOT, ROOT / "src"):
    if (cand / "search_two_mode_joint.py").is_file():
        SCRIPT_DIR = cand
        break
else:
    SCRIPT_DIR = ROOT
sys.path.insert(0, str(SCRIPT_DIR))

from search_two_mode_joint import build_joint_modes  # noqa: E402
from structural_check import find_dark_points        # noqa: E402


def parse_floats(s, n, name):
    """把 '0.47,0.53' 解析成长度 n 的 float 列表。"""
    if s is None:
        return None
    parts = [x.strip() for x in str(s).replace("，", ",").split(",") if x.strip()]
    if len(parts) != n:
        raise SystemExit(f"[error] --{name} 需要 {n} 个值（对应 {n} 个模式），"
                         f"实际给了 {len(parts)} 个：{s}")
    try:
        return [float(x) for x in parts]
    except ValueError:
        raise SystemExit(f"[error] --{name} 含非数字：{s}")


def main():
    ap = argparse.ArgumentParser(
        description="生成一张合成光斑图（用于试跑模式分解流程）")
    ap.add_argument("--out", required=True,
                    help="输出图像路径，如 ./data/demo.png")
    ap.add_argument("--combo", default="HG(0,3)+HG(3,1)",
                    help="模式组合，用 + 连接（默认 HG(0,3)+HG(3,1)）")
    ap.add_argument("--energy", default=None,
                    help="各模能量占比，逗号分隔（默认等分）")
    ap.add_argument("--phi", default=None,
                    help="各模相对相位(rad)，逗号分隔（默认第 2 个起给 1.55）")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="加性高斯噪声标准差（默认 0 = 不加，结构判定可 PASS；"
                         "加 0.01 左右会模拟相机噪声，但暗核检测会出现虚假极小值，"
                         "结构判定可能报失配——这也是真实数据上的常态）")
    ap.add_argument("--size", type=int, default=224,
                    help="图像边长（默认 224，与流程内部网格一致）")
    ap.add_argument("--max_order", type=int, default=10,
                    help="模式基阶数（默认 10）")
    ap.add_argument("--cmap", default="gray",
                    help="存图配色：gray（默认）或 nipy_spectral 等伪彩")
    ap.add_argument("--device", default="cpu", help="cpu / cuda（默认 cpu）")
    ap.add_argument("--seed", type=int, default=0, help="噪声随机种子")
    ap.add_argument("--truth", default=None,
                    help="把真值写入该 JSON 路径，便于与搜索结果对照")
    args = ap.parse_args()

    labels_wanted = [s.strip() for s in args.combo.split("+") if s.strip()]
    if not labels_wanted:
        raise SystemExit("[error] --combo 为空")

    print(f"[1/3] 构建模式基（max_order={args.max_order}, "
          f"size={args.size}, device={args.device}）…")
    modes, labels = build_joint_modes(args.max_order, args.size, 4.0,
                                      args.device)

    idx = []
    for name in labels_wanted:
        if name not in labels:
            near = [l for l in labels if l.startswith(name[:2])][:6]
            raise SystemExit(
                f"[error] 模式基中没有 '{name}'。\n"
                f"        可用示例：{', '.join(near) if near else ', '.join(labels[:6])}\n"
                f"        提示：LG 记为 LG(p,l)，HG 记为 HG(m,n)；"
                f"max_order={args.max_order} 时指数需 <= {args.max_order}")
        idx.append(labels.index(name))

    n = len(idx)
    if n < 2:
        raise SystemExit("[error] 至少需要 2 个模式才有干涉结构（--combo）")

    # ---- 振幅 / 相位 ----
    energies = parse_floats(args.energy, n, "energy")
    if energies is None:
        energies = [1.0 / n] * n
    tot = sum(energies)
    energies = [e / tot for e in energies]
    amps = [float(np.sqrt(max(e, 0.0))) for e in energies]

    phis = parse_floats(args.phi, n, "phi")
    if phis is None:
        phis = [0.0] + [1.55] * (n - 1)
    phis[0] = 0.0          # 约定：全局相位固定在第一个模上

    # ---- 前向模型：E = Σ a_k F_k e^{iφ_k} ----
    print(f"[2/3] 合成 {n} 模叠加：{' + '.join(labels_wanted)}")
    E = None
    for k, (i, a, p) in enumerate(zip(idx, amps, phis)):
        term = modes[i].detach().cpu().numpy() * (a * np.exp(1j * p))
        E = term if E is None else E + term

    I = np.abs(E) ** 2
    span = float(I.max() - I.min())
    I = (I - I.min()) / span if span > 1e-12 else np.zeros_like(I)

    if args.noise > 0:
        rng = np.random.default_rng(args.seed)
        I = np.clip(I + rng.normal(0, args.noise, I.shape), 0, 1)
        print(f"      [note] 已加噪声 σ={args.noise}："
              f"暗核检测可能出现虚假极小值，结构判定可能报失配")

    # ---- 存图 ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if out.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        out = out.with_suffix(".png")
    plt.imsave(out, I, cmap=args.cmap, vmin=0, vmax=1)

    print(f"[3/3] 已写出图像：{out}")
    print(f"      尺寸 {I.shape[0]}x{I.shape[1]}  灰度范围 [0,1]  "
          f"配色 {args.cmap}")

    # ---- 用 pipeline 同款检测器数一下这张图有多少暗核 ----
    # 这一步很关键：pipeline 的 --expect-cores 是"你声明的暗核数"，
    # 声明错了会把本该 PASS 的解判成失配。这里直接复用管线里的
    # 预处理 + 检测器，给出与 pipeline 一致的数。
    n_cores = None
    try:
        from search_two_mode import isot_process
        Ie_proc = isot_process(I)
        n_cores = len(find_dark_points(Ie_proc))
        print(f"      检测到暗核：{n_cores} 个"
              f"（pipeline 请用 --expect-cores {n_cores}）")
    except Exception as e:                                    # pragma: no cover
        print(f"      [warn] 暗核自检失败（不影响出图）：{e}")

    # ---- 真值 ----
    truth = {
        "image": str(out),
        "combo": labels_wanted,
        "energy": [round(e, 4) for e in energies],
        "amp": [round(a, 4) for a in amps],
        "phi_rad": [round(p, 4) for p in phis],
        "noise_sigma": args.noise,
        "size": args.size,
        "max_order": args.max_order,
        "detected_cores": n_cores,
    }
    if args.truth:
        tp = Path(args.truth)
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(truth, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        print(f"      真值已写入：{tp}")

    print("\n  ★ 真值（搜索的目标答案）：")
    print(f"      组合     : {' + '.join(labels_wanted)}")
    print(f"      能量占比 : {truth['energy']}")
    print(f"      相对相位 : {truth['phi_rad']} rad")
    cmap_flag = "none" if args.cmap == "gray" else args.cmap
    cores_flag = n_cores if n_cores else 4
    print("\n  接着跑：")
    print(f"      python pipeline.py --image {out.name} --img_dir {out.parent} \\")
    print(f"          --stage two --cmap {cmap_flag} "
          f"--expect-cores {cores_flag} --out_dir ./run1")
    print("\n  跑完把 search 给出的组合/能量/相位与上面真值对照——"
          "这就是本方法的精度自检。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

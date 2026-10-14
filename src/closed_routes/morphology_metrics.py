# -*- coding: utf-8 -*-
"""
morphology_metrics.py -- 轮廓/中央形态指标（"形态守门员"）
============================================================
背景（2026-10-14 诊断）：C_s02 对低频形态差异不敏感——4 奇点现解
C_s02=0.9163 但中央暗区占比 34%（实验仅 6%）、轮廓瓣形谱也不符
（"高相关、低形态相似"）。本模块提供三个形态指标，作为新的验收
守门员，可对任意一图（实验 vs 模型）计算：

  1. central_dark_frac : 中央 r<=30px 内 I < 0.10*max 的像素占比
                         —— 抓"中央被掏空成暗洞"这类错误；
  2. outline_spectrum  : r(θ)（多阈值外脊追踪，向量化）的角向
                         傅里叶谐波 |A_m|/mean_r，m=1..12
                         —— 抓"轮廓瓣形/对称性"差异；
  3. shape_dist        : 与实验谱的归一化 L1 距离
                         sum|ΔA| / sum A_exp —— 单一可比较的标量。

用法（**务必用未模糊的原始强度图**——σ=2 模糊会抬升暗部与外沿，
使中央/轮廓指标失真）：
    from morphology_metrics import morph_metrics, morph_compare
    rep = morph_compare(Ie, Im)     # 两侧都传 σ=0 图像
    print(rep['morph_pass'], rep['central_delta'], rep['shape_dist'])
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

DEFAULT_FRACS = (0.12, 0.20, 0.30)
DEFAULT_M_MAX = 12
DEFAULT_R_IN = 30.0
DEFAULT_DARK_FRAC = 0.10

# ---- 形态门槛（2026-10-14 由 4 奇点诊断 + 阈值标定确定） ----
GATE_CENTRAL_DELTA = 0.12   # |Δ中央暗区占比| 上限
GATE_CV_RATIO = 0.55        # 模型 cv_r / 实验 cv_r 下限
                            # （实验轮廓起伏大；模型过光滑 = 不像）
GATE_SHAPE_DIST = 0.55      # 形状谱 L1 距离上限（辅助）


def centroid(I: np.ndarray) -> tuple[float, float]:
    H, W = I.shape
    ys, xs = np.mgrid[0:H, 0:W]
    w = np.clip(I, 0, None)
    t = float(w.sum())
    if t < 1e-12:
        return (H - 1) / 2.0, (W - 1) / 2.0
    return float((w * ys).sum() / t), float((w * xs).sum() / t)


def central_dark_frac(I: np.ndarray, r_in: float = DEFAULT_R_IN,
                      dark_frac: float = DEFAULT_DARK_FRAC) -> float:
    """中央圆域内"暗"像素占比（暗 = I < dark_frac * max）。"""
    cy, cx = centroid(I)
    H, W = I.shape
    ys, xs = np.mgrid[0:H, 0:W]
    rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    cen = rr <= r_in
    if not cen.any():
        return 0.0
    return float((I[cen] < dark_frac * I.max()).mean())


def outline_r_theta(I: np.ndarray, frac: float = 0.20, nt: int = 360,
                    rmax: float = 120.0, step: float = 0.5
                    ) -> tuple[np.ndarray, np.ndarray]:
    """r(θ)：每根射线上最外侧"仍亮"（>= frac*max）的半径。

    向量化实现（一次 map_coordinates 处理所有射线）。
    """
    cy, cx = centroid(I)
    thetas = np.linspace(0.0, 2 * np.pi, nt, endpoint=False)
    rr = np.arange(0.0, rmax, step)
    R, T = np.meshgrid(rr, thetas, indexing='ij')       # (nt, nr)
    ys = cy + R * np.sin(T)
    xs = cx + R * np.cos(T)
    vals = map_coordinates(I, [ys.ravel(), xs.ravel()], order=1,
                           mode='constant', cval=0.0).reshape(nt, -1)
    above = vals >= (frac * float(I.max()))
    has = above.any(axis=1)
    idx = np.where(has, above[:, ::-1].argmax(axis=1),
                   -1)                                   # from the end
    rvals = np.full(nt, np.nan)
    ok = idx >= 0
    rvals[ok] = rr[len(rr) - 1 - idx[ok]]
    if not ok.all() and ok.any():
        rvals[~ok] = np.nanmean(rvals[ok])
    return thetas, rvals


def outline_spectrum(I: np.ndarray, fracs=DEFAULT_FRACS, nt: int = 360,
                     rmax: float = 120.0, m_max: int = DEFAULT_M_MAX
                     ) -> np.ndarray:
    """多阈值平均的轮廓形状谱 |A_m|/mean_r（长度 m_max）。"""
    specs = []
    for frac in fracs:
        _th, r = outline_r_theta(I, frac=frac, nt=nt, rmax=rmax)
        if not np.isfinite(r).any():
            specs.append(np.zeros(m_max))
            continue
        r = np.where(np.isfinite(r), r, np.nanmean(r))
        mr = float(r.mean())
        F = np.fft.fft(r - mr) / len(r)
        specs.append(np.abs(F[1:m_max + 1]) / max(mr, 1e-9))
    return np.mean(np.stack(specs, axis=0), axis=0)


def morph_metrics(I: np.ndarray) -> dict:
    """单图形态描述子。"""
    _th, r20 = outline_r_theta(I, frac=0.20)
    cv_r = float(np.nanstd(r20) / max(np.nanmean(r20), 1e-9))
    return {
        'central_dark_frac': central_dark_frac(I),
        'mean_r': float(np.nanmean(r20)),
        'cv_r': cv_r,
        'spectrum': [float(x) for x in outline_spectrum(I)],
    }


def shape_distance(spec_a, spec_b) -> float:
    """归一化 L1 距离：sum|ΔA| / sum A_a（a=参考/实验）。"""
    a = np.asarray(spec_a, dtype=float)
    b = np.asarray(spec_b, dtype=float)
    den = float(np.sum(a)) + 1e-12
    return float(np.sum(np.abs(a - b)) / den)


def morph_compare(Ie: np.ndarray, Im: np.ndarray,
                  gate_central: float = GATE_CENTRAL_DELTA,
                  gate_cv_ratio: float = GATE_CV_RATIO,
                  gate_shape: float = GATE_SHAPE_DIST) -> dict:
    """实验 vs 模型的形态对照 + 门槛判定。返回可直接写入报告。

    三项判据（全部通过才算 morph_pass）：
      1. central  : |Δ中央暗区占比| <= gate_central
      2. cv_ratio : 模型 cv_r / 实验 cv_r >= gate_cv_ratio
                    （实验轮廓起伏大；模型若过光滑 —— 例如纯高斯型
                     或低阶圆对称组合 —— 视为"轮廓不像"）
      3. shape    : 形状谱 L1 距离 <= gate_shape（辅助）
    """
    me = morph_metrics(Ie)
    mm = morph_metrics(Im)
    sd = shape_distance(me['spectrum'], mm['spectrum'])
    cd = abs(mm['central_dark_frac'] - me['central_dark_frac'])
    cv_ratio = (mm['cv_r'] / me['cv_r']) if me['cv_r'] > 1e-9 \
        else float('nan')
    ok_c = cd <= gate_central
    ok_v = (cv_ratio >= gate_cv_ratio) if np.isfinite(cv_ratio) else False
    ok_s = sd <= gate_shape
    return {
        'central_dark_exp': me['central_dark_frac'],
        'central_dark_model': mm['central_dark_frac'],
        'central_delta': cd,
        'central_gate': bool(ok_c),
        'cv_r_exp': me['cv_r'],
        'cv_r_model': mm['cv_r'],
        'cv_ratio': float(cv_ratio),
        'cv_gate': bool(ok_v),
        'shape_dist': sd,
        'shape_gate': bool(ok_s),
        'morph_pass': bool(ok_c and ok_v and ok_s),
        'mean_r_exp': me['mean_r'],
        'mean_r_model': mm['mean_r'],
        'spectrum_exp': me['spectrum'],
        'spectrum_model': mm['spectrum'],
        'gate_central': gate_central,
        'gate_cv_ratio': gate_cv_ratio,
        'gate_shape': gate_shape,
    }


def morph_table_row(cn: str, Ie: np.ndarray, Im: np.ndarray) -> dict:
    """汇总表一行（供八图审计使用）。"""
    rep = morph_compare(Ie, Im)
    return {
        'cn': cn,
        'central_exp': rep['central_dark_exp'],
        'central_model': rep['central_dark_model'],
        'central_delta': rep['central_delta'],
        'cv_ratio': rep['cv_ratio'],
        'shape_dist': rep['shape_dist'],
        'morph_pass': rep['morph_pass'],
        'flags': [k for k, v in (('central', rep['central_gate']),
                                 ('cv', rep['cv_gate']),
                                 ('shape', rep['shape_gate']))
                  if not v],
    }

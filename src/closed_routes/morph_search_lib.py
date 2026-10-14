# -*- coding: utf-8 -*-
"""
morph_search_lib.py -- 形态先验搜索通用库（多图 / 多模数）
============================================================
从 4 奇点专项（stage13a/b/c）提炼的可复用实现：

  - render_dof     : 完整自由度渲染（全局平移 + 逐模 sx/sy + 逐模旋转）
  - make_eval      : C_s02 / 结构 / 形态 三位一体评估器
  - refine         : 坐标下降精修（theta 抖动 + 平移 + 各向异性 +
                     HG 旋转；结构+稳健约束）
  - screen_pool    : GPU 向量化粗筛（解析粗扫 → 64px 渲染 → 形态代理
                     central_dark + 轮廓谱距离）

形态代理为"筛查用近似"（几何中心 + frac=0.20 + 64px）；最终判定一律
用 `morphology_metrics.morph_compare`（质心口径 + 多阈值 + 224px）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate as ndrotate
from scipy.ndimage import shift as ndshift

from search_two_mode import gaussian_blur
from refine_all_new_method import build_base_fields_N, c_table_N
from structural_check import find_dark_points, match_points
from morphology_metrics import morph_compare
from rerun_new_metrics import energy_simplex_N, coarse_grid_params, \
    coarse_scan_N

# 各模数的"筛查用"粗网格（n_energy, n_phase）
SCREEN_GRID = {2: (16, 120), 3: (10, 12), 4: (6, 6), 5: (5, 4)}
RB = 64          # 筛查分辨率
NT, NR = 180, 60  # 极坐标采样


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def blur2(I):
    return gaussian_blur(torch.tensor(I, dtype=torch.float32)
                         .unsqueeze(0), 2.0)[0].numpy()


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def down_complex(x, size):
    t = torch.view_as_real(x).permute(2, 0, 1)[None]
    s = F.interpolate(t, size=(size, size), mode='bilinear',
                      align_corners=False)
    return torch.view_as_complex(s[0].permute(1, 2, 0).contiguous())


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


def render_dof(modes_np, lab2k, combo, theta, sx, sy, shiftv, rot):
    """完整自由度渲染（返回归一化光强）。"""
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


def make_eval(Ie, pts_e, modes_np, lab2k, combo):
    """绑定单图/单组合的三位一体评估器。"""
    Ie_b2 = blur2(Ie)
    N_EXP = len(pts_e)
    N = len(combo)

    def ev(theta, sx, sy, shiftv, rot, want_morph=False):
        Im = render_dof(modes_np, lab2k, combo, theta, sx, sy, shiftv,
                        rot)
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
             'mean_nn': float(mt['mean'])
             if mt['mean'] is not None else 99.0,
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
    return ev


def jitter(theta, rng, N, s_en=0.010, s_ph=0.035):
    u = np.array(theta[:N]) + rng.normal(0, s_en, N)
    u = np.clip(u, 0.005, None)
    u = u / u.sum()
    ph = np.array(theta[N:]) + rng.normal(0, s_ph, N - 1)
    return list(u) + list(ph)


def refine(ev, combo, theta0, sx0, sy0, shift0, rot0, rounds=2,
           budget=120, seed=0, tag=''):
    """坐标下降：theta 抖动 → 平移 → 逐模 sx/sy → HG 逐模旋转。"""
    N = len(combo)
    rot_idx = [i for i, m in enumerate(combo) if m.startswith('HG')]
    rng = np.random.default_rng(9100 + seed)
    base = ev(theta0, sx0, sy0, shift0, rot0, True)
    margin = max(14.0, base['max10'])
    best = base
    cur = base
    for rnd in range(rounds):
        for _ in range(budget):
            th = jitter(cur['theta'], rng, N)
            r = ev(th, cur['sx'], cur['sy'], cur['shift'], cur['rot'])
            if r['struct'] and r['max10'] <= margin \
                    and r['C_s02'] > best['C_s02']:
                best = r
                cur = r
        for step, rad in [(1.0, 2.0), (0.25, 0.5)]:
            for dy in np.arange(-rad, rad + 1e-9, step):
                for dx in np.arange(-rad, rad + 1e-9, step):
                    sh = [cur['shift'][0] + dy, cur['shift'][1] + dx]
                    r = ev(cur['theta'], cur['sx'], cur['sy'], sh,
                           cur['rot'])
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
                        r = ev(cur['theta'], sxv, syv, cur['shift'],
                               cur['rot'])
                        if r['struct'] and r['max10'] <= margin \
                                and r['C_s02'] > best['C_s02']:
                            best = r
                            cur = r
        for step, rad in [(1.0, 5.0), (0.25, 1.0)]:
            for i in rot_idx:
                for d in np.arange(-rad, rad + 1e-9, step):
                    rv = list(cur['rot'])
                    rv[i] = float(np.clip(rv[i] + d, -20.0, 20.0))
                    r = ev(cur['theta'], cur['sx'], cur['sy'],
                           cur['shift'], rv)
                    if r['struct'] and r['max10'] <= margin \
                            and r['C_s02'] > best['C_s02']:
                        best = r
                        cur = r
    final = ev(best['theta'], best['sx'], best['sy'], best['shift'],
               best['rot'], True)
    final['start_C_s02'] = base['C_s02']
    final['start_morph_pass'] = base['morph_pass']
    final['tag'] = tag
    return final


# ---------------------------------------------------------------------------
# GPU 粗筛
# ---------------------------------------------------------------------------
def _polar_grid(cy, cx, rmax):
    rr = np.linspace(0.0, rmax, NR)
    th = np.linspace(0.0, 2 * np.pi, NT, endpoint=False)
    R, T = np.meshgrid(rr, th, indexing='ij')
    ys = cy + R * np.sin(T)
    xs = cx + R * np.cos(T)
    gx = (xs / (RB - 1)) * 2 - 1
    gy = (ys / (RB - 1)) * 2 - 1
    return np.stack([gx, gy], axis=-1).astype(np.float32)


def gpu_morph_proxy(In, dev):
    """central_dark(几何中心, r<=30@224) + 轮廓谱(frac=0.20)。"""
    Bn = In.shape[0]
    mx = In.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    X = In / mx
    c = (RB - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(RB, device=dev),
                            torch.arange(RB, device=dev),
                            indexing='ij')
    rr = ((yy - c) ** 2 + (xx - c) ** 2).sqrt()
    cen = rr <= (30.0 / 224.0 * RB)
    dark = ((X < 0.10) & cen[None]).sum(dim=(1, 2)).float() \
        / cen.sum().float()
    grid0 = torch.tensor(_polar_grid(c, c, RB * 0.45),
                         device=dev)
    amps = []
    for b0 in range(0, Bn, 64):
        sub = X[b0:b0 + 64, None]
        b = sub.shape[0]
        grid = grid0.unsqueeze(0).expand(b, -1, -1, -1)
        samp = F.grid_sample(sub, grid, mode='bilinear',
                             align_corners=True,
                             padding_mode='zeros')[:, 0]
        above = samp >= 0.20
        has = above.any(dim=1)
        idx = torch.where(
            has, NR - 1 - above.flip(1).float().argmax(dim=1),
            torch.full_like(has, -1, dtype=torch.long))
        rr_line = torch.linspace(0.0, RB * 0.45, NR, device=dev)
        rv = torch.where(idx >= 0, rr_line[idx.clamp_min(0)],
                         torch.full_like(idx, -1, dtype=torch.float32)
                         .float())
        rv = torch.where(idx >= 0, rv,
                         torch.full_like(rv, float('nan')))
        mean_r = torch.nanmean(rv, dim=1)
        rv2 = torch.where(torch.isnan(rv), mean_r[:, None], rv)
        Fs = torch.fft.rfft(rv2 - mean_r[:, None], dim=1) / NT
        amps.append(Fs[:, 1:13].abs()
                    / mean_r[:, None].clamp_min(1e-9))
    return dark, torch.cat(amps, 0)


def _coarse_scan(modes_small, Ie_small, combos, c_tab, N, batch=256):
    """本库自带的解析粗扫（不依赖 rerun_new_metrics 的固定批次）。"""
    from search_three_mode import eval_c_grid
    dev = modes_small.device
    Ie = Ie_small
    mu = Ie.mean()
    var = Ie.var(unbiased=False)
    Q = combos.shape[0]
    P = c_tab.shape[0]
    bc = torch.empty(Q, device=dev)
    bi = torch.empty(Q, dtype=torch.long, device=dev)
    for b0 in range(0, Q, batch):
        q = combos[b0:b0 + batch].to(dev)
        Fs = [modes_small[q[:, i]] for i in range(N)]
        G = build_base_fields_N(Fs)
        B = G.shape[0]
        Gf = G.reshape(B, G.shape[1], -1)
        m = Gf.mean(dim=2)
        E = torch.einsum('bki,i->bk', Gf, Ie.reshape(-1)) / Gf.shape[2]
        Gram = torch.einsum('bki,bli->bkl', Gf, Gf) / Gf.shape[2]
        bmx = torch.full((B,), -2.0, device=dev)
        bmi = torch.zeros(B, dtype=torch.long, device=dev)
        for c0 in range(0, P, 100000):
            cc = c_tab[c0:c0 + 100000]
            corr = eval_c_grid(cc, m, E, Gram, mu, var)
            mx, mi = corr.max(dim=1)
            upd = mx > bmx
            bmx = torch.where(upd, mx, bmx)
            bmi = torch.where(upd, mi + c0, bmi)
        bc[b0:b0 + B] = bmx
        bi[b0:b0 + B] = bmi
    return bc, bi


def screen_pool(modes_np, Ie, combos, N, dev, batch=192):
    """对一个组合池做解析粗扫 + GPU 形态粗筛。

    combos: (Q, N) long tensor（全局模式索引）
    返回 dict(corr, central_d, shape_d, params, combos)
    """
    modes = torch.tensor(modes_np)
    Ie_t = torch.tensor(Ie, dtype=torch.float32)
    Ie_small = F.interpolate(Ie_t[None, None], size=(RB, RB),
                             mode='bilinear',
                             align_corners=False)[0, 0]
    modes_small = torch.stack([down_complex(m, RB) for m in modes
                               ]).to(dev)
    Ie_small = Ie_small.to(dev)
    ne, nph = SCREEN_GRID[N]
    pmat = coarse_grid_params(N, ne, nph)
    ct = c_table_N(pmat, N).to(dev)
    bc, bi = _coarse_scan(modes_small, Ie_small, combos, ct, N)
    P_best = pmat[bi.cpu().numpy()]                    # (Q, 2N-1)
    C_all = torch.tensor(c_table_N(P_best, N),
                         dtype=torch.float32)
    # experiment reference
    dark_e, amp_e = gpu_morph_proxy(Ie_small[None], dev)
    dark_e = float(dark_e[0])
    amp_e = amp_e[0].cpu().numpy()
    sum_e = float(np.sum(amp_e)) + 1e-12
    Q = combos.shape[0]
    central = np.zeros(Q, np.float32)
    shaped = np.zeros(Q, np.float32)
    for b0 in range(0, Q, batch):
        q = combos[b0:b0 + batch].to(dev)
        Fs = [modes_small[q[:, i]] for i in range(N)]
        G = build_base_fields_N(Fs)
        Cb = C_all[b0:b0 + batch].to(dev)
        I = torch.einsum('bk,bkhw->bhw', Cb, G).clamp_min(0)
        dark, amp = gpu_morph_proxy(I, dev)
        central[b0:b0 + batch] = dark.cpu().numpy()
        amp_np = amp.cpu().numpy()
        shaped[b0:b0 + batch] = (np.abs(amp_np - amp_e[None]).sum(1)
                                 / sum_e)
    return {'corr': bc.cpu().numpy(), 'central': central,
            'shape': shaped, 'params': P_best, 'combos': combos,
            'central_exp': dark_e}

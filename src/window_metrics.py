# -*- coding: utf-8 -*-
"""
window_metrics.py -- 窗口化解析指标引擎
=========================================

Per-window statistics make each window's PCC an analytic quadratic
form in the coefficient vector c (same math as the global PCC).
W_avg = equal-weight mean of window PCCs over support-region windows,
sigma in {0, 2} -- used both as the refinement objective and as a
reported metric.

Key functions:
  build_window_stats(modes, Ie, tr, scale, W, sigmas)
      -> list of (m, E, Gram, me, ve) per (window, sigma)
  w_avg_batch(win_stats, C)          -> (P,) W_avg for a batch of c rows
  window_pccs(win_stats, params, N)  -> per-window PCC vector (for the
                                        heatmap / rho / C_min)
"""

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from search_two_mode import gaussian_blur
from search_two_mode_lg import rescale_lg
from refine_all_new_method import build_base_fields_N, c_table_N

SIGMAS_W = (0, 2)


def make_windows(H=224, W=56):
    n = H // W
    return [(slice(i * W, (i + 1) * W), slice(j * W, (j + 1) * W))
            for i in range(n) for j in range(n)], n


def _eval_c_grid_b(c_tab, m, E, Gram, mu_e, var_e):
    """eval_c_grid with per-row mu/var (shape (B,) or (B,1))."""
    c_tab = c_tab.to(m.device)
    mu = mu_e.reshape(-1, 1).to(m.device)
    var = var_e.reshape(-1, 1).to(m.device)
    cm = torch.einsum('pk,bk->bp', c_tab, m)
    cE = torch.einsum('pk,bk->bp', c_tab, E)
    cG = torch.einsum('pk,bkl,pl->bp', c_tab, Gram, c_tab)
    cov = cE - cm * mu
    vart = cG - cm * cm
    valid = (var > 1e-12) & (vart > 1e-6 * (cG.abs() + cm * cm + 1e-12))
    corr = torch.where(
        valid,
        cov / torch.sqrt(vart.clamp_min(1e-12) * var),
        torch.full_like(cov, -1.0))
    return corr.clamp(-1.0, 1.0)


def _support_mask(Ie_np, dilate=8):
    return binary_dilation(Ie_np > 0.05 * Ie_np.max(),
                           iterations=dilate)


def build_window_stats(modes, Ie, tr, scale=None, W=56,
                       sigmas=SIGMAS_W, mask=None):
    """Per-(window, sigma) statistics. modes/Ie on device; returns
    (win_stats, win_meta, mask)."""
    dev = modes.device
    mnp = modes if scale is None else rescale_lg(modes, float(scale))
    Fs = [mnp[k].unsqueeze(0) for k in tr]
    G = build_base_fields_N(Fs)[0]          # (K,H,W)
    K = G.shape[0]

    Ie_np = Ie.detach().cpu().numpy()
    if mask is None:
        mask = _support_mask(Ie_np)

    blurred = {}
    Ie_blur = {}
    for sg in sigmas:
        blurred[sg] = gaussian_blur(G, float(sg))
        Ie_blur[sg] = gaussian_blur(Ie.unsqueeze(0), float(sg))[0]

    windows, n_side = make_windows(Ie.shape[0], W)
    win_stats, win_meta = [], []
    for wi, sl in enumerate(windows):
        i, j = wi // n_side, wi % n_side
        if mask[sl].mean() < 0.5:
            continue
        for sg in sigmas:
            Gb = blurred[sg][:, sl[0], sl[1]].reshape(K, -1)   # (K,npx)
            Ib = Ie_blur[sg][sl].reshape(-1)
            npx = Gb.shape[1]
            m = Gb.mean(dim=1).unsqueeze(0)                    # (1,K)
            E = ((Gb * Ib).sum(dim=1) / npx).unsqueeze(0)      # (1,K)
            Gram = (Gb @ Gb.T) / npx
            me = float(Ib.mean())
            ve = float(Ib.var(unbiased=False))
            win_stats.append((m, E, Gram.unsqueeze(0), me, ve))
            win_meta.append({'window': wi, 'row': i, 'col': j,
                             'sigma': sg})
    return win_stats, win_meta, mask


def _stack(win_stats, dev):
    m = torch.cat([s[0] for s in win_stats], 0)
    E = torch.cat([s[1] for s in win_stats], 0)
    Gram = torch.cat([s[2] for s in win_stats], 0)
    me = torch.tensor([s[3] for s in win_stats],
                      dtype=torch.float32, device=dev)
    ve = torch.tensor([s[4] for s in win_stats],
                      dtype=torch.float32, device=dev)
    return m, E, Gram, me, ve


def w_avg_batch(win_stats, C):
    """W_avg for a batch of coefficient rows C (P,K) -> (P,)."""
    m, E, Gram, me, ve = _stack(win_stats, C.device)
    corr = _eval_c_grid_b(C, m, E, Gram, me, ve)   # (Bw,P)
    return corr.mean(dim=0)


def window_pccs(win_stats, params, N):
    """Per-(window,sigma) PCC values for one parameter vector."""
    C = c_table_N(np.array([params], dtype=np.float64), N)
    m, E, Gram, me, ve = _stack(win_stats, C.device)
    corr = _eval_c_grid_b(C, m, E, Gram, me, ve)
    return corr[:, 0].detach().cpu().numpy()


def w_avg_point(win_stats, params, N):
    C = c_table_N(np.array([params], dtype=np.float64), N)
    return float(w_avg_batch(win_stats, C)[0])

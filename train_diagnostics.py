# ─── plotting & formatting upgrades ─────────────────────────────────────
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.stats as st            # for QQ plot

# Global, journal‑ready style (Computer Modern without LaTeX)
mpl.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Computer Modern",       # ← exact name
                          "CMU Serif",             # widely available clone
                          "DejaVu Serif"],         # bundled with Matplotlib
    "mathtext.fontset" : "cm",                     # Computer Modern math
    "font.size"        : 12,
    "axes.labelsize"   : 8,
    "axes.titlesize"   : 9,
    "legend.fontsize"  : 7,
    "xtick.labelsize"  : 7,
    "ytick.labelsize"  : 7,
    "axes.linewidth"   : 0.5,
})

HIST = defaultdict(list)            # keys: 'train', 'val'; values: [rmse]

# ——— general helper to apply grid, minor‑ticks, title ——————
def _style_axes(ax, title: str | None = None) -> None:
    ax.grid(True, which="major", lw=0.3, alpha=0.8)
    ax.grid(True, which="minor", lw=0.2, ls="--", alpha=0.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if title:
        ax.set_title(title, pad=4)

# ╭─────────────────────────── plotting helpers ─────────────────────────╮
def _plot_learning_curve(hist: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot(hist["train"], label="Train")
    ax.plot(hist["val"],   label="Validation")
    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"RMSE $\downarrow$")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _style_axes(ax, r"Learning curve: RMSE vs. Epoch")
    fig.tight_layout()
    plt.show()

def _error_vs_lead(all_pred: np.ndarray, all_true: np.ndarray) -> None:
    mse  = (all_pred - all_true) ** 2
    rmse = np.sqrt(mse.mean(axis=0))

    dt, eps = 600.0, 1e-5
    gamma   = -np.log(eps) / (len(rmse) * dt)
    weights = np.exp(-gamma * np.arange(len(rmse)) * dt)
    od_rmse = np.sqrt((mse * weights).mean(axis=0))

    t_hr = np.arange(len(rmse)) * dt / 3600
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot(t_hr, rmse,    label="RMSE")
    ax.plot(t_hr, od_rmse, label="OD-weighted RMSE")
    ax.set_xlabel(r"Lead time [h]")
    ax.set_ylabel(r"RMSE $\downarrow$")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _style_axes(ax, r"Prediction error vs.\ lead time")
    fig.tight_layout()
    plt.show()

def _scatter_truth_pred(all_pred: np.ndarray, all_true: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    hb = ax.hexbin(all_true.ravel(), all_pred.ravel(),
                   bins="log", gridsize=120)
    lim = [min(all_true.min(), all_pred.min()),
           max(all_true.max(), all_pred.max())]
    ax.plot(lim, lim, "--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"True $\rho$  [kg\,m$^{-3}$]")
    ax.set_ylabel(r"Pred.\ $\rho$  [kg\,m$^{-3}$]")
    cb = fig.colorbar(hb, pad=0.01)
    cb.set_label(r"$\log_{10}\,$(count)")
    _style_axes(ax, r"True vs.\ predicted density")
    fig.tight_layout()
    plt.show()

def _residual_diags(all_pred: np.ndarray, all_true: np.ndarray) -> None:
    resid = (all_pred - all_true).ravel()
    fig, ax = plt.subplots(1, 2, figsize=(5.0, 2.5))

    # PDF
    ax[0].hist(resid, bins=120, density=True, alpha=0.75)
    mu, sigma = resid.mean(), resid.std()
    xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    ax[0].plot(xs, st.norm.pdf(xs, mu, sigma), lw=1)
    ax[0].set_xlabel(r"Residual $\rho$  [kg\,m$^{-3}$]")
    ax[0].set_ylabel(r"PDF")
    _style_axes(ax[0], r"Residual PDF")

    # QQ
    st.probplot(resid, dist="norm", plot=ax[1])
    ax[1].set_xlabel(r"Theoretical quantiles")
    ax[1].set_ylabel(r"Sample quantiles")
    _style_axes(ax[1], r"QQ plot")

    fig.tight_layout()
    plt.show()

# ───── Raw vs Cleaned density distribution ────────────────────────────
def plot_raw_clean(raw, clean):
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.5), sharex=True, sharey=True)
    ax[0].hist(raw.ravel(),   bins=200, alpha=0.6)
    ax[1].hist(clean.ravel(), bins=200, alpha=0.6, color="tab:orange")
    for a in ax:
        a.set_yscale("log")
        a.set_xlabel(r"$\rho$  [kg\,m$^{-3}$]")
        a.set_ylabel(r"Count")
        _style_axes(a)
    ax[0].set_title(r"Raw distribution", pad=4)
    ax[1].set_title(r"Cleaned distribution", pad=4)
    fig.suptitle(r"Density distribution before and after cleaning", y=1.02)
    fig.tight_layout()
    plt.show()

# ───── Correlation matrix on cleaned features ──────────────────────────
def plot_corr(mat):
    mat2d = mat.reshape(-1, mat.shape[-1]) if mat.ndim == 3 else mat
    c = np.corrcoef(mat2d, rowvar=False)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel(r"Feature index")
    ax.set_ylabel(r"Feature index")
    _style_axes(ax, r"Feature correlation matrix")
    cb = fig.colorbar(im, fraction=0.046)
    cb.set_label(r"$\rho$")
    fig.tight_layout()
    plt.show()

# ───── Residual histogram by F10.7 quartile ────────────────────────────
def plot_f107_residuals(pred, truth, f107):
    q = np.quantile(f107, [0, .25, .5, .75, 1])
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    for i in range(4):
        mask  = (f107 >= q[i]) & (f107 < q[i+1])
        resid = (pred[mask] - truth[mask]).ravel()
        ax.hist(resid, bins=120, density=True, alpha=0.5,
                label=rf"$Q_{i+1}$  [{q[i]:.0f}\,--\,{q[i+1]:.0f}]")
    ax.set_xlabel(r"Residual $\rho$  [kg\,m$^{-3}$]")
    ax.set_ylabel(r"PDF")
    _style_axes(ax, r"Residuals by F10.7 quartile")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

# ───── Representative time‑series overlay ──────────────────────────────
def plot_overlay(ts, truth, pred, n=3):
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    for i in range(n):
        ax.plot(ts[i], truth[i], 'k-', lw=.8, alpha=.7)
        ax.plot(ts[i], pred[i],  'r--', lw=.8, alpha=.7)
    ax.set_yscale("log")
    ax.set_xlabel(r"UTC")
    ax.set_ylabel(r"$\rho$  [kg\,m$^{-3}$]")
    _style_axes(ax, r"Representative time‑series overlay")
    fig.tight_layout()
    plt.show()

# ───── Attention heat map (mean over validation) ───────────────────────
def plot_attn_heat(attn_w):
    w_mean = attn_w.mean(axis=0)         # (T,)
    fig, ax = plt.subplots(figsize=(4.8, 1.4))
    im = ax.imshow(w_mean[None, :], cmap="magma", aspect="auto")
    cb = fig.colorbar(im, pad=0.02)
    cb.set_label(r"Weight")
    ax.set_xlabel(r"History step (0 = oldest)")
    _style_axes(ax, r"Mean attention weights")
    fig.tight_layout()
    plt.show()

# ───── Calibration curve (abs residual vs pred) ────────────────────────
def plot_calibration(pred, truth, bins=20):
    """|residual| vs predicted density on logarithmic axes."""
    pred_flat   = pred.ravel()
    truth_flat  = truth.ravel()
    resid       = np.abs(pred_flat - truth_flat)

    mask        = np.isfinite(pred_flat) & np.isfinite(resid)
    pred_flat   = pred_flat[mask]
    resid       = resid[mask]

    q, edges = pd.qcut(pred_flat, bins, retbins=True,
                       labels=False, duplicates="drop")
    mu  = np.array([resid[q == i].mean() for i in range(q.max() + 1)])
    cen = 0.5 * (edges[1:] + edges[:-1])

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(cen, mu, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Predicted $\rho$  [kg\,m$^{-3}$]")
    ax.set_ylabel(r"$\mathbb{E}$  [kg\,m$^{-3}$]")
    _style_axes(ax, r"Calibration curve")
    fig.tight_layout()
    plt.show()
# ╰──────────────────────────────────────────────────────────────────────╯


data = np.load("train_diagnostics.npz")
hist = {"train": data["train_rmse"].tolist(),
        "val":   data["val_rmse"].tolist()}
_plot_learning_curve(hist)
_error_vs_lead(data["val_pred"], data["val_true"])
_scatter_truth_pred(data["val_pred"], data["val_true"])
_residual_diags(data["val_pred"], data["val_true"])
plot_raw_clean(data["raw_hist"], data["clean_hist"])            # replace 2nd arg
plot_corr(data["raw_hist"])
plot_f107_residuals(data["val_pred"], data["val_true"], data["f107"])
plot_attn_heat(data["attn_w"])
plot_calibration(data["val_pred"], data["val_true"])
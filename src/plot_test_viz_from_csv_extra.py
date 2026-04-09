# -*- coding: utf-8 -*-
"""
plot_test_viz_from_csv_extra.py - 从 CSV 预测结果生成增强版可视化

功能：
- 从 test_df_raw_pointwise_predictions.csv 读取预测结果
- 生成多种可视化：奇偶图、三角相图、统计分布等
- 支持高级统计和误差分析
- 生成含注释的可视化图表

用法：
  python plot_test_viz_from_csv_extra.py --csv test_df_raw_pointwise_predictions.csv --out_dir ./results
"""
import os

# ========= Windows OpenMP 冲突 workaround（建议保留）=========
# 解决 Intel MKL 与 OpenMP 的库冲突问题
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
# 必须在 import pyplot 之前设置后端
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 健壮的导入路径设置 ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for p in [_THIS_DIR, _PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 导入本项目的绘图模块
import viz

# metrics fallback
try:
    from metrics import calc_mae_rmse_r2
except Exception:
    def calc_mae_rmse_r2(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
        r2 = 1.0 - ss_res / ss_tot
        return mae, rmse, r2


# -----------------------------
# Style helpers
# -----------------------------

# Nature级别配色方案
NATURE_COLORS = {
    # E相配色：柔和的蓝紫色系
    'E1': '#4A6FE3',  # 宝石蓝
    'E2': '#8E44AD',  # 紫罗兰
    'E3': '#66A61E',  # 天蓝
    'E_primary': '#3457A4',  # 深蓝
    # R相配色：强区分度的红-橙-黄色系
    'R1': '#A6761D',
    'R2': '#FC8D62',
    'R3': '#E7298A',
    'R_primary': '#D35400',  # 深橙
    # 辅助色
    'tieline': '#95A5A6',  # 灰色
    'reference': '#34495E',  # 深灰蓝
    'grid': '#ECF0F1',  # 浅灰

}

def apply_style(font_scale: float = 1.8):
    """
    Nature-like 风格 + 字号整体放大
    """
    try:
        viz.apply_nature_style()
    except TypeError:
        viz.apply_nature_style(plt)

    def _scale(key, default):
        v = plt.rcParams.get(key, default)
        try:
            v = float(v)
        except Exception:
            v = float(default)
        plt.rcParams[key] = v * float(font_scale)

    _scale("font.size", 10)
    _scale("axes.titlesize", 12)
    _scale("axes.labelsize", 11)
    _scale("xtick.labelsize", 10)
    _scale("ytick.labelsize", 10)
    _scale("legend.fontsize", 10)

    # --- thicken axes/ticks globally ---
    plt.rcParams["axes.linewidth"] = max(1.6, float(plt.rcParams.get("axes.linewidth", 1.0)) * 1.35)
    plt.rcParams["xtick.major.width"] = max(1.6, float(plt.rcParams.get("xtick.major.width", 1.0)) * 1.35)
    plt.rcParams["ytick.major.width"] = max(1.6, float(plt.rcParams.get("ytick.major.width", 1.0)) * 1.35)
    plt.rcParams["xtick.major.size"] = max(5.0, float(plt.rcParams.get("xtick.major.size", 4.0)) * 1.15)
    plt.rcParams["ytick.major.size"] = max(5.0, float(plt.rcParams.get("ytick.major.size", 4.0)) * 1.15)

    # --- thicken default line ---
    plt.rcParams["lines.linewidth"] = max(1.6, float(plt.rcParams.get("lines.linewidth", 1.2)) * 1.25)


# -----------------------------
# Column normalization
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "temperature": "T",
        "temperature_raw": "T",
        "true_EX_comp1": "Ex1",
        "true_EX_comp2": "Ex2",
        "true_EX_comp3": "Ex3",
        "true_RX_comp1": "Rx1",
        "true_RX_comp2": "Rx2",
        "true_RX_comp3": "Rx3",
        "pred_EX_comp1": "pred_Ex1",
        "pred_EX_comp2": "pred_Ex2",
        "pred_EX_comp3": "pred_Ex3",
        "pred_RX_comp1": "pred_Rx1",
        "pred_RX_comp2": "pred_Rx2",
        "pred_RX_comp3": "pred_Rx3",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    if "T" not in df.columns:
        raise ValueError("CSV 缺少温度列：需要 'T' 或 'temperature'/'temperature_raw'。")
    if "system_id" not in df.columns:
        raise ValueError("CSV 缺少 system_id 列，无法按体系画相图。")

    if "t" not in df.columns:
        df["t"] = df.groupby(["system_id", "T"]).cumcount().astype(np.float32)
        df["t"] = df.groupby(["system_id", "T"])["t"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
        )

    need = [
        "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3",
        "pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3"
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺少必要列: {miss}")

    return df


# -----------------------------
# Common helpers
# -----------------------------
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _savefig(fig, out_path: str, dpi: int = 350, pad: float = 0.10):
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def _phase_cols(phase: str):
    phase = phase.upper()
    if phase == "E":
        true_cols = ["Ex1", "Ex2", "Ex3"]
        pred_cols = ["pred_Ex1", "pred_Ex2", "pred_Ex3"]
        comp_labels = ["E1", "E2", "E3"]
        phase_name = "Extract"
    else:
        true_cols = ["Rx1", "Rx2", "Rx3"]
        pred_cols = ["pred_Rx1", "pred_Rx2", "pred_Rx3"]
        comp_labels = ["R1", "R2", "R3"]
        phase_name = "Raffinate"
    return true_cols, pred_cols, comp_labels, phase_name


def _compute_err_arrays(df: pd.DataFrame, phase: str):
    true_cols, pred_cols, comp_labels, phase_name = _phase_cols(phase)
    y_true = df[true_cols].to_numpy(dtype=np.float64)
    y_pred = df[pred_cols].to_numpy(dtype=np.float64)
    err = y_pred - y_true
    return y_true, y_pred, err, comp_labels, phase_name


def _downsample_xy(x, y, max_points: int, seed: int = 0):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if max_points <= 0 or n <= max_points:
        return x, y
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_points), replace=False)
    return x[idx], y[idx]


def _gaussian_kde_1d(samples: np.ndarray, x_grid: np.ndarray):
    """
    尽量用 scipy.stats.gaussian_kde；没有 scipy 时用轻量手写 KDE。
    返回在 x_grid 上的密度。
    """
    s = np.asarray(samples, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if len(s) < 5:
        return np.zeros_like(x_grid, dtype=np.float64)

    try:
        from scipy.stats import gaussian_kde  # type: ignore
        kde = gaussian_kde(s)
        return kde.evaluate(x_grid)
    except Exception:
        # Silverman bandwidth
        n = len(s)
        std = float(np.std(s, ddof=1))
        iqr = float(np.subtract(*np.percentile(s, [75, 25])))
        sigma = min(std, iqr / 1.34) if iqr > 0 else std
        sigma = sigma if sigma > 1e-8 else max(std, 1e-3)
        h = 1.06 * sigma * (n ** (-1.0 / 5.0))
        h = max(h, 1e-3)

        xg = np.asarray(x_grid, dtype=np.float64)
        dens = np.zeros_like(xg, dtype=np.float64)
        chunk = 4096
        inv = 1.0 / (np.sqrt(2.0 * np.pi) * h * n)
        for i in range(0, len(xg), chunk):
            xx = xg[i:i + chunk][:, None]
            z = (xx - s[None, :]) / h
            dens[i:i + chunk] = inv * np.sum(np.exp(-0.5 * z * z), axis=1)
        return dens


def _short_label(s: str, max_len: int = 22):
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


# -----------------------------
# Parity plots (reference style)
# -----------------------------
def parity_plot_combined(df: pd.DataFrame, out_path: str):
    """合并E相和R相的parity图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0))
    
    for ax, phase in zip(axes, ["E", "R"]):
        true_cols, pred_cols, comp_labels, phase_name = _phase_cols(phase)
        
        y_true_all = df[true_cols].to_numpy(dtype=np.float64).reshape(-1)
        y_pred_all = df[pred_cols].to_numpy(dtype=np.float64).reshape(-1)
        mae, rmse, r2 = calc_mae_rmse_r2(y_true_all, y_pred_all)
        
        # 使用Nature配色
        colors = [NATURE_COLORS[f'{phase}{i}'] for i in [1, 2, 3]]
        markers = ["o", "s", "^"]
        
        for i, (tc, pc, lb, mk, col) in enumerate(zip(true_cols, pred_cols, comp_labels, markers, colors)):
            ax.scatter(df[tc], df[pc], s=55, alpha=0.7, marker=mk, 
                      color=col, edgecolors='white', linewidths=0.5, label=lb)
        
        lo, hi = -0.05, 1.05
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], color=NATURE_COLORS['reference'], 
               linestyle="--", linewidth=2.2, alpha=0.8)
        
        ax.set_xlabel(f"True composition")
        ax.set_ylabel(f"Predicted composition")
        ax.set_title(f"{phase_name}", fontweight='bold')
        ax.set_aspect('equal')
        
        txt = f"MAE  {mae:.4f}\nRMSE {rmse:.4f}\nR²   {r2:.4f}"
        ax.text(
            0.05, 0.95, txt,
            transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                     alpha=0.92, linewidth=1.2, edgecolor=NATURE_COLORS['reference']),
        )
        
        leg = ax.legend(loc="lower right", frameon=True)
        leg.get_frame().set_edgecolor(NATURE_COLORS['reference'])
        leg.get_frame().set_linewidth(1.3)
        leg.get_frame().set_alpha(0.95)
    
    fig.suptitle("Parity Plot: Extract vs Raffinate", y=0.98, fontweight='bold')
    _savefig(fig, out_path, dpi=350, pad=0.12)


# -----------------------------
# Extra plots requested
# -----------------------------
def plot_error_hist_kde_combined(df: pd.DataFrame, out_path: str, bins: int = 50):
    """
    误差直方图 + KDE（E相和R相合并，分组分）
    """
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), sharex=False, sharey=False)
    
    for row, phase in enumerate(["E", "R"]):
        _, _, err, labels, phase_name = _compute_err_arrays(df, phase)
        
        flat = err.reshape(-1)
        q = np.nanpercentile(flat, [1, 99])
        xlim = (float(q[0]), float(q[1]))
        pad = 0.08 * (xlim[1] - xlim[0] + 1e-12)
        xlim = (xlim[0] - pad, xlim[1] + pad)
        x_grid = np.linspace(xlim[0], xlim[1], 400)
        
        colors = [NATURE_COLORS[f'{phase}{i}'] for i in [1, 2, 3]]
        
        for j, ax in enumerate(axes[row, :]):
            e = err[:, j]
            e = e[np.isfinite(e)]
            
            ax.hist(e, bins=bins, density=True, alpha=0.45, 
                   color=colors[j], edgecolor='white', linewidth=0.5)
            kde_y = _gaussian_kde_1d(e, x_grid)
            ax.plot(x_grid, kde_y, color=colors[j], linewidth=2.8, alpha=0.9)
            ax.axvline(0.0, color=NATURE_COLORS['reference'], 
                      linestyle="--", linewidth=1.8, alpha=0.7)
            
            mae = float(np.mean(np.abs(e))) if len(e) else float("nan")
            mu = float(np.mean(e)) if len(e) else float("nan")
            sd = float(np.std(e, ddof=1)) if len(e) > 1 else float("nan")
            
            ax.set_title(f"{phase_name} - {labels[j]}", fontweight='bold')
            ax.set_xlim(*xlim)
            ax.set_xlabel("Residual (pred - true)")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.2, linestyle=':')
            
            ax.text(
                0.97, 0.95,
                f"MAE {mae:.4f}\nmean {mu:.4f}\nstd {sd:.4f}",
                transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", 
                         alpha=0.9, linewidth=1.0, edgecolor=colors[j])
            )
    
    fig.suptitle("Residual Distribution: Histogram + KDE", y=0.995, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    _savefig(fig, out_path, dpi=350, pad=0.15)


def plot_bland_altman_combined(df: pd.DataFrame, out_path: str, max_points: int = 8000, seed: int = 0):
    """
    Bland–Altman（差值 vs 均值，E相和R相合并）
    """
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), sharex=True, sharey=False)
    
    for row, phase in enumerate(["E", "R"]):
        y_true, y_pred, err, labels, phase_name = _compute_err_arrays(df, phase)
        colors = [NATURE_COLORS[f'{phase}{i}'] for i in [1, 2, 3]]
        
        for j, ax in enumerate(axes[row, :]):
            m = 0.5 * (y_true[:, j] + y_pred[:, j])
            d = err[:, j]
            m, d = _downsample_xy(m, d, max_points=max_points, seed=seed)
            
            ax.scatter(m, d, s=22, alpha=0.4, color=colors[j], 
                      edgecolors='white', linewidths=0.3)
            ax.axhline(0.0, color=NATURE_COLORS['reference'], 
                      linestyle="--", linewidth=2.0, alpha=0.7)
            
            mu = float(np.mean(d)) if len(d) else 0.0
            sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
            loa_up = mu + 1.96 * sd
            loa_dn = mu - 1.96 * sd
            
            ax.axhline(mu, color=colors[j], linewidth=2.2, alpha=0.8)
            ax.axhline(loa_up, color=colors[j], linestyle=":", linewidth=1.8, alpha=0.6)
            ax.axhline(loa_dn, color=colors[j], linestyle=":", linewidth=1.8, alpha=0.6)
            
            ax.set_title(f"{phase_name} - {labels[j]}", fontweight='bold')
            ax.set_xlabel("Mean of (pred, true)")
            ax.set_ylabel("Difference (pred - true)")
            ax.grid(True, alpha=0.2, linestyle=':')
            
            ax.text(
                0.97, 0.95,
                f"mean {mu:.4f}\nLoA ±1.96σ\n[{loa_dn:.4f}, {loa_up:.4f}]",
                transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", 
                         alpha=0.9, linewidth=1.0, edgecolor=colors[j])
            )
    
    fig.suptitle("Bland–Altman Plot", y=0.995, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    _savefig(fig, out_path, dpi=350, pad=0.15)


def plot_cdf_abs_error(df: pd.DataFrame, out_path: str):
    """
    CDF/分位数曲线（误差累计分布）
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=False)

    for ax, phase in zip(axes, ["E", "R"]):
        _, _, err, labels, phase_name = _compute_err_arrays(df, phase)
        colors = [NATURE_COLORS[f'{phase}{i}'] for i in [1, 2, 3]]
        
        for j in range(3):
            e = np.abs(err[:, j])
            e = e[np.isfinite(e)]
            if len(e) == 0:
                continue
            xs = np.sort(e)
            ys = (np.arange(1, len(xs) + 1) / len(xs))
            ax.plot(xs, ys, label=labels[j], color=colors[j], linewidth=2.5, alpha=0.85)

            q90 = np.percentile(e, 90)
            ax.scatter([q90], [0.90], s=60, marker="o", color=colors[j], 
                      alpha=0.8, edgecolors='white', linewidths=1.5, zorder=10)

        ax.set_title(f"{phase_name}", fontweight='bold')
        ax.set_xlabel("Absolute error |pred-true|")
        ax.set_ylabel("Cumulative Distribution Function")
        ax.grid(True, alpha=0.2, linestyle=':')
        
        leg = ax.legend(loc="lower right", frameon=True)
        leg.get_frame().set_edgecolor(NATURE_COLORS['reference'])
        leg.get_frame().set_linewidth(1.3)
        leg.get_frame().set_alpha(0.95)

    fig.suptitle("CDF of Absolute Errors (90th percentile marked)", y=0.98, fontweight="bold")
    _savefig(fig, out_path, dpi=350, pad=0.12)


def _binned_mean_line(x, y, n_bins: int = 20):
    """
    residual-vs-true 的“分箱均值”曲线，用于观察异方差/系统偏差。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 50:
        return None

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xc, yc = [], []
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() < 10:
            continue
        xc.append(0.5 * (bins[i] + bins[i + 1]))
        yc.append(float(np.mean(y[mask])))
    if len(xc) < 5:
        return None
    return np.array(xc), np.array(yc)


def plot_residual_vs_true_combined(df: pd.DataFrame, out_path: str, max_points: int = 8000, seed: int = 0):
    """
    残差 vs 真值散点（heteroscedasticity 检查，E相和R相合并）
    """
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), sharex=True, sharey=False)
    
    for row, phase in enumerate(["E", "R"]):
        y_true, _, err, labels, phase_name = _compute_err_arrays(df, phase)
        colors = [NATURE_COLORS[f'{phase}{i}'] for i in [1, 2, 3]]
        
        for j, ax in enumerate(axes[row, :]):
            x = y_true[:, j]
            y = err[:, j]
            x, y = _downsample_xy(x, y, max_points=max_points, seed=seed)
            
            ax.scatter(x, y, s=22, alpha=0.4, color=colors[j], 
                      edgecolors='white', linewidths=0.3)
            ax.axhline(0.0, color=NATURE_COLORS['reference'], 
                      linestyle="--", linewidth=1.8, alpha=0.7)
            ax.set_xlim(-0.05, 1.05)
            ax.set_title(f"{phase_name} - {labels[j]}", fontweight='bold')
            ax.set_xlabel("True composition")
            ax.set_ylabel("Residual (pred - true)")
            ax.grid(True, alpha=0.2, linestyle=':')
            
            line = _binned_mean_line(y_true[:, j], err[:, j], n_bins=22)
            if line is not None:
                xc, yc = line
                ax.plot(xc, yc, color=colors[j], linewidth=2.5, alpha=0.85, label='Binned mean')
                ax.legend(loc='best', framealpha=0.9)
    
    fig.suptitle("Residual vs True Composition", y=0.995, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    _savefig(fig, out_path, dpi=350, pad=0.15)


def plot_sum_to_one_combined(df: pd.DataFrame, out_path: str, bins: int = 55):
    """
    sum-to-one 约束误差分布：pred_sum - 1（E相和R相合并）
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    
    for ax, phase in zip(axes, ["E", "R"]):
        _, pred_cols, _, phase_name = _phase_cols(phase)
        pred_sum = df[pred_cols].to_numpy(dtype=np.float64).sum(axis=1)
        dev = pred_sum - 1.0
        dev = dev[np.isfinite(dev)]
        
        q = np.nanpercentile(dev, [1, 99])
        xlim = (float(q[0]), float(q[1]))
        pad = 0.12 * (xlim[1] - xlim[0] + 1e-12)
        xlim = (xlim[0] - pad, xlim[1] + pad)
        x_grid = np.linspace(xlim[0], xlim[1], 400)
        
        pred = df[pred_cols].to_numpy(dtype=np.float64)
        out_rate = float(np.mean((pred < -1e-6) | (pred > 1.0 + 1e-6)))
        
        color = NATURE_COLORS[f'{phase}_primary']
        ax.hist(dev, bins=bins, density=True, alpha=0.5, 
               color=color, edgecolor='white', linewidth=0.5)
        kde_y = _gaussian_kde_1d(dev, x_grid)
        ax.plot(x_grid, kde_y, color=color, linewidth=2.8, alpha=0.9)
        ax.axvline(0.0, color=NATURE_COLORS['reference'], 
                  linestyle="--", linewidth=2.0, alpha=0.7)
        
        mu = float(np.mean(dev)) if len(dev) else float("nan")
        sd = float(np.std(dev, ddof=1)) if len(dev) > 1 else float("nan")
        ax.set_xlim(*xlim)
        ax.set_xlabel("Sum deviation (pred_sum - 1)")
        ax.set_ylabel("Density")
        ax.set_title(f"{phase_name}", fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle=':')
        
        ax.text(
            0.97, 0.95,
            f"mean {mu:.4e}\nstd  {sd:.4e}\nOOR rate {out_rate*100:.2f}%",
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", 
                     alpha=0.92, linewidth=1.0, edgecolor=color)
        )
    
    fig.suptitle("Sum-to-one Constraint Deviation", y=0.98, fontweight='bold')
    _savefig(fig, out_path, dpi=350, pad=0.12)


def plot_violin_or_box_by_category(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    out_path: str,
    top_n: int = 12,
    kind: str = "violin",
):
    """
    按类别画误差分布（箱线图 / 小提琴图）
    """
    if category_col not in df.columns:
        return

    data = df[[category_col, value_col]].copy()
    data = data[np.isfinite(data[value_col].to_numpy(dtype=np.float64))]
    data[category_col] = data[category_col].astype(str)

    counts = data[category_col].value_counts()
    cats = list(counts.head(int(top_n)).index)
    data = data[data[category_col].isin(cats)]
    if len(data) == 0:
        return

    cats_sorted = sorted(cats, key=lambda c: (-counts[c], str(c)))
    values = [data.loc[data[category_col] == c, value_col].to_numpy(dtype=np.float64) for c in cats_sorted]

    fig, ax = plt.subplots(figsize=(max(9.5, 0.65 * len(cats_sorted)), 5.4))

    # 使用Nature配色方案的渐变色
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(cats_sorted)))
    
    positions = np.arange(1, len(cats_sorted) + 1)
    if kind.lower().startswith("box"):
        bp = ax.boxplot(
            values,
            positions=positions,
            widths=0.65,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(linewidth=2.0, color=NATURE_COLORS['reference']),
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )
        for i, b in enumerate(bp.get("boxes", [])):
            b.set_facecolor(colors[i])
            b.set_edgecolor(NATURE_COLORS['reference'])
            b.set_alpha(0.7)
        for w in bp.get("whiskers", []) + bp.get("caps", []):
            w.set_color(NATURE_COLORS['reference'])
            w.set_alpha(0.8)
    else:
        parts = ax.violinplot(values, positions=positions, widths=0.75, showmeans=False, showmedians=True, showextrema=False)
        for i, body in enumerate(parts.get("bodies", [])):
            body.set_facecolor(colors[i])
            body.set_edgecolor(NATURE_COLORS['reference'])
            body.set_alpha(0.7)
            body.set_linewidth(1.5)
        if "cmedians" in parts:
            parts["cmedians"].set_color(NATURE_COLORS['reference'])
            parts["cmedians"].set_linewidth(2.0)
            parts["cmedians"].set_alpha(0.9)

    xt = [f"{_short_label(c, 18)}\n(n={int(counts[c])})" for c in cats_sorted]
    ax.set_xticks(positions)
    ax.set_xticklabels(xt, rotation=45, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(f"{value_col} by {category_col} (top {len(cats_sorted)} by count)", fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle=':', axis='y')

    _savefig(fig, out_path, dpi=350, pad=0.25)


def add_error_columns_for_group_plots(df: pd.DataFrame) -> pd.DataFrame:
    """
    为分类箱线/小提琴图准备 3 个值：
    - abs_err_E: mean(|err_E1|,|err_E2|,|err_E3|)
    - abs_err_R: mean(|err_R1|,|err_R2|,|err_R3|)
    - abs_err_all: mean(E+R 共 6 个组分的绝对误差)
    """
    df = df.copy()
    e = np.abs(df[["pred_Ex1", "pred_Ex2", "pred_Ex3"]].to_numpy(dtype=np.float64) - df[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float64))
    r = np.abs(df[["pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float64) - df[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float64))
    df["abs_err_E"] = np.mean(e, axis=1)
    df["abs_err_R"] = np.mean(r, axis=1)
    df["abs_err_all"] = np.mean(np.concatenate([e, r], axis=1), axis=1)
    return df


def plot_violin_combined_categories(
    df: pd.DataFrame,
    out_path: str,
    top_n: int = 12,
    kind: str = "violin",
):
    """
    组合的violin/box图：第一行显示IL abbreviation，第二行两列分别显示Family of component 2和Family of component 3
    """
    categories = [
        ("IL abbreviation", "abs_err_all"),
        ("Family of component 2", "abs_err_all"),
        ("Family of component 3", "abs_err_all"),
    ]
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 0.95],
        hspace=2.20,
        wspace=0.32,
        left=0.07,
        right=0.98,
        top=0.90,
        bottom=0.60,
    )
    
    # 第一行：IL abbreviation（占据整个宽度）
    ax_il = fig.add_subplot(gs[0, :])
    axes = [ax_il, 
            fig.add_subplot(gs[1, 0]),  # 组分2
            fig.add_subplot(gs[1, 1])]  # 组分3
    
    # 为不同的category设置不同的top_n
    top_n_map = {
        "IL abbreviation": 50,
        "Family of component 2": 15,
        "Family of component 3": 10,
    }
    
    # 先计算所有数据的最大值，确保y轴范围一致
    all_values = []
    for category_col, value_col in categories:
        if category_col not in df.columns:
            continue
        data = df[[category_col, value_col]].copy()
        data = data[np.isfinite(data[value_col].to_numpy(dtype=np.float64))]
        all_values.extend(data[value_col].to_numpy(dtype=np.float64).tolist())
    
    y_max = np.max(all_values) if all_values else 0.1
    y_lim = (0, y_max * 1.1)  # 留出10%的上边界空间
    
    for idx, (ax, (category_col, value_col)) in enumerate(zip(axes, categories)):
        if category_col not in df.columns:
            continue
            
        data = df[[category_col, value_col]].copy()
        data = data[np.isfinite(data[value_col].to_numpy(dtype=np.float64))]
        data[category_col] = data[category_col].astype(str)
        
        counts = data[category_col].value_counts()
        current_top_n = top_n_map.get(category_col, top_n)
        cats = list(counts.head(int(current_top_n)).index)
        data = data[data[category_col].isin(cats)]
        if len(data) == 0:
            continue
        
        cats_sorted = sorted(cats, key=lambda c: (-counts[c], str(c)))
        values = [data.loc[data[category_col] == c, value_col].to_numpy(dtype=np.float64) for c in cats_sorted]
        
        # Nature级别高端配色方案
        n_colors = len(cats_sorted)
        if n_colors <= 10:
            # 精心挑选的Nature风格配色：深蓝、青绿、橙红、紫色等
            nature_colors = [
                '#4472C4',  # 深蓝
                '#70AD47',  # 绿色
                '#ED7D31',  # 橙色
                '#A5A5A5',  # 灰色
                '#FFC000',  # 金色
                '#5B9BD5',  # 天蓝
                '#C5E0B4',  # 浅绿
                '#F4B183',  # 浅橙
                '#B4C7E7',  # 浅蓝
                '#FFE699',  # 浅黄
            ]
            colors = [nature_colors[i % len(nature_colors)] for i in range(n_colors)]
            colors = np.array([plt.matplotlib.colors.to_rgba(c) for c in colors])
        else:
            # 多类别时使用精致的渐变色
            nature_palette = [
                '#2E5090', '#3D6CB9', '#5689C7', '#6FA5D5', '#88B9DE',
                '#5A9A7C', '#70AD8C', '#88C09D', '#A0D3AE', '#B8E6BF',
                '#D97845', '#E89A5F', '#F4B183', '#FCC9A7', '#FFDDC1',
                '#7B68A6', '#9483BD', '#AC9ECF', '#C5B9E0', '#DDD4F0',
                '#999999', '#ADADAD', '#C0C0C0', '#D4D4D4', '#E8E8E8'
            ]
            colors = [nature_palette[i % len(nature_palette)] for i in range(n_colors)]
            colors = np.array([plt.matplotlib.colors.to_rgba(c) for c in colors])
        
        positions = np.arange(1, len(cats_sorted) + 1)
        if kind.lower().startswith("box"):
            bp = ax.boxplot(
                values,
                positions=positions,
                widths=0.40,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(linewidth=2.0, color=NATURE_COLORS['reference']),
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
            )
            for i, b in enumerate(bp.get("boxes", [])):
                b.set_facecolor(colors[i])
                b.set_edgecolor(NATURE_COLORS['reference'])
                b.set_alpha(0.7)
            for w in bp.get("whiskers", []) + bp.get("caps", []):
                w.set_color(NATURE_COLORS['reference'])
                w.set_alpha(0.8)
        else:
            parts = ax.violinplot(values, positions=positions, widths=0.49, showmeans=False, showmedians=True, showextrema=False)
            for i, body in enumerate(parts.get("bodies", [])):
                body.set_facecolor(colors[i])
                body.set_edgecolor(NATURE_COLORS['reference'])
                body.set_alpha(0.7)
                body.set_linewidth(1.5)
            if "cmedians" in parts:
                parts["cmedians"].set_color(NATURE_COLORS['reference'])
                parts["cmedians"].set_linewidth(2.0)
                parts["cmedians"].set_alpha(0.9)
        
        if idx == 0:
            # IL：不截断名称，完整显示
            xt = [f"{c}\n(n={int(counts[c])})" for c in cats_sorted]
            ax.set_xticks(positions)
            ax.set_xticklabels(xt, rotation=75, ha="right", fontsize=10)
        else:
            # 组分2/3：名称也完整显示
            xt = [f"{c}\n(n={int(counts[c])})" for c in cats_sorted]
            ax.set_xticks(positions)
            ax.set_xticklabels(xt, rotation=55, ha="right", fontsize=12)
        # Use a publication-ready y-axis label
        ax.set_ylabel("Absolute error", fontsize=14)
        ax.tick_params(axis='y', labelsize=12)
        
        # 根据位置调整标题大小（按你的要求命名）
        if idx == 0:  # IL
            ax.set_title("Absolute error by IL", fontweight='bold', pad=16, fontsize=15.5)
        elif idx == 1:  # 组分2
            ax.set_title("Absolute error by component 2", fontweight='bold', pad=14, fontsize=14.5)
        else:  # 组分3
            ax.set_title("Absolute error by component 3", fontweight='bold', pad=14, fontsize=14.5)
        
        # 为所有图设置相同的y轴范围
        ax.set_ylim(y_lim)
        ax.grid(True, alpha=0.2, linestyle=':', axis='y')
    
    fig.suptitle("Error Distribution by Category", y=0.995, fontweight='bold', fontsize=15)
    _savefig(fig, out_path, dpi=350, pad=0.18)


# -----------------------------
# Ternary plotting (keep your existing style)
# -----------------------------
def plot_group_ternary_from_csv(
    group: pd.DataFrame,
    save_path: str,
    draw_tielines_max: int = 18,
    fig_size=(8.8, 9.6),
):
    """
    你之前调好的 ternary 风格：保留
    """
    g = group.copy().sort_values("t")

    row0 = g.iloc[0].to_dict()
    try:
        label1, label2, label3 = viz._get_component_labels(row0)
    except Exception:
        label1, label2, label3 = "Comp1", "Comp2", "Comp3"

    E_true = g[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float32)
    R_true = g[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
    E_pred = g[["pred_Ex1", "pred_Ex2", "pred_Ex3"]].to_numpy(dtype=np.float32)
    R_pred = g[["pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float32)

    Exy_true = np.array([viz.ternary_to_xy(*p) for p in E_true])
    Rxy_true = np.array([viz.ternary_to_xy(*p) for p in R_true])
    Exy_pred = np.array([viz.ternary_to_xy(*p) for p in E_pred])
    Rxy_pred = np.array([viz.ternary_to_xy(*p) for p in R_pred])

    yE_true = g[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float64)
    yE_pred = g[["pred_Ex1", "pred_Ex2", "pred_Ex3"]].to_numpy(dtype=np.float64)
    yR_true = g[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float64)
    yR_pred = g[["pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float64)

    maeE, rmseE, r2E = calc_mae_rmse_r2(yE_true, yE_pred)
    maeR, rmseR, r2R = calc_mae_rmse_r2(yR_true, yR_pred)

    system_id = row0.get("system_id", "NA")
    T = float(row0.get("T", np.nan))

    cE = "#D4A017"
    cR = "#D1492E"
    cTie = "#9AA0A6"

    fig, ax = plt.subplots(figsize=fig_size)
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.2, top=0.90)
    fig.suptitle(f"TEST | System {system_id} | T={T:.2f} K | n={len(g)}", y=0.985, fontweight="bold")

    viz.draw_ternary_axes(ax, labels=(label1, label2, label3))

    for ln in list(ax.lines):
        col = str(ln.get_color()).lower()
        ls = ln.get_linestyle()
        if col in ("k", "black", "#000000") and ls in ("-", "solid"):
            ln.set_linewidth(2.4)
        elif col in ("k", "black", "#000000") and ls in ("--", "dashed", "-.", "dashdot", ":", "dotted"):
            ln.set_linewidth(2.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 0.90)

    ax.plot(Exy_true[:, 0], Exy_true[:, 1], color=cE, linewidth=4.0, label="True E (curve)")
    ax.plot(Rxy_true[:, 0], Rxy_true[:, 1], color=cR, linewidth=4.0, label="True R (curve)")
    ax.plot(Exy_pred[:, 0], Exy_pred[:, 1], color=cE, linewidth=3.6, linestyle="--", label="Pred E (curve)")
    ax.plot(Rxy_pred[:, 0], Rxy_pred[:, 1], color=cR, linewidth=3.6, linestyle="--", label="Pred R (curve)")

    ax.scatter(Exy_true[:, 0], Exy_true[:, 1], s=150, marker="o", color=cE, alpha=0.92, edgecolors="none", label="True E")
    ax.scatter(Rxy_true[:, 0], Rxy_true[:, 1], s=150, marker="o", color=cR, alpha=0.92, edgecolors="none", label="True R")
    ax.scatter(Exy_pred[:, 0], Exy_pred[:, 1], s=185, marker="^", facecolors="none", edgecolors=cE, linewidths=2.6, alpha=0.98, label="Pred E")
    ax.scatter(Rxy_pred[:, 0], Rxy_pred[:, 1], s=185, marker="^", facecolors="none", edgecolors=cR, linewidths=2.6, alpha=0.98, label="Pred R")

    step = max(1, len(g) // max(1, int(draw_tielines_max)))
    for i in range(0, len(g), step):
        ax.plot([Exy_true[i, 0], Rxy_true[i, 0]], [Exy_true[i, 1], Rxy_true[i, 1]], linewidth=1.8, color=cTie, alpha=0.45)
        ax.plot([Exy_pred[i, 0], Rxy_pred[i, 0]], [Exy_pred[i, 1], Rxy_pred[i, 1]], linewidth=1.8, color=cTie, alpha=0.45, linestyle="--")

    metric_txt = (
        f"MAE   E: {maeE:.4f}   R: {maeR:.4f}\n"
        f"RMSE  E: {rmseE:.4f}   R: {rmseR:.4f}\n"
        f"R²    E: {r2E:.4f}   R: {r2R:.4f}"
    )
    fig.text(
        1.2, 0.64, metric_txt,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.48", facecolor="white", alpha=1.0, linewidth=1.6, edgecolor="black")
    )

    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.88), frameon=True, ncol=1)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.8)
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_facecolor("white")

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=350, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="test_df_raw_pointwise_predictions.csv 路径")
    ap.add_argument("--out_dir", type=str, default="", help="输出目录（默认自动创建）")
    ap.add_argument("--system_id", type=int, default=None, help="只画某个 system（可选）")
    ap.add_argument("--max_groups", type=int, default=0, help="最多画多少个(system,T)组；0=全画")
    ap.add_argument("--font_scale", type=float, default=1.7, help="字体整体放大倍数（建议 1.6~2.2）")
    ap.add_argument("--skip_ternary", action="store_true", help="只画 parity + 额外统计图，不画相图（ternary）")
    ap.add_argument("--skip_extra", action="store_true", help="只画 parity/ternary，跳过额外统计图")
    ap.add_argument("--tielines_max", type=int, default=18, help="每张相图最多画多少条 tie-line（抽样）")

    ap.add_argument("--scatter_max", type=int, default=8000, help="散点图最多点数（下采样），0=不下采样")
    ap.add_argument("--top_categories", type=int, default=12, help="类别箱线/小提琴图展示 top-N 类别")
    ap.add_argument("--cat_kind", type=str, default="violin", choices=["violin", "box"], help="类别误差图类型")
    ap.add_argument("--seed", type=int, default=0, help="下采样随机种子")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = normalize_columns(df)

    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join("eval_output", f"test_viz_from_csv_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    apply_style(args.font_scale)

    # parity（合并版本）
    parity_plot_combined(df, os.path.join(out_dir, "parity_combined.png"))

    # extra requested plots
    if not args.skip_extra:
        extra_dir = os.path.join(out_dir, "extra_plots")
        os.makedirs(extra_dir, exist_ok=True)

        # 所有统计图使用合并版本
        plot_error_hist_kde_combined(df, os.path.join(extra_dir, "err_hist_kde_combined.png"))
        plot_bland_altman_combined(df, os.path.join(extra_dir, "bland_altman_combined.png"), max_points=args.scatter_max, seed=args.seed)
        plot_cdf_abs_error(df, os.path.join(extra_dir, "cdf_abs_error.png"))
        plot_residual_vs_true_combined(df, os.path.join(extra_dir, "residual_vs_true_combined.png"), max_points=args.scatter_max, seed=args.seed)
        plot_sum_to_one_combined(df, os.path.join(extra_dir, "sum_to_one_combined.png"))

        df_cat = add_error_columns_for_group_plots(df)

        # 组合的violin/box图（三行一列）
        plot_violin_combined_categories(
            df_cat,
            out_path=os.path.join(extra_dir, f"{args.cat_kind}_abs_err_combined.png"),
            top_n=args.top_categories,
            kind=args.cat_kind,
        )

        # 也额外给你分相的（IL abbreviation）
        plot_violin_or_box_by_category(
            df_cat,
            category_col="IL abbreviation",
            value_col="abs_err_E",
            out_path=os.path.join(extra_dir, f"{args.cat_kind}_abs_err_E_by_IL_abbrev.png"),
            top_n=args.top_categories,
            kind=args.cat_kind,
        )
        plot_violin_or_box_by_category(
            df_cat,
            category_col="IL abbreviation",
            value_col="abs_err_R",
            out_path=os.path.join(extra_dir, f"{args.cat_kind}_abs_err_R_by_IL_abbrev.png"),
            top_n=args.top_categories,
            kind=args.cat_kind,
        )

    if args.skip_ternary:
        print("✓ Saved:", os.path.join(out_dir, "parity_combined.png"))
        if not args.skip_extra:
            print("✓ Saved extra plots to:", os.path.join(out_dir, "extra_plots"))
        print("DONE:", out_dir)
        return

    # ternary PNG per (system,T)
    df_plot = df
    if args.system_id is not None:
        df_plot = df_plot[df_plot["system_id"] == args.system_id].copy()
        if len(df_plot) == 0:
            raise RuntimeError(f"system_id={args.system_id} 在CSV中不存在。")

    groups = list(df_plot.groupby(["system_id", "T"], sort=True))
    if args.max_groups and args.max_groups > 0:
        groups = groups[: int(args.max_groups)]

    ternary_dir = os.path.join(out_dir, "ternary_png")
    os.makedirs(ternary_dir, exist_ok=True)

    for (sid, T), g in groups:
        safeT = f"{float(T):.6f}".rstrip("0").rstrip(".").replace(".", "p")
        out_png = os.path.join(ternary_dir, f"system_{int(sid)}_T_{safeT}.png")
        plot_group_ternary_from_csv(g, out_png, draw_tielines_max=int(args.tielines_max))

    print("✓ Saved:", os.path.join(out_dir, "parity_combined.png"))
    if not args.skip_extra:
        print("✓ Saved extra plots to:", os.path.join(out_dir, "extra_plots"))
    print("✓ Saved ternary PNGs to:", ternary_dir)
    print("DONE:", out_dir)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
plot_test_viz_from_csv.py - 从 CSV 预测结果生成可视化

功能：
- 从 test_df_raw_pointwise_predictions.csv 读取预测结果
- 生成奇偶图（Parity plots）
- 生成三角相图（Ternary diagrams）
- 支持按体系分组可视化

用法：
  python plot_test_viz_from_csv.py --csv test_df_raw_pointwise_predictions.csv --out_dir ./results
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
# Parity plots (reference style)
# -----------------------------
def parity_plot_phase(df: pd.DataFrame, phase: str, out_path: str):
    if phase.upper() == "E":
        true_cols = ["Ex1", "Ex2", "Ex3"]
        pred_cols = ["pred_Ex1", "pred_Ex2", "pred_Ex3"]
        comp_labels = ["E1", "E2", "E3"]
        title = "Parity (E phase)"
        xlabel = "True composition (E phase)"
        ylabel = "Pred composition (E phase)"
    else:
        true_cols = ["Rx1", "Rx2", "Rx3"]
        pred_cols = ["pred_Rx1", "pred_Rx2", "pred_Rx3"]
        comp_labels = ["R1", "R2", "R3"]
        title = "Parity (R phase)"
        xlabel = "True composition (R phase)"
        ylabel = "Pred composition (R phase)"

    y_true_all = df[true_cols].to_numpy(dtype=np.float64).reshape(-1)
    y_pred_all = df[pred_cols].to_numpy(dtype=np.float64).reshape(-1)
    mae, rmse, r2 = calc_mae_rmse_r2(y_true_all, y_pred_all)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    markers = ["o", "s", "^"]
    for tc, pc, lb, mk in zip(true_cols, pred_cols, comp_labels, markers):
        ax.scatter(df[tc], df[pc], s=46, alpha=0.85, marker=mk, edgecolors="none", label=lb)

    lo, hi = -0.05, 1.05
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    txt = f"MAE  {mae:.4f}\nRMSE {rmse:.4f}\nR²   {r2:.4f}"
    ax.text(
        0.06, 0.94, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.95, linewidth=1.4, edgecolor="black"),
    )

    leg = ax.legend(loc="lower right", frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.6)
    leg.get_frame().set_alpha(1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=350, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


# -----------------------------
# Ternary plotting (thicker axes/lines, bigger points, solid black frames)
# -----------------------------
def plot_group_ternary_from_csv(
    group: pd.DataFrame,
    save_path: str,
    draw_tielines_max: int = 18,
    fig_size=(8.8, 9.6),
):
    """
    你要求的改动：
    - 坐标/三角边框加粗（draw_ternary_axes 画完后统一加粗黑色线）
    - 曲线加粗
    - 点加大
    - 图例使用黑色实线框
    - 指标框底部不截断（加大 bottom 留白 + 提高 text 的 y + 保存时 pad_inches）
    """
    g = group.copy().sort_values("t")

    row0 = g.iloc[0].to_dict()
    try:
        label1, label2, label3 = viz._get_component_labels(row0)
    except Exception:
        label1, label2, label3 = "Comp1", "Comp2", "Comp3"

    # data
    E_true = g[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float32)
    R_true = g[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
    E_pred = g[["pred_Ex1", "pred_Ex2", "pred_Ex3"]].to_numpy(dtype=np.float32)
    R_pred = g[["pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float32)

    Exy_true = np.array([viz.ternary_to_xy(*p) for p in E_true])
    Rxy_true = np.array([viz.ternary_to_xy(*p) for p in R_true])
    Exy_pred = np.array([viz.ternary_to_xy(*p) for p in E_pred])
    Rxy_pred = np.array([viz.ternary_to_xy(*p) for p in R_pred])

    # metrics (E and R separately)
    yE_true = g[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float64)
    yE_pred = g[["pred_Ex1", "pred_Ex2", "pred_Ex3"]].to_numpy(dtype=np.float64)
    yR_true = g[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float64)
    yR_pred = g[["pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float64)

    maeE, rmseE, r2E = calc_mae_rmse_r2(yE_true, yE_pred)
    maeR, rmseR, r2R = calc_mae_rmse_r2(yR_true, yR_pred)

    system_id = row0.get("system_id", "NA")
    T = float(row0.get("T", np.nan))

    # palette
# Premium, print-friendly colors (deep ink blue & muted terracotta)
    # Premium, print-friendly colors (deep teal & raspberry)
    cE = "#D4A017"   # Deep Teal
    cR = "#D1492E"   # Raspberry
    cTie = "#9AA0A6"



    fig, ax = plt.subplots(figsize=fig_size)

    # 重要：给底部留够空间放指标框，避免 bbox_inches='tight' 截断
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.2, top=0.90)

    # Title outside axes
    fig.suptitle(f"TEST | System {system_id} | T={T:.2f} K | n={len(g)}", y=0.985, fontweight="bold")

    # ternary axes
    viz.draw_ternary_axes(ax, labels=(label1, label2, label3))

    # --- thicken ternary coordinate lines (draw_ternary_axes already drew them) ---
    for ln in list(ax.lines):
        col = str(ln.get_color()).lower()
        ls = ln.get_linestyle()
        # mostly the triangle/frame lines are black/gray; keep others unchanged
        if col in ("k", "black", "#000000") and ls in ("-", "solid"):
            ln.set_linewidth(2.4)
        elif col in ("k", "black", "#000000") and ls in ("--", "dashed", "-.", "dashdot", ":", "dotted"):
            ln.set_linewidth(2.0)

    # make triangle fill the main area
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 0.90)

    # curves (thicker)
    ax.plot(Exy_true[:, 0], Exy_true[:, 1], color=cE, linewidth=4.0, label="True E (curve)")
    ax.plot(Rxy_true[:, 0], Rxy_true[:, 1], color=cR, linewidth=4.0, label="True R (curve)")
    ax.plot(Exy_pred[:, 0], Exy_pred[:, 1], color=cE, linewidth=3.6, linestyle="--", label="Pred E (curve)")
    ax.plot(Rxy_pred[:, 0], Rxy_pred[:, 1], color=cR, linewidth=3.6, linestyle="--", label="Pred R (curve)")

    # points (bigger)
    ax.scatter(Exy_true[:, 0], Exy_true[:, 1], s=150, marker="o", color=cE, alpha=0.92,
               edgecolors="none", label="True E")
    ax.scatter(Rxy_true[:, 0], Rxy_true[:, 1], s=150, marker="o", color=cR, alpha=0.92,
               edgecolors="none", label="True R")
    ax.scatter(Exy_pred[:, 0], Exy_pred[:, 1], s=185, marker="^", facecolors="none", edgecolors=cE,
               linewidths=2.6, alpha=0.98, label="Pred E")
    ax.scatter(Rxy_pred[:, 0], Rxy_pred[:, 1], s=185, marker="^", facecolors="none", edgecolors=cR,
               linewidths=2.6, alpha=0.98, label="Pred R")

    # tie-lines (slightly thicker, lighter + subsample)
    step = max(1, len(g) // max(1, int(draw_tielines_max)))
    for i in range(0, len(g), step):
        ax.plot([Exy_true[i, 0], Rxy_true[i, 0]], [Exy_true[i, 1], Rxy_true[i, 1]],
                linewidth=1.8, color=cTie, alpha=0.45)
        ax.plot([Exy_pred[i, 0], Rxy_pred[i, 0]], [Exy_pred[i, 1], Rxy_pred[i, 1]],
                linewidth=1.8, color=cTie, alpha=0.45, linestyle="--")

    # ---- metrics box (move up a bit to avoid cropping) ----
    metric_txt = (
        f"MAE   E: {maeE:.4f}   R: {maeR:.4f}\n"
        f"RMSE  E: {rmseE:.4f}   R: {rmseR:.4f}\n"
        f"R²    E: {r2E:.4f}   R: {r2R:.4f}"
    )
    fig.text(
        1.2, 0.64, metric_txt,
        ha="right", va="top",
        bbox=dict(
            boxstyle="round,pad=0.48",
            facecolor="white",
            alpha=1.0,
            linewidth=1.6,
            edgecolor="black",
        )
    )

    # ---- legend (solid black frame) ----
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.88),
        frameon=True,
        ncol=1
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.8)
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_facecolor("white")

    fig.savefig(save_path, dpi=350, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="test_df_raw_pointwise_predictions.csv 路径")
    ap.add_argument("--out_dir", type=str, default="", help="输出目录（默认自动创建）")
    ap.add_argument("--system_id", type=int, default=None, help="只画某个 system（可选）")
    ap.add_argument("--max_groups", type=int, default=0, help="最多画多少个(system,T)组；0=全画")
    ap.add_argument("--font_scale", type=float, default=1.8, help="字体整体放大倍数（建议 1.6~2.2）")
    ap.add_argument("--skip_ternary", action="store_true", help="只画 parity，不画相图（ternary）")
    ap.add_argument("--tielines_max", type=int, default=18, help="每张相图最多画多少条 tie-line（抽样）")
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

    # parity
    parity_plot_phase(df, "E", os.path.join(out_dir, "parity_E.png"))
    parity_plot_phase(df, "R", os.path.join(out_dir, "parity_R.png"))

    if args.skip_ternary:
        print("✓ Saved:", os.path.join(out_dir, "parity_E.png"))
        print("✓ Saved:", os.path.join(out_dir, "parity_R.png"))
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

    print("✓ Saved:", os.path.join(out_dir, "parity_E.png"))
    print("✓ Saved:", os.path.join(out_dir, "parity_R.png"))
    print("✓ Saved ternary PNGs to:", ternary_dir)
    print("DONE:", out_dir)


if __name__ == "__main__":
    main()

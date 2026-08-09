# -*- coding: utf-8 -*-
import os
import math
from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tqdm.auto import tqdm
import torch

from .config import FP_BITS, FP_RADIUS, N_SWEEP, DRAW_TIELINES_MAX, DEVICE
from .utils import canonicalize_smiles, morgan_fp, renorm3
from .metrics import calc_mae_rmse_r2


# -----------------------------
# Nature-like plotting style
# -----------------------------
def apply_nature_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


apply_nature_style()


def parity_plots(df_pred: pd.DataFrame, out_dir: str, backend: str = "auto", save_pdf: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)

    lo, hi = -0.05, 1.05

    # E parity
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    for k in [1, 2, 3]:
        ax.scatter(df_pred[f"Ex{k}"], df_pred[f"pred_Ex{k}"], s=10, label=f"E{k}")
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True E")
    ax.set_ylabel("Pred E")
    ax.set_title("Parity Plot (E phase)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "parity_E.png"), dpi=300)
    plt.close(fig)

    # R parity
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    for k in [1, 2, 3]:
        ax.scatter(df_pred[f"Rx{k}"], df_pred[f"pred_Rx{k}"], s=10, label=f"R{k}")
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True R")
    ax.set_ylabel("Pred R")
    ax.set_title("Parity Plot (R phase)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "parity_R.png"), dpi=300)
    plt.close(fig)


def ternary_to_xy(x1: float, x2: float, x3: float) -> Tuple[float, float]:
    s = x1 + x2 + x3
    if abs(s - 1.0) > 1e-6 and s > 1e-12:
        x1, x2, x3 = x1 / s, x2 / s, x3 / s
    X = x2 + 0.5 * x3
    Y = (math.sqrt(3) / 2.0) * x3
    return float(X), float(Y)


def draw_ternary_axes(ax, labels=("Comp1(IL)", "Comp2", "Comp3")) -> None:
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (0.5, math.sqrt(3) / 2.0)

    ax.plot([A[0], B[0]], [A[1], B[1]], color="black", linewidth=1.3)
    ax.plot([B[0], C[0]], [B[1], C[1]], color="black", linewidth=1.3)
    ax.plot([C[0], A[0]], [C[1], A[1]], color="black", linewidth=1.3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, math.sqrt(3) / 2 + 0.08)
    ax.axis("off")

    ax.text(A[0] - 0.02, A[1] - 0.035, labels[0], ha="right", va="top", color="black")
    ax.text(B[0] + 0.02, B[1] - 0.035, labels[1], ha="left", va="top", color="black")
    ax.text(C[0], C[1] + 0.04, labels[2], ha="center", va="bottom", color="black")


@torch.no_grad()
def predict_curve_sweep(model, T_scaler,
                        smiles1: str, smiles2: str, smiles3: str, T: float,
                        n_sweep: int = N_SWEEP, backend: str = "auto") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if backend == "auto":
        backend = "torch" if isinstance(model, torch.nn.Module) else "sklearn"
    if backend == "torch":
        model.eval()
    s1 = canonicalize_smiles(smiles1)
    s2 = canonicalize_smiles(smiles2)
    s3 = canonicalize_smiles(smiles3)
    if not (s1 and s2 and s3):
        raise ValueError("Invalid SMILES.")

    fp1 = morgan_fp(s1, radius=FP_RADIUS, n_bits=FP_BITS)
    fp2 = morgan_fp(s2, radius=FP_RADIUS, n_bits=FP_BITS)
    fp3 = morgan_fp(s3, radius=FP_RADIUS, n_bits=FP_BITS)

    t_grid = np.linspace(0.0, 1.0, n_sweep, dtype=np.float32)
    Tn = T_scaler.transform(np.array([T], dtype=np.float32))[0].astype(np.float32)

    X = []
    for t in t_grid:
        feat = np.concatenate([fp1, fp2, fp3, np.array([Tn, t], dtype=np.float32)], axis=0)
        X.append(feat)
        X = np.stack(X, axis=0).astype(np.float32)

    if backend == "torch":
        Xt = torch.from_numpy(X).to(DEVICE)
        pred = model(Xt).detach().cpu().numpy()  # (n,6)
    else:
        pred = np.asarray(model.predict(X), dtype=np.float32)  # (n,6)
        if pred.ndim != 2 or pred.shape[1] != 6:
            raise ValueError(f"model.predict(X) must return (N,6), got {pred.shape}")

    E = np.vstack([renorm3(p[:3]) for p in pred])
    R = np.vstack([renorm3(p[3:]) for p in pred])
    return t_grid, E, R


def plot_test_group_ternary(model: torch.nn.Module, T_scaler,
                            group_true: pd.DataFrame,
                            df_pointwise_pred: pd.DataFrame,
                            system_id: int, T: float,
                            save_path: str) -> None:
    g = group_true.copy().drop_duplicates(
        subset=["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3", "t"]
    ).sort_values("t")

    smiles1, smiles2, smiles3 = g["smiles1"].iloc[0], g["smiles2"].iloc[0], g["smiles3"].iloc[0]

    # 1) Pred curve
    t_grid, E_pred, R_pred = predict_curve_sweep(model, T_scaler, smiles1, smiles2, smiles3, T, backend=backend)

    # 2) True points -> xy
    E_true = g[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float32)
    R_true = g[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
    Exy_true = np.array([ternary_to_xy(*p) for p in E_true])
    Rxy_true = np.array([ternary_to_xy(*p) for p in R_true])

    # 3) Pred curve -> xy
    Exy_pred = np.array([ternary_to_xy(*p) for p in E_pred])
    Rxy_pred = np.array([ternary_to_xy(*p) for p in R_pred])

    # 4) Pred @ true t
    gp = df_pointwise_pred[
        (df_pointwise_pred["system_id"] == system_id) & (np.isclose(df_pointwise_pred["T"], T))
    ].copy().drop_duplicates(
        subset=["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3", "t"]
    ).sort_values("t")

    Exy_pt, Rxy_pt = None, None

    def _fmt(v: float) -> str:
        return "NaN" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.4f}"

    metrics_text = "Overall: N/A\nEx-only: N/A\nRx-only: N/A"

    if len(gp) > 0:
        E_pt = gp[["pred_Ex1", "pred_Ex2", "pred_Ex3"]].to_numpy(dtype=np.float32)
        R_pt = gp[["pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float32)
        Exy_pt = np.array([ternary_to_xy(*p) for p in E_pt])
        Rxy_pt = np.array([ternary_to_xy(*p) for p in R_pt])

        y_true_6 = gp[["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
        y_pred_6 = gp[["pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float32)

        mae_all, rmse_all, r2_all = calc_mae_rmse_r2(y_true_6, y_pred_6)
        mae_E, rmse_E, r2_E = calc_mae_rmse_r2(y_true_6[:, :3], y_pred_6[:, :3])
        mae_R, rmse_R, r2_R = calc_mae_rmse_r2(y_true_6[:, 3:], y_pred_6[:, 3:])

        metrics_text = (
            f"Overall  MAE {_fmt(mae_all)}  RMSE {_fmt(rmse_all)}  R² {_fmt(r2_all)}\n"
            f"Ex-only  MAE {_fmt(mae_E)}  RMSE {_fmt(rmse_E)}  R² {_fmt(r2_E)}\n"
            f"Rx-only  MAE {_fmt(mae_R)}  RMSE {_fmt(rmse_R)}  R² {_fmt(r2_R)}"
        )

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    draw_ternary_axes(ax, labels=("Comp1(IL)", "Comp2", "Comp3"))

    # Pred curves
    ax.plot(Exy_pred[:, 0], Exy_pred[:, 1], linewidth=2.0, label="Pred E (curve)")
    ax.plot(Rxy_pred[:, 0], Rxy_pred[:, 1], linewidth=2.0, label="Pred R (curve)")

    # True points
    ax.scatter(Exy_true[:, 0], Exy_true[:, 1], s=20, marker="o", label="True E")
    ax.scatter(Rxy_true[:, 0], Rxy_true[:, 1], s=20, marker="x", label="True R")

    # True tie-lines (solid)
    step_true = max(1, len(g) // DRAW_TIELINES_MAX)
    for i in range(0, len(g), step_true):
        ax.plot([Exy_true[i, 0], Rxy_true[i, 0]],
                [Exy_true[i, 1], Rxy_true[i, 1]],
                linewidth=1.0)

    # Pred points @ true t + Pred tie-lines (dashed)
    if Exy_pt is not None and Rxy_pt is not None:
        ax.scatter(Exy_pt[:, 0], Exy_pt[:, 1], s=16, marker="^", label="Pred E @ true t")
        ax.scatter(Rxy_pt[:, 0], Rxy_pt[:, 1], s=16, marker="v", label="Pred R @ true t")

        step_pred = max(1, len(gp) // DRAW_TIELINES_MAX)
        first = True
        for i in range(0, len(gp), step_pred):
            ax.plot([Exy_pt[i, 0], Rxy_pt[i, 0]],
                    [Exy_pt[i, 1], Rxy_pt[i, 1]],
                    linewidth=1.0,
                    linestyle="--",
                    label="Pred tie-lines" if first else None)
            first = False
    else:
        step_pred = max(1, len(t_grid) // DRAW_TIELINES_MAX)
        first = True
        for i in range(0, len(t_grid), step_pred):
            ax.plot([Exy_pred[i, 0], Rxy_pred[i, 0]],
                    [Exy_pred[i, 1], Rxy_pred[i, 1]],
                    linewidth=1.0,
                    linestyle="--",
                    label="Pred tie-lines" if first else None)
            first = False

    ax.set_title(f"TEST | System {system_id} | T={T:.2f} K | n={len(g)}")

    # Legend (top-left)
    ax.legend(loc="upper left", fontsize=9)

    # ---- Metrics box: still top-right, but shift DOWN to avoid covering the top vertex label ----
    # ax.text(
    #     0.985, 0.92, metrics_text,
    #     transform=ax.transAxes,
    #     ha="right", va="top",
    #     fontsize=10,
    #     bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85, pad=0.30)
    # )

    # ---- Metrics table (4 rows x 3 cols): header + 3 rows, still top-right ----
    # fallback values
    mae_all = rmse_all = r2_all = np.nan
    mae_E = rmse_E = r2_E = np.nan
    mae_R = rmse_R = r2_R = np.nan

    if len(gp) > 0:
        # Recompute the summary from the displayed pointwise predictions.
        y_true_6 = gp[["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
        y_pred_6 = gp[["pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float32)
        mae_all, rmse_all, r2_all = calc_mae_rmse_r2(y_true_6, y_pred_6)
        mae_E, rmse_E, r2_E = calc_mae_rmse_r2(y_true_6[:, :3], y_pred_6[:, :3])
        mae_R, rmse_R, r2_R = calc_mae_rmse_r2(y_true_6[:, 3:], y_pred_6[:, 3:])

# ---- Metrics box (legend-like frame): 9 lines x 1 col, top-right ----
    # ---- Metrics box (legend-like frame): aligned columns with monospace ----
    def _fmt4(x):
        try:
            x = float(x)
        except Exception:
            return "N/A"
        if np.isnan(x) or np.isinf(x):
            return "N/A"
        return f"{x:.4f}"

    def _line(group: str, metric: str, value: str) -> str:
        return f"{group:<8} {metric:<5} {value:<8}"

    metrics_lines = [
        _line("Overall", "MAE",  _fmt4(mae_all)),
        _line("Overall", "RMSE", _fmt4(rmse_all)),
        _line("Overall", "R²",   _fmt4(r2_all)),
        _line("Ex-only", "MAE",  _fmt4(mae_E)),
        _line("Ex-only", "RMSE", _fmt4(rmse_E)),
        _line("Ex-only", "R²",   _fmt4(r2_E)),
        _line("Rx-only", "MAE",  _fmt4(mae_R)),
        _line("Rx-only", "RMSE", _fmt4(rmse_R)),
        _line("Rx-only", "R²",   _fmt4(r2_R)),
    ]
    metrics_text = "\n".join(metrics_lines)

    ax.text(
        0.985, 0.92, metrics_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        linespacing=1.15,
        fontfamily="DejaVu Sans Mono",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85
        ),
        zorder=10
    )


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def visualize_all_test_groups(model, T_scaler,
                              df_raw: pd.DataFrame,
                              test_system_ids: set,
                              df_pointwise_pred: pd.DataFrame,
                              out_dir: str, backend: str = "auto", save_pdf: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    png_dir = os.path.join(out_dir, "test_ternary_png")
    os.makedirs(png_dir, exist_ok=True)

    df_raw_test = df_raw[df_raw["system_id"].isin(test_system_ids)].copy()
    groups = df_raw_test[["system_id", "T"]].drop_duplicates().sort_values(["system_id", "T"]).to_numpy()

    if not save_pdf:
        for (sid, TT) in tqdm(groups, desc="Plot all test ternary"):
            sid = int(sid)
            TT = float(TT)
            g = df_raw_test[(df_raw_test["system_id"] == sid) & (np.isclose(df_raw_test["T"], TT))].copy()
            if len(g) == 0:
                continue
            fig_path = os.path.join(png_dir, f"test_system_{sid}_T_{TT:.2f}.png")
            plot_test_group_ternary(model, T_scaler, g, df_pointwise_pred, sid, TT, fig_path, backend=backend)
        print("Saved ternary PNGs in:", png_dir)
        return

    pdf_path = os.path.join(out_dir, "test_all_systems_ternary.pdf")
    with PdfPages(pdf_path) as pdf:
        for (sid, TT) in tqdm(groups, desc="Plot all test ternary"):
            sid = int(sid)
            TT = float(TT)
            g = df_raw_test[(df_raw_test["system_id"] == sid) & (np.isclose(df_raw_test["T"], TT))].copy()
            if len(g) == 0:
                continue

            fig_path = os.path.join(png_dir, f"test_system_{sid}_T_{TT:.2f}.png")
            plot_test_group_ternary(model, T_scaler, g, df_pointwise_pred, sid, TT, fig_path, backend=backend)

            # PDF page: embed the PNG
            img = plt.imread(fig_path)
            fig = plt.figure(figsize=(7.2, 6.2))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"System {sid} | T={TT:.2f} K", fontsize=10)
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)

    print("Saved all test ternary PDF:", pdf_path)
    print("Saved per-group PNGs:", png_dir)

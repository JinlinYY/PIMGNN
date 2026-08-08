# -*- coding: utf-8 -*-
"""Create parity plots and ternary phase diagrams for PSMI predictions."""
import math
import os
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

import torch
from . import config as C
from .metrics import calc_mae_rmse_r2
from .utils import canonicalize_smiles, morgan_fp, renorm3, batch_to_device, batch_graphs, batch_mixture_graphs, temperature_scalar_value
from .data import (
    FunctionalGroupCache,
    GraphCache,
    MixGraphCache,
    condition_scalar_values,
)

FP_BITS = getattr(C, 'FP_BITS', 2048)
FP_RADIUS = getattr(C, 'FP_RADIUS', 2)
N_SWEEP = getattr(C, 'N_SWEEP', 80)
DRAW_TIELINES_MAX = getattr(C, 'DRAW_TIELINES_MAX', 14)
DEVICE = getattr(C, 'DEVICE', 'cpu')


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


def parity_plots(df_pred: pd.DataFrame, out_dir: str) -> None:
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


def _get_component_labels(row: dict) -> tuple:
    """Return component labels."""
    label1 = row.get("IL abbreviation") or row.get("IL (Component 1) full name") or row.get("smiles1") or "Comp1"
    label2 = row.get("Component 2") or row.get("smiles2") or "Comp2"
    label3 = row.get("Component 3") or row.get("smiles3") or "Comp3"
    return str(label1), str(label2), str(label3)


def draw_ternary_axes(ax, labels=("Comp1(IL)", "Comp2", "Comp3")) -> None:
    """Draw the ternary plot frame and labels."""
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
def predict_curve_sweep(model: torch.nn.Module, T_scaler,
                        smiles1: str, smiles2: str, smiles3: str, T: float,
                        n_sweep: int = N_SWEEP,
                        g_cache=None,
                        P: float = 101.325, # Added P
                        P_scaler: Optional[Any] = None, # Added P_scaler
                        mix_cache=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep t in [0,1] for one (system_id, T). Supports FP and graph modes."""
    model.eval()
    s1 = canonicalize_smiles(smiles1)
    s2 = canonicalize_smiles(smiles2)
    s3 = canonicalize_smiles(smiles3)
    if not (s1 and s2 and s3):
        # Graceful fallback for invalid systems during batch visualization
        return np.zeros(n_sweep), np.zeros((n_sweep,3)), np.zeros((n_sweep,3))

    t_grid = np.linspace(0.0, 1.0, n_sweep, dtype=np.float32)
    T_feature = temperature_scalar_value(
        np.array([T], dtype=np.float32),
        mode=getattr(C, "TEMPERATURE_ENCODING", "linear_quadratic"),
        reference_k=getattr(C, "TEMPERATURE_REFERENCE_K", 500.0),
    )
    Tn = T_scaler.transform(T_feature)[0].astype(np.float32)
    
    # Process P
    Pn = 0.0
    if P_scaler is not None:
         try:
            Pn = P_scaler.transform(np.array([P], dtype=np.float32))[0].astype(np.float32)
         except Exception:
            Pn = 0.0
            
    device = DEVICE
    use_graph = getattr(C, "USE_GRAPH", False)
    scalar_dim = int(getattr(model, "scalar_dim", 2))

    if use_graph:
        if g_cache is None:
            g_cache = GraphCache(
                add_hs=getattr(C, "GRAPH_ADD_HS", False),
                add_3d=getattr(C, "GRAPH_ADD_3D", False),
                use_gasteiger=getattr(C, "GRAPH_USE_GASTEIGER", True),
                max_atoms=getattr(C, "GRAPH_MAX_ATOMS", 256),
            )
            g_cache.build_from_smiles([s1, s2, s3])

        g1 = g_cache.get(s1)
        g2 = g_cache.get(s2)
        g3 = g_cache.get(s3)

        bg1 = batch_graphs([g1] * n_sweep)
        bg2 = batch_graphs([g2] * n_sweep)
        bg3 = batch_graphs([g3] * n_sweep)

        # Match the scalar contract stored by the selected model profile.
        scalars_np = np.stack(
            [condition_scalar_values(Tn, t, Pn, scalar_dim=scalar_dim) for t in t_grid],
            axis=0,
        )
        scalars = torch.from_numpy(scalars_np)
        
        x = {"g1": bg1, "g2": bg2, "g3": bg3, "scalars": scalars}

        if getattr(C, "USE_MIX_GRAPH", False):
            if mix_cache is None:
                mix_cache = MixGraphCache(C)
            mg = mix_cache.build(s1, s2, s3, float(Tn), float(T))
            x["mix"] = batch_mixture_graphs([mg] * n_sweep)

        x = batch_to_device(x, device)
        pred = model(x).detach().cpu().numpy()
    else:
        fp1 = morgan_fp(s1, radius=FP_RADIUS, n_bits=FP_BITS)
        fp2 = morgan_fp(s2, radius=FP_RADIUS, n_bits=FP_BITS)
        fp3 = morgan_fp(s3, radius=FP_RADIUS, n_bits=FP_BITS)

        X = []
        for t in t_grid:
            scalar_values = condition_scalar_values(Tn, t, Pn, scalar_dim=scalar_dim)
            feat = np.concatenate([fp1, fp2, fp3, scalar_values], axis=0)
            X.append(feat)
        X = torch.from_numpy(np.stack(X, axis=0)).to(device)

        pred = model(X).detach().cpu().numpy()

    E = np.vstack([renorm3(p[:3]) for p in pred])
    R = np.vstack([renorm3(p[3:]) for p in pred])
    return t_grid, E, R


def plot_test_group_ternary(model: torch.nn.Module, T_scaler,
                            group_true: pd.DataFrame,
                            df_pointwise_pred: pd.DataFrame,
                            system_id: int, T: float,
                            save_path: str,
                            g_cache=None,
                            P=101.325,
                            P_scaler=None,
                            mix_cache=None) -> None:
    g = group_true.copy().drop_duplicates(
        subset=["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3", "t"]
    ).sort_values("t")

    
    row = g.iloc[0].to_dict()
    label1, label2, label3 = _get_component_labels(row)
    smiles1, smiles2, smiles3 = row.get("smiles1"), row.get("smiles2"), row.get("smiles3")

    # 1) Pred curve
    t_grid, E_pred, R_pred = predict_curve_sweep(model, T_scaler, smiles1, smiles2, smiles3, T, 
                                                 g_cache=g_cache, P=P, P_scaler=P_scaler, mix_cache=mix_cache)

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
        return "N/A" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.4f}"

    # default metrics
    mae_all = rmse_all = r2_all = np.nan
    mae_E = rmse_E = r2_E = np.nan
    mae_R = rmse_R = r2_R = np.nan

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    draw_ternary_axes(ax, labels=(label1, label2, label3))

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
    if len(gp) > 0:
        E_pt = gp[["pred_Ex1", "pred_Ex2", "pred_Ex3"]].to_numpy(dtype=np.float32)
        R_pt = gp[["pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float32)
        Exy_pt = np.array([ternary_to_xy(*p) for p in E_pt])
        Rxy_pt = np.array([ternary_to_xy(*p) for p in R_pt])

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
    ax.legend(loc="upper left", fontsize=9)

    # metrics box
    if len(gp) > 0:
        y_true_6 = gp[["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
        y_pred_6 = gp[["pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float32)
        mae_all, rmse_all, r2_all = calc_mae_rmse_r2(y_true_6, y_pred_6)
        mae_E, rmse_E, r2_E = calc_mae_rmse_r2(y_true_6[:, :3], y_pred_6[:, :3])
        mae_R, rmse_R, r2_R = calc_mae_rmse_r2(y_true_6[:, 3:], y_pred_6[:, 3:])

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
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="gray", alpha=0.85),
        zorder=10,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def visualize_all_test_groups(model: torch.nn.Module, T_scaler,
                              df_raw: pd.DataFrame,
                              test_system_ids: set,
                              df_pointwise_pred: pd.DataFrame,
                              out_dir: str,
                              P_scaler: Optional[Any] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    png_dir = os.path.join(out_dir, "test_ternary_png")
    os.makedirs(png_dir, exist_ok=True)

    df_raw_test = df_raw[df_raw["system_id"].isin(test_system_ids)].copy()

    # Pre-build graph cache for faster sweeping (graph mode)
    g_cache = None
    mix_cache = None
    if getattr(C, 'USE_GRAPH', False):
        g_cache = GraphCache(
            add_hs=getattr(C, 'GRAPH_ADD_HS', False),
            add_3d=getattr(C, 'GRAPH_ADD_3D', False),
            use_gasteiger=getattr(C, 'GRAPH_USE_GASTEIGER', True),
            max_atoms=getattr(C, 'GRAPH_MAX_ATOMS', 256),
        )
        smiles_all = df_raw_test[['smiles1','smiles2','smiles3']].values.reshape(-1).tolist()
        g_cache.build_from_smiles(smiles_all)
        
        if getattr(C, 'USE_MIX_GRAPH', False):
            mix_cache = MixGraphCache(C)

    groups = df_raw_test[["system_id", "T"]].drop_duplicates().sort_values(["system_id", "T"]).to_numpy()

    pdf_path = os.path.join(out_dir, "test_all_systems_ternary.pdf")
    with PdfPages(pdf_path) as pdf:
        for (sid, TT) in tqdm(groups, desc="Plot all test ternary"):
            sid = int(sid)
            TT = float(TT)
            g = df_raw_test[(df_raw_test["system_id"] == sid) & (np.isclose(df_raw_test["T"], TT))].copy()
            if len(g) == 0:
                continue
            
            # Fetch Pressure P for this group (if available)
            P_val = 101.325
            if "P" in g.columns:
                # Use the first value, assuming P is constant per group
                P_val = g.iloc[0]["P"]
            
            fig_path = os.path.join(png_dir, f"test_system_{sid}_T_{TT:.2f}.png")
            plot_test_group_ternary(model, T_scaler, g, df_pointwise_pred, sid, TT, fig_path, 
                                    g_cache=g_cache, P=P_val, P_scaler=P_scaler, mix_cache=mix_cache)

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

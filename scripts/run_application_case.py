# -*- coding: utf-8 -*-
"""Run prediction, analysis, and plotting for industrial application cases."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


try:
    from _bootstrap import add_src_to_path
except ModuleNotFoundError:  # Support ``import scripts.run_application_case``.
    from scripts._bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "14_industrial_cases" / "00_application_workflow" / "results" / "reproduction"

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
from tqdm.auto import tqdm

from psmi import config as C
from psmi.utils import set_seed, canonicalize_smiles, Scaler, safe_group_apply_t
from psmi.data import FingerprintCache, GraphCache, MixGraphCache, FunctionalGroupCache, GraphLLEDataset, collate_graph_batch
from psmi.predict import predict_pointwise_df_raw
from psmi.train import build_model
from psmi.checkpoints import load_state_dict_compat


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def _norm_col(c):
    """Normalize a column name."""
    c_str = str(c).strip().replace('\n', ' ').replace('\r', ' ')
    c_str = ' '.join(c_str.split())
    return c_str


def _find_col(available_cols, candidates):
    """Find the first matching column."""
    norm_cols = {_norm_col(c).lower(): c for c in available_cols}
    for cand in candidates:
        cand_norm = _norm_col(cand).lower()
        if cand_norm in norm_cols:
            return norm_cols[cand_norm]
    return None


def _require_col(df, name, candidates):
    """Return a required column or raise an error."""
    col = _find_col(df.columns.tolist(), candidates)
    if col is None:
        raise KeyError(
            f"Cannot find column for '{name}'. Tried candidates={candidates}\n"
            f"Available columns ({len(df.columns)}):\n{list(df.columns)}"
        )
    return col


def load_and_prepare_application_excel(path: str) -> pd.DataFrame:
    """Load and prepare application excel."""
    df = pd.read_excel(path)
    df.columns = [_norm_col(c) for c in df.columns]
    
    print(f"Input columns: {df.columns.tolist()}")
    
    
    col_system = _find_col(df.columns.tolist(), [
        "LLE system NO.", "LLE system NO", "LLE system No.", "LLE system No",
        "LLE system number", "LLE system#", "LLE system #",
        "System No.", "System No", "System ID", "system_id",
    ])
    if col_system is None:
        print("Warning: no system identifier column was found; using system_id=1.")
        df["system_id"] = 1
    else:
        df = df.rename(columns={col_system: "system_id"})
    
    
    col_T = _require_col(df, "T", [
        "T/K", "T / K", "T (K)", "T", "Temp", "Temperature", "Temperature/K", "Temperature (K)"
    ])
    
    
    col_s1 = _require_col(df, "smiles1", [
        "IL (Component 1) full name SMILES",
        "IL (Component 1) SMILES",
        "Component 1 SMILES", "Comp 1 SMILES",
        "Component 1",
        "smiles1", "SMILES1", "SMILES 1"
    ])
    col_s2 = _require_col(df, "smiles2", [
        "Component 2 SMILES", "Comp 2 SMILES",
        "Component 2",
        "smiles2", "SMILES2", "SMILES 2"
    ])
    col_s3 = _require_col(df, "smiles3", [
        "Component 3 SMILES", "Comp 3 SMILES",
        "Component 3",
        "smiles3", "SMILES3", "SMILES 3"
    ])
    
    
    def _req_comp(name: str) -> Optional[str]:
        return _find_col(df.columns.tolist(), [name, name.upper(), name.lower(), name.replace("x", "X"), name.replace("X", "x")])
    
    col_Ex1 = _req_comp("Ex1"); col_Ex2 = _req_comp("Ex2"); col_Ex3 = _req_comp("Ex3")
    col_Rx1 = _req_comp("Rx1"); col_Rx2 = _req_comp("Rx2"); col_Rx3 = _req_comp("Rx3")
    
    
    rename_dict = {col_T: "T", col_s1: "smiles1", col_s2: "smiles2", col_s3: "smiles3"}
    if col_Ex1: rename_dict[col_Ex1] = "Ex1"
    if col_Ex2: rename_dict[col_Ex2] = "Ex2"
    if col_Ex3: rename_dict[col_Ex3] = "Ex3"
    if col_Rx1: rename_dict[col_Rx1] = "Rx1"
    if col_Rx2: rename_dict[col_Rx2] = "Rx2"
    if col_Rx3: rename_dict[col_Rx3] = "Rx3"
    
    df = df.rename(columns=rename_dict)
    
    
    for col in ["smiles1", "smiles2", "smiles3"]:
        if col in df.columns:
            sample = str(df[col].iloc[0]).lower()
            if not any(c in sample for c in ['c', 'c', 'n', 'o', 's', 'p', '(', ')', '=']):
                smiles_col = _find_col(df.columns.tolist(), [f"{col} SMILES", f"{col.replace('smiles', '')} SMILES"])
                if smiles_col and smiles_col != col:
                    print(f"Using {smiles_col} as the SMILES source for {col}.")
                    df[col] = df[smiles_col]
    
    
    for c in ["smiles1", "smiles2", "smiles3"]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(canonicalize_smiles)
    
    
    df = df[(df["smiles1"] != "") & (df["smiles2"] != "") & (df["smiles3"] != "")].copy()
    
    
    if "t" not in df.columns:
        print("Computing the phase-path coordinate t with PCA...")
        df = safe_group_apply_t(df)
    
    print(f"Loaded {len(df)} data points.")
    
    return df


def load_application_case_plot_excel(path: str) -> pd.DataFrame:
    """Load application case plot excel."""
    df = pd.read_excel(path)
    df.columns = [_norm_col(c) for c in df.columns]

    col_system = _find_col(df.columns.tolist(), [
        "LLE system NO.", "LLE system NO", "LLE system No.", "LLE system No",
        "LLE system number", "LLE system#", "LLE system #",
        "System No.", "System No", "System ID", "system_id",
    ])
    if col_system is None:
        df["system_id"] = 1
    else:
        df = df.rename(columns={col_system: "system_id"})

    col_T = _require_col(df, "T", [
        "T/K", "T / K", "T (K)", "T", "Temp", "Temperature", "Temperature/K", "Temperature (K)"
    ])
    col_model = _require_col(df, "Model", ["Model", "model"])

    col_c1 = _require_col(df, "Component 1", ["Component 1", "Comp 1", "Component1"])
    col_c2 = _require_col(df, "Component 2", ["Component 2", "Comp 2", "Component2"])
    col_c3 = _require_col(df, "Component 3", ["Component 3", "Comp 3", "Component3"])

    def _req_comp(name: str) -> Optional[str]:
        return _find_col(df.columns.tolist(), [name, name.upper(), name.lower(), name.replace("x", "X"), name.replace("X", "x")])

    col_Ex1 = _req_comp("Ex1"); col_Ex2 = _req_comp("Ex2"); col_Ex3 = _req_comp("Ex3")
    col_Rx1 = _req_comp("Rx1"); col_Rx2 = _req_comp("Rx2"); col_Rx3 = _req_comp("Rx3")

    rename_dict = {
        col_T: "T",
        col_model: "Model",
        col_c1: "Component 1",
        col_c2: "Component 2",
        col_c3: "Component 3",
    }
    if col_Ex1: rename_dict[col_Ex1] = "Ex1"
    if col_Ex2: rename_dict[col_Ex2] = "Ex2"
    if col_Ex3: rename_dict[col_Ex3] = "Ex3"
    if col_Rx1: rename_dict[col_Rx1] = "Rx1"
    if col_Rx2: rename_dict[col_Rx2] = "Rx2"
    if col_Rx3: rename_dict[col_Rx3] = "Rx3"

    df = df.rename(columns=rename_dict)

    need_cols = ["system_id", "T", "Model", "Component 1", "Component 2", "Component 3",
                 "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
    for c in need_cols:
        if c not in df.columns:
            raise KeyError(f" missing columns :{c}")

    return df


def load_model_and_scaler(ckpt_path: str, device: str) -> tuple:
    """Args:

    Returns:
        (model, T_scaler, config_dict)
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f" load model : {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    
    model_state = None
    T_mean = 302.9259948730469  
    T_std = 10.96979808807373   
    config_dict = {}
    
    if isinstance(ckpt, dict):
        
        if "T_mean" in ckpt:
            T_mean = ckpt["T_mean"]
        if "T_std" in ckpt:
            T_std = ckpt["T_std"]
        
        
        if "config" in ckpt:
            config_dict = ckpt["config"]
        
        
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
            
            if isinstance(state, dict) and "model" in state:
                model_state = state["model"]
            else:
                model_state = state
        
        elif "model" in ckpt:
            model_state = ckpt["model"]
        
        else:
            
            if any(k.startswith(("encoder", "backbone", "head_", "token_fuser")) for k in ckpt.keys()):
                model_state = ckpt
            else:
                
                print(f" warning : unable to Identify checkpoint format . checkpoint Key : {list(ckpt.keys())}")
                model_state = ckpt
    else:
        
        model_state = ckpt
    
    if model_state is None:
        raise ValueError(f" unable to from checkpoint in Extract model checkpoint . checkpoint format : {ckpt.keys() if isinstance(ckpt, dict) else type(ckpt)}")
    
    print(f"  T_mean: {T_mean}, T_std: {T_std}")
    
    T_scaler = Scaler(mean=float(T_mean), std=float(T_std))
    model = build_model()
    adaptations = load_state_dict_compat(model, model_state)
    for adaptation in adaptations:
        print(f"[INFO] Checkpoint compatibility: {adaptation}")
    model.eval()
    model.to(device)
    
    return model, T_scaler, config_dict



def augment_prediction_points(df: pd.DataFrame, num_points: int = 50) -> pd.DataFrame:
    """Augment prediction points."""
    augmented_dfs = []
    
    
    grouped = df.groupby(["system_id", "T"])
    
    for _, group in grouped:
        
        augmented_dfs.append(group)
        
        
        if "t" in group.columns and len(group) > 1:
            t_min = group["t"].min()
            t_max = group["t"].max()
            
            
            t_new = np.linspace(t_min, t_max, num_points)
            
            
            new_rows = pd.DataFrame({
                "t": t_new
            })
            
            
            for col in group.columns:
                if col not in ["t", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3", "system_id", "T"]:
                     
                    new_rows[col] = group.iloc[0][col]
            
            new_rows["system_id"] = group.iloc[0]["system_id"]
            new_rows["T"] = group.iloc[0]["T"]
            
            
            for col in ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]:
                new_rows[col] = np.nan
                
            augmented_dfs.append(new_rows)
            
    df_aug = pd.concat(augmented_dfs, ignore_index=True)
    
    df_aug = df_aug.drop_duplicates(subset=["system_id", "T", "t"], keep="first")
    return df_aug



def test_application_case(
    excel_path: str,
    ckpt_path: str,
    out_dir: str = str(DEFAULT_OUTPUT_DIR),
    device: Optional[str] = None
) -> str:
    """Evaluate application case."""
    if device is None:
        device = getattr(C, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f" use device : {device}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    
    print("\n========== load data ==========")
    df_raw = load_and_prepare_application_excel(excel_path)
    print(f" Original data shape : {df_raw.shape}")
    
    
    print(" True at Increase prediction point for smoother Curve ...")
    df_aug = augment_prediction_points(df_raw, num_points=50)
    print(f" Increase after data shape : {df_aug.shape}")
    
    
    print("\n========== load model ==========")
    model, T_scaler, config_dict = load_model_and_scaler(ckpt_path, device)
    print(f" model loaded successfully ")
    
    
    print("\n========== Into rows prediction ==========")
    with torch.no_grad():
        df_pred = predict_pointwise_df_raw(model, T_scaler, df_aug, device=device)
    
    
    output_csv = os.path.join(out_dir, "application_case_predictions.csv")
    df_pred.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f" predictions saved to : {output_csv}")
    
    
    print("\n========== prediction statistics ==========")
    if "pred_Ex1" in df_pred.columns:
        print("\nExtract phase-prediction statistics :")
        for i in range(1, 4):
            print(f"  Ex{i}: min={df_pred[f'pred_Ex{i}'].min():.6f}, max={df_pred[f'pred_Ex{i}'].max():.6f}, mean={df_pred[f'pred_Ex{i}'].mean():.6f}")
        print("\nRaffinate phase-prediction statistics :")
        for i in range(1, 4):
            print(f"  Rx{i}: min={df_pred[f'pred_Rx{i}'].min():.6f}, max={df_pred[f'pred_Rx{i}'].max():.6f}, mean={df_pred[f'pred_Rx{i}'].mean():.6f}")
    
    return output_csv



def ternary_to_xy(x1: float, x2: float, x3: float) -> Tuple[float, float]:
    """Convert ternary coordinates to Cartesian coordinates."""
    p1 = np.array([0.5, np.sqrt(3) / 2.0])  
    p2 = np.array([0.0, 0.0])  
    p3 = np.array([1.0, 0.0])  
    
    point = x1 * p1 + x2 * p2 + x3 * p3
    return (float(point[0]), float(point[1]))


def draw_ternary_axes(ax, labels=("Comp1(IL)", "Comp2", "Comp3")) -> None:
    """Draw the ternary plot frame and labels."""
    p1 = np.array([0.5, np.sqrt(3) / 2.0])  
    p2 = np.array([0.0, 0.0])  
    p3 = np.array([1.0, 0.0])  

    
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k-', lw=1.0)
    ax.plot([p3[0], p1[0]], [p3[1], p1[1]], 'k-', lw=1.0)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=1.0)

    
    
    offset = 0.05
    
    ax.text(p1[0], p1[1] + 0.02, labels[0], ha="center", va="bottom", fontsize=10)
    ax.text(p2[0] - offset, p2[1] - 0.02, labels[1], ha="right", va="center", fontsize=10)
    ax.text(p3[0] + offset, p3[1] - 0.02, labels[2], ha="left", va="center", fontsize=10)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_ternary_detailed(df_pred: pd.DataFrame, out_dir: str) -> None:
    """Plot ternary detailed."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    
    c_true = 'black'
    c_pred = '#D55E00'  # Vermillion
    m_ex = 'o'
    m_rx = '^'
    
    grouped = df_pred.groupby(["system_id", "T"])
    plot_data_list = []
    
    for (system_id, T), group in grouped:
        g = group.copy().sort_values("t", na_position="last")
        
        fig, ax = plt.subplots(figsize=(6, 5)) 
        
        first_row = g.iloc[0].to_dict()
        comp1_name = str(first_row.get("Component 1", "Comp1")).strip() if "Component 1" in g.columns else "Comp1"
        comp2_name = str(first_row.get("Component 2", "Comp2")).strip() if "Component 2" in g.columns else "Comp2"
        comp3_name = str(first_row.get("Component 3", "Comp3")).strip() if "Component 3" in g.columns else "Comp3"
        
        draw_ternary_axes(ax, labels=(comp1_name, comp2_name, comp3_name))
        
        
        if all(col in g.columns for col in ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]):
            g_true = g.dropna(subset=["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"])
            if not g_true.empty:
                E_true = g_true[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float32)
                R_true = g_true[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
                
                Exy_true = np.array([ternary_to_xy(*p) for p in E_true])
                Rxy_true = np.array([ternary_to_xy(*p) for p in R_true])
                
                # Plot Markers
                ax.scatter(Exy_true[:, 0], Exy_true[:, 1], c=c_true, s=36, marker=m_ex, 
                          edgecolors="none", label="Exp Extract", zorder=5, alpha=0.9)
                ax.scatter(Rxy_true[:, 0], Rxy_true[:, 1], c=c_true, s=36, marker=m_rx, 
                          edgecolors="none", label="Exp Raffinate", zorder=5, alpha=0.9)
                
                # Plot Tie-lines
                for i in range(len(Exy_true)):
                    ax.plot([Exy_true[i, 0], Rxy_true[i, 0]], 
                           [Exy_true[i, 1], Rxy_true[i, 1]], 
                           color=c_true, alpha=0.35, linewidth=0.8, zorder=2)
        
        
        if all(col in g.columns for col in ["pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3"]):
            
            if "Ex1" in g.columns:
                g_orig = g.dropna(subset=["Ex1"])
            else:
                g_orig = g 
            
            if not g_orig.empty:
                
                plot_data_list.append(g_orig.copy())

                E_pred = g_orig[["pred_Ex1", "pred_Ex2", "pred_Ex3"]].to_numpy(dtype=np.float32)
                R_pred = g_orig[["pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float32)
                
                Exy_pred = np.array([ternary_to_xy(*p) for p in E_pred])
                Rxy_pred = np.array([ternary_to_xy(*p) for p in R_pred])
                
                # Plot Markers
                ax.scatter(Exy_pred[:, 0], Exy_pred[:, 1], c=c_pred, s=36, marker=m_ex, 
                        edgecolors="none", label="Pred Extract", zorder=4, alpha=0.8)
                ax.scatter(Rxy_pred[:, 0], Rxy_pred[:, 1], c=c_pred, s=36, marker=m_rx, 
                        edgecolors="none", label="Pred Raffinate", zorder=4, alpha=0.8)
                
                # Plot Tie-lines
                for i in range(len(Exy_pred)):
                    ax.plot([Exy_pred[i, 0], Rxy_pred[i, 0]], 
                           [Exy_pred[i, 1], Rxy_pred[i, 1]], 
                           color=c_pred, linestyle="--", alpha=0.35, linewidth=0.8, zorder=1)
        
        
        legend_handles = []
        # Exp
        legend_handles.append(mlines.Line2D([], [], color=c_true, marker=m_ex, linestyle='None', markersize=6, label='Exp Ex'))
        legend_handles.append(mlines.Line2D([], [], color=c_true, marker=m_rx, linestyle='None', markersize=6, label='Exp Rx'))
        # Pred
        legend_handles.append(mlines.Line2D([], [], color=c_pred, marker=m_ex, linestyle='None', markersize=6, label='Pred Ex'))
        legend_handles.append(mlines.Line2D([], [], color=c_pred, marker=m_rx, linestyle='None', markersize=6, label='Pred Rx'))

        ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), 
                  frameon=False, fontsize=9)
        
        ax.set_title(f"System {int(system_id)}, T = {float(T):.2f} K", fontsize=12, pad=10)
        
        save_path = os.path.join(out_dir, f"ternary_system{int(system_id)}_T{float(T):.1f}K.png")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved : {save_path}")

    
    if plot_data_list:
        combined_wide = pd.concat(plot_data_list, ignore_index=True)
        
        
        long_data = []
        
        
        cols = combined_wide.columns
        c1_name_col = _find_col(cols, ["Component 1", "Comp 1", "Component1", "IL"])
        c2_name_col = _find_col(cols, ["Component 2", "Comp 2", "Component2"])
        c3_name_col = _find_col(cols, ["Component 3", "Comp 3", "Component3"])
        
        
        def _add_row(src_row, model_name, ex_cols, rx_cols):
            new_row = {
                "LLE system NO.": src_row.get("system_id", 1),
                "Model": model_name,
                
                "Component 1": src_row[c1_name_col] if c1_name_col else "Comp1",
                "Component 1 SMILES": src_row.get("smiles1", ""),
                
                "Component 2": src_row[c2_name_col] if c2_name_col else "Comp2",
                "Component 2 SMILES": src_row.get("smiles2", ""),
                
                "Component 3": src_row[c3_name_col] if c3_name_col else "Comp3",
                "Component 3 SMILES": src_row.get("smiles3", ""),
                
                "T/K": src_row.get("T", 298.15),
                "P/kPa": src_row.get("P", 101.325), 
            }
            
            
            new_row["Ex1"] = src_row.get(ex_cols[0], np.nan)
            new_row["Ex2"] = src_row.get(ex_cols[1], np.nan)
            new_row["Ex3"] = src_row.get(ex_cols[2], np.nan)
            new_row["Rx1"] = src_row.get(rx_cols[0], np.nan)
            new_row["Rx2"] = src_row.get(rx_cols[1], np.nan)
            new_row["Rx3"] = src_row.get(rx_cols[2], np.nan)
            
            return new_row

        for _, row in combined_wide.iterrows():
            # 1. Experiment Row
            
            if all(pd.notna(row.get(c)) for c in ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]):
                 long_data.append(_add_row(
                     row, "Experiment", 
                     ["Ex1", "Ex2", "Ex3"], 
                     ["Rx1", "Rx2", "Rx3"]
                 ))
            
            # 2. Prediction Row
            if all(pd.notna(row.get(c)) for c in ["pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3"]):
                 long_data.append(_add_row(
                     row, "PSMI", 
                     ["pred_Ex1", "pred_Ex2", "pred_Ex3"], 
                     ["pred_Rx1", "pred_Rx2", "pred_Rx3"]
                 ))
        
        df_long = pd.DataFrame(long_data)
        
        
        
        if not df_long.empty:
            df_long = df_long.sort_values(by=['LLE system NO.', 'T/K', 'Model'])
        
        plot_csv_path = os.path.join(out_dir, "application_case_plot_data_formatted.csv")
        
        target_cols = ['LLE system NO.', 'Model', 
                       'Component 1', 'Component 1 SMILES', 
                       'Component 2', 'Component 2 SMILES', 
                       'Component 3', 'Component 3 SMILES', 
                       'T/K', 'P/kPa', 
                       'Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3']
        
        
        final_cols = [c for c in target_cols if c in df_long.columns]
        df_long = df_long[final_cols]
        
        df_long.to_csv(plot_csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] format Hua Painting graph data saved : {plot_csv_path}")


def plot_ternary_models_vs_experiment(df_plot: pd.DataFrame, out_dir: str) -> None:
    """Plot ternary models vs experiment."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model_colors = {
        "Experiment": "black",
        "COSMO-RS": "#D55E00",
        "NRTL": "#0072B2",
        "UNIFAC": "#009E73",
    }
    
    extra_colors = ['#CC79A7', '#F0E442', '#56B4E9']
    
    phase_markers = {"Extract": "o", "Raffinate": "^"}

    grouped = df_plot.groupby(["system_id", "T"], dropna=False)
    for (system_id, T), group in grouped:
        g = group.copy()
        fig, ax = plt.subplots(figsize=(6, 5))

        first_row = g.iloc[0].to_dict()
        comp1_name = str(first_row.get("Component 1", "Comp1")).strip()
        comp2_name = str(first_row.get("Component 2", "Comp2")).strip()
        comp3_name = str(first_row.get("Component 3", "Comp3")).strip()

        draw_ternary_axes(ax, labels=(comp1_name, comp2_name, comp3_name))

        legend_handles = []
        unique_models = g["Model"].dropna().unique()
        
        for i, model in enumerate(unique_models):
            m_str = str(model)
            if m_str in model_colors:
                color = model_colors[m_str]
            else:
                color = extra_colors[i % len(extra_colors)]
                
            m_df = g[g["Model"] == model]

            E_true = m_df[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float32)
            R_true = m_df[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)

            Exy = np.array([ternary_to_xy(*p) for p in E_true])
            Rxy = np.array([ternary_to_xy(*p) for p in R_true])

            # Markers
            ax.scatter(Exy[:, 0], Exy[:, 1], c=color, s=36, marker=phase_markers["Extract"],
                       edgecolors="none", alpha=0.9, zorder=5)
            ax.scatter(Rxy[:, 0], Rxy[:, 1], c=color, s=36, marker=phase_markers["Raffinate"],
                       edgecolors="none", alpha=0.9, zorder=5)
            
            # Tie-lines
            for k in range(len(Exy)):
                ax.plot([Exy[k, 0], Rxy[k, 0]], 
                       [Exy[k, 1], Rxy[k, 1]], 
                       color=color, alpha=0.35, linewidth=0.8, zorder=2)

            legend_handles.append(mlines.Line2D([], [], color=color, marker=phase_markers["Extract"],
                                                linestyle="None", markersize=6,
                                                label=f"{model} Ex"))
            
            
            

        ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.05, 1.0),
                  frameon=False, fontsize=9)

        ax.set_title(
            f"System {int(system_id)}, T = {float(T):.2f} K",
            fontsize=12, pad=10
        )

        save_path = os.path.join(out_dir, f"ternary_models_system{int(system_id)}_T{float(T):.1f}K.png")
        plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1) 
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved : {save_path}")


def plot_parity_diagrams(df_pred: pd.DataFrame, out_dir: str) -> None:
    """Plot parity diagrams."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    
    components = [
        ("Ex1", "pred_Ex1", "Extract Component 1"),
        ("Ex2", "pred_Ex2", "Extract Component 2"),
        ("Ex3", "pred_Ex3", "Extract Component 3"),
        ("Rx1", "pred_Rx1", "Raffinate Component 1"),
        ("Rx2", "pred_Rx2", "Raffinate Component 2"),
        ("Rx3", "pred_Rx3", "Raffinate Component 3"),
    ]
    
    for idx, (true_col, pred_col, title) in enumerate(components):
        ax = axes[idx // 3, idx % 3]
        
        if true_col in df_pred.columns and pred_col in df_pred.columns:
            
            valid_df = df_pred.dropna(subset=[true_col, pred_col])
            
            true_vals = valid_df[true_col].values
            pred_vals = valid_df[pred_col].values
            
            if len(true_vals) > 0:
                ax.scatter(true_vals, pred_vals, c="blue", s=80, alpha=0.6, edgecolors="black", linewidth=0.5)
                
                lim_min = min(true_vals.min(), pred_vals.min()) - 0.05
                lim_max = max(true_vals.max(), pred_vals.max()) + 0.05
                ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=2, label="Perfect Pred", alpha=0.7)
                
                mae = np.mean(np.abs(true_vals - pred_vals))
                rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
                denom = np.sum((true_vals - true_vals.mean()) ** 2)
                r2 = 1 - np.sum((true_vals - pred_vals) ** 2) / denom if denom > 1e-10 else 0
                
                ax.set_xlabel("True Value", fontsize=10)
                ax.set_ylabel("Predicted Value", fontsize=10)
                ax.set_title(f"{title}\nMAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}", fontsize=11, fontweight="bold")
                ax.set_xlim([lim_min, lim_max])
                ax.set_ylim([lim_min, lim_max])
            else:
                 ax.set_title(f"{title} (No Data)", fontsize=11)

            ax.grid(True, alpha=0.3)
            ax.legend()
    
    fig.suptitle("Parity Plots: True vs Predicted", fontsize=14, fontweight="bold", y=1.00)
    save_path = os.path.join(out_dir, "parity_plots.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved : {save_path}")


def print_statistics(df_pred: pd.DataFrame, out_dir: str) -> None:
    """Print statistics."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("=" * 70)
    report.append(" application case predictions statistics Analytics ")
    report.append("=" * 70)
    report.append("")
    
    report.append(f" total Base Points : {len(df_pred)}")
    report.append(f" System count : {df_pred['system_id'].nunique()}")
    report.append(f" temperature count : {df_pred['T'].nunique()}")
    report.append("")
    
    
    report.append("=" * 70)
    report.append("Extract phase-prediction statistics ")
    report.append("=" * 70)
    
    for comp_idx, (true_col, pred_col) in enumerate([("Ex1", "pred_Ex1"), ("Ex2", "pred_Ex2"), ("Ex3", "pred_Ex3")], 1):
        if true_col in df_pred.columns and pred_col in df_pred.columns:
            
            valid_df = df_pred.dropna(subset=[true_col, pred_col])
            
            true_vals = valid_df[true_col].values
            pred_vals = valid_df[pred_col].values
            
            if len(true_vals) > 0:
                mae = np.mean(np.abs(true_vals - pred_vals))
                rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
                denom = np.sum((true_vals - true_vals.mean()) ** 2)
                r2 = 1 - np.sum((true_vals - pred_vals) ** 2) / denom if denom > 1e-10 else 0
                
                report.append(f"\nComponent {comp_idx} (Ex{comp_idx}):")
                report.append(f" target range : [{true_vals.min():.6f}, {true_vals.max():.6f}]")
                report.append(f" prediction range : [{pred_vals.min():.6f}, {pred_vals.max():.6f}]")
                report.append(f" mean absolute error (MAE): {mae:.6f}")
                report.append(f" root mean squared error (RMSE): {rmse:.6f}")
                report.append(f" R² score : {r2:.6f}")
            else:
                report.append(f"\nComponent {comp_idx} (Ex{comp_idx}): No valid data for comparison")
    
    
    report.append("\n" + "=" * 70)
    report.append("Raffinate phase-prediction statistics ")
    report.append("=" * 70)
    
    for comp_idx, (true_col, pred_col) in enumerate([("Rx1", "pred_Rx1"), ("Rx2", "pred_Rx2"), ("Rx3", "pred_Rx3")], 1):
        if true_col in df_pred.columns and pred_col in df_pred.columns:
            
            valid_df = df_pred.dropna(subset=[true_col, pred_col])
            
            true_vals = valid_df[true_col].values
            pred_vals = valid_df[pred_col].values
            
            if len(true_vals) > 0:
                mae = np.mean(np.abs(true_vals - pred_vals))
                rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
                denom = np.sum((true_vals - true_vals.mean()) ** 2)
                r2 = 1 - np.sum((true_vals - pred_vals) ** 2) / denom if denom > 1e-10 else 0
                
                report.append(f"\nComponent {comp_idx} (Rx{comp_idx}):")
                report.append(f" target range : [{true_vals.min():.6f}, {true_vals.max():.6f}]")
                report.append(f" prediction range : [{pred_vals.min():.6f}, {pred_vals.max():.6f}]")
                report.append(f" mean absolute error (MAE): {mae:.6f}")
                report.append(f" root mean squared error (RMSE): {rmse:.6f}")
                report.append(f" R² score : {r2:.6f}")
            else:
                report.append(f"\nComponent {comp_idx} (Rx{comp_idx}): No valid data for comparison")
    
    report.append("\n" + "=" * 70)
    report_text = "\n".join(report)
    
    print(report_text)
    
    report_path = os.path.join(out_dir, "detailed_analysis.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[OK] Detailed Analysis saved : {report_path}")


def analyze_application_case(csv_path: str, out_dir: str = str(DEFAULT_OUTPUT_DIR)) -> None:
    """Analyze application case."""
    if not os.path.isfile(csv_path):
        print(f" error : CSV file does not exist : {csv_path}")
        sys.exit(1)
    
    print(f" load data : {csv_path}")
    df_pred = pd.read_csv(csv_path)
    print(f"Loaded {len(df_pred)} prediction rows.\n")
    
    print(" generating Visualization and Analytics ...\n")
    
    print("1. generate Triangle phase diagram ...")
    plot_ternary_detailed(df_pred, out_dir)
    
    print("\n2. generate Parity graph ...")
    plot_parity_diagrams(df_pred, out_dir)
    
    print("\n3. generate statistics Analytics ...")
    print_statistics(df_pred, out_dir)
    
    print(f"\n all results saved to : {out_dir}")
    print(" complete !")



def main():
    parser = argparse.ArgumentParser(
        description="Run the complete application-case workflow or analyze existing predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
 # Complete workflow: Excel -> prediction -> analysis
 python scripts/run_application_case.py --excel datasets/raw/application_case_1.xlsx \\
 --ckpt models/paper_historical/figure2a_psmi/best_model.pt \\
 --out_dir experiments/14_industrial_cases/00_application_workflow/results/reproduction
 
 # Analyze an existing prediction table 
 python scripts/run_application_case.py --csv experiments/14_industrial_cases/00_application_workflow/results/application_case_predictions.csv \\
 --out_dir experiments/14_industrial_cases/00_application_workflow/results/reproduction --analyze_only
 """
    )
    
    
    parser.add_argument(
        "--excel", "-e",
        type=str,
        default=None,
        help="Application-case Excel file for the complete workflow."
    )
    parser.add_argument(
        "--ckpt", "-c",
        type=str,
        default=None,
        help="Model checkpoint for the complete workflow."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Existing prediction CSV for analysis-only mode."
    )
    
    
    parser.add_argument(
        "--out_dir", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory."
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Compute device: cuda or cpu."
    )
    
    
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Analyze and visualize an existing prediction table; requires --csv."
    )
    
    args = parser.parse_args()
    
    
    if args.analyze_only:
        if args.csv is None:
            print("Error: --analyze_only requires --csv.")
            sys.exit(1)
        analyze_application_case(args.csv, args.out_dir)
        return

    
    if args.excel is not None and args.ckpt is None:
        if not os.path.isfile(args.excel):
            print(f" error : Excel file does not exist : {args.excel}")
            sys.exit(1)
        df_plot = load_application_case_plot_excel(args.excel)
        plot_ternary_models_vs_experiment(df_plot, args.out_dir)
        return

    
    if args.excel is None or args.ckpt is None:
        print("Error: the complete workflow requires --excel and --ckpt.")
        parser.print_help()
        sys.exit(1)

    if not os.path.isfile(args.excel):
        print(f" error : Excel file does not exist : {args.excel}")
        sys.exit(1)

    if not os.path.isfile(args.ckpt):
        print(f" error : model checkpoint does not exist : {args.ckpt}")
        sys.exit(1)

    
    csv_path = test_application_case(args.excel, args.ckpt, args.out_dir, args.device)

    
    print("\n" + "=" * 50)
    analyze_application_case(csv_path, args.out_dir)


if __name__ == "__main__":
    main()

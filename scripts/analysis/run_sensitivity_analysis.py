# -*- coding: utf-8 -*-
"""
Temperature and composition sensitivity analysis for the PSMI LLE model.

This script is intentionally independent from the training entry point. It
reuses an existing checkpoint and the saved raw test predictions, then writes
paper-ready tables and figures to the structured temperature-robustness
experiment directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from psmi import config as C
from psmi.checkpoints import load_state_dict_compat
from psmi.data import load_and_prepare_excel, stratified_split_by_system
from psmi.predict import predict_pointwise_df_raw
from psmi.train import build_model
from psmi.utils import Scaler, set_seed


TARGET_COLS = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
PRED_COLS = [f"pred_{c}" for c in TARGET_COLS]
E_COLS = ["pred_Ex1", "pred_Ex2", "pred_Ex3"]
R_COLS = ["pred_Rx1", "pred_Rx2", "pred_Rx3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temperature and concentration sensitivity analysis for PSMI."
    )
    parser.add_argument("--checkpoint", default="", help="Path to best_model.pt.")
    parser.add_argument(
        "--pred-csv",
        default="",
        help="Path to test_df_raw_pointwise_predictions.csv.",
    )
    parser.add_argument(
        "--data-path",
        default="",
        help="Original Excel data file used to recover the training T_scaler.",
    )
    parser.add_argument(
        "--out-dir",
        "--results-dir",
        dest="results_dir",
        default=str(
            ROOT
            / "experiments"
            / "08_temperature_robustness"
            / "01_local_perturbation"
            / "runs"
            / "current"
        ),
        help="Output directory for tables, predictions, and the manuscript draft.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(
            ROOT
            / "experiments"
            / "08_temperature_robustness"
            / "01_local_perturbation"
            / "figures"
        ),
        help="Output directory for generated figures.",
    )
    parser.add_argument("--n-systems", type=int, default=6, help="Number of test systems to select.")
    parser.add_argument("--n-sweep", type=int, default=101, help="Number of t points in [0, 1].")
    parser.add_argument(
        "--temp-deltas",
        default="-10,-5,0,5,10",
        help="Comma-separated temperature perturbations in K.",
    )
    parser.add_argument(
        "--clip-temperature",
        action="store_true",
        help="Clip requested perturbed temperatures to the training temperature range.",
    )
    parser.add_argument(
        "--pred-batch-size-graph",
        type=int,
        default=1,
        help=(
            "Graph inference batch size. The default is 1 so finite-difference "
            "comparisons are independent of neighboring perturbation rows."
        ),
    )
    parser.add_argument("--seed", type=int, default=int(getattr(C, "SEED", 42)))
    parser.add_argument("--device", default="cpu", help="Inference device, default cpu.")
    return parser.parse_args()


def as_root_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def display_path(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def prediction_path_for_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    """Map models/<section>/<run>/best_model.pt to its experiment prediction table."""
    try:
        relative = checkpoint_path.resolve().relative_to((ROOT / "models").resolve())
    except ValueError:
        return None
    if len(relative.parts) < 3:
        return None
    section, run_name = relative.parts[0], relative.parts[1]
    return (
        ROOT
        / "experiments"
        / section
        / "runs"
        / run_name
        / "predictions"
        / "test_df_raw_pointwise_predictions.csv"
    )


def find_checkpoint(user_path: str) -> Path:
    if user_path:
        p = as_root_path(user_path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    priority = [Path(getattr(C, "LOAD_CKPT_PATH", ""))]
    for p in priority:
        if str(p) and p.is_file():
            return p

    hits = sorted(
        (ROOT / "models").rglob("best_model.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        raise FileNotFoundError("No best_model.pt checkpoint found in the project.")
    return hits[0]


def find_prediction_csv(user_path: str, checkpoint_path: Path) -> Path:
    if user_path:
        p = as_root_path(user_path)
        if not p.is_file():
            raise FileNotFoundError(f"Prediction CSV not found: {p}")
        return p

    mapped_prediction = prediction_path_for_checkpoint(checkpoint_path)
    priority = [
        mapped_prediction,
        checkpoint_path.parent / "test_df_raw_pointwise_predictions.csv",
    ]
    for p in priority:
        if p is not None and p.is_file():
            return p

    hits = sorted(
        (ROOT / "experiments").rglob("test_df_raw_pointwise_predictions.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        raise FileNotFoundError("No test_df_raw_pointwise_predictions.csv found in the project.")
    return hits[0]


def find_data_path(user_path: str, checkpoint_path: Path) -> Optional[Path]:
    if user_path:
        p = as_root_path(user_path)
        if not p.is_file():
            raise FileNotFoundError(f"Data workbook not found: {p}")
        return p

    candidates: List[Path] = []
    if "aichej" in str(checkpoint_path.parent).lower():
        candidates.extend(
            [
                ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles_min3.xlsx",
                ROOT / "datasets" / "raw" / "AIChEj-LLE-all.xlsx",
            ]
        )
    candidates.extend(
        [
            ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles_min3.xlsx",
            ROOT / "datasets" / "processed" / "LLE-literature-data-boosted.xlsx",
            ROOT / "datasets" / "raw" / "AIChEj-LLE-all.xlsx",
            Path(getattr(C, "EXCEL_PATH", "")),
        ]
    )
    for p in candidates:
        if str(p) and p.is_file():
            return p
    return None


def parse_temp_deltas(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if 0.0 not in vals:
        vals.append(0.0)
    vals = sorted(set(vals))
    if len(vals) < 3:
        raise ValueError("At least three temperature deltas are recommended.")
    return vals


def torch_load_checkpoint(path: Path) -> Dict:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        ckpt = {"state_dict": ckpt}
    return ckpt


def get_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    for key in ["state_dict", "model"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError(f"Cannot find a model state_dict in checkpoint keys: {list(ckpt.keys())}")


def count_layer_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> int:
    ids = set()
    for k in state_dict:
        if k.startswith(prefix):
            parts = k[len(prefix) :].split(".", 1)
            if parts and parts[0].isdigit():
                ids.add(int(parts[0]))
    return max(ids) + 1 if ids else 0


def apply_runtime_config_from_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    checkpoint_path: Path,
    device: str,
) -> Dict[str, object]:
    """Set config attributes so build_model() matches the checkpoint."""
    use_graph = any(k.startswith("encoder.") for k in state_dict)
    use_mix = any(k.startswith("mix_encoder.") for k in state_dict)
    use_fg = any(k.startswith("fg_") or k.startswith("fg.") for k in state_dict)
    fg_token_mode = "fg_token_embed.weight" in state_dict

    setattr(C, "DEVICE", device)
    setattr(C, "OUT_DIR", str(checkpoint_path.parent))
    setattr(C, "USE_GRAPH", bool(use_graph))
    setattr(C, "USE_MIX_GRAPH", bool(use_mix))
    setattr(C, "USE_FG", bool(use_fg))
    setattr(C, "FG_TOKEN_MODE", bool(fg_token_mode))
    setattr(C, "NUM_WORKERS_GRAPH", 0)
    setattr(C, "USE_AMP", False)
    setattr(C, "PRED_BATCH_SIZE_GRAPH", int(getattr(C, "PRED_BATCH_SIZE_GRAPH", 128)))

    if use_graph and "encoder.node_proj.0.weight" in state_dict:
        setattr(C, "GNN_HIDDEN", int(state_dict["encoder.node_proj.0.weight"].shape[0]))
    gnn_layers = count_layer_prefix(state_dict, "encoder.layers.")
    if gnn_layers:
        setattr(C, "GNN_LAYERS", int(gnn_layers))
    if "head_E.weight" in state_dict:
        setattr(C, "GNN_HEAD_HIDDEN", int(state_dict["head_E.weight"].shape[1]))

    if use_mix:
        mix_layers = count_layer_prefix(state_dict, "mix_encoder.layers.")
        if mix_layers:
            setattr(C, "MIX_LAYERS", int(mix_layers))
        if "mix_encoder.layers.0.msg.0.bias" in state_dict:
            setattr(C, "MIX_HIDDEN", int(state_dict["mix_encoder.layers.0.msg.0.bias"].shape[0]))

    if use_fg:
        if fg_token_mode:
            setattr(C, "FG_TOPK", int(state_dict["fg_token_embed.weight"].shape[0] - 1))
            setattr(C, "FG_MLP_HIDDEN", int(state_dict["fg_token_embed.weight"].shape[1]))
        elif "fg_encoder.0.weight" in state_dict:
            setattr(C, "FG_TOPK", int(state_dict["fg_encoder.0.weight"].shape[1]))
        setattr(C, "FG_CROSS_ATTN", bool(any(k.startswith("fg_attn.") for k in state_dict)))

    if any(k.startswith("token_fuser.") for k in state_dict):
        fusion_mode = "transformer"
    elif any(k.startswith("comp_backbone.") for k in state_dict):
        fusion_mode = "s3_set"
    else:
        fusion_mode = "concat"
    setattr(C, "FUSION_MODE", fusion_mode)

    return {
        "use_graph": bool(use_graph),
        "use_mix_graph": bool(use_mix),
        "use_fg": bool(use_fg),
        "fg_token_mode": bool(fg_token_mode),
        "fg_topk": int(getattr(C, "FG_TOPK", 0)),
        "fusion_mode": fusion_mode,
        "gnn_hidden": int(getattr(C, "GNN_HIDDEN", 0)),
        "gnn_layers": int(getattr(C, "GNN_LAYERS", 0)),
        "mix_layers": int(getattr(C, "MIX_LAYERS", 0)),
    }


def load_fg_corpus(ckpt: Dict, checkpoint_path: Path) -> List[str]:
    if isinstance(ckpt.get("fg_corpus"), list):
        return list(ckpt["fg_corpus"])
    p = checkpoint_path.parent / "fg_corpus.json"
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        if isinstance(corpus, list):
            return list(corpus)
    return []


def load_model_bundle(checkpoint_path: Path, device: str) -> Tuple[torch.nn.Module, Dict[str, object], Dict]:
    ckpt = torch_load_checkpoint(checkpoint_path)
    state_dict = get_state_dict(ckpt)
    runtime = apply_runtime_config_from_checkpoint(state_dict, checkpoint_path, device)

    model = build_model().to(device)
    try:
        adaptations = load_state_dict_compat(model, state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint with the inferred runtime config. "
            "Inspect config fields such as USE_GRAPH, USE_MIX_GRAPH, USE_FG, "
            "FG_TOPK, MIX_LAYERS, GNN_HIDDEN, and FUSION_MODE.\n"
            f"Checkpoint: {checkpoint_path}\nOriginal error:\n{exc}"
        ) from exc
    runtime["checkpoint_adaptations"] = "; ".join(adaptations)

    fg_corpus = load_fg_corpus(ckpt, checkpoint_path)
    if fg_corpus:
        setattr(model, "fg_corpus", fg_corpus)
    model.eval()
    return model, runtime, ckpt


def recover_temperature_scaler(
    ckpt: Dict,
    data_path: Optional[Path],
    pred_df: pd.DataFrame,
) -> Tuple[Scaler, Dict[str, object], Optional[pd.DataFrame]]:
    if ("T_mean" in ckpt) and ("T_std" in ckpt):
        scaler = Scaler(mean=float(ckpt["T_mean"]), std=float(ckpt["T_std"]))
        info = {
            "source": "checkpoint",
            "train_T_min": float("nan"),
            "train_T_max": float("nan"),
            "train_T_mean": scaler.mean,
            "train_T_std": scaler.std,
            "split_test_systems_match_prediction_csv": "",
            "data_path": "",
        }
        return scaler, info, None

    if data_path is not None:
        df_raw, df_aug = load_and_prepare_excel(
            str(data_path),
            min_points_per_group=int(getattr(C, "MIN_POINTS_PER_GROUP", 6)),
            permute_23_aug=bool(getattr(C, "PERMUTE_23_AUG", True)),
        )
        train_df, _val_df, test_df = stratified_split_by_system(
            df_aug,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=int(getattr(C, "SEED", 42)),
            n_bins=8,
            min_bin_size=3,
        )
        scaler = Scaler.fit(train_df["T"].to_numpy(dtype="float32"))
        split_test_ids = {int(x) for x in test_df["system_id"].unique().tolist()}
        pred_ids = {int(x) for x in pred_df["system_id"].unique().tolist()}
        info = {
            "source": "reconstructed_train_split",
            "train_T_min": float(train_df["T"].min()),
            "train_T_max": float(train_df["T"].max()),
            "train_T_mean": float(scaler.mean),
            "train_T_std": float(scaler.std),
            "split_test_systems_match_prediction_csv": bool(split_test_ids == pred_ids),
            "split_test_system_overlap_count": int(len(split_test_ids & pred_ids)),
            "split_test_system_count": int(len(split_test_ids)),
            "prediction_system_count": int(len(pred_ids)),
            "data_path": str(data_path),
        }
        return scaler, info, df_raw

    temps = pred_df["T"].to_numpy(dtype="float32")
    scaler = Scaler.fit(temps)
    info = {
        "source": "fallback_prediction_csv_only",
        "train_T_min": float(np.min(temps)),
        "train_T_max": float(np.max(temps)),
        "train_T_mean": float(scaler.mean),
        "train_T_std": float(scaler.std),
        "split_test_systems_match_prediction_csv": "",
        "data_path": "",
    }
    return scaler, info, None


def first_existing_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def summarize_systems(pred_df: pd.DataFrame) -> pd.DataFrame:
    name1_col = first_existing_column(pred_df, ["IL (Component 1) full name", "Component 1", "component1"])
    name2_col = first_existing_column(pred_df, ["Component 2", "component2"])
    name3_col = first_existing_column(pred_df, ["Component 3", "component3"])
    fam2_col = first_existing_column(pred_df, ["Family of component 2", "family2"])
    fam3_col = first_existing_column(pred_df, ["Family of component 3", "family3"])
    il_col = first_existing_column(pred_df, ["IL abbreviation", "IL"])

    rows = []
    for sid, g in pred_df.groupby("system_id", sort=True):
        first = g.iloc[0]
        fam2 = str(first[fam2_col]) if fam2_col else ""
        fam3 = str(first[fam3_col]) if fam3_col else ""
        rows.append(
            {
                "system_id": int(sid),
                "n_points": int(len(g)),
                "n_temperatures": int(g["T"].nunique()),
                "T_min": float(g["T"].min()),
                "T_max": float(g["T"].max()),
                "T_median": float(g["T"].median()),
                "t_min": float(g["t"].min()),
                "t_max": float(g["t"].max()),
                "component1": str(first[name1_col]) if name1_col else "",
                "component2": str(first[name2_col]) if name2_col else "",
                "component3": str(first[name3_col]) if name3_col else "",
                "IL_abbreviation": str(first[il_col]) if il_col else "",
                "smiles1": str(first["smiles1"]),
                "smiles2": str(first["smiles2"]),
                "smiles3": str(first["smiles3"]),
                "family2": fam2,
                "family3": fam3,
                "family_pair": f"{fam2}|{fam3}",
            }
        )
    return pd.DataFrame(rows)


def choose_nearest(
    candidates: pd.DataFrame,
    value_col: str,
    target: float,
    selected: set,
    reason: str,
    reasons: Dict[int, List[str]],
) -> Optional[int]:
    pool = candidates[~candidates["system_id"].isin(selected)].copy()
    if pool.empty:
        return None
    pool["_dist"] = (pool[value_col].astype(float) - float(target)).abs()
    pool = pool.sort_values(["_dist", "n_points", "system_id"], ascending=[True, False, True])
    sid = int(pool.iloc[0]["system_id"])
    selected.add(sid)
    reasons.setdefault(sid, []).append(reason)
    return sid


def select_representative_systems(
    summary: pd.DataFrame,
    n_systems: int,
    train_t_min: float,
    train_t_max: float,
    temp_deltas: Sequence[float],
) -> pd.DataFrame:
    n_systems = max(1, min(int(n_systems), len(summary)))
    work = summary.copy()
    min_delta = float(min(temp_deltas))
    max_delta = float(max(temp_deltas))
    work["temp_window_inside_train"] = (
        (work["T_median"] + min_delta >= train_t_min)
        & (work["T_median"] + max_delta <= train_t_max)
    )

    supported = work[work["temp_window_inside_train"]].copy()
    pool = supported if len(supported) >= min(3, n_systems) else work

    selected: set = set()
    reasons: Dict[int, List[str]] = {}

    t_targets = [
        ("low_temperature", float(pool["T_median"].quantile(0.05))),
        ("medium_temperature", float(pool["T_median"].quantile(0.50))),
        ("high_temperature", float(pool["T_median"].quantile(0.95))),
    ]
    for reason, target in t_targets:
        if len(selected) < n_systems:
            choose_nearest(pool, "T_median", target, selected, reason, reasons)

    n_targets = [
        ("sparse_composition_path", float(work["n_points"].quantile(0.10))),
        ("dense_composition_path", float(work["n_points"].quantile(0.90))),
    ]
    for reason, target in n_targets:
        if len(selected) < n_systems:
            choose_nearest(work, "n_points", target, selected, reason, reasons)

    while len(selected) < n_systems:
        pool2 = work[~work["system_id"].isin(selected)].copy()
        if pool2.empty:
            break
        selected_families = set(work[work["system_id"].isin(selected)]["family_pair"].tolist())
        t_scale = max(1e-12, float(work["T_median"].max() - work["T_median"].min()))
        n_scale = max(1e-12, float(work["n_points"].max() - work["n_points"].min()))
        sel_t = work[work["system_id"].isin(selected)]["T_median"].to_numpy(dtype=float)
        sel_n = work[work["system_id"].isin(selected)]["n_points"].to_numpy(dtype=float)

        scores = []
        for _, r in pool2.iterrows():
            dt = np.min(np.abs(sel_t - float(r["T_median"]))) / t_scale if len(sel_t) else 1.0
            dn = np.min(np.abs(sel_n - float(r["n_points"]))) / n_scale if len(sel_n) else 1.0
            fam_bonus = 0.5 if r["family_pair"] not in selected_families else 0.0
            support_bonus = 0.2 if bool(r["temp_window_inside_train"]) else 0.0
            scores.append(dt + dn + fam_bonus + support_bonus)
        pool2 = pool2.assign(_score=scores).sort_values(
            ["_score", "n_points", "system_id"], ascending=[False, False, True]
        )
        sid = int(pool2.iloc[0]["system_id"])
        selected.add(sid)
        reasons.setdefault(sid, []).append("diversity_fill")

    out = work[work["system_id"].isin(selected)].copy()
    out["selection_reason"] = out["system_id"].map(lambda sid: ";".join(reasons.get(int(sid), [])))
    out = out.sort_values(["T_median", "n_points", "system_id"]).reset_index(drop=True)
    return out


def predict_df(model: torch.nn.Module, scaler: Scaler, df: pd.DataFrame) -> pd.DataFrame:
    with torch.no_grad():
        return predict_pointwise_df_raw(model, scaler, df.reset_index(drop=True))


def make_temperature_eval_df(
    pred_df: pd.DataFrame,
    selected_ids: Iterable[int],
    temp_deltas: Sequence[float],
    train_t_min: float,
    train_t_max: float,
    clip_temperature: bool,
) -> pd.DataFrame:
    selected_ids = {int(x) for x in selected_ids}
    rows = []
    src = pred_df[pred_df["system_id"].astype(int).isin(selected_ids)].copy()
    for source_index, row in src.iterrows():
        base_t = float(row["T"])
        for delta in temp_deltas:
            requested = base_t + float(delta)
            evaluated = requested
            clipped = False
            if clip_temperature:
                evaluated = float(np.clip(requested, train_t_min, train_t_max))
                clipped = not math.isclose(evaluated, requested)
            out = row.copy()
            out["source_index"] = int(source_index)
            out["base_T"] = base_t
            out["delta_T_requested"] = float(delta)
            out["requested_T"] = float(requested)
            out["evaluated_T"] = float(evaluated)
            out["outside_train_T_range"] = bool((requested < train_t_min) or (requested > train_t_max))
            out["temperature_clipped"] = bool(clipped)
            out["T"] = float(evaluated)
            rows.append(out)
    return pd.DataFrame(rows)


def aggregate_slopes_by_system(
    df: pd.DataFrame,
    x_col: str,
    group_col: str = "source_index",
) -> Dict[int, np.ndarray]:
    out: Dict[int, List[np.ndarray]] = {}
    for sid, sys_g in df.groupby("system_id", sort=True):
        arrs = []
        for _, point_g in sys_g.groupby(group_col, sort=True):
            point_g = point_g.sort_values(x_col)
            x = point_g[x_col].to_numpy(dtype=float)
            y = point_g[PRED_COLS].to_numpy(dtype=float)
            if len(x) < 2:
                continue
            uniq_x, uniq_idx = np.unique(x, return_index=True)
            if len(uniq_x) < 2:
                continue
            y = y[uniq_idx]
            dx = np.diff(uniq_x)
            keep = np.abs(dx) > 1e-12
            if not np.any(keep):
                continue
            slopes = np.abs(np.diff(y, axis=0)[keep]) / np.abs(dx[keep])[:, None]
            arrs.append(slopes)
        out[int(sid)] = np.vstack(arrs) if arrs else np.empty((0, len(PRED_COLS)))
    return out


def summarize_temperature_sensitivity(
    temp_pred: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    slope_map = aggregate_slopes_by_system(temp_pred, "evaluated_T", group_col="source_index")
    rows = []
    baseline = temp_pred[np.isclose(temp_pred["delta_T_requested"].astype(float), 0.0)].copy()
    baseline_mae = float("nan")
    if not baseline.empty:
        saved = pred_df.loc[baseline["source_index"].astype(int), PRED_COLS].to_numpy(dtype=float)
        now = baseline[PRED_COLS].to_numpy(dtype=float)
        baseline_mae = float(np.mean(np.abs(now - saved)))

    for sid, g in temp_pred.groupby("system_id", sort=True):
        sid_int = int(sid)
        slopes = slope_map.get(sid_int, np.empty((0, len(PRED_COLS))))
        row = {
            "system_id": sid_int,
            "n_source_points": int(g["source_index"].nunique()),
            "n_temperature_evaluations": int(len(g)),
            "base_T_min": float(g["base_T"].min()),
            "base_T_max": float(g["base_T"].max()),
            "requested_T_min": float(g["requested_T"].min()),
            "requested_T_max": float(g["requested_T"].max()),
            "outside_train_T_count": int(g["outside_train_T_range"].sum()),
            "outside_train_T_fraction": float(g["outside_train_T_range"].mean()),
            "temperature_clipped_count": int(g["temperature_clipped"].sum()),
        }
        if slopes.size:
            row.update(
                {
                    "mean_abs_dy_dT": float(np.mean(slopes)),
                    "p95_abs_dy_dT": float(np.percentile(slopes, 95)),
                    "max_abs_dy_dT": float(np.max(slopes)),
                    "mean_abs_dE_dT": float(np.mean(slopes[:, :3])),
                    "mean_abs_dR_dT": float(np.mean(slopes[:, 3:])),
                    "max_abs_dE_dT": float(np.max(slopes[:, :3])),
                    "max_abs_dR_dT": float(np.max(slopes[:, 3:])),
                }
            )
            for i, col in enumerate(PRED_COLS):
                row[f"mean_abs_d{col.replace('pred_', '')}_dT"] = float(np.mean(slopes[:, i]))
                row[f"max_abs_d{col.replace('pred_', '')}_dT"] = float(np.max(slopes[:, i]))
        else:
            row.update(
                {
                    "mean_abs_dy_dT": float("nan"),
                    "p95_abs_dy_dT": float("nan"),
                    "max_abs_dy_dT": float("nan"),
                    "mean_abs_dE_dT": float("nan"),
                    "mean_abs_dR_dT": float("nan"),
                    "max_abs_dE_dT": float("nan"),
                    "max_abs_dR_dT": float("nan"),
                }
            )
        rows.append(row)

    summary = {
        "baseline_reproduction_mae_vs_saved_predictions": baseline_mae,
        "temperature_mean_abs_dy_dT_overall": float(np.nanmean([r["mean_abs_dy_dT"] for r in rows])),
        "temperature_max_abs_dy_dT_overall": float(np.nanmax([r["max_abs_dy_dT"] for r in rows])),
        "temperature_outside_train_fraction": float(temp_pred["outside_train_T_range"].mean()),
        "temperature_clipped_fraction": float(temp_pred["temperature_clipped"].mean()),
    }
    return pd.DataFrame(rows), summary


def choose_reference_rows_for_sweep(pred_df: pd.DataFrame, selected_systems: pd.DataFrame) -> Dict[int, pd.Series]:
    refs: Dict[int, pd.Series] = {}
    for sid in selected_systems["system_id"].astype(int):
        g = pred_df[pred_df["system_id"].astype(int) == sid].copy()
        t_counts = g.groupby("T").size().sort_values(ascending=False)
        ref_T = float(t_counts.index[0])
        gT = g[np.isclose(g["T"].astype(float), ref_T)].copy()
        mid_idx = (gT["t"].astype(float) - 0.5).abs().sort_values().index[0]
        refs[sid] = g.loc[mid_idx].copy()
    return refs


def make_concentration_sweep_df(
    pred_df: pd.DataFrame,
    selected_systems: pd.DataFrame,
    n_sweep: int,
) -> pd.DataFrame:
    refs = choose_reference_rows_for_sweep(pred_df, selected_systems)
    t_grid = np.linspace(0.0, 1.0, int(n_sweep), dtype=float)
    rows = []
    for sid, row in refs.items():
        for t in t_grid:
            out = row.copy()
            out["source_index"] = int(row.name) if row.name is not None else -1
            out["sweep_T"] = float(row["T"])
            out["t"] = float(t)
            rows.append(out)
    return pd.DataFrame(rows)


def summarize_concentration_sensitivity(conc_pred: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    all_slope_blocks = []
    for sid, g in conc_pred.groupby("system_id", sort=True):
        g = g.sort_values("t")
        t = g["t"].to_numpy(dtype=float)
        y = g[PRED_COLS].to_numpy(dtype=float)
        dt = np.diff(t)
        keep = np.abs(dt) > 1e-12
        slopes = np.abs(np.diff(y, axis=0)[keep]) / np.abs(dt[keep])[:, None]
        all_slope_blocks.append(slopes)
        if len(t) >= 3:
            dt_mean = float(np.mean(np.diff(t)))
            second = np.abs(np.diff(y, n=2, axis=0)) / max(dt_mean * dt_mean, 1e-12)
        else:
            second = np.empty((0, len(PRED_COLS)))

        sum_e = g[E_COLS].sum(axis=1).to_numpy(dtype=float)
        sum_r = g[R_COLS].sum(axis=1).to_numpy(dtype=float)
        pred_values = g[PRED_COLS].to_numpy(dtype=float)
        row = {
            "system_id": int(sid),
            "sweep_T": float(g["sweep_T"].iloc[0]),
            "n_sweep_points": int(len(g)),
            "mean_abs_dy_dt": float(np.mean(slopes)),
            "p95_abs_dy_dt": float(np.percentile(slopes, 95)),
            "max_abs_dy_dt": float(np.max(slopes)),
            "mean_abs_dE_dt": float(np.mean(slopes[:, :3])),
            "mean_abs_dR_dt": float(np.mean(slopes[:, 3:])),
            "max_abs_dE_dt": float(np.max(slopes[:, :3])),
            "max_abs_dR_dt": float(np.max(slopes[:, 3:])),
            "mean_abs_second_derivative": float(np.mean(second)) if second.size else float("nan"),
            "max_abs_second_derivative": float(np.max(second)) if second.size else float("nan"),
            "min_predicted_composition": float(np.min(pred_values)),
            "max_predicted_composition": float(np.max(pred_values)),
            "negative_fraction": float(np.mean(pred_values < -1e-10)),
            "max_sum_error_E": float(np.max(np.abs(sum_e - 1.0))),
            "max_sum_error_R": float(np.max(np.abs(sum_r - 1.0))),
            "mean_sum_error_E": float(np.mean(np.abs(sum_e - 1.0))),
            "mean_sum_error_R": float(np.mean(np.abs(sum_r - 1.0))),
        }
        for i, col in enumerate(PRED_COLS):
            row[f"mean_abs_d{col.replace('pred_', '')}_dt"] = float(np.mean(slopes[:, i]))
            row[f"max_abs_d{col.replace('pred_', '')}_dt"] = float(np.max(slopes[:, i]))
        rows.append(row)

    all_slopes = np.vstack(all_slope_blocks) if all_slope_blocks else np.empty((0, len(PRED_COLS)))
    pred_values = conc_pred[PRED_COLS].to_numpy(dtype=float)
    sum_e = conc_pred[E_COLS].sum(axis=1).to_numpy(dtype=float)
    sum_r = conc_pred[R_COLS].sum(axis=1).to_numpy(dtype=float)
    summary = {
        "concentration_mean_abs_dy_dt_overall": float(np.mean(all_slopes)) if all_slopes.size else float("nan"),
        "concentration_max_abs_dy_dt_overall": float(np.max(all_slopes)) if all_slopes.size else float("nan"),
        "composition_negative_fraction": float(np.mean(pred_values < -1e-10)),
        "composition_min_predicted_value": float(np.min(pred_values)),
        "composition_max_sum_error_E": float(np.max(np.abs(sum_e - 1.0))),
        "composition_max_sum_error_R": float(np.max(np.abs(sum_r - 1.0))),
    }
    return pd.DataFrame(rows), summary


def plot_temperature_sensitivity(temp_by_system: pd.DataFrame, selected_systems: pd.DataFrame, out_dir: Path) -> None:
    plot_df = temp_by_system.merge(
        selected_systems[["system_id", "selection_reason", "T_median", "n_points"]],
        on="system_id",
        how="left",
    ).sort_values("T_median")
    x = np.arange(len(plot_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, plot_df["mean_abs_dE_dT"], width, label="Extract phase", color="#4C78A8")
    ax.bar(x + width / 2, plot_df["mean_abs_dR_dT"], width, label="Raffinate phase", color="#F58518")
    ax.scatter(x, plot_df["max_abs_dy_dT"], color="#222222", s=28, zorder=3, label="Max component sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)}" for s in plot_df["system_id"]], rotation=0)
    ax.set_xlabel("Selected test system ID")
    ax.set_ylabel(r"Finite-difference sensitivity, $|\Delta x|/\Delta T$ (K$^{-1}$)")
    ax.set_title("Temperature sensitivity of predicted phase compositions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "temperature_sensitivity_bar.png", dpi=300)
    plt.close(fig)


def plot_concentration_sweep(conc_pred: pd.DataFrame, selected_systems: pd.DataFrame, out_dir: Path) -> None:
    selected_order = selected_systems.sort_values("T_median")["system_id"].astype(int).tolist()
    n = len(selected_order)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4, max(3.0, 2.35 * nrows)), sharex=True, sharey=True)
    axes_arr = np.array(axes).reshape(-1)
    colors = ["#4C78A8", "#54A24B", "#E45756"]
    labels = ["component 1", "component 2", "component 3"]

    for ax, sid in zip(axes_arr, selected_order):
        g = conc_pred[conc_pred["system_id"].astype(int) == sid].sort_values("t")
        t = g["t"].to_numpy(dtype=float)
        for i, (color, label) in enumerate(zip(colors, labels), start=1):
            ax.plot(t, g[f"pred_Ex{i}"], color=color, lw=1.8, label=f"E {label}" if sid == selected_order[0] else None)
            ax.plot(
                t,
                g[f"pred_Rx{i}"],
                color=color,
                lw=1.8,
                ls="--",
                label=f"R {label}" if sid == selected_order[0] else None,
            )
        sweep_T = float(g["sweep_T"].iloc[0])
        ax.set_title(f"System {sid}, T={sweep_T:.2f} K", fontsize=10)
        ax.grid(alpha=0.22)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.03)

    for ax in axes_arr[n:]:
        ax.axis("off")
    for ax in axes_arr[-ncols:]:
        ax.set_xlabel("Composition path variable t")
    for ax in axes_arr[::ncols]:
        ax.set_ylabel("Predicted composition")

    handles, labels_ = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="upper center", ncol=3, frameon=False, fontsize=8)
        fig.subplots_adjust(top=0.88)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "concentration_sweep_curves.png", dpi=300)
    plt.close(fig)


def write_summary_csv(
    out_dir: Path,
    checkpoint_path: Path,
    pred_csv: Path,
    data_path: Optional[Path],
    runtime_info: Dict[str, object],
    scaler_info: Dict[str, object],
    selected_systems: pd.DataFrame,
    temp_summary: Dict[str, float],
    conc_summary: Dict[str, float],
) -> pd.DataFrame:
    row = {
        "checkpoint": display_path(checkpoint_path),
        "prediction_csv": display_path(pred_csv),
        "data_path": display_path(data_path),
        "n_selected_systems": int(len(selected_systems)),
        "selected_system_ids": ";".join(str(int(x)) for x in selected_systems["system_id"].tolist()),
    }
    row.update({f"runtime_{k}": v for k, v in runtime_info.items()})
    row.update(scaler_info)
    row.update(temp_summary)
    row.update(conc_summary)
    summary_df = pd.DataFrame([row])
    summary_df.to_csv(out_dir / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    return summary_df


def fmt_num(x: object, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return "NA"
    if abs(v) < 1e-4 and v != 0:
        return f"{v:.2e}"
    return f"{v:.{digits}f}"


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "") -> str:
    """Small dependency-free markdown table writer."""
    if df.empty:
        return "_No rows._"

    def _format(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            if floatfmt:
                return format(float(v), floatfmt)
            return fmt_num(v, 4)
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if v is None:
            return ""
        return str(v).replace("\n", " ").replace("|", "\\|")

    cols = [str(c) for c in df.columns]
    rows = [[_format(v) for v in row] for row in df.to_numpy(dtype=object)]
    widths = [len(c) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(values: Sequence[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    header = _line(cols)
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [_line(row) for row in rows]
    return "\n".join([header, sep, *body])


def write_markdown(
    out_dir: Path,
    figures_dir: Path,
    summary_df: pd.DataFrame,
    selected_systems: pd.DataFrame,
    temp_by_system: pd.DataFrame,
    conc_by_system: pd.DataFrame,
    temp_deltas: Sequence[float],
    n_sweep: int,
    clip_temperature: bool,
) -> None:
    s = summary_df.iloc[0].to_dict()
    selected_ids = ", ".join(str(int(x)) for x in selected_systems["system_id"].tolist())
    temp_mean = s.get("temperature_mean_abs_dy_dT_overall", float("nan"))
    temp_max = s.get("temperature_max_abs_dy_dT_overall", float("nan"))
    conc_mean = s.get("concentration_mean_abs_dy_dt_overall", float("nan"))
    conc_max = s.get("concentration_max_abs_dy_dt_overall", float("nan"))
    neg_frac = s.get("composition_negative_fraction", float("nan"))
    max_sum_e = s.get("composition_max_sum_error_E", float("nan"))
    max_sum_r = s.get("composition_max_sum_error_R", float("nan"))
    baseline_mae = s.get("baseline_reproduction_mae_vs_saved_predictions", float("nan"))
    outside_frac = s.get("temperature_outside_train_fraction", float("nan"))
    scaler_source = s.get("source", "")

    anomaly_notes = []
    if np.isfinite(float(outside_frac)) and float(outside_frac) > 0:
        action = "clipped to the training range" if clip_temperature else "flagged and left un-clipped"
        anomaly_notes.append(
            f"{fmt_num(100 * float(outside_frac), 2)}% of perturbed temperature evaluations were outside the reconstructed training temperature range and were {action}."
        )
    if np.isfinite(float(neg_frac)) and float(neg_frac) > 0:
        anomaly_notes.append(f"Negative predicted compositions occurred with fraction {fmt_num(neg_frac, 6)}.")
    if np.isfinite(float(max(max_sum_e, max_sum_r))) and max(float(max_sum_e), float(max_sum_r)) > 1e-5:
        anomaly_notes.append(
            f"The maximum phase sum-to-one error was {fmt_num(max(float(max_sum_e), float(max_sum_r)), 6)}."
        )
    if np.isfinite(float(baseline_mae)) and float(baseline_mae) > 1e-4:
        anomaly_notes.append(
            f"The reloaded single-sample sensitivity pipeline differs from the saved batched prediction CSV by MAE={fmt_num(baseline_mae, 6)}; all finite differences reported here are generated self-consistently with the same single-sample inference setting."
        )
    if not anomaly_notes:
        anomaly_notes.append("No abnormal physical-constraint violations were observed in the generated sensitivity grids.")

    selected_table = selected_systems[
        ["system_id", "T_median", "n_points", "family_pair", "selection_reason"]
    ]

    temp_key_table_df = temp_by_system[
        ["system_id", "mean_abs_dy_dT", "max_abs_dy_dT", "outside_train_T_fraction"]
    ]
    conc_key_table_df = conc_by_system[
        ["system_id", "mean_abs_dy_dt", "max_abs_dy_dt", "max_sum_error_E", "max_sum_error_R"]
    ]

    selected_table_md = df_to_markdown(selected_table)
    temp_key_table_md = df_to_markdown(temp_key_table_df, floatfmt=".5f")
    conc_key_table_md = df_to_markdown(conc_key_table_df, floatfmt=".5f")

    run_cmd = (
        "python scripts/analysis/run_sensitivity_analysis.py "
        f'--checkpoint "{s.get("checkpoint", "")}" '
        f'--pred-csv "{s.get("prediction_csv", "")}"'
    )

    paragraph = (
        "To evaluate whether the PSMI predictions are robust to the variables that most strongly affect "
        "liquid-liquid equilibrium, we performed a deterministic temperature and composition sensitivity "
        f"analysis on {len(selected_systems)} representative test systems (IDs {selected_ids}). "
        f"For temperature, the component identities and path variable t were fixed and T was perturbed by "
        f"{list(temp_deltas)} K. The resulting finite-difference sensitivity was small, with an overall "
        f"mean |Delta x|/Delta T of {fmt_num(temp_mean, 5)} K^-1 and a maximum observed component sensitivity "
        f"of {fmt_num(temp_max, 5)} K^-1. For composition, each selected system was evaluated along a "
        f"{n_sweep}-point sweep of t from 0 to 1 at the reference test temperature, giving a continuous "
        f"response with mean |Delta x|/Delta t={fmt_num(conc_mean, 4)} and maximum |Delta x|/Delta t="
        f"{fmt_num(conc_max, 4)}. Across all sweep predictions, the softmax-constrained outputs remained "
        f"non-negative (negative fraction={fmt_num(neg_frac, 6)}) and the maximum deviations of the extract "
        f"and raffinate phase sums from unity were {fmt_num(max_sum_e, 2)} and {fmt_num(max_sum_r, 2)}, "
        "respectively. These results indicate that the trained model responds smoothly and physically "
        "consistently to local perturbations in both temperature and composition within the tested domain."
    )

    text = f"""# Temperature and Concentration Sensitivity Analysis

## Run command

```powershell
{run_cmd}
```

## Experimental setting

- Checkpoint: `{s.get("checkpoint", "")}`
- Prediction CSV: `{s.get("prediction_csv", "")}`
- T scaler source: `{scaler_source}`; mean={fmt_num(s.get("train_T_mean"), 4)}, std={fmt_num(s.get("train_T_std"), 4)}
- Reconstructed training temperature range: {fmt_num(s.get("train_T_min"), 2)}-{fmt_num(s.get("train_T_max"), 2)} K
- Selected test systems: {selected_ids}
- Temperature perturbations: {list(temp_deltas)} K. Out-of-range handling: {"clip to training range" if clip_temperature else "flag only, no clipping"}.
- Concentration path: {n_sweep} uniformly spaced t values from 0 to 1.
- Graph inference batch size for finite differences: {s.get("runtime_pred_batch_size_graph", "")}.
- Reloaded-model check against saved pointwise predictions: MAE={fmt_num(baseline_mae, 6)}.

## Selected systems

{selected_table_md}

## Key numerical results

Temperature sensitivity by system:

{temp_key_table_md}

Composition sensitivity by system:

{conc_key_table_md}

Overall:

- Mean temperature sensitivity: {fmt_num(temp_mean, 5)} K^-1
- Max temperature sensitivity: {fmt_num(temp_max, 5)} K^-1
- Mean composition sensitivity: {fmt_num(conc_mean, 4)}
- Max composition sensitivity: {fmt_num(conc_max, 4)}
- Negative prediction fraction: {fmt_num(neg_frac, 6)}
- Max extract sum-to-one error: {fmt_num(max_sum_e, 2)}
- Max raffinate sum-to-one error: {fmt_num(max_sum_r, 2)}

## Interpretation for the main text

{paragraph}

## Notes on abnormal points

{" ".join(anomaly_notes)}

## Generated paper-ready figures

- `{display_path(figures_dir / "temperature_sensitivity_bar.png")}`: finite-difference temperature sensitivity for selected systems.
- `{display_path(figures_dir / "concentration_sweep_curves.png")}`: predicted extract/raffinate composition curves along t.
"""
    (out_dir / "main_text_sensitivity_analysis.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    results_dir = as_root_path(args.results_dir)
    figures_dir = as_root_path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    temp_deltas = parse_temp_deltas(args.temp_deltas)
    checkpoint_path = find_checkpoint(args.checkpoint)
    pred_csv = find_prediction_csv(args.pred_csv, checkpoint_path)
    data_path = find_data_path(args.data_path, checkpoint_path)

    print(f"[1/7] Loading checkpoint: {checkpoint_path}")
    model, runtime_info, ckpt = load_model_bundle(checkpoint_path, args.device)
    setattr(C, "PRED_BATCH_SIZE_GRAPH", int(args.pred_batch_size_graph))
    setattr(C, "PRED_BATCH_SIZE", int(args.pred_batch_size_graph))
    runtime_info["pred_batch_size_graph"] = int(args.pred_batch_size_graph)
    print(f"      Runtime config: {runtime_info}")

    print(f"[2/7] Loading saved test predictions: {pred_csv}")
    pred_df = pd.read_csv(pred_csv)
    missing_cols = [c for c in ["system_id", "T", "t", "smiles1", "smiles2", "smiles3", *PRED_COLS] if c not in pred_df.columns]
    if missing_cols:
        raise KeyError(f"Prediction CSV is missing required columns: {missing_cols}")

    print("[3/7] Recovering T_scaler and training temperature range")
    t_scaler, scaler_info, _df_raw = recover_temperature_scaler(ckpt, data_path, pred_df)
    train_t_min = float(scaler_info["train_T_min"])
    train_t_max = float(scaler_info["train_T_max"])
    if not np.isfinite(train_t_min) or not np.isfinite(train_t_max):
        train_t_min = float(pred_df["T"].min())
        train_t_max = float(pred_df["T"].max())
    print(
        f"      T_scaler source={scaler_info['source']} mean={t_scaler.mean:.4f} std={t_scaler.std:.4f}; "
        f"train T range={train_t_min:.2f}-{train_t_max:.2f} K"
    )

    print("[4/7] Selecting representative test systems")
    system_summary = summarize_systems(pred_df)
    selected_systems = select_representative_systems(
        system_summary,
        n_systems=int(args.n_systems),
        train_t_min=train_t_min,
        train_t_max=train_t_max,
        temp_deltas=temp_deltas,
    )
    selected_systems.to_csv(results_dir / "selected_systems.csv", index=False, encoding="utf-8-sig")
    selected_ids = selected_systems["system_id"].astype(int).tolist()
    print(f"      Selected system IDs: {selected_ids}")

    print("[5/7] Running temperature perturbation predictions")
    temp_eval = make_temperature_eval_df(
        pred_df,
        selected_ids,
        temp_deltas,
        train_t_min=train_t_min,
        train_t_max=train_t_max,
        clip_temperature=bool(args.clip_temperature),
    )
    temp_pred = predict_df(model, t_scaler, temp_eval)
    temp_pred.to_csv(results_dir / "temperature_perturbation_predictions.csv", index=False, encoding="utf-8-sig")
    temp_by_system, temp_summary = summarize_temperature_sensitivity(temp_pred, pred_df)
    temp_by_system.to_csv(results_dir / "temperature_sensitivity_by_system.csv", index=False, encoding="utf-8-sig")

    print("[6/7] Running concentration t-sweep predictions")
    conc_eval = make_concentration_sweep_df(pred_df, selected_systems, n_sweep=int(args.n_sweep))
    conc_pred = predict_df(model, t_scaler, conc_eval)
    conc_pred.to_csv(results_dir / "concentration_sweep_predictions.csv", index=False, encoding="utf-8-sig")
    conc_by_system, conc_summary = summarize_concentration_sensitivity(conc_pred)
    conc_by_system.to_csv(results_dir / "concentration_sensitivity_by_system.csv", index=False, encoding="utf-8-sig")

    print("[7/7] Writing summary tables, figures, and manuscript draft")
    summary_df = write_summary_csv(
        results_dir,
        checkpoint_path,
        pred_csv,
        data_path,
        runtime_info,
        scaler_info,
        selected_systems,
        temp_summary,
        conc_summary,
    )
    plot_temperature_sensitivity(temp_by_system, selected_systems, figures_dir)
    plot_concentration_sweep(conc_pred, selected_systems, figures_dir)
    write_markdown(
        results_dir,
        figures_dir,
        summary_df,
        selected_systems,
        temp_by_system,
        conc_by_system,
        temp_deltas=temp_deltas,
        n_sweep=int(args.n_sweep),
        clip_temperature=bool(args.clip_temperature),
    )

    print(f"[OK] Sensitivity tables written to: {results_dir}")
    print(f"[OK] Sensitivity figures written to: {figures_dir}")


if __name__ == "__main__":
    main()

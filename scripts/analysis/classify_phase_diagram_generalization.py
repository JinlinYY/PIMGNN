"""Classify held-out ternary LLE systems by phase-diagram reproduction quality.

The analysis combines pointwise binodal-endpoint errors with tie-line orientation
errors.  It intentionally distinguishes a user-supplied mole-fraction tolerance
from experimental uncertainty metadata: the current benchmark CSV does not contain
system-specific experimental uncertainties, so the default 0.02 threshold is a
sensitivity-analysis proxy and must not be described as measured uncertainty.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


TRUE_COLUMNS = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
PRED_COLUMNS = [
    "pred_Ex1",
    "pred_Ex2",
    "pred_Ex3",
    "pred_Rx1",
    "pred_Rx2",
    "pred_Rx3",
]

CATEGORY_ORDER = [
    "Quantitative within tolerance",
    "Qualitatively correct",
    "Failure",
]
CATEGORY_CN = {
    "Quantitative within tolerance": "在给定容差内定量正确",
    "Qualitatively correct": "定性正确但超出给定容差",
    "Failure": "失效",
}


def ternary_to_cartesian(x: np.ndarray) -> np.ndarray:
    """Map ternary mole fractions to an equilateral Cartesian triangle."""
    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]],
        dtype=float,
    )
    return np.asarray(x, dtype=float) @ vertices


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _symmetric_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def _safe_percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def _component_label(group: pd.DataFrame, column: str, fallback: str) -> str:
    if column not in group.columns:
        return fallback
    values = group[column].dropna().astype(str)
    return values.iloc[0] if not values.empty else fallback


def calculate_system_metrics(
    group: pd.DataFrame,
    min_tie_length: float = 0.05,
) -> Dict[str, float]:
    """Calculate composition, curve-shape, boundary, and tie-line metrics."""
    true = group[TRUE_COLUMNS].to_numpy(dtype=float)
    pred = group[PRED_COLUMNS].to_numpy(dtype=float)
    error = pred - true
    abs_error = np.abs(error)

    column_mean = true.mean(axis=0, keepdims=True)
    sse = float(np.square(error).sum())
    sst = float(np.square(true - column_mean).sum())
    curve_r2 = 1.0 - sse / sst if sst > 1e-12 else float("nan")
    envelope_scale = math.sqrt(sst / true.size) if sst > 1e-12 else float("nan")
    rmse_x = math.sqrt(float(np.square(error).mean()))
    nrmse_envelope = rmse_x / envelope_scale if envelope_scale > 1e-12 else float("nan")

    rank_correlations = []
    for idx in range(true.shape[1]):
        corr = _rank_correlation(true[:, idx], pred[:, idx])
        if np.isfinite(corr):
            rank_correlations.append(corr)
    shape_spearman = (
        float(np.median(rank_correlations)) if rank_correlations else float("nan")
    )

    true_e_xy = ternary_to_cartesian(true[:, :3])
    true_r_xy = ternary_to_cartesian(true[:, 3:])
    pred_e_xy = ternary_to_cartesian(pred[:, :3])
    pred_r_xy = ternary_to_cartesian(pred[:, 3:])

    true_tie = true_e_xy - true_r_xy
    pred_tie = pred_e_xy - pred_r_xy
    true_length = np.linalg.norm(true_tie, axis=1)
    pred_length = np.linalg.norm(pred_tie, axis=1)
    valid_angle = (true_length >= float(min_tie_length)) & (pred_length > 1e-12)

    if valid_angle.any():
        dot = np.sum(true_tie[valid_angle] * pred_tie[valid_angle], axis=1)
        denom = true_length[valid_angle] * pred_length[valid_angle]
        # Tie-lines are unoriented geometric segments, hence abs(cosine).
        cosine = np.clip(np.abs(dot / denom), 0.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
    else:
        angles = np.asarray([], dtype=float)

    endpoint_e = np.linalg.norm(pred_e_xy - true_e_xy, axis=1)
    endpoint_r = np.linalg.norm(pred_r_xy - true_r_xy, axis=1)

    return {
        "n_tie_lines": int(len(group)),
        "mae_x": float(abs_error.mean()),
        "rmse_x": rmse_x,
        "p95_abs_error": _safe_percentile(abs_error.reshape(-1), 95.0),
        "max_abs_error": float(abs_error.max()),
        "curve_r2": curve_r2,
        "nrmse_envelope": nrmse_envelope,
        "shape_spearman": shape_spearman,
        "mean_endpoint_error_E": float(endpoint_e.mean()),
        "mean_endpoint_error_R": float(endpoint_r.mean()),
        "hausdorff_E": _symmetric_hausdorff(true_e_xy, pred_e_xy),
        "hausdorff_R": _symmetric_hausdorff(true_r_xy, pred_r_xy),
        "median_tie_angle_deg": _safe_percentile(angles, 50.0),
        "p90_tie_angle_deg": _safe_percentile(angles, 90.0),
        "fraction_angle_le_15": float(np.mean(angles <= 15.0)) if angles.size else float("nan"),
        "fraction_angle_le_30": float(np.mean(angles <= 30.0)) if angles.size else float("nan"),
        "fraction_angle_le_45": float(np.mean(angles <= 45.0)) if angles.size else float("nan"),
        "n_angle_tie_lines": int(angles.size),
        "median_true_tie_length": float(np.median(true_length)),
    }


def classify_metrics(
    metrics: Dict[str, float],
    composition_tolerance: float,
    quantitative_angle_deg: float = 10.0,
    qualitative_angle_deg: float = 30.0,
    qualitative_p90_angle_deg: float = 45.0,
    qualitative_spearman: float = 0.70,
    qualitative_nrmse: float = 1.0,
) -> Tuple[str, str]:
    """Return an auditable category and the first failed/passed rule summary."""
    quantitative = (
        metrics["rmse_x"] <= composition_tolerance
        and np.isfinite(metrics["median_tie_angle_deg"])
        and metrics["median_tie_angle_deg"] <= quantitative_angle_deg
    )
    if quantitative:
        return (
            "Quantitative within tolerance",
            f"RMSE_x <= {composition_tolerance:.3f} and median tie-line angle <= {quantitative_angle_deg:.1f} deg",
        )

    qualitative = (
        np.isfinite(metrics["shape_spearman"])
        and metrics["shape_spearman"] >= qualitative_spearman
        and np.isfinite(metrics["nrmse_envelope"])
        and metrics["nrmse_envelope"] <= qualitative_nrmse
        and np.isfinite(metrics["median_tie_angle_deg"])
        and metrics["median_tie_angle_deg"] <= qualitative_angle_deg
        and np.isfinite(metrics["p90_tie_angle_deg"])
        and metrics["p90_tie_angle_deg"] <= qualitative_p90_angle_deg
    )
    if qualitative:
        return (
            "Qualitatively correct",
            "shape and tie-line orientation pass qualitative thresholds, but quantitative tolerance is exceeded",
        )

    failed = []
    if not np.isfinite(metrics["shape_spearman"]) or metrics["shape_spearman"] < qualitative_spearman:
        failed.append("shape rank correlation")
    if not np.isfinite(metrics["nrmse_envelope"]) or metrics["nrmse_envelope"] > qualitative_nrmse:
        failed.append("normalized boundary error")
    if not np.isfinite(metrics["median_tie_angle_deg"]) or metrics["median_tie_angle_deg"] > qualitative_angle_deg:
        failed.append("median tie-line angle")
    if not np.isfinite(metrics["p90_tie_angle_deg"]) or metrics["p90_tie_angle_deg"] > qualitative_p90_angle_deg:
        failed.append("90th-percentile tie-line angle")
    return "Failure", "failed: " + ", ".join(failed)


def analyze_predictions(
    predictions: pd.DataFrame,
    composition_tolerance: float,
    min_tie_length: float,
) -> pd.DataFrame:
    required = ["system_id", "T", *TRUE_COLUMNS, *PRED_COLUMNS]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    rows: List[Dict[str, object]] = []
    for (system_id, temperature), group in predictions.groupby(["system_id", "T"], sort=True):
        group = group.sort_values("t") if "t" in group.columns else group
        metrics = calculate_system_metrics(group, min_tie_length=min_tie_length)
        category, reason = classify_metrics(metrics, composition_tolerance)
        row: Dict[str, object] = {
            "system_id": int(system_id),
            "T_K": float(temperature),
            "component_1": _component_label(group, "IL abbreviation", _component_label(group, "IL (Component 1) full name", "component 1")),
            "component_2": _component_label(group, "Component 2", "component 2"),
            "component_3": _component_label(group, "Component 3", "component 3"),
            **metrics,
            "composition_tolerance": float(composition_tolerance),
            "experimental_uncertainty_available": False,
            "category": category,
            "category_cn": CATEGORY_CN[category],
            "classification_reason": reason,
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    order = {category: idx for idx, category in enumerate(CATEGORY_ORDER)}
    result["_category_order"] = result["category"].map(order)
    result = result.sort_values(
        ["_category_order", "rmse_x", "median_tie_angle_deg", "system_id", "T_K"]
    ).drop(columns="_category_order")
    return result.reset_index(drop=True)


def build_summary(system_table: pd.DataFrame) -> pd.DataFrame:
    total = len(system_table)
    rows = []
    for category in CATEGORY_ORDER:
        subset = system_table[system_table["category"] == category]
        rows.append(
            {
                "category": category,
                "category_cn": CATEGORY_CN[category],
                "n_system_temperature_groups": int(len(subset)),
                "fraction": float(len(subset) / total) if total else float("nan"),
                "median_rmse_x": float(subset["rmse_x"].median()) if len(subset) else float("nan"),
                "median_tie_angle_deg": float(subset["median_tie_angle_deg"].median()) if len(subset) else float("nan"),
                "median_shape_spearman": float(subset["shape_spearman"].median()) if len(subset) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity(
    system_table: pd.DataFrame,
    tolerances: Iterable[float],
) -> pd.DataFrame:
    rows = []
    metric_columns = [
        "rmse_x",
        "median_tie_angle_deg",
        "p90_tie_angle_deg",
        "shape_spearman",
        "nrmse_envelope",
    ]
    for tolerance in tolerances:
        counts = {category: 0 for category in CATEGORY_ORDER}
        for _, record in system_table.iterrows():
            metrics = {column: float(record[column]) for column in metric_columns}
            category, _ = classify_metrics(metrics, float(tolerance))
            counts[category] += 1
        total = len(system_table)
        rows.append(
            {
                "composition_tolerance": float(tolerance),
                "quantitative_n": counts["Quantitative within tolerance"],
                "qualitative_n": counts["Qualitatively correct"],
                "failure_n": counts["Failure"],
                "quantitative_fraction": counts["Quantitative within tolerance"] / total,
                "qualitative_fraction": counts["Qualitatively correct"] / total,
                "failure_fraction": counts["Failure"] / total,
            }
        )
    return pd.DataFrame(rows)


def write_markdown_table(system_table: pd.DataFrame, output_path: Path) -> None:
    selected = system_table[
        [
            "system_id",
            "T_K",
            "component_1",
            "component_2",
            "component_3",
            "n_tie_lines",
            "rmse_x",
            "shape_spearman",
            "median_tie_angle_deg",
            "p90_tie_angle_deg",
            "category",
        ]
    ].copy()
    for column in ["rmse_x", "shape_spearman", "median_tie_angle_deg", "p90_tie_angle_deg"]:
        selected[column] = selected[column].map(lambda value: f"{value:.3f}")
    output_path.write_text(selected.to_markdown(index=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--composition_tolerance", type=float, default=0.02)
    parser.add_argument("--min_tie_length", type=float, default=0.05)
    parser.add_argument(
        "--sensitivity_tolerances",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.05],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.composition_tolerance <= 0:
        raise ValueError("composition_tolerance must be positive")
    predictions = pd.read_csv(args.predictions)
    system_table = analyze_predictions(
        predictions,
        composition_tolerance=args.composition_tolerance,
        min_tie_length=args.min_tie_length,
    )
    summary = build_summary(system_table)
    sensitivity = build_sensitivity(system_table, args.sensitivity_tolerances)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    system_table.to_csv(args.out_dir / "system_classification.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_dir / "category_summary.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(args.out_dir / "tolerance_sensitivity.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(system_table, args.out_dir / "system_classification.md")

    print(summary.to_string(index=False))
    print("\nTolerance sensitivity")
    print(sensitivity.to_string(index=False))
    print(f"\nSaved analysis to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

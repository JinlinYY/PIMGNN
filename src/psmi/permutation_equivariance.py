"""Utilities for auditing component-2/3 permutation equivariance."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET_COLUMNS: Tuple[str, ...] = (
    "Ex1",
    "Ex2",
    "Ex3",
    "Rx1",
    "Rx2",
    "Rx3",
)
OUTPUT_PERMUTATION_23 = np.asarray([0, 2, 1, 3, 5, 4], dtype=np.int64)
PHASE_SLICES: Mapping[str, slice] = {
    "overall": slice(0, 6),
    "extract": slice(0, 3),
    "raffinate": slice(3, 6),
}


def swap_component_23_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Exchange Components 2 and 3 while preserving input-label correspondence."""
    required_inputs = {"smiles2", "smiles3"}
    missing_inputs = sorted(required_inputs.difference(frame.columns))
    if missing_inputs:
        raise KeyError(f"Missing component columns: {', '.join(missing_inputs)}")

    swapped = frame.copy()
    swapped[["smiles2", "smiles3"]] = frame[["smiles3", "smiles2"]].to_numpy()
    missing_targets = sorted(set(TARGET_COLUMNS).difference(frame.columns))
    if missing_targets:
        raise KeyError(f"Missing target columns: {', '.join(missing_targets)}")
    swapped[["Ex2", "Ex3"]] = frame[["Ex3", "Ex2"]].to_numpy()
    swapped[["Rx2", "Rx3"]] = frame[["Rx3", "Rx2"]].to_numpy()
    swapped["aug_swap23"] = 1
    return swapped


def restore_component_23_outputs(values: np.ndarray) -> np.ndarray:
    """Map six predicted mole fractions from exchanged to original component order."""
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != 6:
        raise ValueError(f"Expected an array ending in six outputs, got {array.shape}")
    return array[..., OUTPUT_PERMUTATION_23].copy()


def _validate_audit_arrays(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Return float64 audit arrays after validating their shared shape."""
    converted = tuple(np.asarray(value, dtype=np.float64) for value in arrays)
    if not converted:
        raise ValueError("At least one array is required")
    expected_shape = converted[0].shape
    if len(expected_shape) != 2 or expected_shape[1] != 6 or expected_shape[0] == 0:
        raise ValueError(f"Expected a non-empty (n, 6) array, got {expected_shape}")
    if any(value.shape != expected_shape for value in converted[1:]):
        raise ValueError("All audit arrays must have the same shape")
    if any(not np.isfinite(value).all() for value in converted):
        raise ValueError("Audit arrays must contain only finite values")
    return converted


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate flattened MAE, RMSE, and coefficient of determination."""
    residual = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    centered = np.asarray(y_true, dtype=np.float64) - float(np.mean(y_true))
    denominator = float(np.sum(np.square(centered)))
    r2 = float("nan") if denominator <= 1e-12 else float(
        1.0 - np.sum(np.square(residual)) / denominator
    )
    return {"mae": mae, "rmse": rmse, "r2": r2}


def summarize_permutation_audit(
    y_true: np.ndarray,
    original_predictions: np.ndarray,
    swapped_predictions_restored: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize predictive accuracy and paired equivariance deviations by phase."""
    y_true, original_predictions, swapped_predictions_restored = _validate_audit_arrays(
        y_true,
        original_predictions,
        swapped_predictions_restored,
    )

    predictive_rows: List[Dict[str, object]] = []
    evaluations = {
        "original_ordering": original_predictions,
        "components_2_3_swapped": swapped_predictions_restored,
    }
    for evaluation, predictions in evaluations.items():
        for phase, phase_slice in PHASE_SLICES.items():
            predictive_rows.append(
                {
                    "evaluation": evaluation,
                    "phase": phase,
                    **_regression_metrics(
                        y_true[:, phase_slice],
                        predictions[:, phase_slice],
                    ),
                }
            )

    difference = swapped_predictions_restored - original_predictions
    equivariance_rows: List[Dict[str, object]] = []
    for phase, phase_slice in PHASE_SLICES.items():
        phase_difference = difference[:, phase_slice]
        absolute = np.abs(phase_difference)
        equivariance_rows.append(
            {
                "phase": phase,
                "mae": float(np.mean(absolute)),
                "rmse": float(np.sqrt(np.mean(np.square(phase_difference)))),
                "p95_absolute_error": float(np.percentile(absolute, 95.0)),
                "maximum_absolute_error": float(np.max(absolute)),
            }
        )
    return pd.DataFrame(predictive_rows), pd.DataFrame(equivariance_rows)


def _flatten_summary_metrics(
    predictive: pd.DataFrame,
    equivariance: pd.DataFrame,
) -> Dict[Tuple[str, str, str], float]:
    """Convert the two summary tables into bootstrap-addressable scalar metrics."""
    flattened: Dict[Tuple[str, str, str], float] = {}
    for row in predictive.itertuples(index=False):
        for metric in ("mae", "rmse", "r2"):
            flattened[(str(row.evaluation), str(row.phase), metric)] = float(
                getattr(row, metric)
            )
    for row in equivariance.itertuples(index=False):
        for metric in ("mae", "rmse", "p95_absolute_error"):
            flattened[("equivariance", str(row.phase), metric)] = float(
                getattr(row, metric)
            )
    for phase in PHASE_SLICES:
        for metric in ("mae", "rmse", "r2"):
            original = flattened[("original_ordering", phase, metric)]
            swapped = flattened[("components_2_3_swapped", phase, metric)]
            flattened[("swapped_minus_original", phase, metric)] = swapped - original
    return flattened


def cluster_bootstrap_intervals(
    system_ids: Sequence[object],
    y_true: np.ndarray,
    original_predictions: np.ndarray,
    swapped_predictions_restored: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 2026,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Estimate percentile intervals by resampling complete chemical systems."""
    y_true, original_predictions, swapped_predictions_restored = _validate_audit_arrays(
        y_true,
        original_predictions,
        swapped_predictions_restored,
    )
    identifiers = np.asarray(system_ids)
    if identifiers.ndim != 1 or len(identifiers) != len(y_true):
        raise ValueError("system_ids must provide one identifier per prediction row")
    if int(n_resamples) <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    unique_systems = pd.unique(identifiers)
    groups = [np.flatnonzero(identifiers == system_id) for system_id in unique_systems]
    if not groups:
        raise ValueError("At least one system is required")

    point_predictive, point_equivariance = summarize_permutation_audit(
        y_true,
        original_predictions,
        swapped_predictions_restored,
    )
    point_estimates = _flatten_summary_metrics(point_predictive, point_equivariance)
    bootstrap_values: Dict[Tuple[str, str, str], np.ndarray] = {
        key: np.empty(int(n_resamples), dtype=np.float64) for key in point_estimates
    }

    rng = np.random.default_rng(int(seed))
    for bootstrap_index in range(int(n_resamples)):
        sampled_group_indices = rng.integers(0, len(groups), size=len(groups))
        sampled_rows = np.concatenate([groups[index] for index in sampled_group_indices])
        predictive, equivariance = summarize_permutation_audit(
            y_true[sampled_rows],
            original_predictions[sampled_rows],
            swapped_predictions_restored[sampled_rows],
        )
        values = _flatten_summary_metrics(predictive, equivariance)
        for key, value in values.items():
            bootstrap_values[key][bootstrap_index] = value

    tail = (1.0 - float(confidence_level)) / 2.0
    lower_percentile = 100.0 * tail
    upper_percentile = 100.0 * (1.0 - tail)
    rows: List[Dict[str, object]] = []
    for (analysis, phase, metric), estimate in point_estimates.items():
        samples = bootstrap_values[(analysis, phase, metric)]
        rows.append(
            {
                "analysis": analysis,
                "phase": phase,
                "metric": metric,
                "estimate": estimate,
                "ci_lower": float(np.nanpercentile(samples, lower_percentile)),
                "ci_upper": float(np.nanpercentile(samples, upper_percentile)),
                "confidence_level": float(confidence_level),
                "n_resamples": int(n_resamples),
                "bootstrap_seed": int(seed),
                "resampling_unit": "system_id",
                "n_systems": int(len(groups)),
                "n_records": int(len(identifiers)),
            }
        )
    return pd.DataFrame(rows)

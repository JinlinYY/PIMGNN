"""Quantify soft thermodynamic-consistency violations in saved PSMI predictions.

The chemical-potential equilibrium criterion is evaluated with

    r_i = ln(x_i^E gamma_i^E) - ln(x_i^R gamma_i^R)

and a prediction is counted as violating a tolerance ``epsilon`` when
``max_i |r_i| > epsilon``.  Because no universal numerical tolerance exists for
this dimensionless residual, the script reports a sensitivity table rather than
selecting a favorable single cutoff.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from psmi.loss import nrtl_mu_residual  # noqa: E402


PRED_E_COLUMNS = ["pred_Ex1", "pred_Ex2", "pred_Ex3"]
PRED_R_COLUMNS = ["pred_Rx1", "pred_Rx2", "pred_Rx3"]
IDENTITY_COLUMNS = ["system_id", "T"]


def _parse_thresholds(value: str) -> List[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or thresholds[0] < 0:
        raise ValueError("Thresholds must contain one or more non-negative values")
    return thresholds


def _percentiles(values: np.ndarray) -> Dict[str, float]:
    return {
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + (z * z / (2.0 * total))) / denominator
    radius = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) / total)
            + (z * z / (4.0 * total * total))
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _load_nrtl(path: Path) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    params = payload.get("params", {})
    meta = payload.get("meta", {})
    if not params:
        raise ValueError(f"No NRTL parameters found in {path}")
    return meta, params


def evaluate_predictions(
    predictions_path: Path,
    nrtl_meta: Mapping[str, Any],
    nrtl_params: Mapping[str, Any],
    thresholds: Sequence[float],
    *,
    label: str,
    sum_tolerance: float = 1e-6,
    negative_tolerance: float = 1e-12,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(predictions_path)
    required = IDENTITY_COLUMNS + PRED_E_COLUMNS + PRED_R_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {predictions_path}: {missing}")

    system_keys = df["system_id"].map(lambda value: str(int(value)))
    covered = system_keys.isin(nrtl_params)
    evaluated = df.loc[covered].copy()
    evaluated_keys = system_keys.loc[covered]
    if evaluated.empty:
        raise ValueError(f"None of the predictions are covered by {predictions_path}")

    x_e_raw = evaluated[PRED_E_COLUMNS].to_numpy(dtype=np.float64)
    x_r_raw = evaluated[PRED_R_COLUMNS].to_numpy(dtype=np.float64)
    phase_sum_error = np.maximum(
        np.abs(x_e_raw.sum(axis=1) - 1.0),
        np.abs(x_r_raw.sum(axis=1) - 1.0),
    )
    minimum_fraction = np.minimum(x_e_raw.min(axis=1), x_r_raw.min(axis=1))
    composition_violation = (phase_sum_error > sum_tolerance) | (
        minimum_fraction < -negative_tolerance
    )

    temperature = evaluated["T"].to_numpy(dtype=np.float64)
    g_values = np.stack([nrtl_params[key] for key in evaluated_keys])
    residual = nrtl_mu_residual(
        torch.as_tensor(x_e_raw, dtype=torch.float32),
        torch.as_tensor(x_r_raw, dtype=torch.float32),
        torch.as_tensor(temperature, dtype=torch.float32),
        torch.as_tensor(g_values, dtype=torch.float32),
        alpha=float(nrtl_meta.get("alpha", 0.3)),
        R=float(nrtl_meta.get("R", 8.314462618)),
        tau_clip=10.0,
        ln_gamma_clip=20.0,
    ).detach().cpu().numpy()

    absolute_residual = np.abs(residual)
    sample_mean = absolute_residual.mean(axis=1)
    sample_max = absolute_residual.max(axis=1)
    gas_constant = float(nrtl_meta.get("R", 8.314462618))
    delta_mu_max_kj_mol = sample_max * gas_constant * temperature / 1000.0

    per_sample = pd.DataFrame(
        {
            "model": label,
            "source_row": evaluated.index.to_numpy(),
            "system_id": evaluated["system_id"].to_numpy(),
            "T_K": temperature,
            "phase_path_t": evaluated["t"].to_numpy() if "t" in evaluated else np.nan,
            "composition_violation": composition_violation,
            "phase_sum_error_max": phase_sum_error,
            "minimum_predicted_fraction": minimum_fraction,
            "mu_abs_mean": sample_mean,
            "mu_abs_max": sample_max,
            "delta_mu_max_kJ_mol": delta_mu_max_kj_mol,
            "mu_residual_1": residual[:, 0],
            "mu_residual_2": residual[:, 1],
            "mu_residual_3": residual[:, 2],
        }
    )

    threshold_rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        violation = sample_max > threshold
        violating_values = sample_max[violation]
        violating_energy = delta_mu_max_kj_mol[violation]
        ci_low, ci_high = _wilson_interval(int(violation.sum()), len(sample_max))
        threshold_rows.append(
            {
                "model": label,
                "tolerance_max_abs_log_activity_residual": float(threshold),
                "equivalent_max_activity_ratio": float(math.exp(threshold)),
                "evaluated_predictions": int(len(sample_max)),
                "violating_predictions": int(violation.sum()),
                "violation_fraction": float(violation.mean()),
                "violation_fraction_ci95_low": ci_low,
                "violation_fraction_ci95_high": ci_high,
                "violating_mu_abs_max_mean": (
                    float(violating_values.mean()) if violation.any() else 0.0
                ),
                "violating_mu_abs_max_median": (
                    float(np.median(violating_values)) if violation.any() else 0.0
                ),
                "mean_excess_above_tolerance": (
                    float((violating_values - threshold).mean()) if violation.any() else 0.0
                ),
                "violating_delta_mu_max_kJ_mol_mean": (
                    float(violating_energy.mean()) if violation.any() else 0.0
                ),
                "violating_delta_mu_max_kJ_mol_p95": (
                    float(np.percentile(violating_energy, 95)) if violation.any() else 0.0
                ),
            }
        )

    summary: Dict[str, Any] = {
        "label": label,
        "predictions_path": str(predictions_path.resolve()),
        "total_predictions": int(len(df)),
        "evaluated_predictions": int(len(evaluated)),
        "nrtl_parameter_coverage": float(covered.mean()),
        "systems_evaluated": int(evaluated["system_id"].nunique()),
        "composition_violation_count": int(composition_violation.sum()),
        "composition_violation_fraction": float(composition_violation.mean()),
        "phase_sum_error_p95": float(np.percentile(phase_sum_error, 95)),
        "minimum_predicted_fraction": float(minimum_fraction.min()),
        "mu_component_mae": float(absolute_residual.mean()),
        "mu_component_rmse": float(np.sqrt(np.mean(residual**2))),
        "mu_sample_abs_max": _percentiles(sample_max),
        "delta_mu_sample_max_kJ_mol": _percentiles(delta_mu_max_kj_mol),
    }
    return summary, pd.DataFrame(threshold_rows), per_sample


def _format_report(
    summaries: Sequence[Mapping[str, Any]], threshold_table: pd.DataFrame
) -> str:
    lines = [
        "# Thermodynamic-consistency audit",
        "",
        "A test prediction is counted as violating tolerance `epsilon` when",
        "`max_i |ln(x_i^E gamma_i^E) - ln(x_i^R gamma_i^R)| > epsilon`.",
        "The residual is dimensionless; `exp(epsilon)` is the corresponding maximum",
        "activity-ratio mismatch. Multiple tolerances are reported because there is no",
        "universal cutoff for this soft constraint.",
        "",
        "## Model-level summary",
        "",
        "| Model | N | Systems | NRTL coverage | Composition violation | mu MAE | mu RMSE | Median max | P95 max | Max | Median max delta-mu (kJ/mol) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        mu_max = item["mu_sample_abs_max"]
        energy = item["delta_mu_sample_max_kJ_mol"]
        lines.append(
            f"| {item['label']} | {item['evaluated_predictions']} | "
            f"{item['systems_evaluated']} | {item['nrtl_parameter_coverage']:.2%} | "
            f"{item['composition_violation_fraction']:.2%} | "
            f"{item['mu_component_mae']:.4f} | {item['mu_component_rmse']:.4f} | "
            f"{mu_max['median']:.4f} | {mu_max['p95']:.4f} | {mu_max['maximum']:.4f} | "
            f"{energy['median']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Tolerance sensitivity",
            "",
            "| Model | epsilon | exp(epsilon) | Violations | Fraction (95% CI) | Mean max residual among violations | Mean excess | Mean max delta-mu among violations (kJ/mol) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_table.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.tolerance_max_abs_log_activity_residual:.6g} | "
            f"{row.equivalent_max_activity_ratio:.4f} | "
            f"{row.violating_predictions}/{row.evaluated_predictions} | "
            f"{row.violation_fraction:.2%} "
            f"({row.violation_fraction_ci95_low:.2%}-{row.violation_fraction_ci95_high:.2%}) | "
            f"{row.violating_mu_abs_max_mean:.4f} | "
            f"{row.mean_excess_above_tolerance:.4f} | "
            f"{row.violating_delta_mu_max_kJ_mol_mean:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These are post-hoc consistency diagnostics based on system-specific NRTL",
            "parameters. They demonstrate the effect of the penalty but do not convert the",
            "soft regularizer into a hard thermodynamic guarantee.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument(
        "--nrtl-params",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "parameters" / "nrtl_params_all.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "thermodynamic_consistency_audit"
        ),
    )
    parser.add_argument("--thresholds", default="0.000001,0.1,0.25,0.5,1.0,2.0")
    parser.add_argument("--sum-tolerance", type=float, default=1e-6)
    parser.add_argument("--negative-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.thresholds)
    meta, params = _load_nrtl(args.nrtl_params)
    evaluations = [
        evaluate_predictions(
            args.predictions,
            meta,
            params,
            thresholds,
            label="physics_informed",
            sum_tolerance=args.sum_tolerance,
            negative_tolerance=args.negative_tolerance,
        )
    ]
    if args.baseline_predictions:
        evaluations.append(
            evaluate_predictions(
                args.baseline_predictions,
                meta,
                params,
                thresholds,
                label="data_driven_baseline",
                sum_tolerance=args.sum_tolerance,
                negative_tolerance=args.negative_tolerance,
            )
        )

        reference_keys = evaluations[0][2][["system_id", "T_K", "phase_path_t"]]
        baseline_keys = evaluations[1][2][["system_id", "T_K", "phase_path_t"]]
        if len(reference_keys) != len(baseline_keys) or not np.allclose(
            reference_keys.to_numpy(dtype=np.float64),
            baseline_keys.to_numpy(dtype=np.float64),
            equal_nan=True,
        ):
            raise ValueError(
                "Physics-informed and baseline predictions do not use the same ordered test records"
            )

    summaries = [item[0] for item in evaluations]
    threshold_table = pd.concat([item[1] for item in evaluations], ignore_index=True)
    per_sample = pd.concat([item[2] for item in evaluations], ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(args.out_dir / "threshold_sensitivity.csv", index=False)
    per_sample.to_csv(args.out_dir / "per_prediction_consistency.csv", index=False)
    comparison: Dict[str, float] = {}
    if len(summaries) == 2:
        physics_summary, baseline_summary = summaries

        def relative_reduction(metric_path: Sequence[str]) -> float:
            physics_value: Any = physics_summary
            baseline_value: Any = baseline_summary
            for key in metric_path:
                physics_value = physics_value[key]
                baseline_value = baseline_value[key]
            return float((baseline_value - physics_value) / baseline_value)

        comparison = {
            "mu_component_mae_relative_reduction": relative_reduction(["mu_component_mae"]),
            "mu_component_rmse_relative_reduction": relative_reduction(["mu_component_rmse"]),
            "mu_sample_abs_max_median_relative_reduction": relative_reduction(
                ["mu_sample_abs_max", "median"]
            ),
            "delta_mu_median_relative_reduction": relative_reduction(
                ["delta_mu_sample_max_kJ_mol", "median"]
            ),
        }

    payload = {
        "criterion": "max_i |ln(x_i^E gamma_i^E) - ln(x_i^R gamma_i^R)| > epsilon",
        "thresholds": thresholds,
        "sum_tolerance": args.sum_tolerance,
        "negative_tolerance": args.negative_tolerance,
        "nrtl_params_path": str(args.nrtl_params.resolve()),
        "nrtl_meta": dict(meta),
        "models": summaries,
        "physics_informed_relative_reduction_vs_baseline": comparison,
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
    (args.out_dir / "report.md").write_text(
        _format_report(summaries, threshold_table), encoding="utf-8"
    )
    print(f"[OK] Wrote thermodynamic-consistency audit to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

"""Audit Component-2/3 permutation equivariance with a registered PSMI checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from psmi import config as C
from psmi.configuration import apply_config_files, apply_config_overrides
from psmi.permutation_equivariance import (
    PHASE_SLICES,
    TARGET_COLUMNS,
    cluster_bootstrap_intervals,
    restore_component_23_outputs,
    summarize_permutation_audit,
    swap_component_23_frame,
)
from psmi.predict import predict_pointwise_df_raw
from psmi.reproduction import PREDICTION_COLUMNS, prepare_saved_checkpoint, sha256_file
from psmi.utils import set_seed


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "supporting_information"
    / "s3_additional_evaluation_and_validation"
    / "s3_10_component_permutation_equivariance"
)


def parse_args() -> argparse.Namespace:
    """Parse the registered-checkpoint audit options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/reproduction/published_checkpoint_registry.json",
    )
    parser.add_argument("--run-id", default="figure2a_psmi")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default=None, help="cpu, cuda, or a CUDA device")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--expected-test-records", type=int, default=803)
    parser.add_argument("--expected-test-systems", type=int, default=78)
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    """Resolve repository-relative command and registry paths."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _portable_path(path: str | Path) -> str:
    """Represent repository files without machine-specific absolute prefixes."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_registry_run(registry_path: Path, run_id: str) -> Mapping[str, Any]:
    """Return one uniquely identified checkpoint registry entry."""
    with registry_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    matches = [run for run in payload.get("runs", []) if run.get("id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one registry entry for {run_id!r}, found {len(matches)}")
    return matches[0]


def _validate_partition(test_frame: pd.DataFrame, records: int, systems: int) -> None:
    """Stop if the registered run does not resolve to the manuscript test partition."""
    actual_records = int(len(test_frame))
    actual_systems = int(test_frame["system_id"].nunique())
    if actual_records != int(records) or actual_systems != int(systems):
        raise ValueError(
            "Unexpected test partition: "
            f"{actual_records} records/{actual_systems} systems; "
            f"expected {records} records/{systems} systems"
        )


def _paired_prediction_table(
    test_frame: pd.DataFrame,
    original_predictions: np.ndarray,
    swapped_predictions_raw: np.ndarray,
    swapped_predictions_restored: np.ndarray,
) -> pd.DataFrame:
    """Build the row-level evidence table for all paired predictions."""
    output = test_frame.copy().reset_index(drop=True)
    output.insert(0, "audit_record_id", np.arange(len(output), dtype=np.int64))
    output["swapped_smiles1"] = output["smiles1"]
    output["swapped_smiles2"] = output["smiles3"]
    output["swapped_smiles3"] = output["smiles2"]
    swapped_targets = restore_component_23_outputs(
        output[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    )
    for index, target in enumerate(TARGET_COLUMNS):
        output[f"true_swapped_{target}"] = swapped_targets[:, index]
        output[f"pred_original_{target}"] = original_predictions[:, index]
        output[f"pred_swapped_raw_{target}"] = swapped_predictions_raw[:, index]
        output[f"pred_swapped_restored_{target}"] = swapped_predictions_restored[:, index]
        residual = swapped_predictions_restored[:, index] - original_predictions[:, index]
        output[f"equivariance_residual_{target}"] = residual
        output[f"equivariance_absolute_error_{target}"] = np.abs(residual)
    return output


def _empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted values and their empirical cumulative probabilities."""
    sorted_values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    probabilities = np.arange(1, len(sorted_values) + 1, dtype=np.float64) / len(
        sorted_values
    )
    return sorted_values, probabilities


def _plot_equivariance(
    original_predictions: np.ndarray,
    swapped_predictions_restored: np.ndarray,
    equivariance_metrics: pd.DataFrame,
    figure_dir: Path,
) -> Sequence[Path]:
    """Create a two-panel vector and high-resolution raster audit figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    phase_styles = {
        "extract": ("Extract", "#0072B2"),
        "raffinate": ("Raffinate", "#D55E00"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))

    lower = float(min(original_predictions.min(), swapped_predictions_restored.min()))
    upper = float(max(original_predictions.max(), swapped_predictions_restored.max()))
    margin = max(0.02, 0.04 * (upper - lower))
    limits = (lower - margin, upper + margin)
    axes[0].plot(limits, limits, color="#4D4D4D", linewidth=1.0, linestyle="--", zorder=1)
    for phase, (display_name, color) in phase_styles.items():
        phase_slice = PHASE_SLICES[phase]
        axes[0].scatter(
            original_predictions[:, phase_slice].reshape(-1),
            swapped_predictions_restored[:, phase_slice].reshape(-1),
            s=8,
            alpha=0.42,
            color=color,
            edgecolors="none",
            label=display_name,
            rasterized=True,
            zorder=2,
        )
    axes[0].set(xlim=limits, ylim=limits)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Prediction in original ordering")
    axes[0].set_ylabel("Prediction after swap and restoration")
    axes[0].legend(frameon=False, loc="upper left")

    difference = np.abs(swapped_predictions_restored - original_predictions)
    for phase, (display_name, color) in phase_styles.items():
        phase_slice = PHASE_SLICES[phase]
        x_values, y_values = _empirical_cdf(difference[:, phase_slice])
        axes[1].plot(
            x_values,
            y_values,
            color=color,
            linewidth=1.5,
            label=display_name,
        )
    overall = equivariance_metrics.loc[
        equivariance_metrics["phase"] == "overall"
    ].iloc[0]
    axes[1].axhline(0.95, color="#777777", linewidth=0.8, linestyle=":")
    axes[1].axvline(
        float(overall["p95_absolute_error"]),
        color="#777777",
        linewidth=0.8,
        linestyle=":",
    )
    axes[1].text(
        0.98,
        0.38,
        "Overall equivariance\n"
        f"MAE = {float(overall['mae']):.4f}\n"
        f"P95 = {float(overall['p95_absolute_error']):.4f}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    axes[1].set_xlabel("Absolute equivariance deviation")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_ylim(0.0, 1.005)
    axes[1].legend(frameon=False, loc="lower right")

    for label, axis in zip(("a", "b"), axes):
        axis.text(
            -0.18,
            1.03,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
        axis.tick_params(direction="out", length=3, width=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.tight_layout(w_pad=2.2)
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "component_23_permutation_equivariance.png"
    pdf_path = figure_dir / "component_23_permutation_equivariance.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png_path, pdf_path]


def main() -> int:
    """Run paired inference, cluster bootstrap analysis, and figure generation."""
    args = parse_args()
    registry_path = _resolve(args.registry)
    run = _load_registry_run(registry_path, args.run_id)
    config_path = _resolve(run["config"])
    checkpoint_path = _resolve(run["checkpoint"])
    corpus_path = _resolve(run["fg_corpus"]) if run.get("fg_corpus") else None
    output_dir = _resolve(args.output_dir)
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)

    apply_config_files([config_path])
    apply_config_overrides([f"SEED={int(run['seed'])}", *run.get("set", [])])
    context = prepare_saved_checkpoint(
        checkpoint_path,
        device=args.device,
        functional_group_corpus=corpus_path,
        allow_derived_scalers=bool(run.get("allow_derived_scalers", False)),
        require_hash_match=not bool(run.get("allow_input_hash_mismatch", False)),
    )
    _validate_partition(
        context.test_frame,
        args.expected_test_records,
        args.expected_test_systems,
    )

    start_time = time.perf_counter()
    set_seed(int(C.SEED))
    original_output = predict_pointwise_df_raw(
        context.model,
        context.temperature_scaler,
        context.test_frame,
        device=context.device,
        P_scaler=context.pressure_scaler,
    )
    swapped_frame = swap_component_23_frame(context.test_frame)
    set_seed(int(C.SEED))
    swapped_output = predict_pointwise_df_raw(
        context.model,
        context.temperature_scaler,
        swapped_frame,
        device=context.device,
        P_scaler=context.pressure_scaler,
    )
    inference_seconds = time.perf_counter() - start_time

    y_true = original_output[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    original_predictions = original_output[PREDICTION_COLUMNS].to_numpy(dtype=np.float64)
    swapped_predictions_raw = swapped_output[PREDICTION_COLUMNS].to_numpy(dtype=np.float64)
    swapped_predictions_restored = restore_component_23_outputs(swapped_predictions_raw)

    predictive_metrics, equivariance_metrics = summarize_permutation_audit(
        y_true,
        original_predictions,
        swapped_predictions_restored,
    )
    bootstrap_intervals = cluster_bootstrap_intervals(
        original_output["system_id"].to_numpy(),
        y_true,
        original_predictions,
        swapped_predictions_restored,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    paired_predictions = _paired_prediction_table(
        context.test_frame,
        original_predictions,
        swapped_predictions_raw,
        swapped_predictions_restored,
    )

    predictive_path = result_dir / "predictive_metrics.csv"
    equivariance_path = result_dir / "equivariance_metrics.csv"
    bootstrap_path = result_dir / "system_cluster_bootstrap_intervals.csv"
    paired_path = result_dir / "paired_predictions.csv"
    predictive_metrics.to_csv(predictive_path, index=False, encoding="utf-8")
    equivariance_metrics.to_csv(equivariance_path, index=False, encoding="utf-8")
    bootstrap_intervals.to_csv(bootstrap_path, index=False, encoding="utf-8")
    paired_predictions.to_csv(paired_path, index=False, encoding="utf-8")
    figure_paths = _plot_equivariance(
        original_predictions,
        swapped_predictions_restored,
        equivariance_metrics,
        figure_dir,
    )

    input_checks = context.input_verification
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "experiment": "component_2_3_permutation_equivariance",
        "execution_mode": "paired_checkpoint_inference",
        "component_mapping": {
            "model_inputs_after_exchange": [
                "M1",
                "M3",
                "M2",
                "temperature",
                "phase_path_coordinate",
            ],
            "outputs_in_swapped_order": ["Ex1", "Ex3", "Ex2", "Rx1", "Rx3", "Rx2"],
            "unchanged_model_inputs": ["temperature", "phase_path_coordinate"],
            "unchanged_metadata": ["system_id", "pressure"],
        },
        "registry": _portable_path(registry_path),
        "registry_run_id": str(run["id"]),
        "config": _portable_path(config_path),
        "configuration_overrides": [f"SEED={int(run['seed'])}", *run.get("set", [])],
        "checkpoint": {
            "path": _portable_path(context.checkpoint_path),
            "sha256": sha256_file(context.checkpoint_path),
            "epoch": context.checkpoint.get("epoch", context.checkpoint.get("best_epoch")),
            "compatibility_adaptations": list(context.compatibility_adaptations),
        },
        "functional_group_corpus": (
            {
                "path": _portable_path(context.functional_group_corpus_path),
                "sha256": sha256_file(context.functional_group_corpus_path),
            }
            if context.functional_group_corpus_path is not None
            else None
        ),
        "input_identity": {
            "checkpoint_hashes_verified": bool(input_checks.get("verified", False)),
            "checkpoint_provenance_available": isinstance(
                context.checkpoint.get("provenance"), Mapping
            ),
            "dataset_path": _portable_path(C.EXCEL_PATH),
            "dataset_sha256": sha256_file(C.EXCEL_PATH),
            "split_manifest_path": _portable_path(C.SPLIT_MANIFEST_PATH),
            "split_manifest_sha256": sha256_file(C.SPLIT_MANIFEST_PATH),
        },
        "test_partition": {
            "records": int(len(context.test_frame)),
            "systems": int(context.test_frame["system_id"].nunique()),
        },
        "bootstrap": {
            "resampling_unit": "system_id",
            "resamples": int(args.bootstrap_resamples),
            "seed": int(args.bootstrap_seed),
            "confidence_level": 0.95,
        },
        "runtime": {
            "device": context.device,
            "paired_inference_seconds": float(inference_seconds),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": {
            "predictive_metrics": _portable_path(predictive_path),
            "equivariance_metrics": _portable_path(equivariance_path),
            "bootstrap_intervals": _portable_path(bootstrap_path),
            "paired_predictions": _portable_path(paired_path),
            "figures": [_portable_path(path) for path in figure_paths],
        },
    }
    manifest_path = result_dir / "experiment_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    overall = equivariance_metrics.loc[equivariance_metrics["phase"] == "overall"].iloc[0]
    print(
        json.dumps(
            {
                "test_records": int(len(context.test_frame)),
                "test_systems": int(context.test_frame["system_id"].nunique()),
                "equivariance_mae": float(overall["mae"]),
                "equivariance_rmse": float(overall["rmse"]),
                "equivariance_p95": float(overall["p95_absolute_error"]),
                "output_dir": _portable_path(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

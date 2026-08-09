#!/usr/bin/env python
"""Aggregate, bootstrap, and plot the tie-line threshold sensitivity study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.psmi.data import load_and_prepare_excel  # noqa: E402


TRUE_COLS = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
PRED_COLS = [f"pred_{c}" for c in TRUE_COLS]
GROUP_COLS = ["system_id", "T"]
T_BINS = [-1e-12, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-12]
T_LABELS = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]


def grouped_error_stats(df: pd.DataFrame) -> pd.DataFrame:
    abs_error = np.abs(
        df[PRED_COLS].to_numpy(dtype=np.float64)
        - df[TRUE_COLS].to_numpy(dtype=np.float64)
    )
    tmp = df[GROUP_COLS].copy()
    tmp["abs_sum"] = abs_error.sum(axis=1)
    tmp["n_values"] = abs_error.shape[1]
    return tmp.groupby(GROUP_COLS, sort=False).agg(
        abs_sum=("abs_sum", "sum"), n_values=("n_values", "sum")
    ).reset_index()


def bootstrap_mae(
    df: pd.DataFrame, n_boot: int, seed: int
) -> Tuple[float, float, float]:
    grouped = grouped_error_stats(df)
    sums = grouped["abs_sum"].to_numpy(dtype=np.float64)
    counts = grouped["n_values"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(n_boot, len(grouped)))
    values = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    point = float(sums.sum() / counts.sum())
    lo, hi = np.quantile(values, [0.025, 0.975])
    return point, float(lo), float(hi)


def paired_delta_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    a = grouped_error_stats(candidate)
    b = grouped_error_stats(reference)
    merged = a.merge(b, on=GROUP_COLS, suffixes=("_a", "_b"), validate="one_to_one")
    delta = (
        merged["abs_sum_a"].to_numpy() / merged["n_values_a"].to_numpy()
        - merged["abs_sum_b"].to_numpy() / merged["n_values_b"].to_numpy()
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot = delta[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(delta.mean()), float(lo), float(hi)


def threshold_dataset_stats(dataset: Path, thresholds) -> pd.DataFrame:
    clean, _ = load_and_prepare_excel(str(dataset), 1, False)
    counts = clean.groupby(GROUP_COLS, sort=False).size().rename("n_tielines").reset_index()
    rows = []
    for threshold in thresholds:
        keep = counts[counts["n_tielines"] >= threshold]
        selected = clean.merge(keep[GROUP_COLS], on=GROUP_COLS, how="inner")
        rows.append({
            "threshold": threshold,
            "tielines": int(len(selected)),
            "equilibrium_endpoints": int(2 * len(selected)),
            "systems": int(selected["system_id"].nunique()),
            "system_temperature_groups": int(len(keep)),
            "mean_tielines_per_group": float(len(selected) / len(keep)),
            "groups_retained_percent": float(100 * len(keep) / len(counts)),
            "tielines_retained_percent": float(100 * len(selected) / len(clean)),
        })
    return pd.DataFrame(rows)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.7,
        "lines.markersize": 5.2,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "supporting_information"
            / "s3_additional_evaluation_and_validation"
            / "s3_5_tieline_density_and_phase_path"
            / "results"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "tieline_density_sensitivity",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-threshold", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    figure_dir = args.figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(root / "threshold_metrics.csv").sort_values("threshold")
    thresholds = base["threshold"].astype(int).tolist()
    predictions: Dict[int, pd.DataFrame] = {}
    for threshold in thresholds:
        path = root / "runs" / f"seed{args.seed}" / f"threshold_{threshold:02d}" / "test_predictions.csv"
        predictions[threshold] = pd.read_csv(path)

    reference = predictions[args.reference_threshold]
    threshold_records = []
    location_records = []
    for threshold in thresholds:
        pred = predictions[threshold]
        mae, lo, hi = bootstrap_mae(
            pred, args.bootstrap_samples, seed=20260806 + threshold
        )
        delta, delta_lo, delta_hi = paired_delta_bootstrap(
            pred,
            reference,
            args.bootstrap_samples,
            seed=20261806 + threshold,
        )
        threshold_records.append({
            "threshold": threshold,
            "mae": mae,
            "mae_ci_low": lo,
            "mae_ci_high": hi,
            "delta_mae_vs_threshold6_group_balanced": delta,
            "delta_ci_low": delta_lo,
            "delta_ci_high": delta_hi,
        })

        work = pred.copy()
        work["t_bin"] = pd.cut(
            work["t"], bins=T_BINS, labels=T_LABELS, include_lowest=True
        )
        for i, label in enumerate(T_LABELS):
            sub = work[work["t_bin"] == label]
            mae_t, lo_t, hi_t = bootstrap_mae(
                sub, args.bootstrap_samples, seed=20262806 + 100 * threshold + i
            )
            location_records.append({
                "threshold": threshold,
                "t_bin": label,
                "n_tielines": int(len(sub)),
                "mae": mae_t,
                "mae_ci_low": lo_t,
                "mae_ci_high": hi_t,
            })

    threshold_ci = pd.DataFrame(threshold_records)
    location_ci = pd.DataFrame(location_records)
    coverage = threshold_dataset_stats(args.dataset.resolve(), thresholds)
    merged = base.merge(threshold_ci, on="threshold", suffixes=("_raw", ""))
    merged = merged.merge(coverage, on="threshold")
    merged.to_csv(root / "threshold_metrics_with_ci.csv", index=False, encoding="utf-8-sig")
    location_ci.to_csv(root / "location_metrics_with_ci.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(root / "dataset_threshold_counts.csv", index=False, encoding="utf-8-sig")

    configure_style()
    blue = "#0072B2"
    orange = "#E69F00"
    green = "#009E73"
    vermillion = "#D55E00"
    grey = "#666666"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)

    # A: retained data coverage.
    ax = axes[0, 0]
    ax.plot(coverage["threshold"], coverage["tielines"], "o-", color=blue, label="Tie-lines")
    ax.set_xlabel("Minimum tie-lines per system–temperature group")
    ax.set_ylabel("Retained tie-lines", color=blue)
    ax.tick_params(axis="y", colors=blue)
    ax2 = ax.twinx()
    ax2.plot(
        coverage["threshold"], coverage["system_temperature_groups"], "s--", color=orange,
        label="System–temperature groups",
    )
    ax2.set_ylabel("Retained groups", color=orange)
    ax2.tick_params(axis="y", colors=orange)
    ax.axvline(args.reference_threshold, color=grey, linestyle=":", linewidth=1.1)
    ax.set_title("A  Data retained by the density criterion", loc="left", fontweight="bold")
    handles = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(handles, [h.get_label() for h in handles], frameon=False, loc="lower left")

    # B: common-test performance with grouped bootstrap confidence intervals.
    ax = axes[0, 1]
    y = threshold_ci["mae"].to_numpy()
    yerr = np.vstack([
        y - threshold_ci["mae_ci_low"].to_numpy(),
        threshold_ci["mae_ci_high"].to_numpy() - y,
    ])
    ax.errorbar(
        threshold_ci["threshold"], y, yerr=yerr, fmt="o-", color=blue,
        ecolor=blue, capsize=2.5, elinewidth=1.0,
    )
    ref_row = threshold_ci[threshold_ci["threshold"] == args.reference_threshold].iloc[0]
    ax.scatter([args.reference_threshold], [ref_row["mae"]], s=48, color=vermillion, zorder=5, label="Paper threshold")
    ax.axvline(args.reference_threshold, color=grey, linestyle=":", linewidth=1.1)
    ax.set_xlabel("Minimum tie-lines per system–temperature group")
    ax.set_ylabel("Composition MAE on the common test set")
    ax.set_title("B  Threshold sensitivity", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")

    # C: location sensitivity for the manuscript threshold.
    ax = axes[1, 0]
    loc6 = location_ci[location_ci["threshold"] == args.reference_threshold].copy()
    x = np.arange(len(loc6))
    y = loc6["mae"].to_numpy()
    yerr = np.vstack([y - loc6["mae_ci_low"], loc6["mae_ci_high"] - y])
    ax.errorbar(x, y, yerr=yerr, fmt="o-", color=green, capsize=2.5, elinewidth=1.0)
    ax.set_xticks(x, loc6["t_bin"].tolist())
    ax.set_xlabel("Normalized phase-path location, $s$")
    ax.set_ylabel("Composition MAE")
    ax.set_title("C  Location sensitivity at threshold 6", loc="left", fontweight="bold")

    # D: location-by-threshold error map.
    ax = axes[1, 1]
    heat = location_ci.pivot(index="threshold", columns="t_bin", values="mae").reindex(
        index=thresholds, columns=T_LABELS
    )
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(np.arange(len(T_LABELS)), T_LABELS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(thresholds)), thresholds)
    ax.set_xlabel("Normalized phase-path location, $s$")
    ax.set_ylabel("Minimum tie-line threshold")
    ax.set_title("D  Error across thresholds and locations", loc="left", fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Composition MAE")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)
    axes[0, 0].spines["top"].set_visible(False)

    # Keep panels B and C fully boxed.  Their top/right spines were previously
    # hidden by the generic publication style, which left the frames open.
    for ax in (axes[0, 1], axes[1, 0]):
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_linewidth(0.8)
        ax.spines["right"].set_linewidth(0.8)

    png = figure_dir / "tieline_threshold_sensitivity.png"
    pdf = figure_dir / "tieline_threshold_sensitivity.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    summary = {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "system-temperature group",
        "reference_threshold": args.reference_threshold,
        "outputs": {
            "threshold_metrics": str(root / "threshold_metrics_with_ci.csv"),
            "location_metrics": str(root / "location_metrics_with_ci.csv"),
            "coverage": str(root / "dataset_threshold_counts.csv"),
            "figure_png": str(png),
            "figure_pdf": str(pdf),
        },
    }
    (root / "analysis_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

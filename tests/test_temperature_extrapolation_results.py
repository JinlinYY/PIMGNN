"""Tests for the organized temperature-extrapolation evidence package."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plot_temperature_encoding_sensitivity import (
    aggregate_across_seeds,
    load_predictions,
    parse_args as parse_analysis_args,
)
from scripts.run_temperature_encoding_sensitivity import (
    PROJECT_ROOT,
    parse_args as parse_training_args,
    resolve_output_root,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPERATURE_ROOT = (
    REPOSITORY_ROOT
    / "experiments"
    / "supporting_information"
    / "s3_additional_evaluation_and_validation"
    / "s3_3_temperature_robustness"
    / "02_temperature_extrapolation"
)
TRUE_COLUMNS = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
PREDICTION_COLUMNS = [f"pred_{column}" for column in TRUE_COLUMNS]


def test_temperature_extrapolation_has_one_unambiguous_public_tree() -> None:
    """Aggregate evidence and each completed seed must have separate locations."""
    legacy = TEMPERATURE_ROOT.parent / "02_encoding_and_tail"
    assert not legacy.exists()
    assert (TEMPERATURE_ROOT / "README.md").is_file()
    assert (TEMPERATURE_ROOT / "figures" / "temperature_extrapolation_robustness.pdf").is_file()
    assert (TEMPERATURE_ROOT / "figures" / "temperature_extrapolation_robustness.png").is_file()

    aggregate = TEMPERATURE_ROOT / "results" / "aggregate"
    expected_aggregate = {
        "analysis_manifest.json",
        "per_seed_distance_metrics.csv",
        "per_seed_overall_metrics.csv",
        "reciprocal_temperature_approximation.json",
        "seed_42_system_temperature_bootstrap_distance_metrics.csv",
        "seed_42_system_temperature_bootstrap_encoding_metrics.csv",
        "seed_42_system_temperature_bootstrap_paired_differences.csv",
        "three_seed_distance_metrics.csv",
        "three_seed_encoding_metrics.csv",
        "three_seed_paired_differences.csv",
    }
    assert {path.name for path in aggregate.iterdir() if path.is_file()} == expected_aggregate

    run_root = TEMPERATURE_ROOT / "results" / "runs"
    assert {path.name for path in run_root.iterdir() if path.is_dir()} == {
        "seed_7",
        "seed_42",
        "seed_2024",
    }
    for seed in (7, 42, 2024):
        seed_root = run_root / f"seed_{seed}"
        manifest = json.loads((seed_root / "experiment_manifest.json").read_text("utf-8"))
        assert manifest["training"]["seed"] == seed
        assert manifest["system_overlap"] == 0
        assert set(path.name for path in (seed_root / "encodings").iterdir()) == {
            "inverse",
            "linear_quadratic",
        }


def test_three_seed_tables_use_standard_deviation_not_ci_labels() -> None:
    """Across-seed uncertainty must be identified as sample standard deviation."""
    aggregate = TEMPERATURE_ROOT / "results" / "aggregate"
    encoding = pd.read_csv(aggregate / "three_seed_encoding_metrics.csv")
    paired = pd.read_csv(aggregate / "three_seed_paired_differences.csv")

    assert list(encoding.columns) == [
        "subset",
        "encoding",
        "n_seeds",
        "n_tielines_min",
        "n_tielines_max",
        "n_groups_min",
        "n_groups_max",
        "mae_mean",
        "mae_sd",
    ]
    assert list(paired.columns) == [
        "subset",
        "n_seeds",
        "inverse_minus_linear_quadratic_mae_mean",
        "mae_difference_sd",
    ]
    extrapolation = encoding[
        (encoding["subset"] == "Extrapolation")
        & (encoding["encoding"] == "linear_quadratic")
    ].iloc[0]
    assert extrapolation["n_seeds"] == 3
    assert extrapolation["n_tielines_min"] == 712
    assert extrapolation["n_tielines_max"] == 712
    assert np.isclose(extrapolation["mae_mean"], 0.08636752015407627)
    assert np.isclose(extrapolation["mae_sd"], 0.00186447771739956)


def test_archived_predictions_match_the_controlled_partition_counts() -> None:
    """Every seed and encoding must retain both pointwise evaluation partitions."""
    run_root = TEMPERATURE_ROOT / "results" / "runs"
    expected = {
        7: (709, 70, 712, 72),
        42: (698, 70, 712, 72),
        2024: (705, 70, 712, 72),
    }
    for seed, counts in expected.items():
        for encoding in ("linear_quadratic", "inverse"):
            encoding_root = run_root / f"seed_{seed}" / "encodings" / encoding
            interpolation = pd.read_csv(encoding_root / "interpolation_predictions.csv")
            extrapolation = pd.read_csv(encoding_root / "extrapolation_predictions.csv")
            assert len(interpolation) == counts[0]
            assert interpolation["system_id"].nunique() == counts[1]
            assert len(extrapolation) == counts[2]
            assert extrapolation["system_id"].nunique() == counts[3]


def test_archived_metrics_are_recomputed_from_the_pointwise_predictions() -> None:
    """Moved result files must retain exact agreement with their summary metrics."""
    run_root = TEMPERATURE_ROOT / "results" / "runs"
    for seed in (7, 42, 2024):
        seed_root = run_root / f"seed_{seed}"
        summary = pd.read_csv(seed_root / "encoding_metrics.csv").set_index("encoding")
        for encoding in ("linear_quadratic", "inverse"):
            encoding_root = seed_root / "encodings" / encoding
            for subset in ("interpolation", "extrapolation"):
                predictions = pd.read_csv(
                    encoding_root / f"{subset}_predictions.csv"
                )
                mae = np.abs(
                    predictions[PREDICTION_COLUMNS].to_numpy(float)
                    - predictions[TRUE_COLUMNS].to_numpy(float)
                ).mean()
                assert np.isclose(mae, summary.loc[encoding, f"{subset}_mae"])


def test_cross_seed_aggregation_depends_only_on_pointwise_evidence(tmp_path: Path) -> None:
    """Derived per-seed summary CSVs must not be required for recomputation."""
    source_runs = TEMPERATURE_ROOT / "results" / "runs"
    minimal_roots = []
    for seed in (7, 42, 2024):
        source_root = source_runs / f"seed_{seed}"
        target_root = tmp_path / f"seed_{seed}"
        target_root.mkdir()
        shutil.copyfile(
            source_root / "experiment_manifest.json",
            target_root / "experiment_manifest.json",
        )
        for encoding in ("linear_quadratic", "inverse"):
            target_encoding = target_root / "encodings" / encoding
            target_encoding.mkdir(parents=True)
            for subset in ("interpolation", "extrapolation"):
                filename = f"{subset}_predictions.csv"
                shutil.copyfile(
                    source_root / "encodings" / encoding / filename,
                    target_encoding / filename,
                )
        minimal_roots.append(target_root)

    overall, distance, paired, per_seed_overall, per_seed_distance = (
        aggregate_across_seeds(minimal_roots)
    )
    assert len(overall) == 4
    assert len(distance) == 8
    assert len(paired) == 6
    assert len(per_seed_overall) == 12
    assert len(per_seed_distance) == 24
    extrapolation = overall[
        (overall["subset"] == "Extrapolation")
        & (overall["encoding"] == "linear_quadratic")
    ].iloc[0]
    assert np.isclose(extrapolation["mae_mean"], 0.08636752015407627)


def test_analysis_manifest_distinguishes_two_uncertainty_conventions() -> None:
    """System bootstrap intervals and across-seed SD must not be conflated."""
    manifest_path = (
        TEMPERATURE_ROOT / "results" / "aggregate" / "analysis_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["completed_seeds"] == [7, 42, 2024]
    assert manifest["reference_seed"] == 42
    assert manifest["partition_ranges_across_completed_seeds"] == {
        "interpolation": {
            "records": [698, 709],
            "system_temperature_groups": [70, 70],
        },
        "extrapolation": {
            "records": [712, 712],
            "system_temperature_groups": [72, 72],
        },
    }
    assert manifest["system_temperature_bootstrap"]["confidence_level"] == 0.95
    assert manifest["across_seed_summary"]["dispersion"] == "sample_standard_deviation"


def test_analysis_cli_and_loader_follow_the_public_layout() -> None:
    """The analysis command must consume archived seed directories directly."""
    results_root = TEMPERATURE_ROOT / "results"
    args = parse_analysis_args([
        "--results-root",
        str(results_root),
        "--completed-seeds",
        "7",
        "42",
        "2024",
    ])
    assert args.results_root == results_root
    assert args.completed_seeds == [7, 42, 2024]

    frames = load_predictions(results_root / "runs" / "seed_42")
    assert set(frames) == {
        ("linear_quadratic", "Interpolation"),
        ("linear_quadratic", "Extrapolation"),
        ("inverse", "Interpolation"),
        ("inverse", "Extrapolation"),
    }
    assert len(frames[("linear_quadratic", "Interpolation")]) == 698
    assert len(frames[("inverse", "Extrapolation")]) == 712


def test_training_cli_resolves_seed_scoped_and_explicit_output_roots() -> None:
    """Training CLI defaults must keep independently completed seeds separate."""
    defaults = parse_training_args(["--seed", "7"])
    assert defaults.encodings == ["linear_quadratic", "inverse"]
    assert resolve_output_root(defaults.output_root, defaults.seed) == (
        PROJECT_ROOT / "outputs" / "temperature_extrapolation" / "runs" / "seed_7"
    ).resolve()

    explicit = parse_training_args([
        "--seed",
        "2024",
        "--encodings",
        "inverse",
        "--output-root",
        "custom-temperature-run",
    ])
    assert explicit.encodings == ["inverse"]
    assert resolve_output_root(explicit.output_root, explicit.seed) == Path(
        "custom-temperature-run"
    ).resolve()

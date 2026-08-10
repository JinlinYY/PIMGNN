"""Tests for the component-2/3 permutation-equivariance audit."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from psmi.permutation_equivariance import (
    cluster_bootstrap_intervals,
    restore_component_23_outputs,
    summarize_permutation_audit,
    swap_component_23_frame,
)
from scripts.analysis.evaluate_component_permutation_equivariance import parse_args


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    REPOSITORY_ROOT
    / "experiments"
    / "supporting_information"
    / "s3_additional_evaluation_and_validation"
    / "s3_10_component_permutation_equivariance"
)


def test_component_23_swap_preserves_input_label_correspondence() -> None:
    """Molecular inputs and both phase labels must be permuted together."""
    frame = pd.DataFrame(
        {
            "system_id": [17],
            "smiles1": ["A"],
            "smiles2": ["B"],
            "smiles3": ["C"],
            "T": [298.15],
            "P": [1.0],
            "t": [0.4],
            "Ex1": [0.1],
            "Ex2": [0.2],
            "Ex3": [0.7],
            "Rx1": [0.6],
            "Rx2": [0.3],
            "Rx3": [0.1],
        }
    )

    swapped = swap_component_23_frame(frame)

    assert swapped.loc[0, ["smiles1", "smiles2", "smiles3"]].tolist() == [
        "A",
        "C",
        "B",
    ]
    assert swapped.loc[0, ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]].tolist() == [
        0.1,
        0.7,
        0.2,
        0.6,
        0.1,
        0.3,
    ]
    assert swapped.loc[0, ["T", "P", "t", "system_id"]].tolist() == [
        298.15,
        1.0,
        0.4,
        17.0,
    ]
    assert swapped.loc[0, "aug_swap23"] == 1


def test_component_23_output_permutation_is_self_inverse() -> None:
    """Restoring an exchanged output twice must recover the original order."""
    values = np.arange(18, dtype=np.float64).reshape(3, 6)
    restored = restore_component_23_outputs(restore_component_23_outputs(values))
    np.testing.assert_array_equal(restored, values)


def test_summary_reports_zero_gap_for_exactly_equivariant_predictions() -> None:
    """An exactly equivariant synthetic model must have a zero consistency gap."""
    y_true = np.array(
        [
            [0.1, 0.2, 0.7, 0.6, 0.3, 0.1],
            [0.3, 0.4, 0.3, 0.2, 0.5, 0.3],
        ],
        dtype=np.float64,
    )
    prediction = y_true + 0.01
    predictive, equivariance = summarize_permutation_audit(
        y_true,
        prediction,
        prediction.copy(),
    )

    assert set(predictive["evaluation"]) == {
        "original_ordering",
        "components_2_3_swapped",
    }
    assert np.allclose(equivariance["mae"], 0.0)
    assert np.allclose(equivariance["rmse"], 0.0)
    assert np.allclose(equivariance["p95_absolute_error"], 0.0)


def test_cluster_bootstrap_is_deterministic_and_uses_systems() -> None:
    """Bootstrap intervals must be reproducible and identify the cluster unit."""
    system_ids = np.repeat([101, 102, 103], 2)
    y_true = np.linspace(0.05, 0.95, 36).reshape(6, 6)
    original = y_true + 0.02
    swapped_restored = original + np.linspace(-0.01, 0.01, 36).reshape(6, 6)

    first = cluster_bootstrap_intervals(
        system_ids,
        y_true,
        original,
        swapped_restored,
        n_resamples=50,
        seed=9,
    )
    second = cluster_bootstrap_intervals(
        system_ids,
        y_true,
        original,
        swapped_restored,
        n_resamples=50,
        seed=9,
    )

    pd.testing.assert_frame_equal(first, second)
    assert set(first["resampling_unit"]) == {"system_id"}
    assert set(first["n_systems"]) == {3}
    assert set(first["n_resamples"]) == {50}
    assert np.isfinite(first["estimate"]).all()
    assert (first["ci_lower"] <= first["ci_upper"]).all()


def test_archived_audit_matches_the_registered_figure_2a_partition() -> None:
    """The distributed evidence bundle must retain its test-set and metric identity."""
    predictive = pd.read_csv(EXPERIMENT_ROOT / "results" / "predictive_metrics.csv")
    equivariance = pd.read_csv(EXPERIMENT_ROOT / "results" / "equivariance_metrics.csv")
    paired = pd.read_csv(EXPERIMENT_ROOT / "results" / "paired_predictions.csv")

    original_extract = predictive.loc[
        (predictive["evaluation"] == "original_ordering")
        & (predictive["phase"] == "extract")
    ].iloc[0]
    original_raffinate = predictive.loc[
        (predictive["evaluation"] == "original_ordering")
        & (predictive["phase"] == "raffinate")
    ].iloc[0]
    overall_equivariance = equivariance.loc[equivariance["phase"] == "overall"].iloc[0]

    assert len(paired) == 803
    assert paired["system_id"].nunique() == 78
    assert np.isclose(original_extract["mae"], 0.03705783, atol=1e-7)
    assert np.isclose(original_raffinate["mae"], 0.03176129, atol=1e-7)
    assert np.isclose(overall_equivariance["mae"], 0.01098669, atol=1e-7)
    assert np.allclose(paired["true_swapped_Ex2"], paired["Ex3"])
    assert np.allclose(paired["true_swapped_Ex3"], paired["Ex2"])
    assert np.allclose(paired["true_swapped_Rx2"], paired["Rx3"])
    assert np.allclose(paired["true_swapped_Rx3"], paired["Rx2"])
    pdf_figure = EXPERIMENT_ROOT / "figures" / "component_23_permutation_equivariance.pdf"
    png_figure = EXPERIMENT_ROOT / "figures" / "component_23_permutation_equivariance.png"
    assert pdf_figure.stat().st_size > 10_000
    assert png_figure.stat().st_size > 10_000


def test_audit_cli_parses_reproducibility_options(monkeypatch) -> None:
    """The public CLI must expose checkpoint selection and bootstrap controls."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_component_permutation_equivariance.py",
            "--run-id",
            "registered_run",
            "--device",
            "cpu",
            "--bootstrap-resamples",
            "250",
            "--bootstrap-seed",
            "11",
        ],
    )
    args = parse_args()
    assert args.run_id == "registered_run"
    assert args.device == "cpu"
    assert args.bootstrap_resamples == 250
    assert args.bootstrap_seed == 11

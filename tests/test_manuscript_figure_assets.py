"""Regression tests for the reference manuscript figure assets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGED_PUBLIC_ROOT = PROJECT_ROOT / "public_release" / "PSMI-public"
PUBLIC_ROOT = STAGED_PUBLIC_ROOT if STAGED_PUBLIC_ROOT.is_dir() else PROJECT_ROOT


def _phase_metrics(table: pd.DataFrame, phase: str) -> tuple[float, float, float]:
    true = table[[f"{phase}x1", f"{phase}x2", f"{phase}x3"]].to_numpy(float).ravel()
    pred = table[
        [f"pred_{phase}x1", f"pred_{phase}x2", f"pred_{phase}x3"]
    ].to_numpy(float).ravel()
    residual = pred - true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    r2 = float(1.0 - np.sum(residual**2) / np.sum((true - np.mean(true)) ** 2))
    return mae, rmse, r2


def test_figure2d_metrics_match_the_manuscript_panels() -> None:
    """Representative-system source files must reproduce the printed metrics."""
    base = (
        PUBLIC_ROOT
        / "experiments"
        / "section_3_results"
        / "3_1_lle_prediction"
        / "main_benchmark"
    )
    cases = {
        22: (
            base / "data" / "figure_2d_system_22_source_predictions.csv",
            (0.0146, 0.0179, 0.9961),
            (0.0219, 0.0306, 0.9915),
        ),
        826: (
            base / "data" / "figure_2d_system_826_source_predictions.csv",
            (0.0261, 0.0367, 0.9830),
            (0.0168, 0.0258, 0.9952),
        ),
    }
    for system_id, (source, expected_e, expected_r) in cases.items():
        table = pd.read_csv(source)
        table = table.loc[table["system_id"] == system_id]
        assert len(table) > 0
        np.testing.assert_allclose(_phase_metrics(table, "E"), expected_e, atol=5e-5)
        np.testing.assert_allclose(_phase_metrics(table, "R"), expected_r, atol=5e-5)
        assert (
            base / "figures" / f"figure_2d_system_{system_id}.png"
        ).stat().st_size > 0


def test_imported_checkpoint_and_saliency_identity() -> None:
    """The organized assets must retain their verified file identities."""
    checkpoint = (
        PUBLIC_ROOT
        / "experiments"
        / "section_3_results"
        / "3_1_lle_prediction"
        / "main_benchmark"
        / "models"
        / "figure_2a_psmi"
        / "best_model.pt"
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    assert digest == "72C432BC7FD48CB44B52402AD01393BA5A8B47737FA1E286658928670C10F380"

    saliency = pd.read_csv(
        PUBLIC_ROOT
        / "experiments"
        / "section_3_results"
        / "3_2_molecular_interaction_mechanisms"
        / "data"
        / "global_saliency"
        / "mix_edge_feature_importance_grad.csv"
    )
    assert saliency.iloc[0]["name"] == "Aromatic Rings Product"
    assert saliency.iloc[1]["name"] == "Fraction SP3 Product"


def test_supplementary_error_figures_have_source_data() -> None:
    """Figures S1-S3 must be paired with their pointwise prediction source."""
    experiment = (
        PUBLIC_ROOT
        / "experiments"
        / "supporting_information"
        / "s3_additional_evaluation_and_validation"
        / "s3_1_prediction_error_analysis"
    )
    assert (experiment / "results" / "test_pointwise_predictions.csv").stat().st_size > 0
    for name in (
        "figure_s1_category_error_distributions.png",
        "figure_s2_bland_altman.png",
        "figure_s3_residual_distributions.png",
    ):
        assert (experiment / "figures" / name).stat().st_size > 0

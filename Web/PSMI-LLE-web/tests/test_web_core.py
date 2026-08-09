"""Smoke tests for Web configuration, schemas, plotting, and checkpoint loading."""

from __future__ import annotations

import numpy as np

from backend import config
from backend.models import ModelPredictor
from backend.schemas.request import PredictRequest
from backend.utils.plot_generator import generate_ternary_plot


def test_request_uses_atmospheric_pressure_by_default() -> None:
    request = PredictRequest(
        smiles1="O",
        smiles2="CCO",
        smiles3="CCCCCC",
        temperature=298.15,
    )
    assert request.pressure == config.DEFAULT_PRESSURE_KPA


def test_plot_generator_returns_png_data_url() -> None:
    t_grid = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    extract = np.array([[0.7, 0.2, 0.1], [0.6, 0.25, 0.15], [0.5, 0.3, 0.2]])
    raffinate = np.array([[0.1, 0.7, 0.2], [0.15, 0.65, 0.2], [0.2, 0.6, 0.2]])
    result = generate_ternary_plot(
        t_grid,
        extract,
        raffinate,
        temperature=298.15,
        pressure=101.325,
        labels=["A", "B", "C"],
        tie_lines_count=3,
    )
    assert result.startswith("data:image/png;base64,")


def test_legacy_checkpoint_loads_through_psmi_compatibility_layer() -> None:
    predictor = ModelPredictor(device="cpu")
    assert predictor.model is not None
    assert predictor.temperature_scaler.std > 0
    assert predictor.model.scalar_dim == 2
    assert predictor.pressure_supported is False

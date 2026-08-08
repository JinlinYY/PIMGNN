"""Tests for the organized per-component GNN comparison baseline."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from psmi_baselines.common.data import (
    GraphCache,
    LLEGNNDataset,
    augment_component_23,
    gnn_collate_fn,
    smiles_to_graph,
)
from psmi_baselines.common.metrics import collect_preds
from psmi_baselines.common.model import build_torch_model
from psmi_baselines.common.format_summary_table import build_formatted_table
from psmi_baselines.common.utils import Scaler


@pytest.fixture()
def one_sample_dataset() -> LLEGNNDataset:
    frame = pd.DataFrame(
        [
            {
                "smiles1": "O",
                "smiles2": "CCO",
                "smiles3": "CCCCCC",
                "T": 298.15,
                "t": 0.5,
                "Ex1": 0.6,
                "Ex2": 0.3,
                "Ex3": 0.1,
                "Rx1": 0.1,
                "Rx2": 0.2,
                "Rx3": 0.7,
            }
        ]
    )
    return LLEGNNDataset(frame, Scaler(mean=298.15, std=1.0), GraphCache())


def _small_gnn():
    return build_torch_model(
        "gnn",
        in_dim=6146,
        fp_bits=2048,
        hidden=64,
        dropout=0.1,
        GNN_NODE_DIM=11,
        GNN_HIDDEN=32,
        GNN_LAYERS=2,
        GNN_MLP=32,
        GNN_SCALAR_DIM=2,
    )


def test_smiles_to_graph_returns_symmetric_adjacency() -> None:
    nodes, adjacency = smiles_to_graph("CCO")
    assert nodes.shape == (3, 11)
    assert adjacency.shape == (3, 3)
    assert (adjacency == adjacency.T).all()


def test_baseline_permutation_can_be_restricted_to_training(
    one_sample_dataset: LLEGNNDataset,
) -> None:
    """Validation and test frames must remain unaugmented for fair comparison."""
    frame = one_sample_dataset.df
    assert len(augment_component_23(frame, enabled=True)) == 2
    held_out = augment_component_23(frame, enabled=False)
    assert len(held_out) == 1
    assert int(held_out.iloc[0]["aug_swap23"]) == 0


def test_gnn_forward_preserves_phase_simplexes(one_sample_dataset: LLEGNNDataset) -> None:
    batch = next(iter(DataLoader(one_sample_dataset, batch_size=1, collate_fn=gnn_collate_fn)))
    prediction = _small_gnn()(*batch[:4])
    assert prediction.shape == (1, 6)
    torch.testing.assert_close(prediction[:, :3].sum(dim=1), torch.ones(1))
    torch.testing.assert_close(prediction[:, 3:].sum(dim=1), torch.ones(1))


def test_metrics_collector_accepts_gnn_batches(one_sample_dataset: LLEGNNDataset) -> None:
    loader = DataLoader(one_sample_dataset, batch_size=1, collate_fn=gnn_collate_fn)
    targets, predictions = collect_preds(_small_gnn(), loader, "cpu")
    assert targets.shape == predictions.shape == (1, 6)


def test_summary_formatter_uses_english_headers_and_model_names() -> None:
    summary = pd.DataFrame(
        [
            {"Model": "gnn", "Metric": "MAE_E", "Mean": 0.1, "Std": 0.01},
            {"Model": "gnn", "Metric": "MAE", "Mean": 0.2, "Std": 0.02},
        ]
    )
    formatted = build_formatted_table(summary)
    assert formatted.loc[0, "Model"] == "GNN"
    assert formatted.loc[0, "E_phase_MAE"] == "0.1000(0.0100)"
    assert formatted.loc[0, "Overall_MAE"] == "0.2000(0.0200)"

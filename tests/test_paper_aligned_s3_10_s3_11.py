"""Tests for the paper-aligned component-ordering and temperature experiments."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from psmi.identity import (  # noqa: E402
    add_chemical_system_identity,
    merge_nearby_temperature_levels,
)


S3_ROOT = (
    REPOSITORY_ROOT
    / "experiments"
    / "supporting_information"
    / "s3_additional_evaluation_and_validation"
)
PERMUTATION_ROOT = S3_ROOT / "s3_10_component_permutation_equivariance"
EXTRAPOLATION_ROOT = S3_ROOT / "s3_11_conditional_same_system_temperature_extrapolation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_s3_11_runner():
    path = (
        REPOSITORY_ROOT
        / "scripts"
        / "experiments"
        / "run_same_system_temperature_extrapolation.py"
    )
    specification = importlib.util.spec_from_file_location("psmi_s3_11_runner", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_existing_s3_3_experiment_keeps_its_public_path() -> None:
    """Adding S3.11 must not rename or replace the existing S3.3.2 package."""
    temperature_root = S3_ROOT / "s3_3_temperature_robustness"
    assert (temperature_root / "02_encoding_and_tail").is_dir()
    assert not (temperature_root / "02_temperature_extrapolation").exists()


def test_chemical_identity_is_independent_of_component_order() -> None:
    """Permuting the three component positions must preserve system identity."""
    frame = pd.DataFrame(
        {
            "smiles1": ["O", "CCO"],
            "smiles2": ["CCO", "CC"],
            "smiles3": ["CC", "O"],
        }
    )
    identified = add_chemical_system_identity(frame)
    assert identified["chemical_system_signature"].nunique() == 1
    assert identified["chemical_system_id"].nunique() == 1


def test_nominal_temperature_merge_respects_the_full_cluster_span() -> None:
    """Only nominal levels within the configured within-system span are merged."""
    frame = pd.DataFrame(
        {
            "chemical_system_signature": ["system-a"] * 3,
            "T": [298.15, 298.20, 298.40],
        }
    )
    merged = merge_nearby_temperature_levels(frame, tolerance_K=0.1)
    assert np.allclose(merged["T_original"], frame["T"])
    assert merged["T"].nunique() == 2
    assert np.isclose(merged.loc[0, "T"], 298.175)
    assert np.isclose(merged.loc[1, "T"], 298.175)
    assert np.isclose(merged.loc[2, "T"], 298.40)


def test_extreme_temperature_split_is_group_disjoint() -> None:
    """The lowest and highest target temperatures must not enter training."""
    runner = _load_s3_11_runner()
    frame = pd.DataFrame(
        {
            "chemical_system_id": [1] * 6 + [2] * 2 + [3] * 2,
            "T": [280.0, 280.0, 300.0, 300.0, 320.0, 320.0, 290.0, 290.0, 305.0, 305.0],
        }
    )
    train, validation, test, _, manifest = runner.build_extreme_temperature_split(
        frame,
        seed=42,
        validation_fraction=0.5,
        min_temperatures=3,
        system_column="chemical_system_id",
    )
    train_keys = set(zip(train["chemical_system_id"], train["T"]))
    test_keys = set(zip(test["chemical_system_id"], test["T"]))
    assert train_keys.isdisjoint(test_keys)
    assert set(validation["chemical_system_id"]).isdisjoint(
        set(test["chemical_system_id"])
    )
    assert manifest["n_target_systems"] == 1
    assert manifest["n_target_test_groups"] == 2
    assert manifest["n_test_rows_original"] == 4


def test_archived_predictions_recompute_table_s17_summary() -> None:
    """The archived pointwise table must regenerate the distributed summary."""
    runner = _load_s3_11_runner()
    predictions = pd.read_csv(EXTRAPOLATION_ROOT / "results" / "predictions.csv")
    expected = pd.read_csv(EXTRAPOLATION_ROOT / "results" / "summary.csv")
    actual, _ = runner.summarize_predictions(
        predictions, system_column="chemical_system_id"
    )
    order = ["scope", "method"]
    expected = expected.sort_values(order).reset_index(drop=True)
    actual = actual.sort_values(order).reset_index(drop=True)
    assert actual[order].equals(expected[order])
    numeric = [
        "n_tie_lines",
        "n_system_temperature_groups",
        "temperature_gap_median_K",
        "mae",
        "rmse",
        "r2",
        "median_tie_angle_deg",
        "p90_tie_angle_deg",
    ]
    assert np.allclose(actual[numeric], expected[numeric], equal_nan=True)


def test_reference_checkpoint_loader_and_cli_contract() -> None:
    """The default command must evaluate the public checkpoint without training."""
    runner = _load_s3_11_runner()
    arguments = runner.parse_args(["--device", "cpu", "--split-only"])
    assert arguments.split_only is True
    assert arguments.train_from_scratch is False
    assert arguments.checkpoint_dir.name == "reference_checkpoint"
    model, temperature_scaler, pressure_scaler = runner.load_existing_run(
        arguments.checkpoint_dir, "cpu"
    )
    assert type(model).__name__ == "LLEGraphNet"
    assert model.scalar_dim == 3
    assert np.isclose(temperature_scaler.mean, 302.4490661621094)
    assert np.isclose(temperature_scaler.std, 9.739729891286622)
    assert np.isclose(pressure_scaler.mean, 101.32499694824219)
    import torch

    checkpoint = torch.load(
        arguments.checkpoint_dir / "best_model.pt", map_location="cpu"
    )
    assert checkpoint["provenance"]["dataset_path"] is None


def test_evaluation_manifest_tracks_all_frozen_inputs() -> None:
    """Checkpoint, dataset, and split digests must resolve to public files."""
    manifest = json.loads(
        (EXTRAPOLATION_ROOT / "results" / "evaluation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for key in ("checkpoint", "dataset", "split_manifest"):
        artifact = REPOSITORY_ROOT / manifest[key]["path"]
        assert artifact.is_file()
        assert _sha256(artifact) == manifest[key]["sha256"]


def test_s3_10_maps_to_the_final_supporting_information() -> None:
    """The component-ordering audit must map to Tables S15-S16."""
    readme = (PERMUTATION_ROOT / "README.md").read_text(encoding="utf-8")
    assert "S3.10 Sensitivity to Component Ordering" in readme
    assert "Table S15" in readme
    assert "Table S16" in readme


def test_s3_11_distributes_the_table_s17_evidence_and_checkpoint() -> None:
    """The public S3.11 package must reproduce the manuscript-scale evidence."""
    summary = pd.read_csv(EXTRAPOLATION_ROOT / "results" / "summary.csv")
    overall = summary[
        (summary["scope"] == "overall") & (summary["method"] == "PSMI")
    ].iloc[0]
    assert overall["n_tie_lines"] == 623
    assert overall["n_system_temperature_groups"] == 64
    assert np.isclose(overall["mae"], 0.04158497762977967)
    assert np.isclose(overall["rmse"], 0.06425202111088764)
    assert np.isclose(overall["r2"], 0.909608542470772)
    assert np.isclose(overall["median_tie_angle_deg"], 3.7571889719295166)

    split_manifest = json.loads(
        (EXTRAPOLATION_ROOT / "splits" / "split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert split_manifest["n_target_systems"] == 32
    assert split_manifest["n_target_test_groups"] == 64
    assert split_manifest["n_test_rows_original"] == 623
    assert split_manifest["n_synthetic_temperature_interpolation_rows"] == 0
    assert split_manifest["n_target_systems_with_one_interior_temperature"] == 30

    checkpoint = EXTRAPOLATION_ROOT / "models" / "reference_checkpoint" / "best_model.pt"
    assert _sha256(checkpoint) == (
        "b3cc6f5ee5bd5d7a533af938f856afc58920b81f9d3f3a034be4afded5c0fc52"
    )
    assert (EXTRAPOLATION_ROOT / "figures" / "figure_s7.png").is_file()
    assert (EXTRAPOLATION_ROOT / "figures" / "figure_s7.pdf").is_file()


def test_experiment_catalog_indexes_s3_10_and_s3_11() -> None:
    """The public catalog must follow the final SI numbering and terminology."""
    catalog = json.loads(
        (REPOSITORY_ROOT / "experiments" / "experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    entries = {entry["title"]: entry for entry in catalog["experiments"]}
    assert "Sensitivity to Component Ordering" in entries
    assert "Conditional Same-System Temperature Extrapolation" in entries
    assert entries["Sensitivity to Component Ordering"]["paper_sections"] == [
        "SI Section S3.10",
        "Tables S15-S16",
    ]
    assert entries["Conditional Same-System Temperature Extrapolation"][
        "paper_sections"
    ] == ["SI Section S3.11", "Table S17", "Figure S7"]
    assert entries["PSMI-LLE Web Application"]["paper_sections"] == [
        "SI Section S4",
        "Figure S8",
    ]
    assert entries["Dataset Construction and Distribution"]["paper_sections"] == [
        "Main text Section 2.1",
        "SI Section S5",
        "Tables S18-S19",
        "Figure S9",
    ]
    assert entries["Phase-Diagram System Classification"]["paper_sections"] == [
        "SI Section S6",
        "Table S20",
    ]


def test_dataset_compatibility_table_uses_final_s18_label() -> None:
    """The retained historical filename must contain final-SI labels."""
    table = pd.read_csv(
        REPOSITORY_ROOT
        / "experiments"
        / "supporting_information"
        / "s5_dataset_construction_and_distribution"
        / "results"
        / "table_s15_counts.csv"
    )
    assert set(table["paper_item"]) == {"Table S18"}

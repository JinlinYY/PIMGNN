"""Test explicit architecture contracts used by supported PSMI profiles."""

from __future__ import annotations

import pytest
import pandas as pd
import torch
import yaml
from types import SimpleNamespace


def test_fusion_mode_aliases_preserve_published_tf_behavior() -> None:
    """The published ``tf`` label selected concatenation, not Transformer fusion."""
    from psmi.model import normalize_fusion_mode

    assert normalize_fusion_mode("concat") == "concat"
    assert normalize_fusion_mode("tf") == "concat"
    assert normalize_fusion_mode("transformer") == "transformer"
    assert normalize_fusion_mode("s3_set") == "s3_set"


def test_unknown_fusion_mode_is_rejected() -> None:
    """Unknown labels must not silently fall through to a different architecture."""
    from psmi.model import normalize_fusion_mode

    with pytest.raises(ValueError, match="Unsupported fusion mode"):
        normalize_fusion_mode("set-transformer")


def test_command_line_config_overrides_are_typed_and_validated(tmp_path) -> None:
    """Multi-seed runs must override seeds and paths without editing YAML files."""
    from psmi.configuration import apply_config_overrides

    config = SimpleNamespace(SEED=0, OUT_DIR="", FREEZE_BACKBONE=False)
    values = apply_config_overrides(
        ["SEED=43", "OUT_DIR=results/example", "FREEZE_BACKBONE=true"],
        config_module=config,
        project_root=tmp_path,
    )
    assert values["SEED"] == 43
    assert values["FREEZE_BACKBONE"] is True
    assert config.OUT_DIR == str((tmp_path / "results" / "example").resolve())
    with pytest.raises(KeyError, match="Unknown runtime configuration key"):
        apply_config_overrides(["UNKNOWN_KEY=1"], config_module=config, project_root=tmp_path)


def test_sample_major_mixture_node_layout_matches_batched_graph_indices() -> None:
    """Each sample's three component embeddings must be adjacent in a graph batch."""
    from psmi.model import stack_mixture_node_embeddings

    e1 = torch.tensor([[11.0], [21.0]])
    e2 = torch.tensor([[12.0], [22.0]])
    e3 = torch.tensor([[13.0], [23.0]])

    actual = stack_mixture_node_embeddings(e1, e2, e3, layout="sample_major")

    assert actual[:, 0].tolist() == [11.0, 12.0, 13.0, 21.0, 22.0, 23.0]


def test_component_major_layout_remains_available_for_declared_checkpoints() -> None:
    """Checkpoints can select their declared component-major layout."""
    from psmi.model import stack_mixture_node_embeddings

    e1 = torch.tensor([[11.0], [21.0]])
    e2 = torch.tensor([[12.0], [22.0]])
    e3 = torch.tensor([[13.0], [23.0]])

    actual = stack_mixture_node_embeddings(e1, e2, e3, layout="component_major")

    assert actual[:, 0].tolist() == [11.0, 21.0, 12.0, 22.0, 13.0, 23.0]


@pytest.mark.parametrize(
    ("scalar_dim", "expected"),
    [(2, [1.25, 0.4]), (3, [1.25, 0.4, -0.5])],
)
def test_condition_scalar_contract_is_explicit(
    scalar_dim: int,
    expected: list[float],
) -> None:
    """Benchmark and expanded-data profiles must declare their scalar inputs."""
    from psmi.data import condition_scalar_values

    actual = condition_scalar_values(1.25, 0.4, -0.5, scalar_dim=scalar_dim)

    assert actual.tolist() == pytest.approx(expected)


def test_model_scalar_dimension_matches_profile() -> None:
    """A two-scalar published checkpoint must not require a pressure column."""
    from psmi.model import LLEGraphNet

    benchmark_model = LLEGraphNet(
        use_mix_graph=False,
        use_fg=False,
        use_interaction=False,
        scalar_dim=2,
        fusion_mode="concat",
    )
    expanded_model = LLEGraphNet(
        use_mix_graph=False,
        use_fg=False,
        use_interaction=False,
        scalar_dim=3,
        fusion_mode="concat",
    )

    assert benchmark_model.backbone[0].in_features + 1 == expanded_model.backbone[0].in_features


def test_layered_config_resolves_includes_and_project_paths(tmp_path) -> None:
    """Canonical experiment profiles must be composable and path-stable."""
    from psmi.configuration import load_config_file

    data_dir = tmp_path / "data"
    experiment_dir = tmp_path / "experiments"
    data_dir.mkdir()
    experiment_dir.mkdir()
    (data_dir / "primary.yaml").write_text(
        yaml.safe_dump({"EXCEL_PATH": "datasets/primary.xlsx", "SCALAR_DIM": 2}),
        encoding="utf-8",
    )
    profile = experiment_dir / "main.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "include": ["../data/primary.yaml"],
                "OUT_DIR": "results/main_benchmark/stage1",
                "EPOCHS": 300,
            }
        ),
        encoding="utf-8",
    )

    values = load_config_file(profile, project_root=tmp_path)

    assert values["EXCEL_PATH"] == str(tmp_path / "datasets" / "primary.xlsx")
    assert values["OUT_DIR"] == str(tmp_path / "results" / "main_benchmark" / "stage1")
    assert values["SCALAR_DIM"] == 2
    assert values["EPOCHS"] == 300


def test_sample_major_profile_names_s3_scope_without_claiming_full_equivariance() -> None:
    """The public profile must name the component embedding rather than the full model."""
    from psmi.configuration import load_config_file

    profile = load_config_file("configs/model/psmi_sample_major.yaml")
    assert profile["USE_S3_COMPONENT_EMBEDDING"] is True
    assert "S3_EQUIVARIANT" not in profile


def test_manifest_split_is_disjoint_and_complete(tmp_path) -> None:
    """A published split manifest must define every system exactly once."""
    from psmi.data import split_by_manifest

    frame = pd.DataFrame(
        {
            "system_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "value": list(range(8)),
        }
    )
    manifest = tmp_path / "split.json"
    manifest.write_text(
        """{
  "partitions": {
    "train": [1, 2],
    "validation": [3],
    "test": [4]
  }
}
""",
        encoding="utf-8",
    )

    train, validation, test = split_by_manifest(frame, manifest)

    assert set(train["system_id"]) == {1, 2}
    assert set(validation["system_id"]) == {3}
    assert set(test["system_id"]) == {4}


def test_manifest_split_rejects_overlapping_systems(tmp_path) -> None:
    """System leakage must fail before training starts."""
    from psmi.data import split_by_manifest

    frame = pd.DataFrame({"system_id": [1, 2, 3]})
    manifest = tmp_path / "split.json"
    manifest.write_text(
        '{"partitions":{"train":[1,2],"validation":[2],"test":[3]}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap"):
        split_by_manifest(frame, manifest)


def test_checkpoint_provenance_records_architecture_and_hashes(tmp_path) -> None:
    """Released checkpoints must identify their scientific contract and inputs."""
    from psmi.checkpoints import build_checkpoint_provenance
    from psmi.model import LLEGraphNet

    dataset = tmp_path / "dataset.xlsx"
    split = tmp_path / "split.json"
    dataset.write_bytes(b"dataset")
    split.write_text("{}", encoding="utf-8")
    model = LLEGraphNet(
        use_mix_graph=True,
        use_fg=False,
        scalar_dim=2,
        fusion_mode="concat",
        mixture_node_layout="sample_major",
    )

    provenance = build_checkpoint_provenance(
        model,
        dataset_path=dataset,
        split_manifest_path=split,
        source_checkpoint_path=dataset,
        training_contract={"augmentation_scope": "training_only"},
    )

    assert provenance["architecture"]["fusion_mode"] == "concat"
    assert provenance["architecture"]["mixture_node_layout"] == "sample_major"
    assert provenance["architecture"]["scalar_features"] == ["temperature", "phase_path"]
    assert len(provenance["dataset_sha256"]) == 64
    assert len(provenance["split_manifest_sha256"]) == 64
    assert provenance["source_checkpoint_sha256"] == provenance["dataset_sha256"]
    assert provenance["training_contract"]["augmentation_scope"] == "training_only"


def test_curve_sweep_honors_two_scalar_checkpoint_contract(monkeypatch) -> None:
    """Visualization must not append pressure to a two-scalar benchmark model."""
    import numpy as np
    import torch

    from psmi import config as runtime_config
    from psmi.utils import Scaler
    from psmi.viz import predict_curve_sweep

    class TwoScalarFingerprintModel(torch.nn.Module):
        scalar_dim = 2

        def forward(self, features):
            assert features.shape[1] == 3 * runtime_config.FP_BITS + self.scalar_dim
            output = torch.full((features.shape[0], 6), 1.0 / 3.0)
            return output.to(features.device)

    monkeypatch.setattr(runtime_config, "USE_GRAPH", False)
    model = TwoScalarFingerprintModel()
    temperature_scaler = Scaler(mean=298.15, std=1.0)
    sweep, extract, raffinate = predict_curve_sweep(
        model,
        temperature_scaler,
        "CCO",
        "O",
        "CCCC",
        298.15,
        n_sweep=3,
    )

    assert sweep.shape == (3,)
    np.testing.assert_allclose(extract.sum(axis=1), 1.0)
    np.testing.assert_allclose(raffinate.sum(axis=1), 1.0)


def test_component_permutation_augmentation_swaps_inputs_and_targets() -> None:
    """The declared 2/3 augmentation must preserve component-label correspondence."""
    from psmi.data import augment_component_23

    frame = pd.DataFrame(
        {
            "system_id": [7],
            "T": [298.15],
            "t": [0.25],
            "smiles1": ["A"],
            "smiles2": ["B"],
            "smiles3": ["C"],
            "Ex1": [0.1],
            "Ex2": [0.2],
            "Ex3": [0.7],
            "Rx1": [0.6],
            "Rx2": [0.3],
            "Rx3": [0.1],
        }
    )

    augmented = augment_component_23(frame, enabled=True)
    assert len(augmented) == 2
    original = augmented[augmented["aug_swap23"] == 0].iloc[0]
    swapped = augmented[augmented["aug_swap23"] == 1].iloc[0]
    assert original["smiles2"] == "B" and original["smiles3"] == "C"
    assert swapped["smiles2"] == "C" and swapped["smiles3"] == "B"
    assert swapped["Ex2"] == pytest.approx(0.7)
    assert swapped["Ex3"] == pytest.approx(0.2)
    assert swapped["Rx2"] == pytest.approx(0.1)
    assert swapped["Rx3"] == pytest.approx(0.3)
    assert swapped["t"] == pytest.approx(original["t"])


def test_expanded_release_manifest_has_declared_system_counts() -> None:
    """Expanded fine-tuning must use the frozen 575/72/72 split."""
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "datasets" / "splits" / "expanded_lle_system_split.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    partitions = payload["partitions"]
    assert {name: len(ids) for name, ids in partitions.items()} == {
        "train": 575,
        "validation": 72,
        "test": 72,
    }
    sets = [set(partitions[name]) for name in ("train", "validation", "test")]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert len(set.union(*sets)) == 719


def test_new_pressure_channel_keeps_target_dataset_scaler() -> None:
    """A 2D-to-3D checkpoint transfer must not restore a constant source P scaler."""
    from psmi.train import should_restore_pressure_scaler

    assert should_restore_pressure_scaler(()) is True
    assert should_restore_pressure_scaler(
        ("backbone.0.weight: appended zero input column (3330->3331)",)
    ) is False

"""Tests for checkpoint reproduction helpers."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from psmi import config as C
from psmi import reproduction as reproduction_module
from psmi.reproduction import prepare_saved_checkpoint, sha256_file, verify_checkpoint_inputs
from psmi.utils import Scaler


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """The audit digest must match the standard SHA-256 implementation."""
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"PSMI reproducibility")
    assert sha256_file(sample) == hashlib.sha256(sample.read_bytes()).hexdigest()


def test_checkpoint_input_verification_accepts_portable_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only content hashes, not reference absolute paths, define input identity."""
    dataset = tmp_path / "dataset.xlsx"
    split = tmp_path / "split.json"
    dataset.write_bytes(b"dataset")
    split.write_text(json.dumps({"partitions": {}}), encoding="utf-8")
    monkeypatch.setattr(C, "EXCEL_PATH", str(dataset))
    monkeypatch.setattr(C, "SPLIT_STRATEGY", "manifest")
    monkeypatch.setattr(C, "SPLIT_MANIFEST_PATH", str(split))
    checkpoint = {
        "provenance": {
            "dataset_path": "portable/example/dataset.xlsx",
            "dataset_sha256": sha256_file(dataset),
            "split_manifest_path": "portable/example/split.json",
            "split_manifest_sha256": sha256_file(split),
        }
    }
    result = verify_checkpoint_inputs(checkpoint)
    assert result["verified"] is True


def test_checkpoint_input_verification_rejects_modified_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed dataset must stop strict checkpoint reproduction."""
    dataset = tmp_path / "dataset.xlsx"
    dataset.write_bytes(b"modified")
    monkeypatch.setattr(C, "EXCEL_PATH", str(dataset))
    monkeypatch.setattr(C, "SPLIT_STRATEGY", "random")
    checkpoint = {
        "provenance": {
            "dataset_path": "portable/example/dataset.xlsx",
            "dataset_sha256": hashlib.sha256(b"original").hexdigest(),
        }
    }
    with pytest.raises(ValueError, match="dataset"):
        verify_checkpoint_inputs(checkpoint)


def test_prepare_saved_checkpoint_restores_the_inference_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared checkpoint path must return the model, split, and saved scalers."""
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint placeholder")
    checkpoint = {"epoch": 12, "state_dict": {}}
    frame = pd.DataFrame({"system_id": [1, 2, 3]})
    train_frame = frame.iloc[[0]].copy()
    validation_frame = frame.iloc[[1]].copy()
    test_frame = frame.iloc[[2]].copy()
    model = torch.nn.Linear(1, 1)
    temperature_scaler = Scaler(mean=298.15, std=12.0)

    monkeypatch.setattr(C, "USE_FG", False)
    monkeypatch.setattr(reproduction_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(reproduction_module.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        reproduction_module,
        "verify_checkpoint_inputs",
        lambda *args, **kwargs: {"verified": True},
    )
    monkeypatch.setattr(reproduction_module, "build_model", lambda: model)
    monkeypatch.setattr(
        reproduction_module,
        "load_state_dict_compat",
        lambda *args, **kwargs: ["test_adaptation"],
    )
    monkeypatch.setattr(
        reproduction_module,
        "load_and_prepare_excel",
        lambda *args, **kwargs: (frame, {}),
    )
    monkeypatch.setattr(
        reproduction_module,
        "_split_prepared_frame",
        lambda current: (train_frame, validation_frame, test_frame),
    )
    monkeypatch.setattr(
        reproduction_module,
        "_checkpoint_scalers",
        lambda *args, **kwargs: (temperature_scaler, None, "checkpoint"),
    )

    context = prepare_saved_checkpoint(checkpoint_path, device="cpu")

    assert context.checkpoint_path == checkpoint_path.resolve()
    assert context.model is model
    assert context.device == "cpu"
    assert context.test_frame.equals(test_frame)
    assert context.temperature_scaler is temperature_scaler
    assert context.scaler_source == "checkpoint"
    assert context.input_verification == {"verified": True}
    assert context.compatibility_adaptations == ["test_adaptation"]

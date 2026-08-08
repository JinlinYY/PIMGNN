"""Tests for checkpoint reproduction helpers."""

import hashlib
import json
from pathlib import Path

import pytest

from psmi import config as C
from psmi.reproduction import sha256_file, verify_checkpoint_inputs


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """The audit digest must match the standard SHA-256 implementation."""
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"PSMI reproducibility")
    assert sha256_file(sample) == hashlib.sha256(sample.read_bytes()).hexdigest()


def test_checkpoint_input_verification_accepts_portable_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only content hashes, not archived absolute paths, define input identity."""
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

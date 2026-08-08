"""Auditable separation of training and post-hoc NRTL parameter stores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd


def canonical_system_id(value: Any) -> str:
    """Return the stable JSON key used for a numeric system identifier."""
    return str(int(value))


def dataframe_system_ids(frame: pd.DataFrame) -> set[str]:
    if "system_id" not in frame.columns:
        raise KeyError("Data frame does not contain a system_id column")
    return {canonical_system_id(value) for value in frame["system_id"].unique()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parameter_payload(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload.get("params"), dict):
        raise ValueError(f"NRTL parameter file has no params mapping: {path}")
    return payload


def validate_training_parameter_file(
    path: Path,
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_path: Path | None = None,
) -> Dict[str, Any]:
    """Fail if a training-loss parameter file contains held-out systems."""
    path = Path(path)
    payload = load_parameter_payload(path)
    meta = payload.get("meta", {})
    role = meta.get("role")
    if role != "training_loss":
        raise ValueError(
            "Training loss requires role=training_loss so validation/test systems "
            f"cannot be exposed; got role={role!r}"
        )
    dataset_audit = _validate_dataset_provenance(meta, dataset_path)

    parameter_ids = set(payload["params"])
    train_ids = dataframe_system_ids(train_df)
    validation_ids = dataframe_system_ids(val_df)
    test_ids = dataframe_system_ids(test_df)
    held_out_ids = validation_ids | test_ids
    held_out_overlap = parameter_ids & held_out_ids
    unexpected_ids = parameter_ids - train_ids
    missing_train_ids = train_ids - parameter_ids
    if held_out_overlap or unexpected_ids:
        raise ValueError(
            "Training NRTL parameter file contains validation/test systems or "
            f"other non-training identifiers: {sorted(held_out_overlap | unexpected_ids)}"
        )
    if missing_train_ids:
        raise ValueError(
            "Training NRTL parameter file does not cover all training systems: "
            f"{sorted(missing_train_ids)}"
        )

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "role": role,
        "training_system_count": len(train_ids),
        "parameter_system_count": len(parameter_ids),
        "missing_training_parameter_system_ids": sorted(missing_train_ids),
        "unexpected_parameter_system_ids": sorted(unexpected_ids),
        "validation_parameter_overlap": sorted(parameter_ids & validation_ids),
        "test_parameter_overlap": sorted(parameter_ids & test_ids),
        **dataset_audit,
    }


def validate_evaluation_parameter_file(
    path: Path,
    *,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_path: Path | None = None,
) -> Dict[str, Any]:
    """Require an explicitly post-hoc store with complete held-out coverage."""
    path = Path(path)
    payload = load_parameter_payload(path)
    meta = payload.get("meta", {})
    role = meta.get("role")
    if role != "posthoc_evaluation":
        raise ValueError(
            "Post-hoc evaluation requires role=posthoc_evaluation; "
            f"got role={role!r}"
        )
    dataset_audit = _validate_dataset_provenance(meta, dataset_path)

    parameter_ids = set(payload["params"])
    validation_ids = dataframe_system_ids(val_df)
    test_ids = dataframe_system_ids(test_df)
    missing_validation = validation_ids - parameter_ids
    missing_test = test_ids - parameter_ids
    if missing_validation or missing_test:
        raise ValueError(
            "Post-hoc NRTL parameter file does not cover all validation/test "
            f"systems: {sorted(missing_validation | missing_test)}"
        )

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "role": role,
        "parameter_system_count": len(parameter_ids),
        "validation_system_count": len(validation_ids),
        "test_system_count": len(test_ids),
        "missing_validation_parameter_system_ids": sorted(missing_validation),
        "missing_test_parameter_system_ids": sorted(missing_test),
        **dataset_audit,
    }


def _validate_dataset_provenance(
    metadata: Mapping[str, Any], dataset_path: Path | None
) -> Dict[str, Any]:
    """Compare the source workbook hash when the parameter store declares one."""
    expected_hash = metadata.get("dataset_sha256")
    if dataset_path is None or expected_hash is None:
        return {
            "dataset_path": None if dataset_path is None else str(Path(dataset_path).resolve()),
            "dataset_sha256": None,
            "parameter_dataset_sha256": expected_hash,
        }
    dataset_path = Path(dataset_path)
    actual_hash = sha256_file(dataset_path)
    if actual_hash.lower() != str(expected_hash).lower():
        raise ValueError(
            "NRTL parameter file was fitted from a different dataset: "
            f"expected SHA-256 {expected_hash}, got {actual_hash} for {dataset_path}"
        )
    return {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": actual_hash,
        "parameter_dataset_sha256": expected_hash,
    }


def write_usage_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write an audit record for a completed training run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary_path.replace(path)

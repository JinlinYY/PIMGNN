"""Evaluate saved PSMI checkpoints without entering the training loop."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from . import config as C
from .checkpoints import load_state_dict_compat
from .data import (
    load_and_prepare_excel,
    split_by_manifest,
    split_by_system,
    stratified_split_by_system,
)
from .metrics import compute_metrics
from .predict import predict_pointwise_df_raw
from .train import build_model
from .utils import Scaler, set_seed, temperature_scalar_value
from .viz import parity_plots


TRUE_COLUMNS = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
PREDICTION_COLUMNS = [
    "pred_Ex1",
    "pred_Ex2",
    "pred_Ex3",
    "pred_Rx1",
    "pred_Rx2",
    "pred_Rx3",
]


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_prepared_frame(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the configured split without data augmentation."""
    strategy = str(getattr(C, "SPLIT_STRATEGY", "random")).lower()
    common = {
        "train_ratio": float(getattr(C, "TRAIN_RATIO", 0.8)),
        "val_ratio": float(getattr(C, "VAL_RATIO", 0.1)),
        "seed": int(getattr(C, "SEED", 42)),
    }
    if strategy == "random":
        return split_by_system(frame, **common)
    if strategy == "stratified":
        return stratified_split_by_system(
            frame,
            **common,
            n_bins=int(getattr(C, "STRATIFIED_N_BINS", 3)),
            min_bin_size=int(getattr(C, "STRATIFIED_MIN_BIN_SIZE", 5)),
        )
    if strategy == "manifest":
        manifest_path = str(getattr(C, "SPLIT_MANIFEST_PATH", "")).strip()
        if not manifest_path:
            raise ValueError("SPLIT_MANIFEST_PATH is required for manifest evaluation")
        return split_by_manifest(frame, manifest_path)
    raise ValueError(f"Unsupported SPLIT_STRATEGY: {strategy!r}")


def _verify_provenance_file(
    provenance: Mapping[str, Any],
    label: str,
    current_path: Path,
) -> Dict[str, Any]:
    """Verify one current input against the digest stored in a checkpoint."""
    expected = provenance.get(f"{label}_sha256")
    actual = sha256_file(current_path)
    verified = bool(expected) and str(expected).lower() == actual.lower()
    return {
        "current_path": str(current_path.resolve()),
        "checkpoint_path": provenance.get(f"{label}_path"),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "verified": verified,
    }


def verify_checkpoint_inputs(
    checkpoint: Mapping[str, Any],
    *,
    require_hash_match: bool = True,
) -> Dict[str, Any]:
    """Check the configured dataset and split manifest against checkpoint hashes."""
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        if require_hash_match:
            raise ValueError("Checkpoint has no provenance mapping")
        return {"verified": False, "reason": "checkpoint_has_no_provenance"}

    dataset_path = Path(str(getattr(C, "EXCEL_PATH"))).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Configured dataset does not exist: {dataset_path}")
    checks: Dict[str, Any] = {
        "dataset": _verify_provenance_file(provenance, "dataset", dataset_path)
    }

    if str(getattr(C, "SPLIT_STRATEGY", "")).lower() == "manifest":
        split_path = Path(str(getattr(C, "SPLIT_MANIFEST_PATH"))).resolve()
        if not split_path.is_file():
            raise FileNotFoundError(f"Configured split manifest does not exist: {split_path}")
        checks["split_manifest"] = _verify_provenance_file(
            provenance, "split_manifest", split_path
        )

    checks["verified"] = all(
        bool(value.get("verified"))
        for value in checks.values()
        if isinstance(value, Mapping)
    )
    if require_hash_match and not checks["verified"]:
        failed = [
            name
            for name, value in checks.items()
            if isinstance(value, Mapping) and not value.get("verified")
        ]
        raise ValueError(
            "Checkpoint input hash verification failed for: " + ", ".join(failed)
        )
    return checks


def _checkpoint_scalers(
    checkpoint: Mapping[str, Any],
    train_frame: pd.DataFrame,
    *,
    allow_derived: bool = False,
) -> Tuple[Scaler, Optional[Scaler], str]:
    """Restore feature scalers required for exact checkpoint inference."""
    if "T_mean" in checkpoint and "T_std" in checkpoint:
        temperature = Scaler(
            mean=float(checkpoint["T_mean"]),
            std=float(checkpoint["T_std"]),
        )
        source = "checkpoint"
    elif allow_derived:
        temperature_values = temperature_scalar_value(
            train_frame["T"].to_numpy(dtype=np.float32),
            mode=str(getattr(C, "TEMPERATURE_ENCODING", "linear_quadratic")),
            reference_k=float(getattr(C, "TEMPERATURE_REFERENCE_K", 500.0)),
        )
        temperature = Scaler.fit(temperature_values)
        source = "derived_from_verified_training_partition"
    else:
        raise ValueError("Checkpoint does not contain the temperature scaler")
    pressure = None
    if "P_mean" in checkpoint and "P_std" in checkpoint:
        pressure = Scaler(
            mean=float(checkpoint["P_mean"]),
            std=float(checkpoint["P_std"]),
        )
    if int(getattr(C, "SCALAR_DIM", 3)) == 3 and pressure is None:
        if not allow_derived:
            raise ValueError("Three-scalar checkpoint evaluation requires a pressure scaler")
        pressure = Scaler.fit(train_frame["P"].to_numpy(dtype=np.float32))
        source += "+derived_pressure"
    return temperature, pressure, source


def _load_functional_group_corpus(
    model: torch.nn.Module,
    checkpoint_path: Path,
    corpus_path: Path | str | None = None,
) -> Path:
    """Attach the vocabulary saved beside a functional-group checkpoint."""
    corpus_path = (
        Path(corpus_path).resolve()
        if corpus_path is not None
        else checkpoint_path.parent / "fg_corpus.json"
    )
    if not corpus_path.is_file():
        raise FileNotFoundError(
            "Functional-group model requires its saved fg_corpus.json: "
            f"{corpus_path}"
        )
    with corpus_path.open("r", encoding="utf-8") as stream:
        corpus = json.load(stream)
    if not isinstance(corpus, list) or not corpus:
        raise ValueError(f"Invalid functional-group corpus: {corpus_path}")
    setattr(model, "fg_corpus", corpus)
    return corpus_path


def _compare_reference_predictions(
    reproduced: pd.DataFrame,
    reference_path: Path,
) -> Dict[str, Any]:
    """Compare newly inferred values with the reference pointwise predictions."""
    if not reference_path.is_file():
        return {
            "available": False,
            "reference_path": str(reference_path.resolve()),
        }
    reference = pd.read_csv(reference_path)
    if len(reference) != len(reproduced):
        return {
            "available": True,
            "comparable": False,
            "reference_path": str(reference_path.resolve()),
            "reference_rows": int(len(reference)),
            "reproduced_rows": int(len(reproduced)),
        }
    missing = [name for name in PREDICTION_COLUMNS if name not in reference.columns]
    if missing:
        return {
            "available": True,
            "comparable": False,
            "reference_path": str(reference_path.resolve()),
            "missing_prediction_columns": missing,
        }
    current_values = reproduced[PREDICTION_COLUMNS].to_numpy(dtype=np.float64)
    reference_values = reference[PREDICTION_COLUMNS].to_numpy(dtype=np.float64)
    difference = np.abs(current_values - reference_values)
    return {
        "available": True,
        "comparable": True,
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": sha256_file(reference_path),
        "maximum_absolute_prediction_difference": float(difference.max()),
        "mean_absolute_prediction_difference": float(difference.mean()),
    }


@dataclass
class SavedCheckpointContext:
    """Model, fixed partitions, and scalers restored for inference-only analyses."""

    checkpoint_path: Path
    checkpoint: Mapping[str, Any]
    model: torch.nn.Module
    device: str
    raw_frame: pd.DataFrame
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    temperature_scaler: Scaler
    pressure_scaler: Optional[Scaler]
    scaler_source: str
    input_verification: Dict[str, Any]
    functional_group_corpus_path: Optional[Path]
    compatibility_adaptations: Sequence[str]


def prepare_saved_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: Optional[str] = None,
    functional_group_corpus: Path | str | None = None,
    allow_derived_scalers: bool = False,
    require_hash_match: bool = True,
) -> SavedCheckpointContext:
    """Restore a registered checkpoint and its fixed data context without training."""
    resolved_checkpoint_path = Path(checkpoint_path).resolve()
    if not resolved_checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_checkpoint_path}")

    set_seed(int(getattr(C, "SEED", 42)))
    selected_device = str(device or getattr(C, "DEVICE", "cpu"))
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")

    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    input_verification = verify_checkpoint_inputs(
        checkpoint,
        require_hash_match=require_hash_match,
    )

    model = build_model()
    adaptations = load_state_dict_compat(model, checkpoint, strict=True)
    model = model.to(selected_device)
    model.eval()
    corpus_path = None
    if bool(getattr(C, "USE_FG", False)):
        corpus_path = _load_functional_group_corpus(
            model,
            resolved_checkpoint_path,
            functional_group_corpus,
        )

    raw_frame, _ = load_and_prepare_excel(
        getattr(C, "EXCEL_PATH"),
        int(getattr(C, "MIN_POINTS_PER_GROUP", 6)),
        False,
    )
    train_frame, validation_frame, test_frame = _split_prepared_frame(raw_frame)
    temperature_scaler, pressure_scaler, scaler_source = _checkpoint_scalers(
        checkpoint,
        train_frame,
        allow_derived=allow_derived_scalers,
    )
    return SavedCheckpointContext(
        checkpoint_path=resolved_checkpoint_path,
        checkpoint=checkpoint,
        model=model,
        device=selected_device,
        raw_frame=raw_frame,
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        temperature_scaler=temperature_scaler,
        pressure_scaler=pressure_scaler,
        scaler_source=scaler_source,
        input_verification=input_verification,
        functional_group_corpus_path=corpus_path,
        compatibility_adaptations=list(adaptations),
    )


def evaluate_saved_checkpoint(
    checkpoint_path: Path | str,
    output_dir: Path | str,
    *,
    device: Optional[str] = None,
    reference_predictions: Path | str | None = None,
    functional_group_corpus: Path | str | None = None,
    allow_derived_scalers: bool = False,
    require_hash_match: bool = True,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Run deterministic checkpoint inference and save an auditable result bundle."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    context = prepare_saved_checkpoint(
        checkpoint_path,
        device=device,
        functional_group_corpus=functional_group_corpus,
        allow_derived_scalers=allow_derived_scalers,
        require_hash_match=require_hash_match,
    )

    predictions = predict_pointwise_df_raw(
        context.model,
        context.temperature_scaler,
        context.test_frame,
        device=context.device,
        P_scaler=context.pressure_scaler,
    )
    metrics = compute_metrics(
        predictions[TRUE_COLUMNS].to_numpy(dtype=np.float64),
        predictions[PREDICTION_COLUMNS].to_numpy(dtype=np.float64),
    )
    prediction_path = output_dir / "test_predictions.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    if reference_predictions is None:
        reference_path = context.checkpoint_path.parent / "test_df_raw_pointwise_predictions.csv"
    else:
        reference_path = Path(reference_predictions).resolve()
    reference_comparison = _compare_reference_predictions(predictions, reference_path)

    if make_plots:
        parity_plots(predictions, str(output_dir))

    report: Dict[str, Any] = {
        "schema_version": 1,
        "execution_mode": "checkpoint_evaluation",
        "seed": int(getattr(C, "SEED", 42)),
        "device": context.device,
        "checkpoint": {
            "path": str(context.checkpoint_path),
            "sha256": sha256_file(context.checkpoint_path),
            "epoch": context.checkpoint.get(
                "epoch", context.checkpoint.get("best_epoch")
            ),
            "compatibility_adaptations": list(context.compatibility_adaptations),
            "provenance": context.checkpoint.get("provenance"),
        },
        "inputs": context.input_verification,
        "functional_group_corpus": (
            {
                "path": str(context.functional_group_corpus_path),
                "sha256": sha256_file(context.functional_group_corpus_path),
            }
            if context.functional_group_corpus_path is not None
            else None
        ),
        "feature_scalers": {
            "source": context.scaler_source,
            "temperature_mean": float(context.temperature_scaler.mean),
            "temperature_standard_deviation": float(context.temperature_scaler.std),
            "pressure_mean": (
                float(context.pressure_scaler.mean)
                if context.pressure_scaler is not None
                else None
            ),
            "pressure_standard_deviation": (
                float(context.pressure_scaler.std)
                if context.pressure_scaler is not None
                else None
            ),
        },
        "partition_counts": {
            "all_rows": int(len(context.raw_frame)),
            "train_rows_before_augmentation": int(len(context.train_frame)),
            "validation_rows": int(len(context.validation_frame)),
            "test_rows": int(len(context.test_frame)),
            "train_systems": int(context.train_frame["system_id"].nunique()),
            "validation_systems": int(context.validation_frame["system_id"].nunique()),
            "test_systems": int(context.test_frame["system_id"].nunique()),
        },
        "metrics": metrics,
        "prediction_file": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
        },
        "reference_prediction_comparison": reference_comparison,
    }
    report_path = output_dir / "reproduction_report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def summarize_reports(report_paths: Sequence[Path | str]) -> pd.DataFrame:
    """Convert checkpoint reports into one manuscript-ready metric table."""
    rows = []
    for raw_path in report_paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        metrics = report["metrics"]
        rows.append(
            {
                "run_id": path.parent.name,
                "seed": report["seed"],
                "checkpoint_sha256": report["checkpoint"]["sha256"],
                "mae_extract": metrics["mae_E"],
                "rmse_extract": metrics["rmse_E"],
                "r2_extract": metrics["r2_E"],
                "mae_raffinate": metrics["mae_R"],
                "rmse_raffinate": metrics["rmse_R"],
                "r2_raffinate": metrics["r2_R"],
                "mae_overall": metrics["mae"],
                "rmse_overall": metrics["rmse"],
                "r2_overall": metrics["r2"],
                "input_hashes_verified": bool(report["inputs"].get("verified")),
            }
        )
    return pd.DataFrame(rows)

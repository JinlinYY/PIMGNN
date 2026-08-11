#!/usr/bin/env python
"""Run the controlled PSMI temperature-extrapolation experiment.

The experiment deliberately withholds temperatures outside a central training
interval.  Chemical systems are disjoint across the central train/validation/
interpolation splits and the outer-range extrapolation set.  Both encodings
retain identical feature dimensionality and model parameter count.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.psmi import config as C  # noqa: E402
from src.psmi.data import load_and_prepare_excel, stratified_split_by_system  # noqa: E402
from src.psmi.predict import predict_pointwise_df_raw  # noqa: E402
from src.psmi.train import train_or_load  # noqa: E402
from src.psmi.utils import set_seed  # noqa: E402


TRUE_COLS = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
PRED_COLS = [f"pred_{column}" for column in TRUE_COLS]
GROUP_COLS = ["system_id", "T"]
DISTANCE_BINS = [-1e-12, 5.0, 10.0, 20.0, np.inf]
DISTANCE_LABELS = ["0-5 K", "5-10 K", "10-20 K", ">20 K"]


def parse_encodings(text: str) -> List[str]:
    aliases = {
        "poly": "linear_quadratic",
        "polynomial": "linear_quadratic",
        "linear_quadratic": "linear_quadratic",
        "inverse": "inverse",
        "reciprocal": "inverse",
        "1/t": "inverse",
    }
    values = []
    for item in text.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise argparse.ArgumentTypeError(f"unsupported encoding: {item}")
        value = aliases[key]
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one encoding is required")
    return values


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def data_stats(df: pd.DataFrame) -> Dict[str, float]:
    counts = df.groupby(GROUP_COLS, sort=False).size()
    return {
        "rows_tielines": int(len(df)),
        "systems": int(df["system_id"].nunique()),
        "system_temperature_groups": int(len(counts)),
        "temperature_min_k": float(df["T"].min()),
        "temperature_max_k": float(df["T"].max()),
        "temperature_unique": int(df["T"].nunique()),
    }


def regression_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = df[TRUE_COLS].to_numpy(dtype=np.float64)
    p = df[PRED_COLS].to_numpy(dtype=np.float64)
    error = p - y
    denom = float(np.square(y - y.mean()).sum())
    row_mae = np.abs(error).mean(axis=1)
    group_mae = (
        pd.DataFrame({
            "system_id": df["system_id"].to_numpy(),
            "T": df["T"].to_numpy(),
            "mae": row_mae,
        })
        .groupby(GROUP_COLS, sort=False)["mae"]
        .mean()
    )
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / denom) if denom > 0 else float("nan"),
        "mae_extract": float(np.abs(error[:, :3]).mean()),
        "mae_raffinate": float(np.abs(error[:, 3:]).mean()),
        "group_balanced_mae": float(group_mae.mean()),
    }


def add_extrapolation_metadata(
    df: pd.DataFrame, temperature_low_k: float, temperature_high_k: float
) -> pd.DataFrame:
    out = df.copy()
    temperature = out["T"].to_numpy(dtype=np.float64)
    out["distance_from_training_range_k"] = np.maximum.reduce([
        temperature_low_k - temperature,
        temperature - temperature_high_k,
        np.zeros(len(out), dtype=np.float64),
    ])
    out["temperature_side"] = np.where(
        temperature < temperature_low_k,
        "below",
        np.where(temperature > temperature_high_k, "above", "inside"),
    )
    out["distance_bin"] = pd.cut(
        out["distance_from_training_range_k"],
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
        include_lowest=True,
        right=True,
    )
    return out


def distance_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for label in DISTANCE_LABELS:
        subset = predictions[predictions["distance_bin"] == label]
        if subset.empty:
            continue
        records.append({
            "distance_bin": label,
            "n_tielines": int(len(subset)),
            "n_system_temperature_groups": int(subset.groupby(GROUP_COLS).ngroups),
            **regression_metrics(subset),
        })
    return pd.DataFrame(records)


def side_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for side in ["below", "above"]:
        subset = predictions[predictions["temperature_side"] == side]
        if subset.empty:
            continue
        records.append({
            "temperature_side": side,
            "n_tielines": int(len(subset)),
            "n_system_temperature_groups": int(subset.groupby(GROUP_COLS).ngroups),
            **regression_metrics(subset),
        })
    return pd.DataFrame(records)


def configure_training(args, encoding: str, output_dir: Path) -> None:
    C.OUT_DIR = str(output_dir)
    C.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    C.LOAD_CKPT_PATH = ""
    C.TEMPERATURE_ENCODING = encoding
    C.TEMPERATURE_REFERENCE_K = float(args.temperature_reference_k)
    C.EPOCHS = int(args.epochs)
    C.LR = float(args.learning_rate)
    C.BATCH_SIZE_GRAPH = int(args.batch_size)
    C.PRED_BATCH_SIZE_GRAPH = int(args.batch_size)
    C.NUM_WORKERS_GRAPH = int(args.num_workers)
    C.USE_MECH_LOSS = False
    C.USE_PHYSICS_FINETUNE = False
    C.FREEZE_BACKBONE = False
    C.USE_EARLY_STOP = True
    C.EARLY_STOP_PATIENCE = int(args.patience)
    C.EARLY_STOP_METRIC = "rmse"
    C.EARLY_STOP_MIN_DELTA = float(args.min_delta)
    C.EVAL_EVERY = 1
    C.PLOT_EVERY = max(int(args.epochs) + 1, 1000)
    C.COMPUTE_FINAL_PHYSICS_METRICS = False


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse training and output options for one random seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx",
    )
    parser.add_argument(
        "--encodings",
        type=parse_encodings,
        default=parse_encodings("linear_quadratic,inverse"),
    )
    parser.add_argument("--temperature-low-k", type=float, default=293.15)
    parser.add_argument("--temperature-high-k", type=float, default=323.20)
    parser.add_argument("--temperature-reference-k", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=2e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory for one completed seed. The default is "
            "outputs/temperature_extrapolation/runs/seed_<SEED>."
        ),
    )
    return parser.parse_args(argv)


def resolve_output_root(output_root: Path | None, seed: int) -> Path:
    """Resolve the directory contract for one completed training seed."""
    if output_root is None:
        output_root = (
            PROJECT_ROOT
            / "outputs"
            / "temperature_extrapolation"
            / "runs"
            / f"seed_{seed}"
        )
    return output_root.resolve()


def main() -> None:
    args = parse_args()

    args.dataset = args.dataset.resolve()
    args.output_root = resolve_output_root(args.output_root, args.seed)
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    if args.temperature_low_k >= args.temperature_high_k:
        raise ValueError("temperature-low-k must be smaller than temperature-high-k")

    raw, augmented = load_and_prepare_excel(str(args.dataset), 6, True)
    inside_mask_raw = raw["T"].between(
        args.temperature_low_k,
        args.temperature_high_k,
        inclusive="both",
    )
    inside_mask_aug = augmented["T"].between(
        args.temperature_low_k,
        args.temperature_high_k,
        inclusive="both",
    )
    central_raw = raw[inside_mask_raw].copy()
    central_aug = augmented[inside_mask_aug].copy()
    extrapolation_raw = raw[~inside_mask_raw].copy()

    central_train_raw, central_val_raw, interpolation_raw = stratified_split_by_system(
        central_raw,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
        n_bins=3,
        min_bin_size=5,
    )
    train_ids = set(central_train_raw["system_id"].unique())
    val_ids = set(central_val_raw["system_id"].unique())
    interpolation_ids = set(interpolation_raw["system_id"].unique())
    extrapolation_ids = set(extrapolation_raw["system_id"].unique())
    if (
        train_ids & val_ids
        or train_ids & interpolation_ids
        or val_ids & interpolation_ids
        or (train_ids | val_ids | interpolation_ids) & extrapolation_ids
    ):
        raise RuntimeError("system-level overlap detected between experiment partitions")

    train_df = central_aug[central_aug["system_id"].isin(train_ids)].copy()
    val_df = central_raw[central_raw["system_id"].isin(val_ids)].copy()
    interpolation_df = central_raw[central_raw["system_id"].isin(interpolation_ids)].copy()
    extrapolation_df = add_extrapolation_metadata(
        extrapolation_raw,
        args.temperature_low_k,
        args.temperature_high_k,
    )

    manifest = {
        "dataset": str(args.dataset),
        "encodings": args.encodings,
        "temperature_basis": {
            "linear_quadratic": "scalar z-score(T); edge [T/T_ref, (T/T_ref)^2]",
            "inverse": "scalar z-score(T_ref/T); edge [T_ref/T, (T_ref/T)^2]",
            "temperature_reference_k": args.temperature_reference_k,
        },
        "central_training_interval_k": [args.temperature_low_k, args.temperature_high_k],
        "partitions": {
            "train_unaugmented": data_stats(central_train_raw),
            "train_augmented_rows": int(len(train_df)),
            "validation": data_stats(val_df),
            "interpolation_test": data_stats(interpolation_df),
            "extrapolation_test": data_stats(extrapolation_df),
        },
        "system_overlap": 0,
        "training": {
            "seed": args.seed,
            "epochs_max": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "early_stop_min_delta": args.min_delta,
            "component_23_augmentation": True,
            "checkpoint_initialization": "from scratch and identical across encodings",
            "loss": "supervised MSE",
        },
        "software_hardware": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    write_json(args.output_root / "experiment_manifest.json", manifest)
    (
        extrapolation_df.groupby(
            ["distance_bin", "temperature_side"], observed=False
        )
        .size()
        .rename("n_tielines")
        .reset_index()
        .to_csv(
            args.output_root / "extrapolation_bin_counts.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )

    summary_rows = []
    distance_frames = []
    side_frames = []
    for encoding in args.encodings:
        run_dir = args.output_root / "encodings" / encoding
        summary_path = run_dir / "summary.json"
        if summary_path.exists() and not args.force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_rows.append(summary["table_row"])
            distance_frames.append(pd.read_csv(run_dir / "distance_metrics.csv"))
            side_frames.append(pd.read_csv(run_dir / "side_metrics.csv"))
            print(f"[SKIP] {encoding}: completed result found")
            continue

        configure_training(args, encoding, run_dir)
        set_seed(args.seed)
        start = time.perf_counter()
        model, temperature_scaler, _pressure_scaler, history = train_or_load(
            train_df,
            val_df,
            interpolation_df,
        )
        elapsed = time.perf_counter() - start

        interpolation_pred = predict_pointwise_df_raw(
            model, temperature_scaler, interpolation_df, device=C.DEVICE
        )
        extrapolation_pred = predict_pointwise_df_raw(
            model, temperature_scaler, extrapolation_df, device=C.DEVICE
        )
        interpolation_pred.to_csv(
            run_dir / "interpolation_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        extrapolation_pred.to_csv(
            run_dir / "extrapolation_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        distance = distance_metrics(extrapolation_pred)
        distance.insert(0, "encoding", encoding)
        side = side_metrics(extrapolation_pred)
        side.insert(0, "encoding", encoding)
        distance.to_csv(run_dir / "distance_metrics.csv", index=False, encoding="utf-8-sig")
        side.to_csv(run_dir / "side_metrics.csv", index=False, encoding="utf-8-sig")

        best_path = run_dir / "best_metrics.json"
        best_payload = (
            json.loads(best_path.read_text(encoding="utf-8"))
            if best_path.exists()
            else {}
        )
        interpolation_metrics = regression_metrics(interpolation_pred)
        extrapolation_metrics = regression_metrics(extrapolation_pred)
        table_row = {
            "encoding": encoding,
            "best_epoch": int(
                best_payload.get(
                    "best_epoch",
                    history["epoch"][-1] if history["epoch"] else -1,
                )
            ),
            "elapsed_seconds": float(elapsed),
            **{f"interpolation_{k}": v for k, v in interpolation_metrics.items()},
            **{f"extrapolation_{k}": v for k, v in extrapolation_metrics.items()},
        }
        summary = {
            "encoding": encoding,
            "best_epoch": table_row["best_epoch"],
            "elapsed_seconds": elapsed,
            "interpolation_metrics": interpolation_metrics,
            "extrapolation_metrics": extrapolation_metrics,
            "table_row": table_row,
        }
        write_json(summary_path, summary)
        summary_rows.append(table_row)
        distance_frames.append(distance)
        side_frames.append(side)
        print(
            f"[DONE] {encoding}: interpolation MAE={interpolation_metrics['mae']:.6f}, "
            f"extrapolation MAE={extrapolation_metrics['mae']:.6f}, elapsed={elapsed:.1f}s"
        )

        del model, history, interpolation_pred, extrapolation_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(summary_rows).to_csv(
        args.output_root / "encoding_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(distance_frames, ignore_index=True).to_csv(
        args.output_root / "distance_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(side_frames, ignore_index=True).to_csv(
        args.output_root / "side_metrics.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Aggregated results: {args.output_root / 'encoding_metrics.csv'}")


if __name__ == "__main__":
    main()

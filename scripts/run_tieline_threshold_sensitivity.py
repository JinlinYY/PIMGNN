#!/usr/bin/env python
"""Controlled PSMI sensitivity study for the minimum tie-line threshold.

The system split is created once from all groups containing at least three
tie-lines.  Every threshold variant is trained from the same random
initialization and evaluated on the same validation/test groups containing at
least ``common_eval_threshold`` tie-lines.  This prevents changes in test-set
composition from being mistaken for threshold sensitivity.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

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
PRED_COLS = [f"pred_{c}" for c in TRUE_COLS]
GROUP_COLS = ["system_id", "T"]


def parse_thresholds(text: str) -> List[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("thresholds must be positive integers")
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
        "equilibrium_endpoints": int(2 * len(df)),
        "systems": int(df["system_id"].nunique()),
        "system_temperature_groups": int(len(counts)),
        "mean_tielines_per_group": float(counts.mean()),
        "median_tielines_per_group": float(counts.median()),
        "min_tielines_per_group": int(counts.min()),
        "max_tielines_per_group": int(counts.max()),
    }


def regression_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = df[TRUE_COLS].to_numpy(dtype=np.float64)
    p = df[PRED_COLS].to_numpy(dtype=np.float64)
    e = p - y
    row_mae = np.abs(e).mean(axis=1)
    denom = float(np.square(y - y.mean()).sum())
    group_mae = (
        pd.DataFrame({"system_id": df["system_id"].to_numpy(), "T": df["T"].to_numpy(), "mae": row_mae})
        .groupby(GROUP_COLS, sort=False)["mae"]
        .mean()
    )
    true_vec = y[:, :3] - y[:, 3:]
    pred_vec = p[:, :3] - p[:, 3:]
    vec_l2 = np.linalg.norm(pred_vec - true_vec, axis=1)
    endpoint_l2 = 0.5 * (
        np.linalg.norm(e[:, :3], axis=1) + np.linalg.norm(e[:, 3:], axis=1)
    )
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt(np.square(e).mean())),
        "r2": float(1.0 - np.square(e).sum() / denom) if denom > 0 else float("nan"),
        "mae_extract": float(np.abs(e[:, :3]).mean()),
        "mae_raffinate": float(np.abs(e[:, 3:]).mean()),
        "group_balanced_mae": float(group_mae.mean()),
        "endpoint_l2_mean": float(endpoint_l2.mean()),
        "tieline_vector_l2_mean": float(vec_l2.mean()),
    }


def location_metrics(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1e-12, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-12]
    labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    work = df.copy()
    work["t_bin"] = pd.cut(work["t"], bins=bins, labels=labels, include_lowest=True)
    records = []
    for label in labels:
        sub = work[work["t_bin"].astype(str) == label]
        if sub.empty:
            continue
        metrics = regression_metrics(sub)
        records.append({"t_bin": label, "n_tielines": int(len(sub)), **metrics})
    return pd.DataFrame.from_records(records)


def configure_training(args, out_dir: Path) -> None:
    C.OUT_DIR = str(out_dir)
    C.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    C.LOAD_CKPT_PATH = ""  # no cross-threshold leakage from a previous split
    C.EPOCHS = int(args.epochs)
    C.LR = float(args.learning_rate)
    C.BATCH_SIZE_GRAPH = int(args.batch_size)
    C.PRED_BATCH_SIZE_GRAPH = int(args.batch_size)
    C.NUM_WORKERS_GRAPH = int(args.num_workers)
    C.USE_MECH_LOSS = False
    C.USE_PHYSICS_FINETUNE = False
    C.FREEZE_BACKBONE = False
    C.USE_EARLY_STOP = not args.no_early_stop
    C.EARLY_STOP_PATIENCE = int(args.patience)
    C.EARLY_STOP_METRIC = "rmse"
    C.EARLY_STOP_MIN_DELTA = float(args.min_delta)
    C.EVAL_EVERY = 1
    C.PLOT_EVERY = max(int(args.epochs) + 1, 1000)
    C.COMPUTE_FINAL_PHYSICS_METRICS = False


def aggregate_run_summaries(root: Path, thresholds: Iterable[int], seed: int) -> None:
    rows = []
    loc_frames = []
    for threshold in thresholds:
        run_dir = root / "runs" / f"seed{seed}" / f"threshold_{threshold:02d}"
        summary_path = run_dir / "summary.json"
        location_path = run_dir / "location_metrics.csv"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "seed": seed,
            "threshold": threshold,
            **{f"train_{k}": v for k, v in payload["train_stats"].items()},
            **{f"test_{k}": v for k, v in payload["test_metrics"].items()},
            "best_epoch": payload["best_epoch"],
            "elapsed_seconds": payload["elapsed_seconds"],
        })
        if location_path.exists():
            loc = pd.read_csv(location_path)
            loc.insert(0, "threshold", threshold)
            loc.insert(0, "seed", seed)
            loc_frames.append(loc)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["seed", "threshold"]).to_csv(
        root / "threshold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    if loc_frames:
        pd.concat(loc_frames, ignore_index=True).to_csv(
            root / "location_metrics.csv", index=False, encoding="utf-8-sig"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx",
    )
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("3,4,5,6,7,8,9"))
    parser.add_argument("--split-threshold", type=int, default=3)
    parser.add_argument("--common-eval-threshold", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tieline_density_sensitivity",
    )
    args = parser.parse_args()

    args.dataset = args.dataset.resolve()
    args.output_root = args.output_root.resolve()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    if args.common_eval_threshold < max(args.thresholds):
        raise ValueError("common evaluation threshold must be >= every training threshold")

    set_seed(args.seed)
    split_raw, _ = load_and_prepare_excel(str(args.dataset), args.split_threshold, False)
    split_train, split_val, split_test = stratified_split_by_system(
        split_raw,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
        n_bins=3,
        min_bin_size=5,
    )
    train_ids = set(split_train["system_id"].unique())
    val_ids = set(split_val["system_id"].unique())
    test_ids = set(split_test["system_id"].unique())
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("system-level data leakage detected")

    common_raw, _ = load_and_prepare_excel(str(args.dataset), args.common_eval_threshold, False)
    common_val = common_raw[common_raw["system_id"].isin(val_ids)].copy()
    common_test = common_raw[common_raw["system_id"].isin(test_ids)].copy()
    if common_val.empty or common_test.empty:
        raise RuntimeError("common validation/test subset is empty")

    manifest = {
        "dataset": str(args.dataset),
        "thresholds": args.thresholds,
        "split_threshold": args.split_threshold,
        "common_eval_threshold": args.common_eval_threshold,
        "seed": args.seed,
        "system_split": {
            "train_systems": len(train_ids),
            "validation_systems": len(val_ids),
            "test_systems": len(test_ids),
        },
        "common_validation_stats": data_stats(common_val),
        "common_test_stats": data_stats(common_test),
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "component_23_augmentation": not args.no_augmentation,
            "initialization": "from scratch; identical seed for every threshold",
            "loss": "supervised MSE",
        },
        "software_hardware": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    write_json(args.output_root / "experiment_manifest.json", manifest)

    for threshold in args.thresholds:
        run_dir = args.output_root / "runs" / f"seed{args.seed}" / f"threshold_{threshold:02d}"
        summary_path = run_dir / "summary.json"
        if summary_path.exists() and not args.force:
            print(f"[SKIP] threshold={threshold}: {summary_path} already exists")
            continue

        raw_k, augmented_k = load_and_prepare_excel(
            str(args.dataset), threshold, not args.no_augmentation
        )
        train_source = augmented_k if not args.no_augmentation else raw_k
        train_df = train_source[train_source["system_id"].isin(train_ids)].copy()
        if train_df.empty:
            raise RuntimeError(f"training data empty for threshold {threshold}")

        configure_training(args, run_dir)
        set_seed(args.seed)  # identical initialization and loader RNG per variant
        start = time.perf_counter()
        model, t_scaler, _p_scaler, history = train_or_load(train_df, common_val, common_test)
        elapsed = time.perf_counter() - start

        predictions = predict_pointwise_df_raw(model, t_scaler, common_test, device=C.DEVICE)
        predictions.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
        loc = location_metrics(predictions)
        loc.to_csv(run_dir / "location_metrics.csv", index=False, encoding="utf-8-sig")

        best_metrics_path = run_dir / "best_metrics.json"
        best_payload = (
            json.loads(best_metrics_path.read_text(encoding="utf-8"))
            if best_metrics_path.exists()
            else {}
        )
        summary = {
            "threshold": threshold,
            "seed": args.seed,
            "train_stats": data_stats(raw_k[raw_k["system_id"].isin(train_ids)]),
            "train_rows_after_augmentation": int(len(train_df)),
            "validation_stats": data_stats(common_val),
            "test_stats": data_stats(common_test),
            "test_metrics": regression_metrics(predictions),
            "best_epoch": int(best_payload.get("best_epoch", history["epoch"][-1] if history["epoch"] else -1)),
            "elapsed_seconds": float(elapsed),
        }
        write_json(summary_path, summary)
        print(
            f"[DONE] threshold={threshold} best_epoch={summary['best_epoch']} "
            f"test_MAE={summary['test_metrics']['mae']:.6f} elapsed={elapsed:.1f}s"
        )

        del model, predictions, history
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate_run_summaries(args.output_root, args.thresholds, args.seed)
    print(f"Aggregated results: {args.output_root / 'threshold_metrics.csv'}")


if __name__ == "__main__":
    main()

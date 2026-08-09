"""Aggregate PSMI metrics and checkpoint hashes across fixed benchmark seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_rows(seeds: list[int]) -> list[dict]:
    """Collect selected validation and final test metrics for every stage."""
    rows = []
    for seed in seeds:
        locations = {
            "supervised": PROJECT_ROOT / "results" / "main_benchmark" / "sample_major" / f"seed{seed}" / "stage1_supervised",
            "physics": PROJECT_ROOT / "results" / "main_benchmark" / "sample_major" / f"seed{seed}" / "stage2_physics",
            "expanded": PROJECT_ROOT / "results" / "transfer_evaluation" / "expanded_lle" / "sample_major" / f"seed{seed}",
        }
        for stage, run_dir in locations.items():
            metrics_path = run_dir / "best_metrics.json"
            checkpoint_path = run_dir / "best_model.pt"
            if not metrics_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Incomplete benchmark run: {run_dir}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            validation = metrics["best_val"]
            test = metrics["best_test"]
            rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "best_epoch": int(metrics["best_epoch"]),
                    "validation_rmse": float(validation["rmse"]),
                    "test_mae": float(test["mae"]),
                    "test_rmse": float(test["rmse"]),
                    "test_r2": float(test["r2"]),
                    "test_mu_residual_mae": test.get("mu_res_mae"),
                    "test_tpd_violation_rate": test.get("tpd_viol_rate"),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "run_directory": str(run_dir.relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Compute sample mean and sample standard deviation by training stage."""
    summary = {}
    for stage in ("supervised", "physics", "expanded"):
        stage_rows = [row for row in rows if row["stage"] == stage]
        metrics = {}
        for key in (
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_mu_residual_mae",
            "test_tpd_violation_rate",
        ):
            values = np.asarray(
                [row[key] for row in stage_rows if row[key] is not None], dtype=float
            )
            if values.size:
                metrics[key] = {
                    "mean": float(values.mean()),
                    "sample_standard_deviation": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                }
        summary[stage] = metrics
    return summary


def parse_args() -> argparse.Namespace:
    """Parse seeds and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "summaries" / "multiseed_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    """Write row-level CSV and aggregate JSON artifacts."""
    args = parse_args()
    rows = collect_rows(args.seeds)
    summary = {
        "schema_version": 1,
        "seeds": args.seeds,
        "standard_deviation": "sample (ddof=1)",
        "stages": aggregate(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "runs.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

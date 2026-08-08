"""Aggregate repeated PSMI efficiency benchmark JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    numbers = [float(value) for value in values]
    return statistics.fmean(numbers), statistics.stdev(numbers) if len(numbers) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    payloads: List[Dict[str, Any]] = []
    for path in args.inputs:
        with path.open("r", encoding="utf-8") as stream:
            payloads.append(json.load(stream))
    if not payloads:
        raise ValueError("At least one benchmark result is required")

    devices = {payload["metadata"]["device"] for payload in payloads}
    checkpoints = {payload["metadata"]["checkpoint"] for payload in payloads}
    parameter_counts = {payload["parameters"]["total_parameters"] for payload in payloads}
    if len(devices) != 1 or len(checkpoints) != 1 or len(parameter_counts) != 1:
        raise ValueError("Benchmarks do not describe the same device, checkpoint, and model")

    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for payload in payloads:
        for row in payload["latency"]:
            key = (row["mode"], row["precision"], int(row["batch_size"]))
            grouped.setdefault(key, []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    metrics = [
        "mean_ms_per_batch",
        "median_ms_per_batch",
        "p95_ms_per_batch",
        "mean_ms_per_sample",
        "throughput_samples_per_s",
        "peak_allocated_mib",
    ]
    for (mode, precision, batch_size), rows in sorted(grouped.items()):
        if len(rows) != len(payloads):
            raise ValueError(f"Incomplete repeated measurements for {(mode, precision, batch_size)}")
        result: Dict[str, Any] = {
            "mode": mode,
            "precision": precision,
            "batch_size": batch_size,
            "n_runs": len(rows),
        }
        for metric in metrics:
            mean, sd = mean_sd([row[metric] for row in rows])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
        aggregate_rows.append(result)

    flops = [
        payload["computation"]["profiler_supported_flops_batch1_fp32"]
        for payload in payloads
        if payload["computation"]["profiler_supported_flops_batch1_fp32"] is not None
    ]
    flop_mean, flop_sd = mean_sd(flops)
    metadata = {
        "n_runs": len(payloads),
        "seeds": [payload["metadata"]["seed"] for payload in payloads],
        "device": next(iter(devices)),
        "checkpoint": next(iter(checkpoints)),
        "dataset": payloads[0]["metadata"]["dataset"],
        "pool_records_per_run": payloads[0]["metadata"]["benchmark_pool_records"],
        "torch": payloads[0]["metadata"]["torch"],
        "cuda_runtime": payloads[0]["metadata"]["cuda_runtime"],
    }
    output = {
        "metadata": metadata,
        "parameters": payloads[0]["parameters"],
        "computation": {
            "profiler_supported_flops_batch1_fp32_mean": flop_mean,
            "profiler_supported_flops_batch1_fp32_sd": flop_sd,
            "profiler_supported_gflops_batch1_fp32_mean": flop_mean / 1e9,
            "profiler_supported_gflops_batch1_fp32_sd": flop_sd / 1e9,
            "note": payloads[0]["computation"]["note"],
        },
        "latency": aggregate_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_aggregate.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)

    csv_path = args.output_dir / "latency_aggregate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Saved {json_path.resolve()}")
    print(f"Saved {csv_path.resolve()}")


if __name__ == "__main__":
    main()

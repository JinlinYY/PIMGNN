"""Benchmark PSMI parameter count, computational cost, and inference latency."""
from __future__ import annotations

try:
    from _bootstrap import add_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_src_to_path

add_src_to_path()

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from psmi import config as C
from psmi.checkpoints import load_state_dict_compat
from psmi.data import (
    FunctionalGroupCache,
    GraphCache,
    GraphLLEDataset,
    MixGraphCache,
    collate_graph_batch,
    load_and_prepare_excel,
)
from psmi.train import build_model
from psmi.utils import Scaler, batch_to_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "efficiency_benchmark" / "best_model.pt"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "supporting_information"
    / "s3_additional_evaluation_and_validation"
    / "s3_9_inference_efficiency"
    / "results"
    / "psmi_rtx3090_ti"
)


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_latencies(values_ms: Sequence[float], batch_size: int) -> Dict[str, float]:
    values = [float(v) for v in values_ms]
    mean_ms = float(statistics.fmean(values))
    return {
        "mean_ms_per_batch": mean_ms,
        "std_ms_per_batch": float(statistics.pstdev(values)),
        "median_ms_per_batch": float(statistics.median(values)),
        "p95_ms_per_batch": _percentile(values, 95.0),
        "min_ms_per_batch": min(values),
        "max_ms_per_batch": max(values),
        "mean_ms_per_sample": mean_ms / float(batch_size),
        "throughput_samples_per_s": 1000.0 * float(batch_size) / mean_ms,
    }


def _autocast_context(precision: str):
    if precision == "amp_fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _checkpoint_state(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Scaler, Scaler, Dict[str, Any], float, List[str]]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model()
    adaptations = list(load_state_dict_compat(model, _checkpoint_state(checkpoint)))
    model.to(device).eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started

    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
    t_scaler = Scaler(
        mean=float(checkpoint_dict.get("T_mean", 302.9259948730469)),
        std=float(checkpoint_dict.get("T_std", 10.96979808807373)),
    )
    p_scaler = Scaler(
        mean=float(checkpoint_dict.get("P_mean", 101.325)),
        std=float(checkpoint_dict.get("P_std", 1.0)),
    )
    return model, t_scaler, p_scaler, checkpoint_dict, load_seconds, adaptations


def parameter_statistics(model: torch.nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    state_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    by_module: Dict[str, int] = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        by_module[group] = by_module.get(group, 0) + parameter.numel()
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "nontrainable_parameters": int(total - trainable),
        "buffer_elements": int(buffers),
        "state_size_mib": float(state_bytes / 2**20),
        "parameters_by_top_level_module": dict(
            sorted(by_module.items(), key=lambda item: item[1], reverse=True)
        ),
    }


def resolve_fg_corpus(checkpoint_path: Path, explicit_path: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    try:
        relative = checkpoint_path.resolve().relative_to((PROJECT_ROOT / "models").resolve())
        section = relative.parts[0]
        run = relative.parts[1]
        candidates.append(
            PROJECT_ROOT
            / "experiments"
            / section
            / "runs"
            / run
            / "artifacts"
            / "fg_corpus.json"
        )
    except (ValueError, IndexError):
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Functional-group corpus not found. Pass --fg-corpus with the corpus used by the checkpoint."
    )


def load_corpus(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise TypeError(f"Expected a list in {path}, got {type(value).__name__}")
    return [str(item) for item in value]


def build_dataset(
    frame,
    t_scaler: Scaler,
    p_scaler: Scaler,
    corpus: Sequence[str],
    prebuild_graphs: bool,
) -> GraphLLEDataset:
    graph_cache = GraphCache(
        add_hs=bool(C.GRAPH_ADD_HS),
        add_3d=bool(C.GRAPH_ADD_3D),
        use_gasteiger=bool(C.GRAPH_USE_GASTEIGER),
        max_atoms=int(C.GRAPH_MAX_ATOMS),
    )
    if prebuild_graphs:
        smiles = frame[["smiles1", "smiles2", "smiles3"]].to_numpy().reshape(-1).tolist()
        graph_cache.build_from_smiles(smiles)
    fg_cache = FunctionalGroupCache(
        corpus=list(corpus),
        vocab_size=int(C.FG_TOPK),
        min_freq=int(C.FG_MIN_FREQ),
    )
    return GraphLLEDataset(
        frame,
        t_scaler,
        graph_cache,
        P_scaler=p_scaler,
        mix_cache=MixGraphCache(C),
        fg_cache=fg_cache,
        use_fg=bool(C.USE_FG),
        use_mix_graph=bool(C.USE_MIX_GRAPH),
        scalar_dim=int(getattr(C, "SCALAR_DIM", 3)),
        precompute_scalars=True,
    )


def materialize_samples(dataset: GraphLLEDataset) -> List[Any]:
    return [dataset[index] for index in range(len(dataset))]


def make_variant_batches(
    samples: Sequence[Any],
    batch_size: int,
    variants: int,
) -> List[Any]:
    result = []
    n = len(samples)
    for variant in range(variants):
        start = (variant * batch_size) % n
        selected = [samples[(start + offset) % n] for offset in range(batch_size)]
        inputs, _ = collate_graph_batch(selected)
        result.append(inputs)
    return result


def benchmark_model_only(
    model: torch.nn.Module,
    gpu_batches: Sequence[Any],
    batch_size: int,
    precision: str,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Dict[str, Any]:
    with torch.inference_mode():
        for index in range(warmup):
            with _autocast_context(precision):
                model(gpu_batches[index % len(gpu_batches)])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    with torch.inference_mode():
        for index in range(repeats):
            starts[index].record()
            with _autocast_context(precision):
                model(gpu_batches[index % len(gpu_batches)])
            ends[index].record()
    torch.cuda.synchronize(device)
    values = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    result: Dict[str, Any] = {
        "mode": "model_only_cached_gpu",
        "precision": precision,
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
    }
    result.update(summarize_latencies(values, batch_size))
    return result


def benchmark_cached_pipeline(
    model: torch.nn.Module,
    dataset: GraphLLEDataset,
    index_groups: Sequence[Sequence[int]],
    batch_size: int,
    precision: str,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Dict[str, Any]:
    def run_once(indices: Sequence[int]) -> None:
        inputs, _ = collate_graph_batch([dataset[index] for index in indices])
        inputs = batch_to_device(inputs, str(device))
        with _autocast_context(precision):
            model(inputs)

    with torch.inference_mode():
        for index in range(warmup):
            run_once(index_groups[index % len(index_groups)])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    values: List[float] = []
    with torch.inference_mode():
        for index in range(repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            run_once(index_groups[index % len(index_groups)])
            torch.cuda.synchronize(device)
            values.append((time.perf_counter() - started) * 1000.0)
    result: Dict[str, Any] = {
        "mode": "cached_features_collate_transfer_forward",
        "precision": precision,
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
    }
    result.update(summarize_latencies(values, batch_size))
    return result


def benchmark_uncached_single(
    model: torch.nn.Module,
    frame,
    t_scaler: Scaler,
    p_scaler: Scaler,
    corpus: Sequence[str],
    precision: str,
    repeats: int,
    device: torch.device,
) -> Dict[str, Any]:
    def run_row(row_index: int) -> None:
        one_row = frame.iloc[[row_index]].reset_index(drop=True)
        dataset = build_dataset(
            one_row,
            t_scaler=t_scaler,
            p_scaler=p_scaler,
            corpus=corpus,
            prebuild_graphs=False,
        )
        inputs, _ = collate_graph_batch([dataset[0]])
        inputs = batch_to_device(inputs, str(device))
        with _autocast_context(precision):
            model(inputs)

    # Remove one-time RDKit and CUDA initialization from the measured repetitions.
    with torch.inference_mode():
        run_row(0)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    values: List[float] = []
    with torch.inference_mode():
        for index in range(repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            run_row(index % len(frame))
            torch.cuda.synchronize(device)
            values.append((time.perf_counter() - started) * 1000.0)
    result: Dict[str, Any] = {
        "mode": "uncached_new_system_end_to_end",
        "precision": precision,
        "batch_size": 1,
        "warmup": 1,
        "repeats": int(repeats),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
    }
    result.update(summarize_latencies(values, 1))
    return result


def profile_supported_flops(
    model: torch.nn.Module,
    gpu_batch: Any,
    device: torch.device,
) -> Tuple[Optional[int], Optional[str]]:
    try:
        from torch.profiler import ProfilerActivity, profile

        with torch.inference_mode():
            model(gpu_batch)
        torch.cuda.synchronize(device)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as profiler:
            with torch.inference_mode():
                model(gpu_batch)
        torch.cuda.synchronize(device)
        flops = int(sum(int(event.flops or 0) for event in profiler.key_averages()))
        return flops, None
    except Exception as exc:  # Profiling support varies across PyTorch builds.
        return None, f"{type(exc).__name__}: {exc}"


def make_index_groups(pool_size: int, batch_size: int, variants: int) -> List[List[int]]:
    return [
        [((variant * batch_size) + offset) % pool_size for offset in range(batch_size)]
        for variant in range(variants)
    ]


def write_latency_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = [
        "mode",
        "precision",
        "batch_size",
        "warmup",
        "repeats",
        "mean_ms_per_batch",
        "std_ms_per_batch",
        "median_ms_per_batch",
        "p95_ms_per_batch",
        "min_ms_per_batch",
        "max_ms_per_batch",
        "mean_ms_per_sample",
        "throughput_samples_per_s",
        "peak_allocated_mib",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_batch_sizes(value: str) -> List[int]:
    sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=Path(C.EXCEL_PATH))
    parser.add_argument("--fg-corpus", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("1,8,32,128"))
    parser.add_argument("--precisions", default="fp32,amp_fp16")
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--pipeline-repeats", type=int, default=50)
    parser.add_argument("--cold-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this RTX 3090 benchmark")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    corpus_path = resolve_fg_corpus(args.checkpoint, args.fg_corpus)
    corpus = load_corpus(corpus_path)
    model, t_scaler, p_scaler, _, load_seconds, adaptations = load_model(args.checkpoint, device)
    parameters = parameter_statistics(model)

    frame, _ = load_and_prepare_excel(
        str(args.dataset),
        min_points_per_group=int(C.MIN_POINTS_PER_GROUP),
        permute_23_aug=False,
    )
    pool_size = max(max(args.batch_sizes), min(int(args.pool_size), len(frame)))
    rng = np.random.RandomState(args.seed)
    selected = rng.choice(len(frame), size=pool_size, replace=False)
    pool_frame = frame.iloc[selected].reset_index(drop=True)

    preparation_started = time.perf_counter()
    dataset = build_dataset(
        pool_frame,
        t_scaler=t_scaler,
        p_scaler=p_scaler,
        corpus=corpus,
        prebuild_graphs=True,
    )
    samples = materialize_samples(dataset)
    cache_preparation_seconds = time.perf_counter() - preparation_started

    precisions = [item.strip() for item in args.precisions.split(",") if item.strip()]
    invalid_precisions = set(precisions) - {"fp32", "amp_fp16"}
    if invalid_precisions:
        raise ValueError(f"Unsupported precisions: {sorted(invalid_precisions)}")

    latency_rows: List[Dict[str, Any]] = []
    gpu_batches_by_size: Dict[int, List[Any]] = {}
    for batch_size in args.batch_sizes:
        cpu_batches = make_variant_batches(samples, batch_size, args.variants)
        gpu_batches = [batch_to_device(batch, str(device)) for batch in cpu_batches]
        gpu_batches_by_size[batch_size] = gpu_batches
        index_groups = make_index_groups(len(dataset), batch_size, args.variants)
        for precision in precisions:
            latency_rows.append(
                benchmark_model_only(
                    model,
                    gpu_batches,
                    batch_size,
                    precision,
                    args.warmup,
                    args.repeats,
                    device,
                )
            )
            latency_rows.append(
                benchmark_cached_pipeline(
                    model,
                    dataset,
                    index_groups,
                    batch_size,
                    precision,
                    max(5, args.warmup // 3),
                    args.pipeline_repeats,
                    device,
                )
            )

    for precision in precisions:
        latency_rows.append(
            benchmark_uncached_single(
                model,
                pool_frame,
                t_scaler,
                p_scaler,
                corpus,
                precision,
                args.cold_repeats,
                device,
            )
        )

    flops, flop_error = profile_supported_flops(model, gpu_batches_by_size[1][0], device)
    properties = torch.cuda.get_device_properties(device)
    metadata = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_gib": float(properties.total_memory / 2**30),
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "fg_corpus": str(corpus_path.resolve()),
        "dataset_records_available": int(len(frame)),
        "benchmark_pool_records": int(len(pool_frame)),
        "seed": int(args.seed),
        "model_load_seconds": float(load_seconds),
        "cache_preparation_seconds": float(cache_preparation_seconds),
        "checkpoint_adaptations": adaptations,
    }
    computation = {
        "profiler_supported_flops_batch1_fp32": flops,
        "profiler_supported_gflops_batch1_fp32": None if flops is None else float(flops / 1e9),
        "profiler_error": flop_error,
        "note": (
            "PyTorch profiler FLOPs include only supported operators and are therefore a lower-bound "
            "estimate for this custom graph model."
        ),
    }
    output = {
        "metadata": metadata,
        "parameters": parameters,
        "computation": computation,
        "latency": latency_rows,
    }
    with (args.output_dir / "benchmark_results.json").open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    write_latency_csv(args.output_dir / "latency_summary.csv", latency_rows)

    print(json.dumps({"metadata": metadata, "parameters": parameters, "computation": computation}, indent=2))
    print("\nLatency summary")
    for row in latency_rows:
        print(
            f"{row['mode']:42s} {row['precision']:8s} batch={row['batch_size']:3d} "
            f"median={row['median_ms_per_batch']:.3f} ms "
            f"p95={row['p95_ms_per_batch']:.3f} ms "
            f"throughput={row['throughput_samples_per_s']:.1f} samples/s"
        )
    print(f"\nSaved benchmark outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

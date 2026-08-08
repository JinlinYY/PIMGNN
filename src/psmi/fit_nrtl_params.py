"""Fit split-bound, system-specific NRTL interaction parameters."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import (
    load_and_prepare_excel,
    split_by_manifest,
    split_by_system,
    stratified_split_by_system,
)
from .loss import renorm3_torch, nrtl_mu_residual
from .nrtl_isolation import canonical_system_id, sha256_file


REQ_COLS = ["T", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]


def fit_one_system(
    df_sys: pd.DataFrame,
    alpha: float = 0.30,
    steps: int = 3000,
    lr: float = 5e-2,
    g_max: float = 8000.0,
    device: str = "cpu",
) -> np.ndarray:
    """Fit one system independently by minimizing its chemical-potential residual."""
    df_sys = df_sys.copy()[REQ_COLS].apply(pd.to_numeric, errors="coerce")
    df_sys = df_sys.dropna(axis=0, how="any")
    if len(df_sys) < 3:
        raise ValueError("Too few valid rows after dropna")

    temperature = torch.from_numpy(df_sys["T"].to_numpy(dtype=np.float32)).to(device).view(-1)
    x_e = torch.from_numpy(
        df_sys[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float32)
    ).to(device)
    x_r = torch.from_numpy(
        df_sys[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
    ).to(device)
    x_e = renorm3_torch(x_e.view(-1, 3))
    x_r = renorm3_torch(x_r.view(-1, 3))

    unconstrained = torch.zeros(
        (3, 3), device=device, dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.Adam([unconstrained], lr=lr)
    off_diagonal = 1.0 - torch.eye(3, device=device)

    for iteration in range(steps):
        optimizer.zero_grad(set_to_none=True)
        interaction = g_max * torch.tanh(unconstrained) * off_diagonal
        interaction_batch = interaction.unsqueeze(0).expand(x_e.size(0), 3, 3)
        residual = nrtl_mu_residual(
            x_e,
            x_r,
            temperature,
            interaction_batch,
            alpha=alpha,
            R=8.314462618,
        )
        residual_loss = (residual**2).mean()
        regularization = 1e-4 * (unconstrained**2).mean()
        (residual_loss + regularization).backward()
        optimizer.step()

        if (iteration + 1) % 500 == 0 and float(residual_loss.detach().cpu()) < 1e-6:
            break

    result = (g_max * torch.tanh(unconstrained)).detach().cpu().numpy()
    np.fill_diagonal(result, 0.0)
    return result.astype(np.float32)


def _sorted_ids(values: Iterable[Any]) -> list[str]:
    return sorted({canonical_system_id(value) for value in values}, key=int)


def build_split_manifest(
    df_raw: pd.DataFrame,
    *,
    dataset_path: Path,
    split_strategy: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    n_bins: int,
    min_bin_size: int,
    min_points: int,
    split_manifest_path: Path | None = None,
) -> Dict[str, Any]:
    if split_strategy == "random":
        train_df, val_df, test_df = split_by_system(
            df_raw, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
    elif split_strategy == "stratified":
        train_df, val_df, test_df = stratified_split_by_system(
            df_raw,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            n_bins=n_bins,
            min_bin_size=min_bin_size,
        )
    elif split_strategy == "manifest":
        if split_manifest_path is None:
            raise ValueError("split_manifest_path is required for manifest splitting")
        train_df, val_df, test_df = split_by_manifest(df_raw, split_manifest_path)
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    partitions = {
        "train": _sorted_ids(train_df["system_id"].unique()),
        "validation": _sorted_ids(val_df["system_id"].unique()),
        "test": _sorted_ids(test_df["system_id"].unique()),
    }
    split_sets = [set(partitions[name]) for name in ("train", "validation", "test")]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("System-level split contains overlapping partitions")

    result = {
        "schema_version": 1,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_rows_after_filtering": int(len(df_raw)),
        "dataset_system_count_after_filtering": int(df_raw["system_id"].nunique()),
        "minimum_tie_lines_per_system_temperature": int(min_points),
        "split_strategy": split_strategy,
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "validation_ratio": float(val_ratio),
        "stratification_n_bins": int(n_bins) if split_strategy == "stratified" else None,
        "stratification_min_bin_size": (
            int(min_bin_size) if split_strategy == "stratified" else None
        ),
        "partitions": partitions,
        "partition_system_counts": {
            name: len(system_ids) for name, system_ids in partitions.items()
        },
    }
    if split_strategy == "manifest" and split_manifest_path is not None:
        result["source_split_manifest_path"] = str(split_manifest_path.resolve())
        result["source_split_manifest_sha256"] = sha256_file(split_manifest_path)
    return result


def _fit_worker(
    system_id: str,
    records: Sequence[Mapping[str, Any]],
    alpha: float,
    steps: int,
    lr: float,
    g_max: float,
    device: str,
) -> Tuple[str, list[list[float]]]:
    if device == "cpu":
        torch.set_num_threads(1)
    matrix = fit_one_system(
        pd.DataFrame.from_records(records),
        alpha=alpha,
        steps=steps,
        lr=lr,
        g_max=g_max,
        device=device,
    )
    return system_id, matrix.tolist()


def fit_selected_systems(
    df_raw: pd.DataFrame,
    system_ids: Sequence[str],
    *,
    alpha: float,
    steps: int,
    lr: float,
    g_max: float,
    device: str,
    workers: int,
) -> Dict[str, Any]:
    requested = set(system_ids)
    work_items = []
    for raw_system_id, group in df_raw.groupby("system_id", sort=True):
        system_id = canonical_system_id(raw_system_id)
        if system_id not in requested:
            continue
        valid = group.dropna(subset=REQ_COLS, how="any")
        if len(valid) < 3:
            raise ValueError(f"System {system_id} has fewer than three valid tie lines")
        work_items.append((system_id, valid[REQ_COLS].to_dict(orient="records")))

    available = {system_id for system_id, _ in work_items}
    missing = requested - available
    if missing:
        raise ValueError(f"Requested systems are absent after preprocessing: {sorted(missing)}")

    if device != "cpu":
        ordered_ids = [system_id for system_id, _ in work_items]
        frames = []
        row_system_indices = []
        for system_index, (system_id, records) in enumerate(work_items):
            frame = pd.DataFrame.from_records(records)
            frames.append(frame)
            row_system_indices.extend([system_index] * len(frame))
        selected = pd.concat(frames, ignore_index=True)
        temperature = torch.tensor(
            selected["T"].to_numpy(dtype=np.float32), device=device
        )
        x_e = renorm3_torch(
            torch.tensor(
                selected[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float32),
                device=device,
            )
        )
        x_r = renorm3_torch(
            torch.tensor(
                selected[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32),
                device=device,
            )
        )
        row_indices = torch.tensor(row_system_indices, dtype=torch.long, device=device)
        unconstrained = torch.zeros(
            (len(ordered_ids), 3, 3),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([unconstrained], lr=lr)
        off_diagonal = 1.0 - torch.eye(3, dtype=torch.float32, device=device)
        row_counts = torch.bincount(row_indices, minlength=len(ordered_ids)).clamp_min(1)
        for iteration in range(steps):
            optimizer.zero_grad(set_to_none=True)
            interactions = g_max * torch.tanh(unconstrained) * off_diagonal
            residual = nrtl_mu_residual(
                x_e,
                x_r,
                temperature,
                interactions[row_indices],
                alpha=alpha,
                R=8.314462618,
            )
            row_loss = residual.square().mean(dim=-1)
            per_system_sum = torch.zeros(
                len(ordered_ids), dtype=torch.float32, device=device
            ).scatter_add_(0, row_indices, row_loss)
            residual_loss = (per_system_sum / row_counts).mean()
            regularization = 1e-4 * unconstrained.square().mean()
            (residual_loss + regularization).backward()
            optimizer.step()
            if (iteration + 1) % 500 == 0:
                print(
                    f"[fit_nrtl] vectorized step {iteration + 1}/{steps} | "
                    f"residual_mse={float(residual_loss.detach().cpu()):.6g}"
                )
                if float(residual_loss.detach().cpu()) < 1e-6:
                    break
        fitted_all = (
            g_max * torch.tanh(unconstrained) * off_diagonal
        ).detach().cpu().numpy()
        return {
            system_id: fitted.astype(np.float32).tolist()
            for system_id, fitted in zip(ordered_ids, fitted_all)
        }

    params: Dict[str, Any] = {}
    if workers <= 1:
        for index, (system_id, records) in enumerate(work_items, 1):
            fitted_id, matrix = _fit_worker(
                system_id, records, alpha, steps, lr, g_max, device
            )
            params[fitted_id] = matrix
            if index % 50 == 0 or index == len(work_items):
                print(f"[fit_nrtl] fitted {index}/{len(work_items)} systems")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _fit_worker,
                    system_id,
                    records,
                    alpha,
                    steps,
                    lr,
                    g_max,
                    device,
                ): system_id
                for system_id, records in work_items
            }
            for index, future in enumerate(as_completed(futures), 1):
                fitted_id, matrix = future.result()
                params[fitted_id] = matrix
                if index % 50 == 0 or index == len(futures):
                    print(f"[fit_nrtl] fitted {index}/{len(futures)} systems")

    return {system_id: params[system_id] for system_id in sorted(params, key=int)}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _parameter_payload(
    *,
    role: str,
    params: Mapping[str, Any],
    manifest_path: Path,
    manifest: Mapping[str, Any],
    alpha: float,
    g_max: float,
    steps: int,
    lr: float,
) -> Dict[str, Any]:
    allowed_uses = (
        ["training_loss"]
        if role == "training_loss"
        else ["posthoc_validation_diagnostics", "posthoc_test_diagnostics"]
    )
    return {
        "meta": {
            "schema_version": 2,
            "role": role,
            "allowed_uses": allowed_uses,
            "fitted_independently_by_system": True,
            "dataset_path": manifest["dataset_path"],
            "dataset_sha256": manifest["dataset_sha256"],
            "split_manifest_path": str(manifest_path.resolve()),
            "split_manifest_sha256": sha256_file(manifest_path),
            "split_strategy": manifest["split_strategy"],
            "seed": manifest["seed"],
            "alpha": float(alpha),
            "R": 8.314462618,
            "g_max": float(g_max),
            "optimization_steps": int(steps),
            "optimization_learning_rate": float(lr),
            "parameter_system_count": len(params),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Each g_ij matrix was fitted independently using only tie lines from "
                "the same chemical system."
            ),
        },
        "params": dict(params),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel_path", type=Path, default=Path(C.EXCEL_PATH))
    parser.add_argument("--out_dir", type=Path, default=Path(C.DATASETS_DIR) / "parameters")
    parser.add_argument("--scope", choices=["train", "all", "both"], default="both")
    parser.add_argument(
        "--split-strategy",
        choices=["random", "stratified", "manifest"],
        default="random",
    )
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=int(getattr(C, "SEED", 42)))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--n-bins", type=int, default=3)
    parser.add_argument("--min-bin-size", type=int, default=5)
    parser.add_argument(
        "--min-points", type=int, default=int(getattr(C, "MIN_POINTS_PER_GROUP", 6))
    )
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--g_max", type=float, default=8000.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max_systems", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_raw, _ = load_and_prepare_excel(
        str(args.excel_path),
        min_points_per_group=args.min_points,
        permute_23_aug=False,
    )
    manifest = build_split_manifest(
        df_raw,
        dataset_path=args.excel_path,
        split_strategy=args.split_strategy,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        n_bins=args.n_bins,
        min_bin_size=args.min_bin_size,
        min_points=args.min_points,
        split_manifest_path=args.split_manifest,
    )
    manifest_path = args.out_dir / "nrtl_split_manifest.json"

    all_ids = _sorted_ids(df_raw["system_id"].unique())
    train_ids = list(manifest["partitions"]["train"])
    train_params = None
    all_params = None

    if args.scope in {"train", "both"}:
        selected_train_ids = train_ids
        if args.max_systems > 0:
            selected_train_ids = selected_train_ids[: args.max_systems]
        print(
            "[fit_nrtl] starting isolated training-partition fit: "
            f"{len(selected_train_ids)} systems"
        )
        train_params = fit_selected_systems(
            df_raw[
                df_raw["system_id"].astype(int).isin(
                    [int(system_id) for system_id in selected_train_ids]
                )
            ],
            selected_train_ids,
            alpha=args.alpha,
            steps=args.steps,
            lr=args.lr,
            g_max=args.g_max,
            device=args.device,
            workers=args.workers,
        )
        missing_train = set(selected_train_ids) - set(train_params)
        if missing_train:
            raise RuntimeError(f"Training systems were not fitted: {sorted(missing_train)}")
    if args.scope in {"all", "both"}:
        selected_all_ids = all_ids
        if args.max_systems > 0:
            selected_all_ids = selected_all_ids[: args.max_systems]
        print(
            "[fit_nrtl] starting independent all-system post-hoc fit: "
            f"{len(selected_all_ids)} systems"
        )
        all_params = fit_selected_systems(
            df_raw,
            selected_all_ids,
            alpha=args.alpha,
            steps=args.steps,
            lr=args.lr,
            g_max=args.g_max,
            device=args.device,
            workers=args.workers,
        )
    # Do not replace the published manifest or either parameter store until all
    # requested fits have completed successfully.
    _write_json_atomic(manifest_path, manifest)

    if train_params is not None:
        train_payload = _parameter_payload(
            role="training_loss",
            params=train_params,
            manifest_path=manifest_path,
            manifest=manifest,
            alpha=args.alpha,
            g_max=args.g_max,
            steps=args.steps,
            lr=args.lr,
        )
        _write_json_atomic(args.out_dir / "nrtl_params_train.json", train_payload)

    if all_params is not None:
        all_payload = _parameter_payload(
            role="posthoc_evaluation",
            params=all_params,
            manifest_path=manifest_path,
            manifest=manifest,
            alpha=args.alpha,
            g_max=args.g_max,
            steps=args.steps,
            lr=args.lr,
        )
        _write_json_atomic(args.out_dir / "nrtl_params_all.json", all_payload)

    print(f"[OK] NRTL parameter files written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

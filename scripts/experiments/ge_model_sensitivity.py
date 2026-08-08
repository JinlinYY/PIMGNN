"""Run the excess-Gibbs-energy model sensitivity experiment.

The experiment compares a supervised-only control with otherwise identical head
fine-tuning regularized by NRTL, pairwise three-suffix Margules, or pairwise van Laar.
GE parameters are fitted only for systems assigned to the training split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from psmi import config as C
from psmi.data import load_and_prepare_excel, stratified_split_by_system
from psmi.ge_fit import fit_parameter_store
from psmi.ge_models import ge_mu_residual
from psmi.ge_parameters import GEParameterStore
from psmi.train import train_or_load
from psmi.utils import set_seed


DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles_min3.xlsx"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "03_physics_constraints"
    / "lle_run_混合物图-Cross-s3-tf-纯数据驱动test2"
    / "best_model.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "11_ge_model_sensitivity"
    / "runs"
    / "current"
)
GE_MODELS = ("nrtl", "margules", "van_laar")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fit_residual_metrics(
    data: pd.DataFrame,
    parameter_path: Path,
    model: str,
    device: str,
) -> dict[str, float]:
    store = GEParameterStore(parameter_path, device=device)
    system_ids = torch.tensor(data["system_id"].astype(int).to_numpy(), device=device)
    parameters, mask = store.get_batch(system_ids, device=device)
    extract = torch.tensor(
        data[["Ex1", "Ex2", "Ex3"]].to_numpy(), dtype=torch.float32, device=device
    )
    raffinate = torch.tensor(
        data[["Rx1", "Rx2", "Rx3"]].to_numpy(), dtype=torch.float32, device=device
    )
    temperature = torch.tensor(data["T"].to_numpy(), dtype=torch.float32, device=device)
    with torch.enable_grad():
        residual = ge_mu_residual(
            model,
            extract[mask],
            raffinate[mask],
            temperature[mask],
            parameters[mask],
            nrtl_alpha=store.alpha,
            gas_constant=store.R,
        ).detach()
    return {
        "parameter_coverage": float(mask.float().mean().cpu()),
        "fit_mu_mae": float(residual.abs().mean().cpu()),
        "fit_mu_rmse": float(residual.square().mean().sqrt().cpu()),
    }


def _configure_training(
    *,
    seed: int,
    model: str,
    parameter_path: Path | None,
    output_dir: Path,
    checkpoint: Path,
    device: str,
    epochs: int,
    learning_rate: float,
    lambda_phy: float,
) -> None:
    C.SEED = int(seed)
    C.DEVICE = device
    C.OUT_DIR = str(output_dir)
    C.LOAD_CKPT_PATH = str(checkpoint)
    C.EPOCHS = int(epochs)
    C.LR = float(learning_rate)
    C.USE_MECH_LOSS = model != "data_only"
    C.GE_MODEL = "nrtl" if model == "data_only" else model
    if parameter_path is not None:
        C.NRTL_TRAIN_PARAMS_PATH = str(parameter_path)
    C.LAMBDA_PHY = float(lambda_phy)
    C.WARMUP_EPOCHS = 0
    C.RAMP_EPOCHS = 5
    C.MECH_W_EQ = 1.0
    C.MECH_W_GD = 0.0
    C.MECH_W_STAB = 0.0
    C.FREEZE_BACKBONE = True
    C.USE_PHYSICS_FINETUNE = False
    C.USE_EARLY_STOP = False
    C.EVAL_EVERY = 1
    C.PLOT_EVERY = max(int(epochs), 1)
    C.COMPUTE_FINAL_PHYSICS_METRICS = False
    C.NUM_WORKERS_GRAPH = 0


def _aggregate(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model, group in pd.DataFrame(results).groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "n_seeds": len(group)}
        for metric in ("test_mae", "test_rmse", "test_r2", "fit_mu_mae", "fit_mu_rmse"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the most recent row for each model/seed pair."""
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in results:
        indexed[(str(row["model"]), int(row["seed"]))] = row
    return [indexed[key] for key in sorted(indexed, key=lambda value: (value[1], value[0]))]


def _paired_comparisons(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    control = frame[frame["model"] == "data_only"].set_index("seed")
    rows: list[dict[str, Any]] = []
    for model in [value for value in frame["model"].unique() if value != "data_only"]:
        treatment = frame[frame["model"] == model].set_index("seed")
        shared_seeds = sorted(control.index.intersection(treatment.index))
        row: dict[str, Any] = {"model": model, "n_paired_seeds": len(shared_seeds)}
        for metric in ("test_mae", "test_rmse", "test_r2"):
            delta = treatment.loc[shared_seeds, metric] - control.loc[shared_seeds, metric]
            row[f"delta_{metric}_mean"] = float(delta.mean())
            row[f"delta_{metric}_std"] = (
                float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir).resolve()
    dataset_path = Path(args.dataset).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    raw, augmented = load_and_prepare_excel(
        str(dataset_path), min_points_per_group=6, permute_23_aug=True
    )
    metrics_table_path = output_root / "per_seed_metrics.csv"
    if metrics_table_path.exists() and not args.overwrite:
        all_results = pd.read_csv(metrics_table_path).to_dict(orient="records")
    else:
        all_results: list[dict[str, Any]] = []
    models = [str(value).lower() for value in args.models]
    invalid_models = sorted(set(models) - {"data_only", *GE_MODELS})
    if invalid_models:
        raise ValueError(f"Unsupported experiment models: {invalid_models}")

    recorded_seeds = {int(row["seed"]) for row in all_results}
    manifest_seeds = sorted(recorded_seeds | {int(value) for value in args.seeds})
    recorded_models = {str(row["model"]) for row in all_results}
    manifest_models = sorted(recorded_models | set(models))

    manifest = {
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "models": manifest_models,
        "seeds": manifest_seeds,
        "split": {"train": 0.8, "validation": 0.1, "test": 0.1, "by": "system_id"},
        "min_tie_lines_per_system_temperature": 6,
        "permutation_augmentation": "identity_plus_swap_components_2_and_3",
        "parameter_fit_scope": "training_systems_only",
        "parameter_fit_steps": int(args.fit_steps),
        "finetune_epochs": int(args.epochs),
        "finetune_learning_rate": float(args.learning_rate),
        "lambda_phy": float(args.lambda_phy),
        "freeze_backbone": True,
    }
    _write_json(output_root / "manifest.json", manifest)

    for seed in [int(value) for value in args.seeds]:
        set_seed(seed)
        train_df, val_df, test_df = stratified_split_by_system(
            augmented,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=seed,
            n_bins=3,
            min_bin_size=5,
        )
        train_ids = sorted(train_df["system_id"].astype(int).unique().tolist())
        raw_training = raw[raw["system_id"].astype(int).isin(train_ids)].copy()

        split_manifest = {
            "seed": seed,
            "train_systems": train_ids,
            "validation_systems": sorted(val_df["system_id"].astype(int).unique().tolist()),
            "test_systems": sorted(test_df["system_id"].astype(int).unique().tolist()),
            "rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        }
        _write_json(output_root / "splits" / f"seed_{seed}.json", split_manifest)

        parameter_paths: dict[str, Path] = {}
        fit_metrics: dict[str, dict[str, float]] = {}
        for model in [value for value in models if value in GE_MODELS]:
            parameter_path = output_root / "parameters" / f"{model}_seed_{seed}.json"
            if not parameter_path.exists() or args.overwrite:
                fitted = fit_parameter_store(
                    raw_training,
                    model=model,
                    training_system_ids=train_ids,
                    steps=args.fit_steps,
                    learning_rate=args.fit_learning_rate,
                    maximum_energy=args.maximum_energy,
                    alpha=args.nrtl_alpha,
                    device=args.fit_device,
                    vectorized=True,
                )
                _write_json(parameter_path, fitted)
            parameter_paths[model] = parameter_path
            fit_metrics[model] = _fit_residual_metrics(
                raw_training, parameter_path, model, args.fit_device
            )

        for model in models:
            run_dir = output_root / "runs" / model / f"seed_{seed}"
            metrics_path = run_dir / "best_metrics.json"
            parameter_path = parameter_paths.get(model)
            if not metrics_path.exists() or args.overwrite:
                run_dir.mkdir(parents=True, exist_ok=True)
                set_seed(seed)
                _configure_training(
                    seed=seed,
                    model=model,
                    parameter_path=parameter_path,
                    output_dir=run_dir,
                    checkpoint=checkpoint_path,
                    device=args.device,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    lambda_phy=args.lambda_phy,
                )
                train_or_load(train_df, val_df, test_df)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            result = {
                "model": model,
                "seed": seed,
                "best_epoch": metrics["best_epoch"],
                "test_mae": metrics["best_test"]["mae"],
                "test_rmse": metrics["best_test"]["rmse"],
                "test_r2": metrics["best_test"]["r2"],
                "fit_mu_mae": np.nan,
                "fit_mu_rmse": np.nan,
                "parameter_coverage": np.nan,
            }
            if model in fit_metrics:
                result.update(fit_metrics[model])
            all_results.append(result)
            all_results = _deduplicate_results(all_results)
            pd.DataFrame(all_results).to_csv(metrics_table_path, index=False)

    summary = _aggregate(all_results)
    summary.to_csv(output_root / "summary.csv", index=False)
    (output_root / "summary.md").write_text(
        summary.to_markdown(index=False, floatfmt=".6f") + "\n", encoding="utf-8"
    )
    comparisons = _paired_comparisons(all_results)
    comparisons.to_csv(output_root / "paired_comparisons.csv", index=False)
    (output_root / "paired_comparisons.md").write_text(
        comparisons.to_markdown(index=False, floatfmt=".6f") + "\n", encoding="utf-8"
    )
    return output_root / "summary.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--models", nargs="+", default=["data_only", "nrtl", "margules", "van_laar"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--fit_steps", type=int, default=1000)
    parser.add_argument("--fit_learning_rate", type=float, default=5e-2)
    parser.add_argument("--maximum_energy", type=float, default=8000.0)
    parser.add_argument("--nrtl_alpha", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lambda_phy", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fit_device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())


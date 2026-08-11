"""Run a strict within-system extreme-temperature extrapolation experiment.

For every chemical system measured at three or more distinct temperatures, the
lowest and highest temperatures are held out.  Only the interior temperatures
for those same chemical systems are available during training.  This isolates
temperature extrapolation from molecular-identity extrapolation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from psmi import config as C  # noqa: E402
from psmi.checkpoints import load_state_dict_compat  # noqa: E402
from psmi.data import load_and_prepare_excel  # noqa: E402
from psmi.identity import (  # noqa: E402
    add_chemical_system_identity,
    merge_nearby_temperature_levels,
)
from psmi.predict import predict_pointwise_df_raw  # noqa: E402
from psmi.train import build_model, train_or_load  # noqa: E402
from psmi.utils import Scaler, set_seed  # noqa: E402


TRUE_COLUMNS = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
PRED_COLUMNS = [
    "pred_Ex1",
    "pred_Ex2",
    "pred_Ex3",
    "pred_Rx1",
    "pred_Rx2",
    "pred_Rx3",
]


def orient_phase_path_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Give the PCA path coordinate a deterministic direction in every curve.

    PCA eigenvectors have an arbitrary sign.  Without this correction, two
    temperatures of the same chemistry can receive opposite ``t`` directions.
    The convention used here makes component-1 extract composition decrease as
    ``t`` increases.  It uses no information from another temperature group.
    """
    output = df.copy()
    output["t"] = output["t"].astype(float)
    for indices in output.groupby(["system_id", "T"], sort=False).groups.values():
        group = output.loc[indices]
        if len(group) < 2 or group["Ex1"].nunique() < 2:
            continue
        correlation = np.corrcoef(
            group["t"].to_numpy(dtype=float), group["Ex1"].to_numpy(dtype=float)
        )[0, 1]
        if np.isfinite(correlation) and correlation > 0:
            output.loc[indices, "t"] = 1.0 - group["t"].to_numpy(dtype=float)
    return output


def augment_swap23(df: pd.DataFrame) -> pd.DataFrame:
    original = df.copy()
    original["aug_swap23"] = 0
    swapped = df.copy()
    swapped["aug_swap23"] = 1
    swapped[["smiles2", "smiles3"]] = swapped[["smiles3", "smiles2"]].to_numpy()
    swapped[["Ex2", "Ex3"]] = swapped[["Ex3", "Ex2"]].to_numpy()
    swapped[["Rx2", "Rx3"]] = swapped[["Rx3", "Rx2"]].to_numpy()
    return pd.concat([original, swapped], ignore_index=True)


def build_extreme_temperature_split(
    df: pd.DataFrame,
    seed: int,
    validation_fraction: float,
    min_temperatures: int,
    system_column: str = "system_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if system_column not in df.columns:
        raise KeyError(f"System grouping column is missing: {system_column}")
    per_system = df.groupby(system_column)["T"].nunique()
    target_ids = sorted(per_system[per_system >= int(min_temperatures)].index.tolist())
    if not target_ids:
        raise ValueError("No systems have enough temperatures for extreme-temperature holdout")

    target_mask = df[system_column].isin(target_ids)
    target = df[target_mask].copy()
    background = df[~target_mask].copy()

    test_parts: List[pd.DataFrame] = []
    target_train_parts: List[pd.DataFrame] = []
    split_rows: List[Dict[str, object]] = []
    for system_id, group in target.groupby(system_column, sort=True):
        temperatures = sorted(float(value) for value in group["T"].unique())
        cold_temperature = temperatures[0]
        hot_temperature = temperatures[-1]
        interior = temperatures[1:-1]
        if not interior:
            continue
        system_train = group[group["T"].isin(interior)].copy()
        cold = group[np.isclose(group["T"], cold_temperature)].copy()
        hot = group[np.isclose(group["T"], hot_temperature)].copy()
        cold["temperature_direction"] = "cold_extrapolation"
        hot["temperature_direction"] = "hot_extrapolation"
        cold["multiple_interior_temperatures"] = len(interior) >= 2
        hot["multiple_interior_temperatures"] = len(interior) >= 2
        cold["temperature_gap_K"] = float(min(interior) - cold_temperature)
        hot["temperature_gap_K"] = float(hot_temperature - max(interior))
        target_train_parts.append(system_train)
        test_parts.extend([cold, hot])
        split_rows.extend(
            [
                {
                    system_column: int(system_id),
                    "split": "test_cold",
                    "temperature_K": cold_temperature,
                    "nearest_training_temperature_K": float(min(interior)),
                    "temperature_gap_K": float(min(interior) - cold_temperature),
                    "n_tie_lines": int(len(cold)),
                },
                {
                    system_column: int(system_id),
                    "split": "test_hot",
                    "temperature_K": hot_temperature,
                    "nearest_training_temperature_K": float(max(interior)),
                    "temperature_gap_K": float(hot_temperature - max(interior)),
                    "n_tie_lines": int(len(hot)),
                },
            ]
        )
        for temperature in interior:
            split_rows.append(
                {
                    system_column: int(system_id),
                    "split": "train_interior",
                    "temperature_K": float(temperature),
                    "nearest_training_temperature_K": float(temperature),
                    "temperature_gap_K": 0.0,
                    "n_tie_lines": int(np.isclose(system_train["T"], temperature).sum()),
                }
            )

    rng = np.random.RandomState(int(seed))
    background_ids = np.asarray(sorted(background[system_column].unique().tolist()))
    rng.shuffle(background_ids)
    n_validation = max(1, int(round(len(background_ids) * float(validation_fraction))))
    validation_ids = set(background_ids[:n_validation].tolist())
    validation = background[background[system_column].isin(validation_ids)].copy()
    background_train = background[~background[system_column].isin(validation_ids)].copy()

    train = pd.concat([background_train, *target_train_parts], ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    split_table = pd.DataFrame(split_rows).sort_values([system_column, "temperature_K"])

    train_keys = set(zip(train[system_column], train["T"]))
    test_keys = set(zip(test[system_column], test["T"]))
    if train_keys & test_keys:
        raise AssertionError("A system-temperature group appears in both train and test")
    if set(validation[system_column]) & set(test[system_column]):
        raise AssertionError("Validation systems overlap target extrapolation systems")

    manifest: Dict[str, object] = {
        "seed": int(seed),
        "system_grouping_column": str(system_column),
        "validation_fraction": float(validation_fraction),
        "minimum_temperatures_per_target_system": int(min_temperatures),
        "n_target_systems": int(test[system_column].nunique()),
        "n_target_test_groups": int(test.groupby([system_column, "T"]).ngroups),
        "n_train_systems": int(train[system_column].nunique()),
        "n_validation_systems": int(validation[system_column].nunique()),
        "n_train_rows_original": int(len(train)),
        "n_validation_rows_original": int(len(validation)),
        "n_test_rows_original": int(len(test)),
        "test_temperature_min_K": float(test["T"].min()),
        "test_temperature_max_K": float(test["T"].max()),
        "temperature_gap_min_K": float(test["temperature_gap_K"].min()),
        "temperature_gap_median_K": float(test["temperature_gap_K"].median()),
        "temperature_gap_max_K": float(test["temperature_gap_K"].max()),
        "n_target_systems_with_multiple_interior_temperatures": int(
            test.loc[test["multiple_interior_temperatures"], system_column].nunique()
        ),
        "n_target_systems_with_one_interior_temperature": int(
            test.loc[~test["multiple_interior_temperatures"], system_column].nunique()
        ),
    }
    return train, validation, test, split_table, manifest


def build_target_system_table(
    full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    system_column: str,
) -> pd.DataFrame:
    """Create an auditable inventory of the extrapolation chemistries."""
    rows: List[Dict[str, object]] = []
    target_ids = sorted(test_df[system_column].unique().tolist())
    for system_id in target_ids:
        group = full_df[full_df[system_column] == system_id]
        temperatures = sorted(group["T"].astype(float).unique().tolist())
        row: Dict[str, object] = {
            system_column: int(system_id),
            "source_system_ids": ";".join(
                str(value) for value in sorted(group["system_id"].unique().tolist())
            ),
            "smiles1": str(group["smiles1"].iloc[0]),
            "smiles2": str(group["smiles2"].iloc[0]),
            "smiles3": str(group["smiles3"].iloc[0]),
            "n_temperatures": int(len(temperatures)),
            "temperatures_K": ";".join(f"{value:.3f}" for value in temperatures),
            "temperature_span_K": float(max(temperatures) - min(temperatures)),
            "n_tie_lines_all_temperatures": int(len(group)),
        }
        if "chemical_system_signature" in group.columns:
            row["chemical_system_signature"] = str(
                group["chemical_system_signature"].iloc[0]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def configure_training(
    out_dir: Path,
    seed: int,
    epochs: int,
    device: str,
    initial_checkpoint: Path | None = None,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
) -> None:
    C.SEED = int(seed)
    C.OUT_DIR = str(out_dir)
    C.DEVICE = str(device)
    C.LOAD_CKPT_PATH = str(initial_checkpoint) if initial_checkpoint is not None else ""
    C.USE_FINE_TUNE = False
    C.USE_MECH_LOSS = False
    C.USE_PHYSICS_FINETUNE = False
    C.FREEZE_BACKBONE = False
    C.EPOCHS = int(epochs)
    C.LR = float(learning_rate)
    C.WEIGHT_DECAY = float(weight_decay)
    C.BATCH_SIZE_GRAPH = 256
    C.USE_AMP = str(device).startswith("cuda")
    C.GRAD_CLIP = 1.0
    C.USE_EARLY_STOP = True
    C.EARLY_STOP_METRIC = "rmse"
    C.EARLY_STOP_PATIENCE = 15
    C.EARLY_STOP_MIN_DELTA = 1e-4
    C.EVAL_EVERY = 1
    C.PLOT_EVERY = 10
    C.NUM_WORKERS_GRAPH = 0
    C.NUM_WORKERS = 0


def load_existing_run(run_dir: Path, device: str) -> Tuple[torch.nn.Module, Scaler, Scaler]:
    best_path = run_dir / "best_model.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {best_path}")
    best = torch.load(best_path, map_location="cpu")
    model = build_model().to(device)
    load_state_dict_compat(model, best)
    corpus_path = run_dir / "fg_corpus.json"
    if corpus_path.is_file():
        model.fg_corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    model.eval()
    T_scaler = Scaler(mean=float(best["T_mean"]), std=float(best["T_std"]))
    P_scaler = Scaler(
        mean=float(best.get("P_mean", 101.325)),
        std=float(best.get("P_std", 1.0)),
    )
    return model, T_scaler, P_scaler


def add_nearest_temperature_baseline(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    system_column: str = "system_id",
) -> pd.DataFrame:
    output = test_raw.copy()
    baseline_values = np.empty((len(output), len(TRUE_COLUMNS)), dtype=float)
    grouping_columns = [system_column, "T"]
    if system_column != "system_id":
        grouping_columns.append("system_id")
    for group_key, indices in output.groupby(grouping_columns).groups.items():
        system_id = group_key[0] if isinstance(group_key, tuple) else group_key
        temperature = group_key[1] if isinstance(group_key, tuple) else output.loc[indices, "T"].iloc[0]
        target_rows = output.loc[indices]
        candidates = train_raw[train_raw[system_column] == system_id]
        available = np.asarray(sorted(candidates["T"].unique()), dtype=float)
        if available.size == 0:
            raise ValueError(f"No training temperature remains for target system {system_id}")
        nearest_temperature = float(available[np.argmin(np.abs(available - float(temperature)))])
        reference = candidates[np.isclose(candidates["T"], nearest_temperature)].copy()
        reference["_t_rounded"] = reference["t"].round(6)
        reference = (
            reference.groupby("_t_rounded", as_index=False)[TRUE_COLUMNS]
            .mean()
            .rename(columns={"_t_rounded": "t"})
            .sort_values("t")
        )
        x_ref = reference["t"].to_numpy(dtype=float)
        x_target = target_rows["t"].to_numpy(dtype=float)
        for column_index, column in enumerate(TRUE_COLUMNS):
            baseline_values[output.index.get_indexer(indices), column_index] = np.interp(
                x_target,
                x_ref,
                reference[column].to_numpy(dtype=float),
            )
    for index, column in enumerate(PRED_COLUMNS):
        output[f"nearest_{column}"] = baseline_values[:, index]
    return output


def ternary_to_xy(composition: np.ndarray) -> np.ndarray:
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
    return composition @ vertices


def tie_line_angles(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true_vector = ternary_to_xy(true[:, :3]) - ternary_to_xy(true[:, 3:])
    pred_vector = ternary_to_xy(pred[:, :3]) - ternary_to_xy(pred[:, 3:])
    true_length = np.linalg.norm(true_vector, axis=1)
    pred_length = np.linalg.norm(pred_vector, axis=1)
    valid = (true_length >= 0.05) & (pred_length > 1e-12)
    if not valid.any():
        return np.asarray([], dtype=float)
    cosine = np.abs(np.sum(true_vector[valid] * pred_vector[valid], axis=1))
    cosine /= true_length[valid] * pred_length[valid]
    return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))


def summarize_predictions(
    predictions: pd.DataFrame,
    system_column: str = "system_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    method_columns = {
        "PSMI": PRED_COLUMNS,
        "Nearest observed temperature": [f"nearest_{column}" for column in PRED_COLUMNS],
    }
    summary_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    groupings: List[Tuple[str, pd.DataFrame]] = [("overall", predictions)]
    groupings.extend((name, group) for name, group in predictions.groupby("temperature_direction"))

    for scope, frame in groupings:
        true = frame[TRUE_COLUMNS].to_numpy(dtype=float)
        for method, columns in method_columns.items():
            pred = frame[columns].to_numpy(dtype=float)
            error = pred - true
            sse = float(np.square(error).sum())
            centered = true - true.mean(axis=0, keepdims=True)
            sst = float(np.square(centered).sum())
            angles = tie_line_angles(true, pred)
            summary_rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "n_tie_lines": int(len(frame)),
                    "n_system_temperature_groups": int(frame.groupby([system_column, "T"]).ngroups),
                    "temperature_gap_median_K": float(frame["temperature_gap_K"].median()),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.square(error).mean())),
                    "r2": float(1.0 - sse / sst) if sst > 1e-12 else float("nan"),
                    "median_tie_angle_deg": float(np.median(angles)) if angles.size else float("nan"),
                    "p90_tie_angle_deg": float(np.percentile(angles, 90.0)) if angles.size else float("nan"),
                }
            )

    for (system_id, temperature), frame in predictions.groupby([system_column, "T"], sort=True):
        true = frame[TRUE_COLUMNS].to_numpy(dtype=float)
        for method, columns in method_columns.items():
            pred = frame[columns].to_numpy(dtype=float)
            error = pred - true
            angles = tie_line_angles(true, pred)
            group_rows.append(
                {
                    system_column: int(system_id),
                    "temperature_K": float(temperature),
                    "temperature_direction": str(frame["temperature_direction"].iloc[0]),
                    "temperature_gap_K": float(frame["temperature_gap_K"].iloc[0]),
                    "method": method,
                    "n_tie_lines": int(len(frame)),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.square(error).mean())),
                    "median_tie_angle_deg": float(np.median(angles)) if angles.size else float("nan"),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(group_rows)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a frozen experiment input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_relative_path(path: Path) -> str:
    """Return a portable repository-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def write_report(summary: pd.DataFrame, manifest: Dict[str, object], out_path: Path) -> None:
    lines = ["# Extreme-temperature extrapolation", "", "## Protocol", ""]
    lines.extend(
        [
            f"- Target systems: {manifest['n_target_systems']}",
            f"- Held-out extreme-temperature groups: {manifest['n_target_test_groups']}",
            f"- Original held-out tie-lines: {manifest['n_test_rows_original']}",
            f"- Temperature gap: {manifest['temperature_gap_min_K']:.2f}-{manifest['temperature_gap_max_K']:.2f} K (median {manifest['temperature_gap_median_K']:.2f} K)",
            "- Split rule: lowest and highest temperatures held out; interior temperatures retained for the same chemical system.",
            "- Checkpoint selection: validation RMSE on background systems only.",
            f"- Target systems with one retained interior temperature: {manifest.get('n_target_systems_with_one_interior_temperature', 0)}",
            f"- Synthetic temperature-interpolation rows: {manifest.get('n_synthetic_temperature_interpolation_rows', 0)}",
            "",
            "## Results",
            "",
            summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Interpretation boundary",
            "",
            "This experiment isolates conditional temperature extrapolation for chemical systems observed at interior temperatures. It does not establish extrapolation to simultaneously unseen chemistry and unseen temperature.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce SI Section S3.11 with either the distributed reference "
            "checkpoint or a newly trained model."
        )
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "same_system_temperature_extrapolation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min_temperatures", type=int, default=3)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument(
        "--system-grouping",
        "--system_grouping",
        dest="system_grouping",
        choices=["chemical_identity", "workbook_id"],
        default="chemical_identity",
        help="Use order-invariant canonical-SMILES identity for leakage-free temperature transfer",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--temperature-merge-tolerance-K",
        "--temperature_merge_tolerance_K",
        dest="temperature_merge_tolerance_K",
        type=float,
        default=0.1,
        help="Merge nominal temperature levels whose full span is within this tolerance",
    )
    parser.add_argument(
        "--no-orient-t",
        "--no_orient_t",
        dest="no_orient_t",
        action="store_true",
        help="Keep the arbitrary PCA sign of the original phase-path coordinate",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--initial-checkpoint",
        "--initial_checkpoint",
        dest="initial_checkpoint",
        type=Path,
        default=None,
        help="Optional PSMI checkpoint used only for weight initialization",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "supporting_information"
            / "s3_additional_evaluation_and_validation"
            / "s3_11_conditional_same_system_temperature_extrapolation"
            / "models"
            / "reference_checkpoint"
        ),
        help=(
            "Directory containing best_model.pt and fg_corpus.json. This reference "
            "checkpoint is used unless --train-from-scratch is supplied."
        ),
    )
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help="Train a new model instead of evaluating the distributed checkpoint.",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Write and validate the split artifacts without training or inference.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    excel = args.excel if args.excel.is_absolute() else PROJECT_ROOT / args.excel
    C.EXCEL_PATH = str(excel.resolve())
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    initial_checkpoint = args.initial_checkpoint
    if initial_checkpoint is not None and not initial_checkpoint.is_absolute():
        initial_checkpoint = PROJECT_ROOT / initial_checkpoint
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is not None and not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    run_suffix = "_pretrained" if initial_checkpoint is not None else ""
    tolerance_tag = f"{float(args.temperature_merge_tolerance_K):g}".replace(".", "p")
    run_tag = f"seed_{args.seed}_{args.system_grouping}_tol{tolerance_tag}K{run_suffix}"
    run_dir = out_dir / "runs" / run_tag
    result_dir = out_dir / "results" / run_tag
    split_dir = out_dir / "splits"
    for path in [run_dir, result_dir, split_dir]:
        path.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    raw, _ = load_and_prepare_excel(
        str(excel), min_points_per_group=6, permute_23_aug=False
    )
    if args.system_grouping == "chemical_identity":
        raw = add_chemical_system_identity(raw)
        system_column = "chemical_system_id"
    else:
        system_column = "system_id"
    raw = merge_nearby_temperature_levels(
        raw,
        system_column=system_column,
        tolerance_K=float(args.temperature_merge_tolerance_K),
    )
    if not args.no_orient_t:
        raw = orient_phase_path_coordinates(raw)
    train_raw, validation_raw, test_raw, split_table, manifest = build_extreme_temperature_split(
        raw,
        seed=int(args.seed),
        validation_fraction=float(args.validation_fraction),
        min_temperatures=int(args.min_temperatures),
        system_column=system_column,
    )
    experimental_train_raw = train_raw.copy()
    train_raw["data_origin"] = "experimental"
    target_system_table = build_target_system_table(raw, test_raw, system_column)
    if "chemical_system_signature" in target_system_table.columns:
        split_table = split_table.merge(
            target_system_table[[system_column, "chemical_system_signature"]],
            on=system_column,
            how="left",
        )
    manifest["temperature_merge_tolerance_K"] = float(
        args.temperature_merge_tolerance_K
    )
    manifest["n_synthetic_temperature_interpolation_rows"] = 0
    manifest["n_train_rows"] = int(len(train_raw))
    split_table.to_csv(
        split_dir / f"temperature_split_{run_tag}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    target_system_table.to_csv(
        split_dir / f"target_systems_{run_tag}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (split_dir / f"manifest_{run_tag}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.split_only:
        print(json.dumps(manifest, indent=2))
        print(f"Saved split audit to: {split_dir.resolve()}")
        return

    configure_training(
        run_dir,
        int(args.seed),
        int(args.epochs),
        str(args.device),
        initial_checkpoint=initial_checkpoint,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train = augment_swap23(train_raw)
    validation = augment_swap23(validation_raw)
    test = augment_swap23(test_raw)

    if args.train_from_scratch:
        model, T_scaler, P_scaler, _ = train_or_load(train, validation, test)
        selected_checkpoint_dir = run_dir
    else:
        if checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required unless --train-from-scratch is used")
        model, T_scaler, P_scaler = load_existing_run(
            checkpoint_dir, str(args.device)
        )
        selected_checkpoint_dir = checkpoint_dir

    prediction = predict_pointwise_df_raw(
        model,
        T_scaler,
        test_raw,
        device=str(args.device),
        P_scaler=P_scaler,
    )
    prediction = add_nearest_temperature_baseline(
        experimental_train_raw, prediction, system_column=system_column
    )
    prediction.to_csv(
        result_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    summary, group_metrics = summarize_predictions(prediction, system_column=system_column)
    summary.to_csv(result_dir / "summary.csv", index=False, encoding="utf-8-sig")
    group_metrics.to_csv(
        result_dir / "by_system_temperature.csv", index=False, encoding="utf-8-sig"
    )
    checkpoint_path = selected_checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    split_manifest_path = split_dir / f"manifest_{run_tag}.json"
    evaluation_manifest = {
        "checkpoint": {
            "path": repository_relative_path(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "dataset": {
            "path": repository_relative_path(excel),
            "sha256": sha256_file(excel),
        },
        "split_manifest": {
            "path": repository_relative_path(split_manifest_path),
            "sha256": sha256_file(split_manifest_path),
        },
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_validation_metrics": checkpoint.get("val_metrics", {}),
        "test_evaluated_during_training": bool(
            checkpoint.get("test_evaluated_during_training", False)
        ),
        "system_grouping_column": system_column,
        "temperature_merge_tolerance_K": float(args.temperature_merge_tolerance_K),
        "test_summary": json.loads(summary.to_json(orient="records")),
    }
    (result_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(summary, manifest, result_dir / "report.md")
    print(summary.to_string(index=False))
    print(f"Saved temperature extrapolation experiment to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

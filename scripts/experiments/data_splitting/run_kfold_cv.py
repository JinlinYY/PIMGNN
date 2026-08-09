# -*- coding: utf-8 -*-
"""Run system-level k-fold cross-validation for PSMI.

The experiment quantifies split uncertainty under an approximately 8:1:1 ratio.
With K=10, fold i uses one system fold as test, the next fold as validation,
and the remaining eight folds as training. Exact counts can differ because
whole systems remain indivisible.

Example:
  python scripts/experiments/data_splitting/run_kfold_cv.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._bootstrap import add_src_to_path

add_src_to_path()

from psmi import config as C
from psmi.data import load_and_prepare_excel
from psmi.train import train_or_load
from psmi.utils import set_seed


METRIC_KEYS = [
    "mae",
    "rmse",
    "r2",
    "mae_E",
    "mae_R",
    "rmse_E",
    "rmse_R",
    "r2_E",
    "r2_R",
    "mu_res_mae",
    "mu_res_rmse",
    "tpd_viol_rate",
]


@contextmanager
def temporary_config(**kwargs):
    old = {k: getattr(C, k, None) for k in kwargs}
    missing = {k for k in kwargs if not hasattr(C, k)}
    try:
        for k, v in kwargs.items():
            setattr(C, k, v)
        yield
    finally:
        for k, v in old.items():
            if k in missing:
                delattr(C, k)
            else:
                setattr(C, k, v)


def _qbin(s: pd.Series, q: int) -> pd.Series:
    uniq = int(s.nunique())
    q = int(max(1, min(q, uniq)))
    if q <= 1:
        return pd.Series(["ALL"] * len(s), index=s.index)
    try:
        return pd.qcut(s, q=q, duplicates="drop").astype(str)
    except Exception:
        return pd.Series(["ALL"] * len(s), index=s.index)


def build_stratified_system_folds(
    df: pd.DataFrame,
    folds: int = 10,
    seed: int = 42,
    n_bins: int = 8,
) -> List[List[int]]:
    """Return balanced system_id folds using the same proxies as the 8:1:1 split."""
    if folds < 3:
        raise ValueError("--folds must be at least 3 so train/val/test are all non-empty.")

    stats = (
        df.groupby("system_id")
        .agg(
            n_rows=("system_id", "size"),
            n_groups=("T", lambda x: x.nunique()),
            T_min=("T", "min"),
            T_max=("T", "max"),
        )
        .reset_index()
    )
    stats["T_span"] = (stats["T_max"] - stats["T_min"]).astype(float)
    stats["bin_rows"] = _qbin(stats["n_rows"], n_bins)
    stats["bin_span"] = _qbin(stats["T_span"], n_bins)
    stats["bin_groups"] = _qbin(stats["n_groups"], max(2, n_bins // 2))
    stats["stratum"] = (
        stats["bin_rows"].astype(str)
        + "|"
        + stats["bin_span"].astype(str)
        + "|"
        + stats["bin_groups"].astype(str)
    )

    rng = np.random.RandomState(seed)
    out: List[List[int]] = [[] for _ in range(folds)]
    for _, sub in stats.groupby("stratum", sort=False):
        sids = sub["system_id"].tolist()
        rng.shuffle(sids)
        for j, sid in enumerate(sids):
            out[j % folds].append(sid)

    for f in out:
        f.sort()
    if any(len(f) == 0 for f in out):
        raise ValueError(
            f"At least one fold is empty. Got {len(stats)} systems for {folds} folds."
        )
    return out


def make_811_fold_split(
    df: pd.DataFrame,
    system_folds: List[List[int]],
    fold_idx: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(system_folds)
    test_sys = set(system_folds[fold_idx])
    val_sys = set(system_folds[(fold_idx + 1) % n])
    train_sys = set()
    for j, f in enumerate(system_folds):
        if j not in {fold_idx, (fold_idx + 1) % n}:
            train_sys.update(f)

    train_df = df[df["system_id"].isin(train_sys)].copy()
    val_df = df[df["system_id"].isin(val_sys)].copy()
    test_df = df[df["system_id"].isin(test_sys)].copy()
    return train_df, val_df, test_df


def _safe_get(d: Dict, key: str) -> float:
    try:
        return float(d.get(key, np.nan))
    except Exception:
        return float("nan")


def _format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def summarize_rows(rows: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_fold = pd.DataFrame(rows)
    summary_rows = []
    for key in METRIC_KEYS:
        if key not in per_fold.columns:
            continue
        vals = pd.to_numeric(per_fold[key], errors="coerce").dropna()
        if vals.empty:
            continue
        summary_rows.append(
            {
                "metric": key,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean_std": _format_mean_std(
                    float(vals.mean()),
                    float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                ),
            }
        )
    return per_fold, pd.DataFrame(summary_rows)


def write_markdown_table(summary: pd.DataFrame, path: Path) -> None:
    preferred = [
        "mae",
        "rmse",
        "r2",
        "mae_E",
        "mae_R",
        "rmse_E",
        "rmse_R",
        "r2_E",
        "r2_R",
        "mu_res_mae",
    ]
    rows = summary[summary["metric"].isin(preferred)].copy()
    rows["order"] = rows["metric"].map({k: i for i, k in enumerate(preferred)})
    rows = rows.sort_values("order")
    lines = [
        "| Metric | Mean +/- SD | Min | Max |",
        "|---|---:|---:|---:|",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"| {r['metric']} | {r['mean_std']} | {float(r['min']):.4f} | {float(r['max']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run system-level, approximately 8:1:1 k-fold CV for PSMI."
    )
    ap.add_argument(
        "--excel",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx",
        help="Processed ternary LLE workbook.",
    )
    ap.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "data_splitting"
            / "kfold_cv"
        ),
        help="Directory for fold manifests, metrics, and run artifacts.",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=10,
        help="10 approximates an 8:1:1 train/validation/test ratio.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-fold", "--start_fold", dest="start_fold", type=int, default=0)
    ap.add_argument(
        "--end-fold",
        "--end_fold",
        dest="end_fold",
        type=int,
        default=None,
        help="Exclusive. Default: folds.",
    )
    ap.add_argument("--epochs", type=int, default=None, help="Override config.EPOCHS.")
    ap.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override config.EARLY_STOP_PATIENCE.",
    )
    ap.add_argument(
        "--batch-size-graph",
        "--batch_size_graph",
        dest="batch_size_graph",
        type=int,
        default=None,
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--min-points-per-group",
        type=int,
        default=C.MIN_POINTS_PER_GROUP,
        help="Minimum tie-line points retained for each system-temperature group.",
    )
    ap.add_argument(
        "--skip-existing",
        "--skip_existing",
        dest="skip_existing",
        action="store_true",
        help="Reuse fold metrics if already present.",
    )
    ap.add_argument(
        "--no-permute23",
        "--no_permute23",
        dest="no_permute23",
        action="store_true",
        help="Disable component-2/3 augmentation.",
    )
    ap.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Only write the split manifest; do not train.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 3:
        raise ValueError("--folds must be at least 3.")
    if args.min_points_per_group < 1:
        raise ValueError("--min-points-per-group must be at least 1.")
    end_fold = args.end_fold if args.end_fold is not None else args.folds
    if not (0 <= args.start_fold < end_fold <= args.folds):
        raise ValueError("Fold range must satisfy 0 <= start-fold < end-fold <= folds.")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    excel = args.excel
    if not excel.is_absolute():
        excel = PROJECT_ROOT / excel
    if not excel.is_file():
        raise FileNotFoundError(f"Dataset not found: {excel}")
    nrtl_path = PROJECT_ROOT / "datasets" / "parameters" / "nrtl_params_all.json"

    config_updates = {
        "EXCEL_PATH": str(excel),
        "OUT_DIR": str(out_root),
        "SEED": int(args.seed),
        "MIN_POINTS_PER_GROUP": int(args.min_points_per_group),
        "PERMUTE_23_AUG": not bool(args.no_permute23),
        "LOAD_CKPT_PATH": "",
        "NRTL_EVAL_PARAMS_PATH": str(nrtl_path),
    }
    if args.epochs is not None:
        config_updates["EPOCHS"] = int(args.epochs)
    if args.patience is not None:
        config_updates["EARLY_STOP_PATIENCE"] = int(args.patience)
    if args.batch_size_graph is not None:
        config_updates["BATCH_SIZE_GRAPH"] = int(args.batch_size_graph)
    if args.device is not None:
        config_updates["DEVICE"] = str(args.device)

    with temporary_config(**config_updates):
        set_seed(args.seed)
        df_raw, df_aug = load_and_prepare_excel(
            str(excel),
            min_points_per_group=C.MIN_POINTS_PER_GROUP,
            permute_23_aug=C.PERMUTE_23_AUG,
        )
        system_folds = build_stratified_system_folds(df_aug, folds=args.folds, seed=args.seed)
        manifest = []
        for i in range(args.folds):
            train_df, val_df, test_df = make_811_fold_split(df_aug, system_folds, i)
            manifest.append(
                {
                    "fold": i + 1,
                    "min_points_per_group": int(args.min_points_per_group),
                    "component_permutation_augmented": not bool(args.no_permute23),
                    "train_systems": int(train_df["system_id"].nunique()),
                    "val_systems": int(val_df["system_id"].nunique()),
                    "test_systems": int(test_df["system_id"].nunique()),
                    "train_points": int(len(train_df)),
                    "val_points": int(len(val_df)),
                    "test_points": int(len(test_df)),
                    "test_system_ids": ",".join(
                        map(str, sorted(set(test_df["system_id"].tolist())))
                    ),
                    "val_system_ids": ",".join(
                        map(str, sorted(set(val_df["system_id"].tolist())))
                    ),
                }
            )
        pd.DataFrame(manifest).to_csv(
            out_root / "split_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if args.dry_run:
            print(f"[DRY RUN] Wrote split manifest to {out_root / 'split_manifest.csv'}")
            return

        rows: List[Dict] = []
        for fold_idx in range(args.start_fold, end_fold):
            fold_no = fold_idx + 1
            fold_dir = out_root / f"fold_{fold_no:02d}"
            metrics_path = fold_dir / "best_metrics.json"
            if args.skip_existing and metrics_path.is_file():
                print(f"[Fold {fold_no}] Reusing existing metrics: {metrics_path}")
            else:
                print(f"\n========== Fold {fold_no}/{args.folds} ==========")
                train_df, val_df, test_df = make_811_fold_split(
                    df_aug, system_folds, fold_idx
                )
                print(
                    f"train={len(train_df)} ({train_df['system_id'].nunique()} systems), "
                    f"val={len(val_df)} ({val_df['system_id'].nunique()} systems), "
                    f"test={len(test_df)} ({test_df['system_id'].nunique()} systems)"
                )
                fold_dir.mkdir(parents=True, exist_ok=True)
                with temporary_config(OUT_DIR=str(fold_dir), SEED=int(args.seed + fold_idx)):
                    set_seed(int(args.seed + fold_idx))
                    train_or_load(train_df, val_df, test_df)

            if metrics_path.is_file():
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
                train_df, val_df, test_df = make_811_fold_split(
                    df_aug, system_folds, fold_idx
                )
                best_test = metrics.get("best_test", {}) or {}
                row = {
                    "fold": fold_no,
                    "best_epoch": metrics.get("best_epoch", np.nan),
                    "train_systems": int(train_df["system_id"].nunique()),
                    "val_systems": int(val_df["system_id"].nunique()),
                    "test_systems": int(test_df["system_id"].nunique()),
                    "train_points": int(len(train_df)),
                    "val_points": int(len(val_df)),
                    "test_points": int(len(test_df)),
                }
                for key in METRIC_KEYS:
                    row[key] = _safe_get(best_test, key)
                rows.append(row)

        if rows:
            per_fold, summary = summarize_rows(rows)
            per_fold.to_csv(
                out_root / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig"
            )
            summary.to_csv(out_root / "cv_summary.csv", index=False, encoding="utf-8-sig")
            write_markdown_table(summary, out_root / "cv_summary.md")
            try:
                with pd.ExcelWriter(out_root / "cv_results.xlsx", engine="openpyxl") as writer:
                    per_fold.to_excel(writer, sheet_name="per_fold", index=False)
                    summary.to_excel(writer, sheet_name="summary", index=False)
                    pd.DataFrame(manifest).to_excel(
                        writer, sheet_name="split_manifest", index=False
                    )
            except Exception as exc:
                print(f"[WARN] Could not write Excel summary: {exc}")
            print(f"\nSaved CV results to: {out_root}")
            print(summary[["metric", "mean_std", "min", "max"]].to_string(index=False))
        else:
            print("[WARN] No fold metrics were collected.")


if __name__ == "__main__":
    main()

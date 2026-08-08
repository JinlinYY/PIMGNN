"""Run the shared tabular, sequence, and classical baseline comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ._bootstrap import add_baseline_package_to_path
except ImportError:
    from _bootstrap import add_baseline_package_to_path

add_baseline_package_to_path()

from psmi_baselines.common import config as C
from psmi_baselines.common.main_compare import main as run_comparison
from psmi_baselines.paths import EXPERIMENT_ROOT, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    """Parse repository-level overrides for the classical comparison."""
    parser = argparse.ArgumentParser(description="Run classical PSMI comparison models.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "runs" / "classical",
    )
    parser.add_argument("--models", nargs="+", default=C.MODELS_TO_RUN)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--device", default=C.DEVICE)
    parser.add_argument("--no-permute23", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Apply CLI overrides and run the comparison pipeline."""
    args = parse_args()
    if not args.excel.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.excel}")
    C.EXCEL_PATH = str(args.excel)
    C.OUT_DIR = str(args.out_dir)
    C.MODELS_TO_RUN = list(args.models)
    C.SEED = int(args.seed)
    C.EPOCHS = int(args.epochs)
    C.DEVICE = str(args.device)
    C.PERMUTE_23_AUG = not bool(args.no_permute23)
    run_comparison()


if __name__ == "__main__":
    main()

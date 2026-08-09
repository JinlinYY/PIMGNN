"""Run the PSMI sample-major benchmark for multiple fixed seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
STAGE1_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "main_benchmark_stage1.yaml"
STAGE2_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "main_benchmark_stage2.yaml"
EXPANDED_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "expanded_lle_finetune.yaml"


def _run(config: Path, seed: int, output: Path, source: Path | None, force: bool) -> None:
    """Run one stage unless its best checkpoint already exists."""
    best_checkpoint = output / "best_model.pt"
    if best_checkpoint.is_file() and not force:
        print(f"Skip completed stage: {best_checkpoint}")
        return
    if source is not None and not source.is_file():
        raise FileNotFoundError(f"Required source checkpoint is missing: {source}")

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(config),
        "--set",
        f"SEED={seed}",
        "--set",
        f"OUT_DIR={output.relative_to(PROJECT_ROOT).as_posix()}",
    ]
    if source is not None:
        command.extend(
            ["--set", f"LOAD_CKPT_PATH={source.relative_to(PROJECT_ROOT).as_posix()}"]
        )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    """Parse seeds and pipeline stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("stage1", "stage2", "expanded"),
        default=["stage1", "stage2", "expanded"],
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute requested stages in dependency order for every seed."""
    args = parse_args()
    requested = set(args.stages)
    for seed in args.seeds:
        stage1_dir = PROJECT_ROOT / "results" / "main_benchmark" / "sample_major" / f"seed{seed}" / "stage1_supervised"
        stage2_dir = PROJECT_ROOT / "results" / "main_benchmark" / "sample_major" / f"seed{seed}" / "stage2_physics"
        expanded_dir = PROJECT_ROOT / "results" / "transfer_evaluation" / "expanded_lle" / "sample_major" / f"seed{seed}"
        if "stage1" in requested:
            _run(STAGE1_CONFIG, seed, stage1_dir, None, args.force)
        if "stage2" in requested:
            _run(STAGE2_CONFIG, seed, stage2_dir, stage1_dir / "best_model.pt", args.force)
        if "expanded" in requested:
            _run(EXPANDED_CONFIG, seed, expanded_dir, stage2_dir / "best_model.pt", args.force)


if __name__ == "__main__":
    main()

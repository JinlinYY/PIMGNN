"""Run single-seed or multi-seed GLAM comparison experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ._bootstrap import add_baseline_package_to_path
except ImportError:
    from _bootstrap import add_baseline_package_to_path

add_baseline_package_to_path()

from psmi_baselines.glam.config import Config
from psmi_baselines.glam.train import main as run_single_seed
from psmi_baselines.glam.train import main_multi_seed
from psmi_baselines.paths import EXPERIMENT_ROOT


def parse_args() -> argparse.Namespace:
    """Parse GLAM experiment options."""
    parser = argparse.ArgumentParser(description="Run the GLAM LLE baseline.")
    parser.add_argument("--single-seed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "runs" / "glam",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Dispatch to the selected GLAM training mode."""
    args = parse_args()
    if args.single_seed:
        run_single_seed(
            config=Config(),
            seed=args.seed,
            base_output_dir=str(args.output_dir),
            resume_checkpoint=str(args.resume) if args.resume else None,
            auto_resume=args.auto_resume,
        )
        return
    main_multi_seed(seeds=args.seeds, base_output_dir=str(args.output_dir))


if __name__ == "__main__":
    main()

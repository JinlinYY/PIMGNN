"""Evaluate a trained GLAM comparison model."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ._bootstrap import add_baseline_package_to_path
except ImportError:
    from _bootstrap import add_baseline_package_to_path

add_baseline_package_to_path()

from psmi_baselines.glam.test import main as evaluate


def parse_args() -> argparse.Namespace:
    """Parse GLAM evaluation options."""
    parser = argparse.ArgumentParser(description="Evaluate a trained GLAM model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip plots derived from the training history.",
    )
    return parser.parse_args()


def main() -> None:
    """Run GLAM evaluation with an explicit checkpoint."""
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")
    evaluate(model_path=str(args.model), test_only=args.test_only)


if __name__ == "__main__":
    main()

"""Export the canonical main-text Figure 2a image."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "section_3_results"
    / "3_1_lle_prediction"
    / "main_benchmark"
    / "figures"
    / "figure_2a_parity.png"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "figure_2a.png"


def parse_args() -> argparse.Namespace:
    """Parse optional source and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    """Copy the section-aligned Figure 2a image into the public results directory."""
    args = parse_args()
    source = _resolve(args.source)
    output = _resolve(args.output)
    if not source.is_file():
        raise FileNotFoundError(f"Missing Figure 2a source image: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    print(f"Exported Figure 2a to {output}")
    print(f"SHA-256: {_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

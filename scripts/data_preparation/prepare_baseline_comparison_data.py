"""Export one shared system-level split for all LLE comparison models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts._bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from psmi.data import load_and_prepare_excel, split_by_manifest


DEFAULT_EXCEL = (
    PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "splits" / "main_benchmark_corrected_v2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "processed" / "baseline_comparison"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest used to identify an input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_schema(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Add the legacy headers required by the imported comparison models."""
    output = frame.copy()
    output["LLE system NO."] = output["system_id"]
    output["T/K"] = output["T"]
    output["IL (Component 1) full name SMILES"] = output["smiles1"]
    output["Component 2 SMILES"] = output["smiles2"]
    output["Component 3 SMILES"] = output["smiles3"]
    output["smiles"] = (
        output["smiles1"].astype(str)
        + "."
        + output["smiles2"].astype(str)
        + "."
        + output["smiles3"].astype(str)
    )
    output["split"] = split_name
    preferred = [
        "system_id",
        "LLE system NO.",
        "T",
        "T/K",
        "P",
        "smiles1",
        "smiles2",
        "smiles3",
        "IL (Component 1) full name SMILES",
        "Component 2 SMILES",
        "Component 3 SMILES",
        "smiles",
        "Ex1",
        "Ex2",
        "Ex3",
        "Rx1",
        "Rx2",
        "Rx3",
        "t",
        "split",
    ]
    return output[[column for column in preferred if column in output.columns]]


def parse_args() -> argparse.Namespace:
    """Parse dataset export options."""
    parser = argparse.ArgumentParser(description="Prepare shared baseline comparison CSV files.")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-points-per-group", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    """Create total and system-exclusive split tables."""
    args = parse_args()
    if not args.excel.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.excel}")
    if not args.split_manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {args.split_manifest}")
    if args.min_points_per_group < 1:
        raise ValueError("--min-points-per-group must be at least 1.")

    raw, _ = load_and_prepare_excel(
        str(args.excel),
        min_points_per_group=args.min_points_per_group,
        permute_23_aug=False,
    )
    train, validation, test = split_by_manifest(raw, args.split_manifest)

    split_frames = {
        "train": _export_schema(train, "train"),
        "validation": _export_schema(validation, "validation"),
        "test": _export_schema(test, "test"),
    }
    system_sets = {name: set(frame["system_id"]) for name, frame in split_frames.items()}
    if system_sets["train"] & system_sets["validation"]:
        raise RuntimeError("Train and validation systems overlap.")
    if system_sets["train"] & system_sets["test"]:
        raise RuntimeError("Train and test systems overlap.")
    if system_sets["validation"] & system_sets["test"]:
        raise RuntimeError("Validation and test systems overlap.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in split_frames.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    total = pd.concat(split_frames.values(), ignore_index=True)
    total.to_csv(args.out_dir / "total.csv", index=False, encoding="utf-8-sig")

    with args.split_manifest.open("r", encoding="utf-8") as stream:
        canonical_manifest = json.load(stream)
    manifest_rows = []
    for split_name, systems in system_sets.items():
        manifest_rows.extend(
            {
                "system_id": system_id,
                "split": split_name,
                "source_manifest": str(args.split_manifest.relative_to(PROJECT_ROOT)),
                "source_manifest_sha256": _sha256(args.split_manifest),
                "dataset_sha256": _sha256(args.excel),
                "seed": canonical_manifest.get("seed"),
                "min_points_per_group": args.min_points_per_group,
            }
            for system_id in sorted(systems)
        )
    pd.DataFrame(manifest_rows).to_csv(
        args.out_dir / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Saved {len(total)} rows and {len(manifest_rows)} systems to: {args.out_dir}")


if __name__ == "__main__":
    main()

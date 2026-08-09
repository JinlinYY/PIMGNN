"""Create the fixed 575/72/72 system split for expanded-LLE fine-tuning."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from psmi.data import (
    load_and_prepare_excel,
    split_by_manifest,
    stratified_split_by_system,
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse deterministic split-construction arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "LLE-literature-data-boosted.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "splits" / "expanded_lle_system_split.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--n-bins", type=int, default=3)
    parser.add_argument("--min-bin-size", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Build, serialize, and independently validate the expanded-data split."""
    args = parse_args()
    dataset_path = args.dataset.resolve()
    raw, augmented = load_and_prepare_excel(
        str(dataset_path),
        args.min_points,
        True,
    )
    train, validation, test = stratified_split_by_system(
        augmented,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
        n_bins=args.n_bins,
        min_bin_size=args.min_bin_size,
    )
    partitions = {
        "train": sorted(int(value) for value in train["system_id"].unique()),
        "validation": sorted(int(value) for value in validation["system_id"].unique()),
        "test": sorted(int(value) for value in test["system_id"].unique()),
    }
    counts = {name: len(ids) for name, ids in partitions.items()}
    if counts != {"train": 575, "validation": 72, "test": 72}:
        raise ValueError(f"Unexpected expanded split counts: {counts}")

    payload = {
        "schema_version": 1,
        "name": "expanded_lle_system_split",
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": sha256_file(dataset_path),
        "filter": {"minimum_tie_lines_per_system_temperature": int(args.min_points)},
        "augmentation": "component_2_3_permutation_after_system_split",
        "seed": int(args.seed),
        "strategy": "stratified_by_system_size",
        "stratification": {
            "n_bins": int(args.n_bins),
            "minimum_bin_size": int(args.min_bin_size),
        },
        "selection_policy": (
            "All checkpoint selection uses validation systems only; test systems are evaluated once."
        ),
        "counts": {
            "records_without_augmentation": int(len(raw)),
            "systems": int(raw["system_id"].nunique()),
            "train_systems": counts["train"],
            "validation_systems": counts["validation"],
            "test_systems": counts["test"],
        },
        "partitions": partitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    mapping = pd.DataFrame(
        [
            (system_id, partition)
            for partition, ids in partitions.items()
            for system_id in ids
        ],
        columns=["system_id", "partition"],
    ).sort_values("system_id")
    mapping.to_csv(args.output.with_suffix(".csv"), index=False)

    checked = split_by_manifest(raw, args.output)
    print(
        "Validated expanded split: "
        + ", ".join(str(frame["system_id"].nunique()) for frame in checked)
    )


if __name__ == "__main__":
    main()

"""Build a disjoint PSMI split manifest with a locked historical test set."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from psmi.data import load_and_prepare_excel, split_by_manifest


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse split-construction arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx",
    )
    parser.add_argument("--locked-test-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-systems", type=int, default=612)
    parser.add_argument("--min-points", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    """Create and validate the canonical corrected-model split."""
    args = parse_args()
    dataset_path = args.dataset.resolve()
    prediction_path = args.locked_test_predictions.resolve()
    raw, _ = load_and_prepare_excel(str(dataset_path), args.min_points, False)
    predictions = pd.read_csv(prediction_path)
    if "system_id" not in predictions.columns:
        raise KeyError(f"Prediction table has no system_id column: {prediction_path}")

    all_ids = {int(value) for value in raw["system_id"].unique()}
    test_ids = {int(value) for value in predictions["system_id"].unique()}
    if not test_ids or not test_ids <= all_ids:
        raise ValueError("Locked test systems are empty or absent from the filtered dataset")
    test_rows = int(raw[raw["system_id"].isin(test_ids)].shape[0])
    if test_rows != len(predictions):
        raise ValueError(
            f"Locked prediction rows ({len(predictions)}) do not match filtered test rows ({test_rows})"
        )

    remaining = np.asarray(sorted(all_ids - test_ids), dtype=np.int64)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(remaining)
    if args.train_systems <= 0 or args.train_systems >= len(remaining):
        raise ValueError("train-systems must leave at least one validation system")
    train_ids = sorted(int(value) for value in remaining[: args.train_systems])
    validation_ids = sorted(int(value) for value in remaining[args.train_systems :])
    test_ids_sorted = sorted(test_ids)

    payload = {
        "schema_version": 1,
        "name": "main_benchmark_corrected_v2",
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": sha256_file(dataset_path),
        "filter": {"minimum_tie_lines_per_system_temperature": int(args.min_points)},
        "seed": int(args.seed),
        "strategy": "locked_historical_test_then_seeded_train_validation",
        "test_lock_source": str(prediction_path.relative_to(PROJECT_ROOT)),
        "test_lock_source_sha256": sha256_file(prediction_path),
        "selection_policy": (
            "The historical 78-system test set is locked for comparability. "
            "Corrected-model selection must use validation data only; the test set is evaluated once."
        ),
        "counts": {
            "records_without_augmentation": int(len(raw)),
            "systems": int(len(all_ids)),
            "train_systems": int(len(train_ids)),
            "validation_systems": int(len(validation_ids)),
            "test_systems": int(len(test_ids_sorted)),
            "test_records_without_augmentation": test_rows,
        },
        "partitions": {
            "train": train_ids,
            "validation": validation_ids,
            "test": test_ids_sorted,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    mapping = pd.DataFrame(
        [(value, "train") for value in train_ids]
        + [(value, "validation") for value in validation_ids]
        + [(value, "test") for value in test_ids_sorted],
        columns=["system_id", "partition"],
    ).sort_values("system_id")
    mapping.to_csv(args.output.with_suffix(".csv"), index=False)

    train, validation, test = split_by_manifest(raw, args.output)
    print(
        "Validated split: "
        f"train={train['system_id'].nunique()}, "
        f"validation={validation['system_id'].nunique()}, "
        f"test={test['system_id'].nunique()}, test_rows={len(test)}"
    )


if __name__ == "__main__":
    main()

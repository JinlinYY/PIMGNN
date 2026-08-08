"""Remove system-temperature groups with too few tie-line records."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "update-LLE-all-with-smiles_no-missing-smiles.xlsx"
)
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles_min3.xlsx"
REMOVED_REPORT_PATH = PROJECT_ROOT / "datasets" / "processed" / "removed_systems_min3.csv"
MIN_POINTS = 3


def main() -> None:
    """Filter sparse groups and save both retained data and removal metadata."""
    data = pd.read_excel(INPUT_PATH)

    # Temperature is part of the group key when it is available.
    group_keys = ["LLE system NO.", "T/K"] if "T/K" in data.columns else ["LLE system NO."]
    group_sizes = data.groupby(group_keys, dropna=False).size().reset_index(name="n_points")

    # Split group keys by the configured minimum number of tie-line records.
    retained_keys = group_sizes[group_sizes["n_points"] >= MIN_POINTS].copy()
    removed_keys = group_sizes[group_sizes["n_points"] < MIN_POINTS].copy()
    retained_data = data.merge(retained_keys[group_keys], on=group_keys, how="inner")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    retained_data.to_excel(OUTPUT_PATH, index=False)
    removed_keys.to_csv(REMOVED_REPORT_PATH, index=False, encoding="utf-8-sig")

    print("Grouping columns:", group_keys)
    print("Input rows:", len(data))
    print("Retained rows:", len(retained_data))
    print("Input groups:", len(group_sizes))
    print(f"Retained groups (>= {MIN_POINTS}):", len(retained_keys))
    print(f"Removed groups (< {MIN_POINTS}):", len(removed_keys))
    print("Output file:", OUTPUT_PATH)
    print("Removal report:", REMOVED_REPORT_PATH)


if __name__ == "__main__":
    main()

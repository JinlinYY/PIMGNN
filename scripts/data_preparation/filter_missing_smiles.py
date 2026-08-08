"""Remove LLE records with missing component SMILES values."""

from pathlib import Path
import re

import pandas as pd


# Keep preprocessing inputs and outputs under the structured dataset directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "update-LLE-all-with-smiles_no-missing-smiles.xlsx"
)


def is_blank(value: object) -> bool:
    """Return whether a spreadsheet value is empty or a null marker."""
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def main() -> None:
    """Filter the workbook and write the retained records."""
    data = pd.read_excel(INPUT_PATH)
    smiles_columns = [
        column for column in data.columns if re.search(r"smiles", str(column), re.IGNORECASE)
    ]
    if not smiles_columns:
        raise ValueError("No column containing 'SMILES' was found in the workbook header.")

    # A record is removed when any required component SMILES value is missing.
    missing_any = data[smiles_columns].apply(lambda column: column.map(is_blank)).any(axis=1)
    cleaned = data.loc[~missing_any].copy()
    cleaned.to_excel(OUTPUT_PATH, index=False)

    print("SMILES columns:", smiles_columns)
    print("Input rows:", len(data))
    print("Removed rows:", int(missing_any.sum()))
    print("Retained rows:", len(cleaned))
    print("Output file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

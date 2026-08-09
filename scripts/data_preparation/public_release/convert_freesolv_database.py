"""Convert the semicolon-delimited FreeSolv database text file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_database(path: Path) -> pd.DataFrame:
    """Extract SMILES and experimental values from the FreeSolv text format."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split(";")
            if len(parts) < 4:
                continue
            smiles = parts[1].strip()
            if not smiles or smiles.replace(".", "", 1).isdigit():
                continue
            try:
                value = float(parts[3].strip())
            except ValueError:
                continue
            index = len(rows)
            rows.append(
                {
                    "system_id": 10000 + index,
                    "smiles1": smiles,
                    "smiles2": "O",
                    "smiles3": "O",
                    "T": 298.15,
                    "Ex1": 0.01,
                    "Ex2": 0.99,
                    "Ex3": 0.0,
                    "Rx1": 0.01,
                    "Rx2": 0.99,
                    "Rx3": 0.0,
                    "value": value,
                    "split": "train" if index % 10 < 8 else "test",
                }
            )
    if not rows:
        raise ValueError(f"No valid FreeSolv records were found in {path}")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """Parse text conversion options."""
    parser = argparse.ArgumentParser(description="Convert a FreeSolv database.txt file.")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/external/freesolv/FreeSolv_Ready.xlsx"),
    )
    return parser.parse_args()


def main() -> None:
    """Convert the source text file to an Excel workbook."""
    args = parse_args()
    frame = parse_database(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(args.output, index=False)
    print(f"Saved {len(frame)} FreeSolv rows to {args.output}")


if __name__ == "__main__":
    main()

"""Create the ten-row FreeSolv example included in the public archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXAMPLE_ROWS = [
    ("CN(C)C(=O)c1ccc(cc1)C#N", -11.01),
    ("CS(=O)(=O)Cl", -4.87),
    ("CC(C)C=C", 1.83),
    ("CCc1cnccn1", -5.45),
    ("CCCCCCCO", -4.21),
    ("Cc1cc(C)n(C)c1", -4.53),
    ("CC(C)C(C)C", 2.25),
    ("C1CCNC1", -4.16),
    ("CCOc1ccc(cc1)CC", -4.08),
    ("Cc1cccc(c1)N(C)C", -4.22),
]


def build_example() -> pd.DataFrame:
    """Return the deterministic example table."""
    rows = []
    for index, (smiles, value) in enumerate(EXAMPLE_ROWS):
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
                "split": "train" if index < 6 else "test",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Write the example workbook."""
    parser = argparse.ArgumentParser(description="Build the archived ten-row FreeSolv example.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/external/freesolv/FreeSolv_example.xlsx"),
    )
    args = parser.parse_args()
    frame = build_example()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(args.output, index=False)
    print(f"Saved {len(frame)} example rows to {args.output}")


if __name__ == "__main__":
    main()

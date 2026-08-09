"""Download FreeSolv and export the checkpoint-compatible pseudo-ternary workbook."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests


DEFAULT_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/freesolv.csv"


def convert_freesolv(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert FreeSolv rows to the schema used by binary transfer experiments."""
    required = {"smiles", "expt"}
    if not required.issubset(frame.columns):
        raise KeyError(f"FreeSolv table must contain {sorted(required)}")
    output = pd.DataFrame(
        {
            "system_id": range(10000, 10000 + len(frame)),
            "smiles1": frame["smiles"],
            "smiles2": "O",
            "smiles3": "O",
            "T": 298.15,
            "Ex1": 0.01,
            "Ex2": 0.99,
            "Ex3": 0.0,
            "Rx1": 0.01,
            "Rx2": 0.99,
            "Rx3": 0.0,
            "value": frame["expt"],
        }
    )
    split_at = int(0.8 * len(output))
    output["split"] = ["train" if index < split_at else "test" for index in range(len(output))]
    return output


def parse_args() -> argparse.Namespace:
    """Parse FreeSolv download options."""
    parser = argparse.ArgumentParser(description="Download and convert FreeSolv.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/external/freesolv/FreeSolv_Ready.xlsx"),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    """Download, convert, and save FreeSolv."""
    args = parse_args()
    response = requests.get(args.url, timeout=args.timeout)
    response.raise_for_status()
    source = pd.read_csv(io.StringIO(response.text))
    output = convert_freesolv(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_excel(args.output, index=False)
    print(f"Saved {len(output)} FreeSolv rows to {args.output}")


if __name__ == "__main__":
    main()

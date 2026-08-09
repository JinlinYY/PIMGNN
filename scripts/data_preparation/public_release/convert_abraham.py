"""Convert an Abraham CSV table to the checkpoint-compatible pseudo-ternary schema."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from ._bootstrap import add_src_to_path
except ImportError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from psmi_checkpoint_compat.utils import canonicalize_smiles


def load_abraham_csv_as_pseudo_ternary(
    path: str | Path,
    target_col: str = "L",
    missing_value: float = -123.0,
    t_value: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Return cleaned pseudo-ternary rows and the system-name mapping."""
    frame = pd.read_csv(path)
    required = ["system_id", "smiles1", "smiles2", "smiles3", "T", target_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Abraham CSV is missing columns: {missing}")

    frame = frame.copy()
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna(subset=[target_col])
    frame = frame[frame[target_col] != float(missing_value)].copy()
    for column in ("smiles1", "smiles2", "smiles3"):
        frame[column] = frame[column].astype(str).map(canonicalize_smiles)
    frame = frame[(frame[["smiles1", "smiles2", "smiles3"]] != "").all(axis=1)]

    names = frame["system_id"].astype(str)
    name_to_id = {name: index for index, name in enumerate(sorted(names.unique()))}
    frame["system_id_str"] = names
    frame["system_id"] = names.map(name_to_id).astype(int)
    frame["T"] = pd.to_numeric(frame["T"], errors="coerce")
    frame = frame.dropna(subset=["T"])
    frame["t"] = float(t_value)
    frame["y"] = frame[target_col].astype(np.float32)

    columns = [
        "system_id",
        "system_id_str",
        "smiles1",
        "smiles2",
        "smiles3",
        "T",
        "t",
        "y",
    ]
    return frame[columns].reset_index(drop=True), name_to_id


def parse_args() -> argparse.Namespace:
    """Parse Abraham conversion options."""
    parser = argparse.ArgumentParser(description="Convert Abraham data for checkpoint-compatible PSMI workflows.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--target-column", default="L")
    parser.add_argument("--missing-value", type=float, default=-123.0)
    parser.add_argument("--t-value", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Convert and save an Abraham dataset."""
    args = parse_args()
    frame, mapping = load_abraham_csv_as_pseudo_ternary(
        args.input_csv,
        target_col=args.target_column,
        missing_value=args.missing_value,
        t_value=args.t_value,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {len(frame)} rows and {len(mapping)} systems to {args.output_csv}")


if __name__ == "__main__":
    main()

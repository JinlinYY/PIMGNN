"""Shared safeguards for reproducible comparison experiments."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def canonical_split_indices(frame: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
    """Return validated row indices from the exported canonical split column."""
    if "split" not in frame.columns:
        return None

    labels = frame["split"].astype(str).str.strip().str.lower().replace({"val": "validation"})
    expected = {"train", "validation", "test"}
    observed = set(labels.unique())
    if observed != expected:
        raise ValueError(f"Expected split labels {sorted(expected)}, got {sorted(observed)}.")

    indices = {label: np.flatnonzero(labels.to_numpy() == label) for label in expected}
    if any(len(values) == 0 for values in indices.values()):
        raise ValueError("Every canonical partition must contain at least one row.")

    system_column = next(
        (column for column in ("system_id", "LLE system NO.") if column in frame.columns),
        None,
    )
    if system_column is not None:
        systems = {
            label: set(frame.iloc[row_indices][system_column].tolist())
            for label, row_indices in indices.items()
        }
        if (
            systems["train"] & systems["validation"]
            or systems["train"] & systems["test"]
            or systems["validation"] & systems["test"]
        ):
            raise ValueError("Canonical comparison partitions overlap by system identifier.")
    return indices

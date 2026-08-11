"""Chemical-system identity and nominal-temperature grouping utilities."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .utils import canonicalize_smiles


SMILES_COLUMNS = ("smiles1", "smiles2", "smiles3")


def chemical_system_signature(smiles: Sequence[object]) -> str:
    """Return an order-invariant signature for a ternary chemical system."""
    canonical = [canonicalize_smiles(value) for value in smiles]
    if any(not value for value in canonical):
        raise ValueError("All three component SMILES must be valid")
    return "||".join(sorted(canonical))


def add_chemical_system_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic canonical-chemistry signatures and integer IDs."""
    missing = [column for column in SMILES_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing SMILES columns required for chemical identity: {missing}")
    output = df.copy()
    output["chemical_system_signature"] = output[list(SMILES_COLUMNS)].apply(
        lambda row: chemical_system_signature(row.tolist()), axis=1
    )
    signatures = sorted(output["chemical_system_signature"].unique().tolist())
    identifiers = {signature: index + 1 for index, signature in enumerate(signatures)}
    output["chemical_system_id"] = (
        output["chemical_system_signature"].map(identifiers).astype(int)
    )
    return output


def merge_nearby_temperature_levels(
    df: pd.DataFrame,
    system_column: str = "chemical_system_signature",
    tolerance_K: float = 0.1,
) -> pd.DataFrame:
    """Merge nominal temperatures whose full within-system span is within tolerance."""
    if system_column not in df.columns:
        raise KeyError(f"System grouping column is missing: {system_column}")
    if tolerance_K <= 0:
        output = df.copy()
        output["T_original"] = pd.to_numeric(output["T"], errors="coerce")
        return output

    output = df.copy()
    output["T"] = pd.to_numeric(output["T"], errors="coerce")
    output["T_original"] = output["T"].astype(float)
    for _, indices in output.groupby(system_column, sort=False).groups.items():
        temperatures = sorted(output.loc[indices, "T"].astype(float).unique().tolist())
        clusters: List[List[float]] = []
        for temperature in temperatures:
            if not clusters or temperature - clusters[-1][0] > float(tolerance_K) + 1e-9:
                clusters.append([temperature])
            else:
                clusters[-1].append(temperature)
        mapping: Dict[float, float] = {}
        for cluster in clusters:
            representative = float(np.mean(cluster))
            mapping.update({temperature: representative for temperature in cluster})
        output.loc[indices, "T"] = output.loc[indices, "T"].map(mapping)
    return output

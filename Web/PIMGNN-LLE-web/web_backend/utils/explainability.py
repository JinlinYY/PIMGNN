from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

import config as C


def _read_importance_csv(path: Path, limit: int = 8) -> List[dict]:
    if not path.is_file():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "name" not in df.columns or "importance" not in df.columns:
        return []
    df = df.sort_values("importance", ascending=False).head(limit)
    return [
        {"name": str(row["name"]), "importance": float(row["importance"])}
        for _, row in df.iterrows()
    ]


def explainability_summary() -> dict:
    explain_dir = Path(getattr(C, "EXPLAIN_DIR", ""))
    target_dir = explain_dir / "target_ALL" if (explain_dir / "target_ALL").is_dir() else explain_dir

    return {
        "source": str(target_dir),
        "mechanism_notes": [
            "The model reconstructs liquid-liquid equilibrium as a continuous family of tie-lines across composition space.",
            "Global molecular descriptors capture polarity, lipophilicity, aromaticity, size, and hydrogen-bonding capacity.",
            "Mixture interaction features encode compatibility and mismatch signals between component pairs.",
            "Importance values are precomputed saliency summaries from evaluation runs, not per-request attribution.",
        ],
        "global_features": _read_importance_csv(target_dir / "g1_global_feature_importance.csv"),
        "mixture_features": _read_importance_csv(target_dir / "mix_edge_feature_importance_gradxnorm.csv"),
        "atom_features": {
            "component_1": _read_importance_csv(target_dir / "g1_atom_feature_importance.csv", limit=6),
            "component_2": _read_importance_csv(target_dir / "g2_atom_feature_importance.csv", limit=6),
            "component_3": _read_importance_csv(target_dir / "g3_atom_feature_importance.csv", limit=6),
        },
    }

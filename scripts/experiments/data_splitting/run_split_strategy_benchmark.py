# -*- coding: utf-8 -*-
"""Benchmark PSMI under different train/validation/test split principles.

The script reuses the project training pipeline and changes only the DataFrame
split passed to ``train_or_load``.

Supported split schemes:
  - point_random: random equilibrium-point split; systems can appear in multiple sets.
  - system_random: random system_id split; each system appears in only one set.
  - structure_family_triple: hold out RDKit-derived three-component structure-family groups.
  - temperature_high: hold out high-temperature systems for testing.
  - temperature_low: hold out low-temperature systems for testing.

Example:
  python scripts/experiments/data_splitting/run_split_strategy_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._bootstrap import add_src_to_path

add_src_to_path()

from psmi import config as C
from psmi.data import load_and_prepare_excel, stratified_split_by_system
from psmi.train import train_or_load
from psmi.utils import canonicalize_smiles, set_seed


DEFAULT_SCHEMES = [
    "point_random",
    "system_random",
    "structure_family_triple",
    "temperature_high",
    "temperature_low",
]

DEFAULT_REPEAT_SEEDS = [42, 7, 13, 23, 31, 47, 59, 73, 89, 97]

METRIC_KEYS = [
    "mae",
    "rmse",
    "r2",
    "mae_E",
    "mae_R",
    "rmse_E",
    "rmse_R",
    "r2_E",
    "r2_R",
    "mu_res_mae",
    "mu_res_rmse",
    "tpd_viol_rate",
]


@contextmanager
def temporary_config(**kwargs):
    old = {k: getattr(C, k, None) for k in kwargs}
    missing = {k for k in kwargs if not hasattr(C, k)}
    try:
        for k, v in kwargs.items():
            setattr(C, k, v)
        yield
    finally:
        for k, v in old.items():
            if k in missing:
                delattr(C, k)
            else:
                setattr(C, k, v)


def _split_list(items: Sequence, train_ratio: float, val_ratio: float) -> Tuple[set, set, set]:
    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    elif n == 2:
        n_train, n_val = 1, 0
    elif n == 1:
        n_train, n_val = 1, 0
    train = set(items[:n_train])
    val = set(items[n_train:n_train + n_val])
    test = set(items[n_train + n_val:])
    return train, val, test


def split_point_random(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    rng = np.random.RandomState(seed)
    idx = df.index.to_numpy()
    rng.shuffle(idx)
    train_idx, val_idx, test_idx = _split_list(idx.tolist(), train_ratio, val_ratio)
    train_df = df.loc[list(train_idx)].copy()
    val_df = df.loc[list(val_idx)].copy()
    test_df = df.loc[list(test_idx)].copy()
    meta = {
        "split_unit": "equilibrium_point",
        "leakage_note": "systems may appear in multiple subsets",
    }
    return train_df, val_df, test_df, meta


def split_system_random(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    rng = np.random.RandomState(seed)
    systems = sorted(df["system_id"].unique().tolist())
    rng.shuffle(systems)
    train_sys, val_sys, test_sys = _split_list(systems, train_ratio, val_ratio)
    train_df, val_df, test_df = _subset_by_systems(df, train_sys, val_sys, test_sys)
    return train_df, val_df, test_df, {
        "split_unit": "system_id",
        "leakage_note": "system-exclusive",
    }


def split_system_stratified(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    train_df, val_df, test_df = stratified_split_by_system(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        n_bins=8,
        min_bin_size=3,
    )
    return train_df, val_df, test_df, {
        "split_unit": "system_id",
        "leakage_note": "system-exclusive; stratified by rows/T-span/T-groups",
    }


_FG_SMARTS = {
    "carboxylic_acid": "C(=O)[OX2H1]",
    "acid_derivative": "C(=O)[OX2,NX3,SX2]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[OX2H][CX4]",
    "phenol": "c[OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "nitrile": "[CX2]#N",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "halogenated": "[F,Cl,Br,I]",
    "sulfur_containing": "[SX2,SX3,SX4,SX6]",
    "phosphorus_containing": "[PX3,PX4,PX5]",
}
_FG_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in _FG_SMARTS.items()}

_IL_CATION_SMARTS = {
    "imidazolium": "[n+;r5]",
    "pyridinium": "[n+;r6]",
    "pyrrolidinium": "[N+;R;X4;r5]",
    "piperidinium": "[N+;R;X4;r6]",
    "ammonium": "[N+;X4]",
    "phosphonium": "[P+]",
    "sulfonium": "[S+]",
}
_IL_ANION_SMARTS = {
    "bis_triflimide": "S(=O)(=O)[N-]S(=O)(=O)",
    "tetrafluoroborate": "[B-](F)(F)(F)F",
    "hexafluorophosphate": "[P-](F)(F)(F)(F)(F)F",
    "dicyanamide": "N#C[N-]C#N",
    "nitrate": "[N+](=O)([O-])[O-]",
    "alkylsulfate": "[O-]S(=O)(=O)O[#6]",
    "sulfate_sulfonate": "S(=O)(=O)[O-]",
    "carboxylate": "C(=O)[O-]",
    "halide": "[F-,Cl-,Br-,I-]",
}
_IL_CATION_PATTERNS = {
    name: Chem.MolFromSmarts(smarts) for name, smarts in _IL_CATION_SMARTS.items()
}
_IL_ANION_PATTERNS = {
    name: Chem.MolFromSmarts(smarts) for name, smarts in _IL_ANION_SMARTS.items()
}


def _formal_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _charged_fragments(mol: Chem.Mol) -> List[Chem.Mol]:
    return [
        frag
        for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if _formal_charge(frag) != 0
    ]


def _is_ionic_liquid_like(smiles: str, mol: Chem.Mol) -> bool:
    frags = _charged_fragments(mol)
    charges = [_formal_charge(frag) for frag in frags]
    if any(c > 0 for c in charges) and any(c < 0 for c in charges):
        return True
    if "." in str(smiles) and any(c != 0 for c in charges):
        return True
    return False


def _first_matching_family(mol: Chem.Mol, patterns: Dict[str, Chem.Mol], default: str) -> str:
    for name, patt in patterns.items():
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
    return default


def _ionic_liquid_family(smiles: str, mol: Chem.Mol) -> str:
    cation_family = "other_cation"
    anion_family = "other_anion"
    for frag in _charged_fragments(mol):
        charge = _formal_charge(frag)
        if charge > 0:
            cation_family = _first_matching_family(frag, _IL_CATION_PATTERNS, cation_family)
        elif charge < 0:
            anion_family = _first_matching_family(frag, _IL_ANION_PATTERNS, anion_family)
    return f"IL:{cation_family}|{anion_family}"


def _is_hydrocarbon(mol: Chem.Mol) -> bool:
    return all(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


def _has_hetero_aromatic_atom(mol: Chem.Mol) -> bool:
    return any(
        atom.GetIsAromatic() and atom.GetAtomicNum() not in {1, 6}
        for atom in mol.GetAtoms()
    )


def _has_unsaturated_cc_bond(mol: Chem.Mol) -> bool:
    return any(
        bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in mol.GetBonds()
    )


def _functional_group_family(smiles: str) -> str:
    """Assign a structure-derived family label using RDKit SMARTS rules."""
    smi = canonicalize_smiles(str(smiles))
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return "unknown"

    if smi == "O":
        return "water"

    matches = []
    for name, patt in _FG_PATTERNS.items():
        if patt is not None and mol.HasSubstructMatch(patt):
            matches.append(name)

    if _has_hetero_aromatic_atom(mol):
        matches.append("heteroaromatic")
    elif any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
        matches.append("aromatic")

    if not matches and _is_hydrocarbon(mol):
        if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
            return "aromatic_hydrocarbon"
        if _has_unsaturated_cc_bond(mol):
            return "unsaturated_hydrocarbon"
        if mol.GetRingInfo().NumRings() > 0:
            return "cycloalkane"
        return "alkane"

    if not matches:
        return "other"

    return "+".join(sorted(set(matches)))


def _functional_group_pair_label(row: pd.Series) -> str:
    f2 = _functional_group_family(row.get("smiles2", ""))
    f3 = _functional_group_family(row.get("smiles3", ""))
    return " + ".join(sorted([f2, f3]))


def _component_structure_family(smiles: str) -> str:
    """Classify each component as IL-like or molecular, then assign a structural family."""
    smi = canonicalize_smiles(str(smiles))
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return "unknown"
    if _is_ionic_liquid_like(smi, mol):
        return _ionic_liquid_family(smi, mol)
    return f"MOL:{_functional_group_family(smi)}"


def _structure_family_triple_label(row: pd.Series) -> str:
    families = [
        _component_structure_family(row.get("smiles1", "")),
        _component_structure_family(row.get("smiles2", "")),
        _component_structure_family(row.get("smiles3", "")),
    ]
    return " + ".join(sorted(families))


def _greedy_group_split(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[set, set, set]:
    """Assign whole groups to train/val/test while approximating row-count ratios."""
    rng = np.random.RandomState(seed)
    counts = df.groupby(group_col).size().reset_index(name="n")
    records = counts.to_dict("records")
    rng.shuffle(records)
    records.sort(key=lambda r: r["n"], reverse=True)

    total = float(counts["n"].sum())
    targets = {
        "train": train_ratio * total,
        "val": val_ratio * total,
        "test": (1.0 - train_ratio - val_ratio) * total,
    }
    buckets = {"train": [], "val": [], "test": []}
    sizes = {"train": 0.0, "val": 0.0, "test": 0.0}

    for rec in records:
        # Choose the bucket farthest below its target after normalizing by target.
        choices = sorted(
            targets.keys(),
            key=lambda k: (sizes[k] / max(targets[k], 1.0), sizes[k]),
        )
        chosen = choices[0]
        buckets[chosen].append(rec[group_col])
        sizes[chosen] += float(rec["n"])

    # Avoid empty validation/test if group counts are very small.
    for dst in ["val", "test"]:
        if not buckets[dst]:
            src = max(["train", "val", "test"], key=lambda k: sizes[k])
            moved = buckets[src].pop()
            buckets[dst].append(moved)
    return set(buckets["train"]), set(buckets["val"]), set(buckets["test"])


def split_structure_family_triple(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    work = df.copy()
    work["_structure_family_triple"] = work.apply(_structure_family_triple_label, axis=1)
    train_g, val_g, test_g = _greedy_group_split(
        work, "_structure_family_triple", seed, train_ratio, val_ratio
    )
    train_df = work[work["_structure_family_triple"].isin(train_g)].drop(
        columns=["_structure_family_triple"]
    ).copy()
    val_df = work[work["_structure_family_triple"].isin(val_g)].drop(
        columns=["_structure_family_triple"]
    ).copy()
    test_df = work[work["_structure_family_triple"].isin(test_g)].drop(
        columns=["_structure_family_triple"]
    ).copy()
    meta = {
        "split_unit": "rdkit_structure_family_triple",
        "leakage_note": "three-component RDKit-derived structure-family triples are exclusive",
        "train_groups": len(train_g),
        "val_groups": len(val_g),
        "test_groups": len(test_g),
    }
    return train_df, val_df, test_df, meta


def split_functional_group_pair(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Backward-compatible alias for the three-component structure-family split."""
    return split_structure_family_triple(df, seed, train_ratio, val_ratio)


def split_family_pair(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Backward-compatible alias for the three-component structure-family split."""
    return split_structure_family_triple(df, seed, train_ratio, val_ratio)


def _temperature_system_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("system_id")
        .agg(T_mean=("T", "mean"), n_rows=("system_id", "size"))
        .reset_index()
        .sort_values(["T_mean", "system_id"])
    )


def split_temperature_ordered(
    df: pd.DataFrame,
    seed: int,
    holdout: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    stats = _temperature_system_table(df)
    systems = stats["system_id"].tolist()
    n = len(systems)
    n_test = max(1, int(round(n * (1.0 - train_ratio - val_ratio))))
    n_val = max(1, int(round(n * val_ratio)))

    if holdout == "high":
        test_sys = set(systems[-n_test:])
        val_sys = set(systems[-(n_test + n_val):-n_test])
        train_sys = set(systems[:-(n_test + n_val)])
    elif holdout == "low":
        test_sys = set(systems[:n_test])
        val_sys = set(systems[n_test:n_test + n_val])
        train_sys = set(systems[n_test + n_val:])
    else:
        raise ValueError(f"Unknown temperature holdout: {holdout}")

    meta = {
        "split_unit": "system_id_ordered_by_mean_temperature",
        "leakage_note": f"system-exclusive; {holdout}-temperature systems held out for test",
        "train_T_mean_min": float(stats[stats["system_id"].isin(train_sys)]["T_mean"].min()),
        "train_T_mean_max": float(stats[stats["system_id"].isin(train_sys)]["T_mean"].max()),
        "test_T_mean_min": float(stats[stats["system_id"].isin(test_sys)]["T_mean"].min()),
        "test_T_mean_max": float(stats[stats["system_id"].isin(test_sys)]["T_mean"].max()),
    }
    train_df, val_df, test_df = _subset_by_systems(df, train_sys, val_sys, test_sys)
    return train_df, val_df, test_df, meta


def _subset_by_systems(
    df: pd.DataFrame,
    train_sys: Iterable,
    val_sys: Iterable,
    test_sys: Iterable,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_sys, val_sys, test_sys = set(train_sys), set(val_sys), set(test_sys)
    train_df = df[df["system_id"].isin(train_sys)].copy()
    val_df = df[df["system_id"].isin(val_sys)].copy()
    test_df = df[df["system_id"].isin(test_sys)].copy()
    return train_df, val_df, test_df


def make_split(
    scheme: str,
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    if scheme == "point_random":
        return split_point_random(df, seed, train_ratio, val_ratio)
    if scheme == "system_random":
        return split_system_random(df, seed, train_ratio, val_ratio)
    if scheme == "system_stratified":
        return split_system_stratified(df, seed, train_ratio, val_ratio)
    if scheme == "structure_family_triple":
        return split_structure_family_triple(df, seed, train_ratio, val_ratio)
    if scheme == "functional_group_pair":
        return split_functional_group_pair(df, seed, train_ratio, val_ratio)
    if scheme == "family_pair":
        return split_family_pair(df, seed, train_ratio, val_ratio)
    if scheme == "temperature_high":
        return split_temperature_ordered(df, seed, "high", train_ratio, val_ratio)
    if scheme == "temperature_low":
        return split_temperature_ordered(df, seed, "low", train_ratio, val_ratio)
    raise ValueError(f"Unknown split scheme: {scheme}")


def _safe_metric(metrics: Dict, key: str) -> float:
    try:
        return float(metrics.get(key, np.nan))
    except Exception:
        return float("nan")


def collect_existing_metrics(
    metrics_path: Path,
    scheme: str,
    run_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta: Dict,
    split_seed: int,
    train_seed: int,
) -> Dict:
    with metrics_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    best_test = obj.get("best_test", {}) or {}
    row = {
        "row_type": "run",
        "scheme": scheme,
        "run_id": run_id,
        "split_seed": int(split_seed),
        "train_seed": int(train_seed),
        "best_epoch": obj.get("best_epoch", np.nan),
        "split_unit": meta.get("split_unit", ""),
        "leakage_note": meta.get("leakage_note", ""),
        "train_systems": int(train_df["system_id"].nunique()),
        "val_systems": int(val_df["system_id"].nunique()),
        "test_systems": int(test_df["system_id"].nunique()),
        "train_points": int(len(train_df)),
        "val_points": int(len(val_df)),
        "test_points": int(len(test_df)),
    }
    for key, value in meta.items():
        if key not in row and isinstance(value, (int, float, str)):
            row[key] = value
    for key in METRIC_KEYS:
        row[key] = _safe_metric(best_test, key)
    return row


def build_combined_table(rows: List[Dict]) -> pd.DataFrame:
    run_df = pd.DataFrame(rows)
    summary_rows: List[Dict] = []
    group_cols = ["scheme"]
    numeric_cols = [
        "best_epoch",
        "train_systems",
        "val_systems",
        "test_systems",
        "train_points",
        "val_points",
        "test_points",
    ] + METRIC_KEYS

    for scheme, sub in run_df.groupby(group_cols, dropna=False):
        scheme_name = scheme[0] if isinstance(scheme, tuple) else scheme
        for stat_name, func in [("mean", "mean"), ("SD", "std"), ("min", "min"), ("max", "max")]:
            out = {"row_type": stat_name, "scheme": scheme_name, "run_id": stat_name}
            for c in numeric_cols:
                vals = pd.to_numeric(sub.get(c), errors="coerce")
                if vals.notna().sum() == 0:
                    continue
                if func == "mean":
                    out[c] = float(vals.mean())
                elif func == "std":
                    out[c] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
                elif func == "min":
                    out[c] = float(vals.min())
                elif func == "max":
                    out[c] = float(vals.max())
            summary_rows.append(out)
    return pd.concat([run_df, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run PSMI under multiple data-split principles.")
    ap.add_argument(
        "--excel",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx",
        help="Processed ternary LLE workbook.",
    )
    ap.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "data_splitting"
            / "split_strategy_benchmark"
        ),
        help="Directory for manifests, metrics, and run artifacts.",
    )
    ap.add_argument("--schemes", nargs="+", default=DEFAULT_SCHEMES)
    ap.add_argument("--repeats", type=int, default=3, help="Repeats for stochastic schemes.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--train-ratio", "--train_ratio", dest="train_ratio", type=float, default=0.8
    )
    ap.add_argument(
        "--val-ratio", "--val_ratio", dest="val_ratio", type=float, default=0.1
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument(
        "--batch-size-graph",
        "--batch_size_graph",
        dest="batch_size_graph",
        type=int,
        default=None,
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--min-points-per-group",
        type=int,
        default=C.MIN_POINTS_PER_GROUP,
        help="Minimum tie-line points retained for each system-temperature group.",
    )
    ap.add_argument("--skip-existing", "--skip_existing", dest="skip_existing", action="store_true")
    ap.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    ap.add_argument("--no-permute23", "--no_permute23", dest="no_permute23", action="store_true")
    return ap.parse_args()


def is_stochastic_scheme(scheme: str) -> bool:
    return scheme in {"point_random", "system_random", "structure_family_triple"}


def build_repeat_seeds(base_seed: int, n_repeats: int) -> List[int]:
    seeds: List[int] = []

    def add_seed(value: int) -> None:
        value = int(value)
        if value not in seeds:
            seeds.append(value)

    add_seed(base_seed)
    for value in DEFAULT_REPEAT_SEEDS:
        add_seed(value)
        if len(seeds) >= n_repeats:
            return seeds[:n_repeats]

    for value in range(0, 101):
        add_seed(value)
        if len(seeds) >= n_repeats:
            return seeds[:n_repeats]

    value = 101
    while len(seeds) < n_repeats:
        add_seed(value)
        value += 1
    return seeds[:n_repeats]


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio and --val-ratio must sum to less than 1.")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.min_points_per_group < 1:
        raise ValueError("--min-points-per-group must be at least 1.")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    excel = args.excel
    if not excel.is_absolute():
        excel = PROJECT_ROOT / excel
    if not excel.is_file():
        raise FileNotFoundError(f"Dataset not found: {excel}")
    nrtl_path = PROJECT_ROOT / "datasets" / "parameters" / "nrtl_params_all.json"

    cfg_updates = {
        "EXCEL_PATH": str(excel),
        "OUT_DIR": str(out_root),
        "SEED": int(args.seed),
        "MIN_POINTS_PER_GROUP": int(args.min_points_per_group),
        "PERMUTE_23_AUG": not bool(args.no_permute23),
        "LOAD_CKPT_PATH": "",
        "NRTL_EVAL_PARAMS_PATH": str(nrtl_path),
    }
    if args.epochs is not None:
        cfg_updates["EPOCHS"] = int(args.epochs)
    if args.patience is not None:
        cfg_updates["EARLY_STOP_PATIENCE"] = int(args.patience)
    if args.batch_size_graph is not None:
        cfg_updates["BATCH_SIZE_GRAPH"] = int(args.batch_size_graph)
    if args.device is not None:
        cfg_updates["DEVICE"] = str(args.device)

    with temporary_config(**cfg_updates):
        df_raw, df_aug = load_and_prepare_excel(
            str(excel),
            min_points_per_group=C.MIN_POINTS_PER_GROUP,
            permute_23_aug=C.PERMUTE_23_AUG,
        )
        df = df_aug

        rows: List[Dict] = []
        manifest_rows: List[Dict] = []
        for scheme in args.schemes:
            n_repeats = int(args.repeats) if is_stochastic_scheme(scheme) else 1
            repeat_seeds = build_repeat_seeds(int(args.seed), n_repeats)
            for rep, seed in enumerate(repeat_seeds):
                train_df, val_df, test_df, meta = make_split(
                    scheme,
                    df,
                    seed=seed,
                    train_ratio=float(args.train_ratio),
                    val_ratio=float(args.val_ratio),
                )

                run_id = f"rep{rep + 1:02d}" if n_repeats > 1 else "run01"
                run_dir = out_root / scheme / run_id
                manifest = {
                    "scheme": scheme,
                    "run_id": run_id,
                    "split_seed": seed,
                    "train_seed": seed,
                    "min_points_per_group": int(args.min_points_per_group),
                    "component_permutation_augmented": not bool(args.no_permute23),
                    "train_systems": int(train_df["system_id"].nunique()),
                    "val_systems": int(val_df["system_id"].nunique()),
                    "test_systems": int(test_df["system_id"].nunique()),
                    "train_points": int(len(train_df)),
                    "val_points": int(len(val_df)),
                    "test_points": int(len(test_df)),
                }
                manifest.update(meta)
                manifest_rows.append(manifest)

                print(
                    f"[{scheme}/{run_id}] seed={seed}, "
                    f"train={len(train_df)} ({train_df['system_id'].nunique()} systems), "
                    f"val={len(val_df)} ({val_df['system_id'].nunique()} systems), "
                    f"test={len(test_df)} ({test_df['system_id'].nunique()} systems)"
                )

                if args.dry_run:
                    continue

                metrics_path = run_dir / "best_metrics.json"
                if not (args.skip_existing and metrics_path.is_file()):
                    run_dir.mkdir(parents=True, exist_ok=True)
                    with temporary_config(OUT_DIR=str(run_dir), SEED=seed):
                        set_seed(seed)
                        train_or_load(train_df, val_df, test_df)
                else:
                    print(f"  Reusing existing metrics: {metrics_path}")

                if metrics_path.is_file():
                    rows.append(
                        collect_existing_metrics(
                            metrics_path,
                            scheme,
                            run_id,
                            train_df,
                            val_df,
                            test_df,
                            meta,
                            seed,
                            seed,
                        )
                    )
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df.to_csv(
            out_root / "split_strategy_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if args.dry_run:
            print(f"[DRY RUN] Wrote manifest: {out_root / 'split_strategy_manifest.csv'}")
            return

        if not rows:
            print("[WARN] No metrics collected.")
            return

        run_df = pd.DataFrame(rows)
        run_df.to_csv(
            out_root / "split_strategy_run_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        combined = build_combined_table(rows)
        combined.to_csv(
            out_root / "split_strategy_benchmark.csv", index=False, encoding="utf-8-sig"
        )

        summary = combined[
            combined["row_type"].isin(["mean", "SD", "min", "max"])
        ].copy()
        summary.to_csv(
            out_root / "split_strategy_summary.csv", index=False, encoding="utf-8-sig"
        )
        print(f"\nSaved combined CSV: {out_root / 'split_strategy_benchmark.csv'}")
        cols = [
            "row_type",
            "scheme",
            "run_id",
            "mae",
            "rmse",
            "r2",
            "mae_E",
            "mae_R",
            "rmse_E",
            "rmse_R",
            "r2_E",
            "r2_R",
        ]
        print(combined[[c for c in cols if c in combined.columns]].to_string(index=False))


if __name__ == "__main__":
    main()

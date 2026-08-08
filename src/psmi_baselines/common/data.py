# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import rdchem

from .config import (
    FP_BITS,
    FP_RADIUS,
    GNN_NODE_DIM,
    MIN_POINTS_PER_GROUP,
    PERMUTE_23_AUG,
    SEED,
    SMILES_MAX_LEN,
)
from .utils import canonicalize_smiles, renorm3, safe_group_apply_t, Scaler, morgan_fp


def augment_component_23(df: pd.DataFrame, *, enabled: bool = True) -> pd.DataFrame:
    """Add the component-2/3 training permutation with aligned labels."""
    original = df.copy()
    original["aug_swap23"] = 0
    if not enabled:
        return original
    swapped = df.copy()
    swapped["aug_swap23"] = 1
    swapped[["smiles2", "smiles3"]] = swapped[["smiles3", "smiles2"]].to_numpy()
    swapped[["Ex2", "Ex3"]] = swapped[["Ex3", "Ex2"]].to_numpy()
    swapped[["Rx2", "Rx3"]] = swapped[["Rx3", "Rx2"]].to_numpy()
    return pd.concat([original, swapped], ignore_index=True)


def load_and_prepare_excel(
    path: str,
    min_points_per_group: int = MIN_POINTS_PER_GROUP,
    permute_23_aug: bool = PERMUTE_23_AUG
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    return:
      df_raw: original orientation (no swap augmentation) -> for test visualization / CSV
      df_aug: for training (optional swap(2,3) augmentation)
    """
    df = pd.read_excel(path)
    df = df.rename(columns={
        "LLE system NO.": "system_id",
        "T/K": "T",
        "IL (Component 1) full name SMILES": "smiles1",
        "Component 2 SMILES": "smiles2",
        "Component 3 SMILES": "smiles3",
    })

    needed = ["system_id", "T", "smiles1", "smiles2", "smiles3",
              "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Found columns={list(df.columns)}")

    for c in ["smiles1", "smiles2", "smiles3"]:
        # Keep missing spreadsheet cells as missing values so they are rejected
        # without sending the literal string "nan" to RDKit.
        df[c] = df[c].map(canonicalize_smiles)
    df = df[(df["smiles1"] != "") & (df["smiles2"] != "") & (df["smiles3"] != "")].copy()

    for c in ["T", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["T", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]).copy()

    # Defensive renormalization
    E = df[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float32)
    R = df[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
    E = np.vstack([renorm3(e) for e in E])
    R = np.vstack([renorm3(r) for r in R])
    df[["Ex1", "Ex2", "Ex3"]] = E
    df[["Rx1", "Rx2", "Rx3"]] = R

    # Filter groups by min size
    counts = df.groupby(["system_id", "T"]).size().reset_index(name="n")
    keep = counts[counts["n"] >= min_points_per_group][["system_id", "T"]]
    df = df.merge(keep, on=["system_id", "T"], how="inner")

    # Assign t
    df = safe_group_apply_t(df)

    df_raw = df.copy()
    df_aug = augment_component_23(df, enabled=permute_23_aug)

    return df_raw, df_aug


def split_by_system(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    systems = sorted(df["system_id"].unique().tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(systems)

    n = len(systems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_sys = set(systems[:n_train])
    val_sys = set(systems[n_train:n_train + n_val])
    test_sys = set(systems[n_train + n_val:])

    train_df = df[df["system_id"].isin(train_sys)].copy()
    val_df = df[df["system_id"].isin(val_sys)].copy()
    test_df = df[df["system_id"].isin(test_sys)].copy()
    return train_df, val_df, test_df


def split_by_manifest(
    df: pd.DataFrame,
    manifest_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the canonical system-level split used by the PSMI benchmark."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as stream:
        partitions = json.load(stream).get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError(f"Split manifest has no partitions mapping: {path}")

    train_systems = {int(value) for value in partitions["train"]}
    validation_systems = {int(value) for value in partitions["validation"]}
    test_systems = {int(value) for value in partitions["test"]}
    if (
        train_systems & validation_systems
        or train_systems & test_systems
        or validation_systems & test_systems
    ):
        raise ValueError(f"Split manifest partitions overlap: {path}")

    observed = {int(value) for value in df["system_id"].unique()}
    declared = train_systems | validation_systems | test_systems
    if observed != declared:
        raise ValueError(
            "Split manifest does not match the prepared dataset; "
            f"missing={sorted(observed - declared)}, unexpected={sorted(declared - observed)}"
        )
    return (
        df[df["system_id"].isin(train_systems)].copy(),
        df[df["system_id"].isin(validation_systems)].copy(),
        df[df["system_id"].isin(test_systems)].copy(),
    )


class FingerprintCache:
    def __init__(self, radius: int = FP_RADIUS, n_bits: int = FP_BITS):
        self.cache: Dict[str, np.ndarray] = {}
        self.radius = radius
        self.n_bits = n_bits

    def get(self, smi: str) -> np.ndarray:
        if smi not in self.cache:
            self.cache[smi] = morgan_fp(smi, radius=self.radius, n_bits=self.n_bits)
        return self.cache[smi]


_HYBRIDIZATION_TYPES = (
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
)


def _atom_features(atom: rdchem.Atom) -> np.ndarray:
    """Return the normalized 11-dimensional atom vector used by the GNN baseline."""
    hybridization = atom.GetHybridization()
    features = [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 4.0,
        atom.GetTotalNumHs() / 4.0,
        (atom.GetFormalCharge() + 5) / 10.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        *(float(hybridization == value) for value in _HYBRIDIZATION_TYPES),
    ]
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (GNN_NODE_DIM,):
        raise ValueError(
            f"GNN_NODE_DIM={GNN_NODE_DIM} does not match atom feature size {result.size}."
        )
    return result


def smiles_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one SMILES string to node features and a binary adjacency matrix."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES for GNN baseline: {smiles}")

    atom_count = molecule.GetNumAtoms()
    nodes = np.zeros((atom_count, GNN_NODE_DIM), dtype=np.float32)
    for index, atom in enumerate(molecule.GetAtoms()):
        nodes[index] = _atom_features(atom)

    adjacency = np.zeros((atom_count, atom_count), dtype=np.float32)
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adjacency[begin, end] = 1.0
        adjacency[end, begin] = 1.0
    return nodes, adjacency


class GraphCache:
    """Cache graph features by canonical SMILES string."""

    def __init__(self) -> None:
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get(self, smiles: str) -> Tuple[np.ndarray, np.ndarray]:
        if smiles not in self.cache:
            self.cache[smiles] = smiles_to_graph(smiles)
        return self.cache[smiles]


class LLEDataset(Dataset):
    """Represent the LLEDataset baseline component."""
    def __init__(
        self,
        df: pd.DataFrame,
        T_scaler: Scaler,
        fp_cache: FingerprintCache,
        precompute: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        self.df = df.reset_index(drop=True)
        self.T_scaler = T_scaler
        self.fp_cache = fp_cache
        self.precompute = precompute
        self.dtype = dtype

        self._X: Optional[torch.Tensor] = None
        self._Y: Optional[torch.Tensor] = None

        if self.precompute:
            self._build_cache()

    def _build_cache(self) -> None:
        n = len(self.df)
        in_dim = 3 * FP_BITS + 2

        X = np.empty((n, in_dim), dtype=np.float32)
        Y = np.empty((n, 6), dtype=np.float32)

        # Baseline workflow step.
        for i in range(n):
            r = self.df.iloc[i]
            fp1 = self.fp_cache.get(r["smiles1"])
            fp2 = self.fp_cache.get(r["smiles2"])
            fp3 = self.fp_cache.get(r["smiles3"])

            Tn = self.T_scaler.transform(np.array([r["T"]], dtype=np.float32))[0].astype(np.float32)
            t = float(r["t"])

            X[i, :] = np.concatenate([fp1, fp2, fp3, np.array([Tn, t], dtype=np.float32)], axis=0)
            Y[i, :] = np.array([r["Ex1"], r["Ex2"], r["Ex3"], r["Rx1"], r["Rx2"], r["Rx3"]], dtype=np.float32)

        self._X = torch.from_numpy(X).to(dtype=self.dtype)
        self._Y = torch.from_numpy(Y).to(dtype=self.dtype)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        if self._X is not None and self._Y is not None:
            return self._X[idx], self._Y[idx]

        # Baseline workflow step.
        r = self.df.iloc[idx]
        fp1 = self.fp_cache.get(r["smiles1"])
        fp2 = self.fp_cache.get(r["smiles2"])
        fp3 = self.fp_cache.get(r["smiles3"])
        Tn = self.T_scaler.transform(np.array([r["T"]], dtype=np.float32))[0].astype(np.float32)
        t = float(r["t"])

        x = np.concatenate([fp1, fp2, fp3, np.array([Tn, t], dtype=np.float32)], axis=0).astype(np.float32)
        y = np.array([r["Ex1"], r["Ex2"], r["Ex3"], r["Rx1"], r["Rx2"], r["Rx3"]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class LLEGNNDataset(Dataset):
    """Return three molecular graphs, thermodynamic scalars, and phase targets."""

    def __init__(
        self,
        df: pd.DataFrame,
        T_scaler: Scaler,
        graph_cache: GraphCache,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.T_scaler = T_scaler
        self.graph_cache = graph_cache
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        graphs = tuple(
            self.graph_cache.get(row[f"smiles{index}"])
            for index in range(1, 4)
        )
        temperature = self.T_scaler.transform(
            np.array([row["T"]], dtype=np.float32)
        )[0].astype(np.float32)
        scalars = torch.tensor([temperature, float(row["t"])], dtype=self.dtype)
        target = torch.tensor(
            [row["Ex1"], row["Ex2"], row["Ex3"], row["Rx1"], row["Rx2"], row["Rx3"]],
            dtype=self.dtype,
        )
        return graphs[0], graphs[1], graphs[2], scalars, target


def _pad_graph_batch(
    graphs: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-size molecular graphs and return node masks."""
    if not graphs:
        raise ValueError("Cannot collate an empty graph batch.")
    maximum_atoms = max(nodes.shape[0] for nodes, _ in graphs)
    batch_size = len(graphs)
    nodes = torch.zeros((batch_size, maximum_atoms, GNN_NODE_DIM), dtype=torch.float32)
    adjacency = torch.zeros((batch_size, maximum_atoms, maximum_atoms), dtype=torch.float32)
    mask = torch.zeros((batch_size, maximum_atoms), dtype=torch.float32)

    for index, (node_features, graph_adjacency) in enumerate(graphs):
        atom_count = node_features.shape[0]
        nodes[index, :atom_count] = torch.from_numpy(node_features)
        adjacency[index, :atom_count, :atom_count] = torch.from_numpy(graph_adjacency)
        mask[index, :atom_count] = 1.0
    return nodes, adjacency, mask


def gnn_collate_fn(batch):
    """Collate ternary GNN samples into three padded graph batches."""
    graph_lists = [[], [], []]
    scalars = []
    targets = []
    for graph1, graph2, graph3, scalar, target in batch:
        for index, graph in enumerate((graph1, graph2, graph3)):
            graph_lists[index].append(graph)
        scalars.append(scalar)
        targets.append(target)
    return (
        _pad_graph_batch(graph_lists[0]),
        _pad_graph_batch(graph_lists[1]),
        _pad_graph_batch(graph_lists[2]),
        torch.stack(scalars, dim=0),
        torch.stack(targets, dim=0),
    )


# -------------------------
# SMILES-only dataset for RNN models (no fingerprints)
# -------------------------
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<sep>": 3,
    "<unk>": 4,
}


def build_smiles_vocab(dfs: List[pd.DataFrame]) -> Dict[str, int]:
    chars = set()
    for df in dfs:
        for col in ["smiles1", "smiles2", "smiles3"]:
            chars.update("".join(df[col].astype(str).tolist()))
    vocab = dict(SPECIAL_TOKENS)
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


class SmilesRNNDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Dict[str, int],
        T_scaler: Scaler,
        max_len: int = SMILES_MAX_LEN,
        use_t: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.T_scaler = T_scaler
        self.max_len = max_len
        self.use_t = use_t
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.df)

    def _encode(self, s: str) -> List[int]:
        ids = [self.vocab["<bos>"]]
        for ch in s:
            ids.append(self.vocab.get(ch, self.vocab["<unk>"]))
        ids.append(self.vocab["<eos>"])
        return ids

    def _concat_smiles(self, r) -> List[int]:
        # concat smi1 <sep> smi2 <sep> smi3
        ids = []
        ids.extend(self._encode(r["smiles1"]))
        ids.append(self.vocab["<sep>"])
        ids.extend(self._encode(r["smiles2"]))
        ids.append(self.vocab["<sep>"])
        ids.extend(self._encode(r["smiles3"]))
        return ids

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        tok_ids = self._concat_smiles(r)

        # truncate/pad
        tok_ids = tok_ids[: self.max_len]
        pad_len = self.max_len - len(tok_ids)
        if pad_len > 0:
            tok_ids = tok_ids + [self.vocab["<pad>"]] * pad_len

        tokens = torch.tensor(tok_ids, dtype=torch.long)

        Tn = self.T_scaler.transform(np.array([r["T"]], dtype=np.float32))[0].astype(np.float32)
        if self.use_t:
            scalar = torch.tensor([Tn, float(r["t"])], dtype=self.dtype)
        else:
            scalar = torch.tensor([Tn], dtype=self.dtype)

        y = torch.tensor(
            [r["Ex1"], r["Ex2"], r["Ex3"], r["Rx1"], r["Rx2"], r["Rx3"]],
            dtype=self.dtype,
        )

        return tokens, scalar, y

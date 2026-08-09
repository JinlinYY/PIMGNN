# -*- coding: utf-8 -*-
import random
import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

def set_seed(seed: int = 42) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def canonicalize_smiles(smi: str) -> str:
    if not isinstance(smi, str) or not smi.strip():
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)

def morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def renorm3(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(np.sum(a))
    if s < eps:
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)
    return (a / s).astype(np.float32)

def assign_t_by_pca(group: pd.DataFrame) -> pd.DataFrame:
    """
    Within each (system_id, T), sort tie-lines by PC1 in 6D (E+R) space,
    and assign a pseudo-parameter t in [0, 1].
    """
    cols = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
    X = group[cols].to_numpy(dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    pc1 = vt[0]
    proj = X @ pc1
    order = np.argsort(proj)

    n = len(group)
    t = np.empty((n,), dtype=np.float32)
    if n == 1:
        t[:] = 0.5
    else:
        t[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)

    out = group.copy()
    out["t"] = t
    return out

@dataclass
class Scaler:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-12)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Scaler":
        return cls(mean=float(state.get("mean", 0.0)), std=float(state.get("std", 1.0)))

    @staticmethod
    def fit(x: np.ndarray) -> "Scaler":
        return Scaler(mean=float(np.mean(x)), std=float(np.std(x)))

def safe_group_apply_t(df: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = df.groupby(["system_id", "T"], group_keys=False).apply(assign_t_by_pca)
    return result.reset_index(drop=True)

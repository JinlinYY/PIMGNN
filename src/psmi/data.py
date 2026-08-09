# -*- coding: utf-8 -*-
"""Load, preprocess, split, and batch ternary LLE datasets."""
from __future__ import annotations

from typing import Dict, Tuple, Optional, List, Any
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import OrderedDict, defaultdict
import types
from types import SimpleNamespace

from .config import (
    FP_BITS, FP_RADIUS, MIN_POINTS_PER_GROUP, PERMUTE_23_AUG, SEED,
    GRAPH_ADD_HS, GRAPH_ADD_3D, GRAPH_USE_GASTEIGER, GRAPH_MAX_ATOMS,
    USE_MIX_GRAPH, USE_FG, FG_TOPK, FG_MIN_FREQ, PRECOMPUTE_FG,
    FG_TOKEN_MODE, FG_MAX_TOKENS
)
from . import config as C

from .utils import (
    canonicalize_smiles, renorm3, safe_group_apply_t, Scaler, morgan_fp, fg_smiles_from_smiles,
    smiles_to_graph, batch_graphs,
    build_mixture_graph, batch_mixture_graphs, temperature_scalar_value
)


def condition_scalar_values(
    temperature_normalized: float,
    phase_path: float,
    pressure_normalized: float = 0.0,
    *,
    scalar_dim: int = 3,
) -> np.ndarray:
    """Build the declared thermodynamic scalar vector for one model profile."""
    if int(scalar_dim) == 2:
        values = [temperature_normalized, phase_path]
    elif int(scalar_dim) == 3:
        values = [temperature_normalized, phase_path, pressure_normalized]
    else:
        raise ValueError(f"scalar_dim must be 2 or 3, got {scalar_dim!r}")
    return np.asarray(values, dtype=np.float32)


def augment_component_23(df: pd.DataFrame, *, enabled: bool = True) -> pd.DataFrame:
    """Add a component-2/3 permutation while keeping inputs and targets aligned."""
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



def _norm_col(c: Any) -> str:
    """Normalize a column name."""
    
    c_str = str(c).strip().replace('\n', ' ').replace('\r', ' ')
    
    c_str = ' '.join(c_str.split())
    return c_str


def _find_col(available_cols: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first matching column."""
    
    norm_cols = {_norm_col(c).lower(): c for c in available_cols}
    
    
    for cand in candidates:
        cand_norm = _norm_col(cand).lower()
        if cand_norm in norm_cols:
            return norm_cols[cand_norm]
    
    return None


def _require_col(df: pd.DataFrame, name: str, candidates: List[str]) -> str:
    """Return a required column or raise an error."""
    col = _find_col(df.columns.tolist(), candidates)
    if col is None:
        raise KeyError(
            f"Cannot find column for '{name}'. Tried candidates={candidates}\n"
            f"Available columns ({len(df.columns)}):\n{list(df.columns)}"
        )
    return col


def load_and_prepare_excel(
    path: str,
    min_points_per_group: int = MIN_POINTS_PER_GROUP,
    permute_23_aug: bool = PERMUTE_23_AUG
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    return:
      df_raw: original orientation (no swap augmentation)
      df_aug: for training (optional swap(2,3) augmentation)
    """
    df = pd.read_excel(path)
    df.columns = [_norm_col(c) for c in df.columns]

    col_system = _require_col(df, "system_id", [
        "system_id", "system id", "System ID", "System_ID",
        "LLE system NO.", "LLE system NO", "LLE system No.", "LLE system No",
        "LLE system number", "LLE system#", "LLE system #",
        "System No.", "System No",
    ])
    col_T = _require_col(df, "T", [
        "T/K", "T / K", "T (K)", "T", "Temp", "Temperature", "Temperature/K", "Temperature (K)"
    ])
    
    
    col_P = _find_col(df.columns.tolist(), [
        "P/kPa", "P / kPa", "P(kPa)", "P", "Pressure", "Pressure/kPa", "Pressure (kPa)",
        "P/bar", "P(bar)", "P / bar" 
    ])

    col_s1 = _require_col(df, "smiles1", [
        "Component1-SMILES", "Component1 SMILES",
        "IL (Component 1) full name SMILES",
        "IL (Component 1) SMILES",
        "Component 1 SMILES", "Comp 1 SMILES",
        "smiles1", "SMILES1", "SMILES 1"
    ])
    col_s2 = _require_col(df, "smiles2", [
        "Component2-SMILES", "Component2 SMILES",
        "Component 2 SMILES", "Comp 2 SMILES",
        "smiles2", "SMILES2", "SMILES 2"
    ])
    col_s3 = _require_col(df, "smiles3", [
        "Component3-SMILES", "Component3 SMILES",
        "Component 3 SMILES", "Comp 3 SMILES",
        "smiles3", "SMILES3", "SMILES 3"
    ])

    def _req_comp(name: str) -> str:
        
        candidates = [name, name.upper(), name.lower(), name.replace("x", "X"), name.replace("X", "x")]
        if "x" in name.lower():
             candidates.append(name.replace("x", "").replace("X", ""))
        return _require_col(df, name, candidates)

    col_Ex1 = _req_comp("Ex1"); col_Ex2 = _req_comp("Ex2"); col_Ex3 = _req_comp("Ex3")
    col_Rx1 = _req_comp("Rx1"); col_Rx2 = _req_comp("Rx2"); col_Rx3 = _req_comp("Rx3")

    df = df.rename(columns={
        col_system: "system_id",
        col_T: "T",
        col_s1: "smiles1",
        col_s2: "smiles2",
        col_s3: "smiles3",
        col_Ex1: "Ex1", col_Ex2: "Ex2", col_Ex3: "Ex3",
        col_Rx1: "Rx1", col_Rx2: "Rx2", col_Rx3: "Rx3",
    })

    
    if col_P:
        df = df.rename(columns={col_P: "P"})
        df["P"] = pd.to_numeric(df["P"], errors="coerce")
        
        if df["P"].isnull().all():
             df["P"] = 101.325
        else:
             df["P"] = df["P"].fillna(101.325)
    else:
        df["P"] = 101.325

    for c in ["smiles1", "smiles2", "smiles3"]:
        df[c] = df[c].astype(str).map(canonicalize_smiles)
    df = df[(df["smiles1"] != "") & (df["smiles2"] != "") & (df["smiles3"] != "")].copy()

    for c in ["T", "P", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["T", "P", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]).copy()

    E = df[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float32)
    R = df[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
    E = np.vstack([renorm3(e) for e in E])
    R = np.vstack([renorm3(r) for r in R])
    df[["Ex1", "Ex2", "Ex3"]] = E
    df[["Rx1", "Rx2", "Rx3"]] = R

    counts = df.groupby(["system_id", "T"]).size().reset_index(name="n")
    keep = counts[counts["n"] >= min_points_per_group][["system_id", "T"]]
    df = df.merge(keep, on=["system_id", "T"], how="inner")

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
    manifest_path: Path | str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows using an auditable, system-level JSON manifest."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError(f"Split manifest has no partitions mapping: {path}")

    try:
        train_sys = {int(value) for value in partitions["train"]}
        val_sys = {int(value) for value in partitions["validation"]}
        test_sys = {int(value) for value in partitions["test"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid system identifiers in split manifest: {path}") from exc

    if train_sys & val_sys or train_sys & test_sys or val_sys & test_sys:
        raise ValueError(f"Split manifest partitions overlap: {path}")
    observed = {int(value) for value in df["system_id"].unique()}
    declared = train_sys | val_sys | test_sys
    missing = observed - declared
    unexpected = declared - observed
    if missing or unexpected:
        raise ValueError(
            f"Split manifest does not match dataset systems; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    train_df = df[df["system_id"].isin(train_sys)].copy()
    val_df = df[df["system_id"].isin(val_sys)].copy()
    test_df = df[df["system_id"].isin(test_sys)].copy()
    return train_df, val_df, test_df

def stratified_split_by_system(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = SEED,
    n_bins: int = 8,
    min_bin_size: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process stratified split by system."""
    assert 0 < train_ratio < 1
    assert 0 <= val_ratio < 1
    assert train_ratio + val_ratio < 1

    # --- per-system stats ---
    stats = (
        df.groupby("system_id")
          .agg(
              n_rows=("system_id", "size"),
              n_groups=("T", lambda x: x.nunique()),
              T_min=("T", "min"),
              T_max=("T", "max"),
          )
          .reset_index()
    )
    stats["T_span"] = (stats["T_max"] - stats["T_min"]).astype(float)

    def _qbin(s: pd.Series, q: int) -> pd.Series:
        
        uniq = int(s.nunique())
        q = int(max(1, min(q, uniq)))
        if q <= 1:
            return pd.Series(["ALL"] * len(s), index=s.index)
        try:
            return pd.qcut(s, q=q, duplicates="drop").astype(str)
        except Exception:
            return pd.Series(["ALL"] * len(s), index=s.index)

    
    stats["bin_rows"] = _qbin(stats["n_rows"], n_bins)
    stats["bin_span"] = _qbin(stats["T_span"], n_bins)
    
    stats["bin_groups"] = _qbin(stats["n_groups"], max(2, n_bins // 2))

    stats["stratum"] = (
        stats["bin_rows"].astype(str) + "|" +
        stats["bin_span"].astype(str) + "|" +
        stats["bin_groups"].astype(str)
    )

    # --- merge rare strata to avoid empty val/test in tiny bins ---
    counts = stats["stratum"].value_counts()
    rare = set(counts[counts < min_bin_size].index.tolist())
    if rare:
        stats.loc[stats["stratum"].isin(rare), "stratum"] = "RARE"

    rng = np.random.RandomState(seed)
    train_sys, val_sys, test_sys = set(), set(), set()

    for _, sub in stats.groupby("stratum"):
        sids = sub["system_id"].tolist()
        rng.shuffle(sids)
        n = len(sids)

        
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        elif n == 2:
            n_train = 1
            n_val = 0
        else:  # n == 1
            n_train = 1
            n_val = 0

        train_part = sids[:n_train]
        val_part = sids[n_train:n_train + n_val]
        test_part = sids[n_train + n_val:]

        train_sys.update(train_part)
        val_sys.update(val_part)
        test_sys.update(test_part)

    
    val_sys = val_sys - train_sys
    test_sys = test_sys - train_sys - val_sys

    train_df = df[df["system_id"].isin(train_sys)].copy()
    val_df = df[df["system_id"].isin(val_sys)].copy()
    test_df = df[df["system_id"].isin(test_sys)].copy()
    return train_df, val_df, test_df



class FingerprintCache:
    def __init__(self, radius: int = FP_RADIUS, n_bits: int = FP_BITS):
        self.cache: Dict[str, np.ndarray] = {}
        self.radius = radius
        self.n_bits = n_bits

    def get(self, smi: str) -> np.ndarray:
        if smi not in self.cache:
            self.cache[smi] = morgan_fp(smi, radius=self.radius, n_bits=self.n_bits)
        return self.cache[smi]


class FunctionalGroupCache:
    """Build a fixed-length (FG_TOPK) multi-hot vector for each SMILES based on an FG corpus.

    - During training: build corpus from TRAIN molecules (frequency >= FG_MIN_FREQ), keep top FG_TOPK.
    - During inference: load the same corpus to keep consistent dimensionality.
    """
    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        vocab_size: int = FG_TOPK,
        min_freq: int = FG_MIN_FREQ,
    ):
        self.vocab_size = int(vocab_size)
        self.min_freq = int(min_freq)
        self.corpus: List[str] = list(corpus) if corpus is not None else []
        self.fg2idx: Dict[str, int] = {fg: i for i, fg in enumerate(self.corpus[: self.vocab_size])}
        self.cache: Dict[str, np.ndarray] = {}
        self.token_cache: Dict[str, List[int]] = {}

    def build_corpus_from_smiles(self, smiles_list: List[str]) -> List[str]:
        """Return a corpus list (length <= vocab_size) built from smiles_list."""
        fg_freq = defaultdict(int)
        for smi in smiles_list:
            for fg in fg_smiles_from_smiles(smi):
                fg_freq[fg] += 1
        # filter by min_freq then take top-k
        items = [(fg, c) for fg, c in fg_freq.items() if c >= self.min_freq]
        items.sort(key=lambda x: x[1], reverse=True)
        corpus = [fg for fg, _ in items[: self.vocab_size]]
        return corpus

    def set_corpus(self, corpus: List[str]) -> None:
        self.corpus = list(corpus)[: self.vocab_size]
        self.fg2idx = {fg: i for i, fg in enumerate(self.corpus)}
        self.cache.clear()
        self.token_cache.clear()

    def get(self, smi: str) -> np.ndarray:
        """Return FG multi-hot vector (float32) of shape (vocab_size,)."""
        smi = str(smi) if smi is not None else ""
        if smi not in self.cache:
            v = np.zeros((self.vocab_size,), dtype=np.float32)
            for fg in fg_smiles_from_smiles(smi):
                idx = self.fg2idx.get(fg, None)
                if idx is not None and 0 <= idx < self.vocab_size:
                    v[idx] = 1.0
            self.cache[smi] = v
        return self.cache[smi]

    def get_token_ids(self, smi: str, max_tokens: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return FG token ids (int64) and mask (float32) with padding id=0."""
        smi = str(smi) if smi is not None else ""
        if smi not in self.token_cache:
            ids: List[int] = []
            for fg in sorted(fg_smiles_from_smiles(smi)):
                idx = self.fg2idx.get(fg, None)
                if idx is not None and 0 <= idx < self.vocab_size:
                    ids.append(int(idx) + 1)  # 0 is pad
            self.token_cache[smi] = ids

        ids = self.token_cache[smi]
        n = int(max_tokens)
        out = np.zeros((n,), dtype=np.int64)
        mask = np.zeros((n,), dtype=np.float32)
        if n <= 0:
            return out, mask
        keep = ids[:n]
        if keep:
            out[: len(keep)] = np.asarray(keep, dtype=np.int64)
            mask[: len(keep)] = 1.0
        return out, mask


class LLEDataset(Dataset):
    """
    Checkpoint-compatible FP dataset:
      x = [fp1, fp2, fp3, T_norm, t], y = [Ex1..3, Rx1..3]
    """
    def __init__(
        self,
        df: pd.DataFrame,
        T_scaler: Scaler,
        fp_cache: FingerprintCache,
        P_scaler: Optional[Scaler] = None, 
        fg_cache: Optional[FunctionalGroupCache] = None,
        use_fg: Optional[bool] = None,
        precompute: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        self.df = df.reset_index(drop=True)
        self.T_scaler = T_scaler
        self.P_scaler = P_scaler 
        self.fp_cache = fp_cache
        self.fg_cache = fg_cache
        self.use_fg = bool(getattr(C, "USE_FG", False) if use_fg is None else use_fg)
        self.use_fg = self.use_fg and (self.fg_cache is not None)
        self.fg_dim = int(getattr(C, "FG_TOPK", FG_TOPK)) if self.use_fg else 0
        self.precompute = precompute
        self.dtype = dtype

        self._X: Optional[torch.Tensor] = None
        self._Y: Optional[torch.Tensor] = None

        if self.precompute:
            self._build_cache()

    def _build_cache(self) -> None:
        n = len(self.df)
        in_dim = 3 * FP_BITS + 2
        if self.use_fg:
            in_dim += 3 * self.fg_dim

        X = np.empty((n, in_dim), dtype=np.float32)
        Y = np.empty((n, 6), dtype=np.float32)

        for i in range(n):
            r = self.df.iloc[i]
            
            fp1 = self.fp_cache.get(r["smiles1"])
            fp2 = self.fp_cache.get(r["smiles2"])
            fp3 = self.fp_cache.get(r["smiles3"])

            T_raw_feature = temperature_scalar_value(
                [r["T"]],
                mode=getattr(C, "TEMPERATURE_ENCODING", "linear_quadratic"),
                reference_k=float(getattr(C, "TEMPERATURE_REFERENCE_K", 500.0)),
            )
            Tn = self.T_scaler.transform(T_raw_feature)[0].astype(np.float32)
            t = float(r["t"])

            parts = [fp1, fp2, fp3]
            if self.use_fg:
                parts.extend([
                    self.fg_cache.get(r["smiles1"]),
                    self.fg_cache.get(r["smiles2"]),
                    self.fg_cache.get(r["smiles3"]),
                ])
            parts.append(np.array([Tn, t], dtype=np.float32))
            X[i, :] = np.concatenate(parts, axis=0)
            Y[i, :] = np.array([r["Ex1"], r["Ex2"], r["Ex3"], r["Rx1"], r["Rx2"], r["Rx3"]], dtype=np.float32)

        self._X = torch.from_numpy(X).to(dtype=self.dtype)
        self._Y = torch.from_numpy(Y).to(dtype=self.dtype)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        if self._X is not None and self._Y is not None:
            return self._X[idx], self._Y[idx]

        r = self.df.iloc[idx]
        fp1 = self.fp_cache.get(r["smiles1"])
        fp2 = self.fp_cache.get(r["smiles2"])
        fp3 = self.fp_cache.get(r["smiles3"])
        T_raw_feature = temperature_scalar_value(
            [r["T"]],
            mode=getattr(C, "TEMPERATURE_ENCODING", "linear_quadratic"),
            reference_k=float(getattr(C, "TEMPERATURE_REFERENCE_K", 500.0)),
        )
        Tn = self.T_scaler.transform(T_raw_feature)[0].astype(np.float32)
        t = float(r["t"])

        parts = [fp1, fp2, fp3]
        if self.use_fg:
            parts.extend([
                self.fg_cache.get(r["smiles1"]),
                self.fg_cache.get(r["smiles2"]),
                self.fg_cache.get(r["smiles3"]),
            ])
        parts.append(np.array([Tn, t], dtype=np.float32))
        x = np.concatenate(parts, axis=0).astype(np.float32)
        y = np.array([r["Ex1"], r["Ex2"], r["Ex3"], r["Rx1"], r["Rx2"], r["Rx3"]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================
# Graph cache + graph dataset
# ============================================================

class GraphCache:
    """Cache RDKit->Graph conversion for unique SMILES."""
    def __init__(
        self,
        add_hs: bool = GRAPH_ADD_HS,
        add_3d: bool = GRAPH_ADD_3D,
        use_gasteiger: bool = GRAPH_USE_GASTEIGER,
        max_atoms: int = GRAPH_MAX_ATOMS,
    ):
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.add_hs = add_hs
        self.add_3d = add_3d
        self.use_gasteiger = use_gasteiger
        self.max_atoms = max_atoms

    def build_from_smiles(self, smiles_list: List[str]) -> None:
        uniq = sorted({s for s in smiles_list if isinstance(s, str) and s})
        for i, smi in enumerate(uniq):
            if smi not in self.cache:
                self.cache[smi] = smiles_to_graph(
                    smi,
                    add_hs=self.add_hs,
                    add_3d=self.add_3d,
                    use_gasteiger=self.use_gasteiger,
                    max_atoms=self.max_atoms,
                    seed=i,
                )

    def get(self, smi: str) -> Dict[str, np.ndarray]:
        if smi not in self.cache:
            self.cache[smi] = smiles_to_graph(
                smi,
                add_hs=self.add_hs,
                add_3d=self.add_3d,
                use_gasteiger=self.use_gasteiger,
                max_atoms=self.max_atoms,
                seed=0,
            )
        return self.cache[smi]


# =========================
# Mix graph cache (pickle-safe + triple LRU)
# =========================
def _cfg_to_namespace(cfg_module_or_obj: Any) -> Any:
    """Convert configuration values to a namespace."""
    if cfg_module_or_obj is None:
        cfg_module_or_obj = C

    if isinstance(cfg_module_or_obj, types.ModuleType):
        keys = [k for k in dir(cfg_module_or_obj) if k.isupper()]
        return SimpleNamespace(**{k: getattr(cfg_module_or_obj, k) for k in keys})

    return cfg_module_or_obj


class MixGraphCache:
    """
    Cache mixture interaction graphs (pickle-safe):
      - mol_cache: per-molecule 3D packages
      - pair_cache: per ordered pair interaction
      - triple_cache: (smi1,smi2,smi3,T_raw) -> mixture graph LRU
    """
    def __init__(self, cfg_module: Any = None):
        self.cfg = _cfg_to_namespace(cfg_module)

        self.mol_cache: Dict[str, Any] = {}
        self.pair_cache: Dict[str, Any] = {}

        self.triple_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self.triple_cache_size: int = int(getattr(self.cfg, "MIX_TRIPLE_CACHE_SIZE", 4096))

    def _triple_key(self, s1: str, s2: str, s3: str, T_raw: float) -> str:
        s1c = canonicalize_smiles(s1)
        s2c = canonicalize_smiles(s2)
        s3c = canonicalize_smiles(s3)
        return f"{s1c}||{s2c}||{s3c}||{float(T_raw):.6f}"

    def build(
        self,
        smi1: str,
        smi2: str,
        smi3: str,
        T_norm: float,
        T_raw: float,
    ) -> Dict[str, np.ndarray]:
        k = self._triple_key(smi1, smi2, smi3, T_raw)

        hit = self.triple_cache.get(k, None)
        if hit is not None:
            self.triple_cache.move_to_end(k, last=True)
            return hit

        
        g = build_mixture_graph(
            smi1, smi2, smi3,
            T_norm=float(T_norm), T_raw=float(T_raw),
            cfg=self.cfg,
            mol_cache=self.mol_cache,
            pair_cache=self.pair_cache,
        )

        
        if "x" in g:
            g["x"] = np.asarray(g["x"], dtype=np.float32)
        if "edge_attr" in g:
            g["edge_attr"] = np.asarray(g["edge_attr"], dtype=np.float32)
        if "g" in g:
            g["g"] = np.asarray(g["g"], dtype=np.float32)
        if "edge_index" in g:
            g["edge_index"] = np.asarray(g["edge_index"], dtype=np.int64)

        self.triple_cache[k] = g
        self.triple_cache.move_to_end(k, last=True)
        if len(self.triple_cache) > self.triple_cache_size:
            self.triple_cache.popitem(last=False)

        return g


class GraphLLEDataset(Dataset):
    """
    Graph dataset output (compatible with training loop):
        x_dict = {
            'g1': single_graph_dict,
            'g2': single_graph_dict,
            'g3': single_graph_dict,
            'scalars': tensor([T_norm, t]) or tensor([T_norm, t, P_norm]),
            'mix': mixture_graph_dict (optional)
        }
        y = tensor([Ex1..3, Rx1..3])
    """
    def __init__(
        self,
        df: pd.DataFrame,
        T_scaler: Scaler,
        g_cache: GraphCache,
        P_scaler: Optional[Scaler] = None, 
        mix_cache: Optional[MixGraphCache] = None,
        fg_cache: Optional[FunctionalGroupCache] = None,
        use_fg: Optional[bool] = None,
        use_mix_graph: Optional[bool] = None,
        scalar_dim: int = 3,
        dtype: torch.dtype = torch.float32,
        precompute_scalars: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.T_scaler = T_scaler
        self.P_scaler = P_scaler 
        self.scalar_dim = int(scalar_dim)
        if self.scalar_dim not in {2, 3}:
            raise ValueError(f"scalar_dim must be 2 or 3, got {scalar_dim!r}")
        self.g_cache = g_cache
        self.fg_cache = fg_cache
        self.use_fg = bool(getattr(C, "USE_FG", False) if use_fg is None else use_fg)
        self.use_fg = self.use_fg and (self.fg_cache is not None)
        self.fg_dim = int(getattr(C, "FG_TOPK", FG_TOPK)) if self.use_fg else 0
        self.fg_token_mode = bool(getattr(C, "FG_TOKEN_MODE", FG_TOKEN_MODE)) if self.use_fg else False
        self.fg_max_tokens = int(getattr(C, "FG_MAX_TOKENS", FG_MAX_TOKENS)) if self.use_fg else 0
        self.use_mix_graph = bool(USE_MIX_GRAPH if use_mix_graph is None else use_mix_graph)

        
        self.mix_cache = mix_cache if mix_cache is not None else (MixGraphCache(C) if self.use_mix_graph else None)

        self.dtype = dtype
        self._scalars: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None
        self._fg1: Optional[torch.Tensor] = None
        self._fg2: Optional[torch.Tensor] = None
        self._fg3: Optional[torch.Tensor] = None
        self._fg1_ids: Optional[torch.Tensor] = None
        self._fg2_ids: Optional[torch.Tensor] = None
        self._fg3_ids: Optional[torch.Tensor] = None
        self._fg1_mask: Optional[torch.Tensor] = None
        self._fg2_mask: Optional[torch.Tensor] = None
        self._fg3_mask: Optional[torch.Tensor] = None

        if precompute_scalars:
            self._build_cache()

    def _build_cache(self) -> None:
        n = len(self.df)
        scalars = np.empty((n, self.scalar_dim), dtype=np.float32)
        y = np.empty((n, 6), dtype=np.float32)
        fg1 = fg2 = fg3 = None
        fg1_ids = fg2_ids = fg3_ids = None
        fg1_mask = fg2_mask = fg3_mask = None
        if self.use_fg and bool(getattr(C, "PRECOMPUTE_FG", True)):
            if self.fg_token_mode:
                fg1_ids = np.zeros((n, self.fg_max_tokens), dtype=np.int64)
                fg2_ids = np.zeros((n, self.fg_max_tokens), dtype=np.int64)
                fg3_ids = np.zeros((n, self.fg_max_tokens), dtype=np.int64)
                fg1_mask = np.zeros((n, self.fg_max_tokens), dtype=np.float32)
                fg2_mask = np.zeros((n, self.fg_max_tokens), dtype=np.float32)
                fg3_mask = np.zeros((n, self.fg_max_tokens), dtype=np.float32)
            else:
                fg1 = np.zeros((n, self.fg_dim), dtype=np.float32)
                fg2 = np.zeros((n, self.fg_dim), dtype=np.float32)
                fg3 = np.zeros((n, self.fg_dim), dtype=np.float32)
        for i in range(n):
            r = self.df.iloc[i]
            T_raw_feature = temperature_scalar_value(
                [r["T"]],
                mode=getattr(C, "TEMPERATURE_ENCODING", "linear_quadratic"),
                reference_k=float(getattr(C, "TEMPERATURE_REFERENCE_K", 500.0)),
            )
            Tn = self.T_scaler.transform(T_raw_feature)[0].astype(np.float32)
            t = float(r["t"])

            # P processing (Graph mode)
            Pn = 0.0
            if self.P_scaler:
                 try:
                    Pn = self.P_scaler.transform(np.array([r["P"]], dtype=np.float32))[0].astype(np.float32)
                 except Exception:
                    Pn = 0.0
            
            scalars[i, :] = condition_scalar_values(
                Tn,
                t,
                Pn,
                scalar_dim=self.scalar_dim,
            )
            
            if fg1 is not None:
                fg1[i, :] = self.fg_cache.get(r["smiles1"])
                fg2[i, :] = self.fg_cache.get(r["smiles2"])
                fg3[i, :] = self.fg_cache.get(r["smiles3"])
            if fg1_ids is not None:
                ids1, m1 = self.fg_cache.get_token_ids(r["smiles1"], self.fg_max_tokens)
                ids2, m2 = self.fg_cache.get_token_ids(r["smiles2"], self.fg_max_tokens)
                ids3, m3 = self.fg_cache.get_token_ids(r["smiles3"], self.fg_max_tokens)
                fg1_ids[i, :] = ids1
                fg2_ids[i, :] = ids2
                fg3_ids[i, :] = ids3
                fg1_mask[i, :] = m1
                fg2_mask[i, :] = m2
                fg3_mask[i, :] = m3
            y[i, :] = np.array([r["Ex1"], r["Ex2"], r["Ex3"], r["Rx1"], r["Rx2"], r["Rx3"]], dtype=np.float32)
        self._scalars = torch.from_numpy(scalars).to(dtype=self.dtype)
        self._y = torch.from_numpy(y).to(dtype=self.dtype)
        if fg1 is not None:
            self._fg1 = torch.from_numpy(fg1).to(dtype=self.dtype)
            self._fg2 = torch.from_numpy(fg2).to(dtype=self.dtype)
            self._fg3 = torch.from_numpy(fg3).to(dtype=self.dtype)
        if fg1_ids is not None:
            self._fg1_ids = torch.from_numpy(fg1_ids)
            self._fg2_ids = torch.from_numpy(fg2_ids)
            self._fg3_ids = torch.from_numpy(fg3_ids)
            self._fg1_mask = torch.from_numpy(fg1_mask)
            self._fg2_mask = torch.from_numpy(fg2_mask)
            self._fg3_mask = torch.from_numpy(fg3_mask)

    def __len__(self):
        return len(self.df)

    def get_fg_token_ids(self, idx: int, comp: int) -> torch.Tensor:
        if not self.fg_token_mode:
            raise ValueError("FG token mode is disabled.")
        if comp == 1 and self._fg1_ids is not None:
            return self._fg1_ids[idx]
        if comp == 2 and self._fg2_ids is not None:
            return self._fg2_ids[idx]
        if comp == 3 and self._fg3_ids is not None:
            return self._fg3_ids[idx]
        r = self.df.iloc[idx]
        smi = r["smiles1"] if comp == 1 else r["smiles2"] if comp == 2 else r["smiles3"]
        ids, _ = self.fg_cache.get_token_ids(smi, self.fg_max_tokens)
        return torch.from_numpy(ids)

    def get_fg_token_mask(self, idx: int, comp: int) -> torch.Tensor:
        if not self.fg_token_mode:
            raise ValueError("FG token mode is disabled.")
        if comp == 1 and self._fg1_mask is not None:
            return self._fg1_mask[idx]
        if comp == 2 and self._fg2_mask is not None:
            return self._fg2_mask[idx]
        if comp == 3 and self._fg3_mask is not None:
            return self._fg3_mask[idx]
        r = self.df.iloc[idx]
        smi = r["smiles1"] if comp == 1 else r["smiles2"] if comp == 2 else r["smiles3"]
        _, mask = self.fg_cache.get_token_ids(smi, self.fg_max_tokens)
        return torch.from_numpy(mask)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        # Keep system metadata for physics-informed loss
        system_id = torch.tensor(int(r["system_id"]), dtype=torch.long)
        aug_swap23 = torch.tensor(int(r.get("aug_swap23", 0)), dtype=torch.long)
        sample_weight = torch.tensor(float(r.get("sample_weight", 1.0)), dtype=self.dtype)
        g1 = self.g_cache.get(r["smiles1"])
        g2 = self.g_cache.get(r["smiles2"])
        g3 = self.g_cache.get(r["smiles3"])

        if self._scalars is not None:
            scalars = self._scalars[idx]
            y = self._y[idx]
        else:
            T_raw_feature = temperature_scalar_value(
                [r["T"]],
                mode=getattr(C, "TEMPERATURE_ENCODING", "linear_quadratic"),
                reference_k=float(getattr(C, "TEMPERATURE_REFERENCE_K", 500.0)),
            )
            Tn = self.T_scaler.transform(T_raw_feature)[0].astype(np.float32)
            t = float(r["t"])
            Pn = 0.0
            if self.P_scaler:
                 try:
                     Pn = self.P_scaler.transform(np.array([r["P"]], dtype=np.float32))[0].astype(np.float32)
                 except Exception:
                     Pn = 0.0
            scalars = torch.from_numpy(
                condition_scalar_values(Tn, t, Pn, scalar_dim=self.scalar_dim)
            ).to(dtype=self.dtype)
            y = torch.tensor([r["Ex1"], r["Ex2"], r["Ex3"], r["Rx1"], r["Rx2"], r["Rx3"]], dtype=self.dtype)

        x = {"g1": g1, "g2": g2, "g3": g3, "scalars": scalars,
             "system_id": system_id, "aug_swap23": aug_swap23,
             "sample_weight": sample_weight}
        if self.use_fg:
            if self.fg_token_mode:
                x["fg1_ids"] = self.get_fg_token_ids(idx, 1)
                x["fg2_ids"] = self.get_fg_token_ids(idx, 2)
                x["fg3_ids"] = self.get_fg_token_ids(idx, 3)
                x["fg1_mask"] = self.get_fg_token_mask(idx, 1)
                x["fg2_mask"] = self.get_fg_token_mask(idx, 2)
                x["fg3_mask"] = self.get_fg_token_mask(idx, 3)
            else:
                if self._fg1 is not None:
                    x["fg1"] = self._fg1[idx]
                    x["fg2"] = self._fg2[idx]
                    x["fg3"] = self._fg3[idx]
                else:
                    x["fg1"] = torch.from_numpy(self.fg_cache.get(r["smiles1"]))
                    x["fg2"] = torch.from_numpy(self.fg_cache.get(r["smiles2"]))
                    x["fg3"] = torch.from_numpy(self.fg_cache.get(r["smiles3"]))

        if self.use_mix_graph and (self.mix_cache is not None):
            mix = self.mix_cache.build(
                r["smiles1"], r["smiles2"], r["smiles3"],
                float(scalars[0].item()), float(r["T"])
            )
            x["mix"] = mix

        return x, y


def collate_graph_batch(batch: List[Tuple[Dict[str, Any], torch.Tensor]]):
    """
    Collate list[(x_dict, y)] into:
        x = {'g1': batched_graph, 'g2': ..., 'g3': ..., 'scalars': (B,2|3), 'mix': ... (optional)}
        y = (B,6)
    """
    xs, ys = zip(*batch)
    g1 = batch_graphs([x["g1"] for x in xs])
    g2 = batch_graphs([x["g2"] for x in xs])
    g3 = batch_graphs([x["g3"] for x in xs])
    scalars = torch.stack([x["scalars"] for x in xs], dim=0).to(dtype=torch.float32)
    y = torch.stack(list(ys), dim=0).to(dtype=torch.float32)

    out = {
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "scalars": scalars,
        "system_id": torch.stack([x["system_id"] for x in xs], dim=0),
        "aug_swap23": torch.stack([x["aug_swap23"] for x in xs], dim=0),
        "sample_weight": torch.stack([x["sample_weight"] for x in xs], dim=0),
    }
    if isinstance(xs[0], dict) and ("fg1_ids" in xs[0]):
        out["fg1_ids"] = torch.stack([x["fg1_ids"] for x in xs], dim=0).to(dtype=torch.long)
        out["fg2_ids"] = torch.stack([x["fg2_ids"] for x in xs], dim=0).to(dtype=torch.long)
        out["fg3_ids"] = torch.stack([x["fg3_ids"] for x in xs], dim=0).to(dtype=torch.long)
        out["fg1_mask"] = torch.stack([x["fg1_mask"] for x in xs], dim=0).to(dtype=torch.float32)
        out["fg2_mask"] = torch.stack([x["fg2_mask"] for x in xs], dim=0).to(dtype=torch.float32)
        out["fg3_mask"] = torch.stack([x["fg3_mask"] for x in xs], dim=0).to(dtype=torch.float32)
    elif isinstance(xs[0], dict) and ("fg1" in xs[0]):
        out["fg1"] = torch.stack([x["fg1"] for x in xs], dim=0).to(dtype=torch.float32)
        out["fg2"] = torch.stack([x["fg2"] for x in xs], dim=0).to(dtype=torch.float32)
        out["fg3"] = torch.stack([x["fg3"] for x in xs], dim=0).to(dtype=torch.float32)
    if isinstance(xs[0], dict) and ("mix" in xs[0]):
        out["mix"] = batch_mixture_graphs([x["mix"] for x in xs])
    return out, y

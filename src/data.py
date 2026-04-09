# -*- coding: utf-8 -*-
"""
data.py - 数据加载和处理

核心功能：
1. Excel 数据加载和预处理
   - load_and_prepare_excel(): 加载 LLE 数据，处理缺失值，可选数据增强
   - split_by_system()/stratified_split_by_system(): 按体系分割训练/验证/测试集

2. 特征缓存类
   - FingerprintCache: Morgan 指纹缓存
   - GraphCache: RDKit 分子图缓存
   - FunctionalGroupCache: 功能基团（FG）编码缓存
   - MixGraphCache: 混合物图（混合三个分子）缓存

3. PyTorch 数据集类
   - LLEDataset: 指纹模式数据集（返回指纹特征和标签）
   - GraphLLEDataset: 图模式数据集（返回分子图和标签）
   
4. Batch 处理
   - collate_graph_batch(): 自定义 collate 函数，处理变长图数据
"""
from typing import Dict, Tuple, Optional, List, Any
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import OrderedDict, defaultdict
import types
from types import SimpleNamespace

from config import (
    FP_BITS, FP_RADIUS, MIN_POINTS_PER_GROUP, PERMUTE_23_AUG, SEED,
    GRAPH_ADD_HS, GRAPH_ADD_3D, GRAPH_USE_GASTEIGER, GRAPH_MAX_ATOMS,
    USE_MIX_GRAPH, USE_FG, FG_TOPK, FG_MIN_FREQ, PRECOMPUTE_FG,
    FG_TOKEN_MODE, FG_MAX_TOKENS
)
import config as C

from utils import (
    canonicalize_smiles, renorm3, safe_group_apply_t, Scaler, morgan_fp, fg_smiles_from_smiles,
    smiles_to_graph, batch_graphs,
    build_mixture_graph, batch_mixture_graphs
)


# =========================
# 列名处理工具函数
# =========================

def _norm_col(c: Any) -> str:
    """
    标准化列名：移除换行符、多余空格
    
    Args:
        c: 原始列名
    
    Returns:
        str: 标准化后的列名
    """
    # 移除换行符和多余空格
    c_str = str(c).strip().replace('\n', ' ').replace('\r', ' ')
    # 合并多个空格为单个空格
    c_str = ' '.join(c_str.split())
    return c_str


def _find_col(available_cols: List[str], candidates: List[str]) -> Optional[str]:
    """
    从可用列中查找候选列名
    
    按优先级尝试匹配候选列名，返回第一个存在的列名。
    支持不同大小写和空格变化的列名匹配。
    
    Args:
        available_cols: 当前DataFrame中的所有列名
        candidates: 候选列名列表（按优先级排序）
    
    Returns:
        str: 找到的列名，如果没找到则返回 None
    """
    # 构建规范化的列名映射（用于不区分大小写的匹配）
    norm_cols = {_norm_col(c).lower(): c for c in available_cols}
    
    # 按优先级尝试匹配
    for cand in candidates:
        cand_norm = _norm_col(cand).lower()
        if cand_norm in norm_cols:
            return norm_cols[cand_norm]
    
    return None


def _require_col(df: pd.DataFrame, name: str, candidates: List[str]) -> str:
    """
    从 DataFrame 列名中找到指定的列
    
    尝试从候选列名中找到第一个存在的列。
    如果都找不到，则抛出 KeyError 并显示所有可用列名。
    """
    col = _find_col(df.columns.tolist(), candidates)
    if col is None:
        raise KeyError(
            f"Cannot find column for '{name}'. Tried candidates={candidates}\n"
            f"Available columns ({len(df.columns)}):\n{list(df.columns)}"
        )
    return col


def _try_get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return _find_col(df.columns.tolist(), candidates)


def _read_excel_with_smiles_fallback(path: str) -> pd.DataFrame:
    """
    Prefer the requested workbook, but if it is a raw AIChEJ sheet without SMILES
    columns, automatically switch to a sibling workbook that already includes
    SMILES columns.
    """
    df = pd.read_excel(path)
    df.columns = [_norm_col(c) for c in df.columns]

    smiles1 = _try_get_col(df, [
        "IL (Component 1) full name SMILES",
        "IL (Component 1) SMILES",
        "Component 1 SMILES", "Comp 1 SMILES",
        "Component1-SMILES",
        "smiles1", "SMILES1", "SMILES 1"
    ])
    smiles2 = _try_get_col(df, [
        "Component 2 SMILES", "Comp 2 SMILES",
        "Component2-SMILES",
        "smiles2", "SMILES2", "SMILES 2"
    ])
    smiles3 = _try_get_col(df, [
        "Component 3 SMILES", "Comp 3 SMILES",
        "Component3-SMILES",
        "smiles3", "SMILES3", "SMILES 3"
    ])
    if smiles1 and smiles2 and smiles3:
        return df

    folder = os.path.dirname(path) or "."
    fallback_candidates = [
        os.path.join(folder, "update-LLE-all-with-smiles_min3.xlsx"),
        os.path.join(folder, "update-LLE-all-with-smiles.xlsx"),
    ]
    for alt_path in fallback_candidates:
        if not os.path.isfile(alt_path):
            continue
        alt_df = pd.read_excel(alt_path)
        alt_df.columns = [_norm_col(c) for c in alt_df.columns]
        alt_s1 = _try_get_col(alt_df, ["IL (Component 1) full name SMILES", "smiles1", "Component1-SMILES"])
        alt_s2 = _try_get_col(alt_df, ["Component 2 SMILES", "smiles2", "Component2-SMILES"])
        alt_s3 = _try_get_col(alt_df, ["Component 3 SMILES", "smiles3", "Component3-SMILES"])
        if alt_s1 and alt_s2 and alt_s3:
            print(f"[data] Missing SMILES columns in '{path}'. Using '{alt_path}' instead.")
            return alt_df

    return df


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
    df = _read_excel_with_smiles_fallback(path)

    col_system = _require_col(df, "system_id", [
        "system_id", "system id", "System ID", "System_ID",
        "LLE system NO.", "LLE system NO", "LLE system No.", "LLE system No",
        "LLE system number", "LLE system#", "LLE system #",
        "System No.", "System No", "System NO."
    ])
    col_T = _require_col(df, "T", [
        "T/K", "T / K", "T (K)", "T", "Temp", "Temperature", "Temperature/K", "Temperature (K)"
    ])

    col_s1 = _require_col(df, "smiles1", [
        "IL (Component 1) full name SMILES",
        "IL (Component 1) SMILES",
        "Component 1 SMILES", "Comp 1 SMILES",
        "Component1-SMILES",
        "smiles1", "SMILES1", "SMILES 1"
    ])
    col_s2 = _require_col(df, "smiles2", [
        "Component 2 SMILES", "Comp 2 SMILES",
        "Component2-SMILES",
        "smiles2", "SMILES2", "SMILES 2"
    ])
    col_s3 = _require_col(df, "smiles3", [
        "Component 3 SMILES", "Comp 3 SMILES",
        "Component3-SMILES",
        "smiles3", "SMILES3", "SMILES 3"
    ])

    def _req_comp(name: str, extra: Optional[List[str]] = None) -> str:
        candidates = [name, name.upper(), name.lower(), name.replace("x", "X"), name.replace("X", "x")]
        if extra:
            candidates = list(extra) + candidates
        return _require_col(df, name, candidates)

    # Older AIChEJ sheets may store Ex1 under "Rx1" and the actual Rx1 under "Rx1.1".
    if (_try_get_col(df, ["Ex1"]) is None) and (_try_get_col(df, ["Rx1.1"]) is not None):
        col_Ex1 = _req_comp("Ex1", extra=["Rx1"])
        col_Rx1 = _req_comp("Rx1", extra=["Rx1.1"])
    else:
        col_Ex1 = _req_comp("Ex1")
        col_Rx1 = _req_comp("Rx1")
    col_Ex2 = _req_comp("Ex2")
    col_Ex3 = _req_comp("Ex3")
    col_Rx2 = _req_comp("Rx2")
    col_Rx3 = _req_comp("Rx3")

    df = df.rename(columns={
        col_system: "system_id",
        col_T: "T",
        col_s1: "smiles1",
        col_s2: "smiles2",
        col_s3: "smiles3",
        col_Ex1: "Ex1", col_Ex2: "Ex2", col_Ex3: "Ex3",
        col_Rx1: "Rx1", col_Rx2: "Rx2", col_Rx3: "Rx3",
    })

    for c in ["smiles1", "smiles2", "smiles3"]:
        df[c] = df[c].astype(str).map(canonicalize_smiles)
    df = df[(df["smiles1"] != "") & (df["smiles2"] != "") & (df["smiles3"] != "")].copy()

    for c in ["T", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["T", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]).copy()

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
    df_aug = df.copy()
    df_aug["aug_swap23"] = 0

    if permute_23_aug:
        df2 = df.copy()
        df2["aug_swap23"] = 1
        df2[["smiles2", "smiles3"]] = df2[["smiles3", "smiles2"]]
        df2[["Ex2", "Ex3"]] = df2[["Ex3", "Ex2"]]
        df2[["Rx2", "Rx3"]] = df2[["Rx3", "Rx2"]]
        df_aug = pd.concat([df_aug, df2], ignore_index=True)

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

def stratified_split_by_system(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = SEED,
    n_bins: int = 8,
    min_bin_size: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分层按 system_id 切分：让 train/val/test 在“难度代理”分布上更一致
    难度代理（每个 system_id）：
      - n_rows: 该体系的样本点总数
      - n_groups: 该体系 (system_id, T) 组的数量
      - T_span: 该体系温度范围跨度
    """
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
        # qcut 避免 unique 太少时报错
        uniq = int(s.nunique())
        q = int(max(1, min(q, uniq)))
        if q <= 1:
            return pd.Series(["ALL"] * len(s), index=s.index)
        try:
            return pd.qcut(s, q=q, duplicates="drop").astype(str)
        except Exception:
            return pd.Series(["ALL"] * len(s), index=s.index)

    # 两个主要难度代理做分箱，再拼成分层标签
    stats["bin_rows"] = _qbin(stats["n_rows"], n_bins)
    stats["bin_span"] = _qbin(stats["T_span"], n_bins)
    # 可选：把 n_groups 也纳入（更“细”但也更容易产生小分层）
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

        # 分配数量（保证不会把某个分层全部塞进 train 导致 val/test 空）
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        # 约束：尽量保证 test 至少 1（当 n 足够大时）
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

    # 最后再做一次防交叉（理论上不会重叠，这里防御性处理）
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
    Legacy FP dataset:
      x = [fp1, fp2, fp3, T_norm, t], y = [Ex1..3, Rx1..3]
    """
    def __init__(
        self,
        df: pd.DataFrame,
        T_scaler: Scaler,
        fp_cache: FingerprintCache,
        fg_cache: Optional[FunctionalGroupCache] = None,
        use_fg: Optional[bool] = None,
        precompute: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        self.df = df.reset_index(drop=True)
        self.T_scaler = T_scaler
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

            Tn = self.T_scaler.transform(np.array([r["T"]], dtype=np.float32))[0].astype(np.float32)
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
        Tn = self.T_scaler.transform(np.array([r["T"]], dtype=np.float32))[0].astype(np.float32)
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
    """
    Windows DataLoader(num_workers>0) 需要 pickle Dataset。
    这里把 config 模块转换为可 pickle 的 SimpleNamespace（只保留大写常量）。
    """
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

        # IMPORTANT: utils.build_mixture_graph() 不支持 dtype 参数
        g = build_mixture_graph(
            smi1, smi2, smi3,
            T_norm=float(T_norm), T_raw=float(T_raw),
            cfg=self.cfg,
            mol_cache=self.mol_cache,
            pair_cache=self.pair_cache,
        )

        # 强制 dtype 一致（避免后续 AMP/拼接出问题）
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
            'scalars': tensor([T_norm, t]),
            'mix': mixture_graph_dict (optional)
        }
        y = tensor([Ex1..3, Rx1..3])
    """
    def __init__(
        self,
        df: pd.DataFrame,
        T_scaler: Scaler,
        g_cache: GraphCache,
        mix_cache: Optional[MixGraphCache] = None,
        fg_cache: Optional[FunctionalGroupCache] = None,
        use_fg: Optional[bool] = None,
        use_mix_graph: Optional[bool] = None,
        dtype: torch.dtype = torch.float32,
        precompute_scalars: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.T_scaler = T_scaler
        self.g_cache = g_cache
        self.fg_cache = fg_cache
        self.use_fg = bool(getattr(C, "USE_FG", False) if use_fg is None else use_fg)
        self.use_fg = self.use_fg and (self.fg_cache is not None)
        self.fg_dim = int(getattr(C, "FG_TOPK", FG_TOPK)) if self.use_fg else 0
        self.fg_token_mode = bool(getattr(C, "FG_TOKEN_MODE", FG_TOKEN_MODE)) if self.use_fg else False
        self.fg_max_tokens = int(getattr(C, "FG_MAX_TOKENS", FG_MAX_TOKENS)) if self.use_fg else 0
        self.use_mix_graph = bool(USE_MIX_GRAPH if use_mix_graph is None else use_mix_graph)

        # MixGraphCache 已经是 pickle-safe（cfg 转 namespace），可以直接放在 Dataset 里
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
        scalars = np.empty((n, 2), dtype=np.float32)
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
            Tn = self.T_scaler.transform(np.array([r["T"]], dtype=np.float32))[0].astype(np.float32)
            t = float(r["t"])
            scalars[i, :] = np.array([Tn, t], dtype=np.float32)
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
        g1 = self.g_cache.get(r["smiles1"])
        g2 = self.g_cache.get(r["smiles2"])
        g3 = self.g_cache.get(r["smiles3"])

        if self._scalars is not None:
            scalars = self._scalars[idx]
            y = self._y[idx]
        else:
            Tn = self.T_scaler.transform(np.array([r["T"]], dtype=np.float32))[0].astype(np.float32)
            t = float(r["t"])
            scalars = torch.tensor([Tn, t], dtype=self.dtype)
            y = torch.tensor([r["Ex1"], r["Ex2"], r["Ex3"], r["Rx1"], r["Rx2"], r["Rx3"]], dtype=self.dtype)

        x = {"g1": g1, "g2": g2, "g3": g3, "scalars": scalars,
             "system_id": system_id, "aug_swap23": aug_swap23}
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
        x = {'g1': batched_graph, 'g2': ..., 'g3': ..., 'scalars': (B,2), 'mix': ... (optional)}
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

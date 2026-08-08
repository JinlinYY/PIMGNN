# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch

from . import config as C
from .data import (
    FingerprintCache,
    GraphCache,
    LLEGNNDataset,
    SmilesRNNDataset,
    build_smiles_vocab,
    gnn_collate_fn,
)
from .utils import renorm3


def build_X_from_df_raw(df_raw: pd.DataFrame, T_scaler, fp_cache: FingerprintCache) -> np.ndarray:
    """
    Build model input X from df_raw rows (uses true T and t).
    X shape: (N, 3*FP_BITS + 2)
    """
    n = len(df_raw)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # infer fp_bits from cache
    fp_bits = fp_cache.n_bits
    X = np.zeros((n, 3 * fp_bits + 2), dtype=np.float32)

    # ensure required columns exist
    need_cols = ["smiles1", "smiles2", "smiles3", "T", "t"]
    for c in need_cols:
        if c not in df_raw.columns:
            raise KeyError(f"df_raw missing column: {c}")

    for i, r in enumerate(df_raw[need_cols].to_dict("records")):
        fp1 = fp_cache.get(r["smiles1"])
        fp2 = fp_cache.get(r["smiles2"])
        fp3 = fp_cache.get(r["smiles3"])
        Tn = T_scaler.transform(np.array([r["T"]], dtype=np.float32))[0].astype(np.float32)
        t = float(r["t"])
        X[i, :fp_bits] = fp1
        X[i, fp_bits:2*fp_bits] = fp2
        X[i, 2*fp_bits:3*fp_bits] = fp3
        X[i, 3*fp_bits:] = np.array([Tn, t], dtype=np.float32)
    return X


@torch.no_grad()
def predict_pointwise_df_raw(model: torch.nn.Module, T_scaler, df_raw_test: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper (torch backend)."""
    return predict_pointwise_df_raw_generic(model, T_scaler, df_raw_test, backend="torch")


@torch.no_grad()
def predict_pointwise_df_raw_generic(model, T_scaler, df_raw_test: pd.DataFrame, backend: str = "auto") -> pd.DataFrame:
    """
    Pointwise prediction on df_raw_test using its true (T, t).

    backend:
      - "torch": model(x_torch)->(N,6) or model(tokens, scalars)->(N,6) for SMILES RNN
      - "sklearn": model.predict(X_np)->(N,6)
      - "auto": detect by checking torch.nn.Module
    """
    if backend == "auto":
        backend = "torch" if isinstance(model, torch.nn.Module) else "sklearn"

    if len(df_raw_test) == 0:
        out = df_raw_test.copy()
        for k in ["pred_Ex1","pred_Ex2","pred_Ex3","pred_Rx1","pred_Rx2","pred_Rx3"]:
            out[k] = []
        return out

    if backend == "torch":
        model.eval()
        # Detect if SMILES RNN by checking model class name
        is_smiles_rnn = "SmilesRNN" in model.__class__.__name__
        is_gnn = bool(getattr(model, "is_gnn", False))
        bs = int(getattr(C, "BATCH_SIZE", 1024))
        preds = []

        if is_smiles_rnn:
            # SMILES RNN: need vocab and scalar features
            smiles_vocab = build_smiles_vocab([df_raw_test])
            max_len = getattr(C, "SMILES_MAX_LEN", 256)
            scalar_dim = 2 if getattr(C, "SMILES_USE_TIE_T", True) else 1
            ds = SmilesRNNDataset(df_raw_test, smiles_vocab, T_scaler, max_len=max_len, use_t=scalar_dim==2)
            
            for i in tqdm(range(0, len(ds), bs), desc="Predict (torch SMILES RNN)"):
                batch_tokens = []
                batch_scalars = []
                for j in range(i, min(i+bs, len(ds))):
                    tokens, scalars, _ = ds[j]
                    batch_tokens.append(tokens)
                    batch_scalars.append(scalars)
                
                tokens_b = torch.stack(batch_tokens).to(C.DEVICE)
                scalars_b = torch.stack(batch_scalars).to(C.DEVICE)
                yb = model(tokens_b, scalars_b).detach().cpu().numpy()
                preds.append(yb)
        elif is_gnn:
            dataset = LLEGNNDataset(df_raw_test, T_scaler, GraphCache())
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                shuffle=False,
                num_workers=0,
                collate_fn=gnn_collate_fn,
            )
            for graph1, graph2, graph3, scalars, _ in tqdm(
                loader, desc="Predict (torch GNN)"
            ):
                graph1 = tuple(value.to(C.DEVICE) for value in graph1)
                graph2 = tuple(value.to(C.DEVICE) for value in graph2)
                graph3 = tuple(value.to(C.DEVICE) for value in graph3)
                prediction = model(
                    graph1, graph2, graph3, scalars.to(C.DEVICE)
                ).detach().cpu().numpy()
                preds.append(prediction)
        else:
            # Fingerprint-based models
            fp_cache = FingerprintCache()
            X = build_X_from_df_raw(df_raw_test, T_scaler, fp_cache)
            
            for i in tqdm(range(0, X.shape[0], bs), desc="Predict (torch FP)"):
                xb = torch.from_numpy(X[i:i+bs]).to(C.DEVICE)
                yb = model(xb).detach().cpu().numpy()
                preds.append(yb)

        preds = np.concatenate(preds, axis=0).astype(np.float32)

    elif backend == "sklearn":
        # sklearn-like API: predict(X)->(N,6)
        fp_cache = FingerprintCache()
        X = build_X_from_df_raw(df_raw_test, T_scaler, fp_cache)
        preds = np.asarray(model.predict(X), dtype=np.float32)
        if preds.ndim != 2 or preds.shape[1] != 6:
            raise ValueError(f"sklearn model.predict(X) must return (N,6), got {preds.shape}")

        # enforce compositional constraints
        E = preds[:, :3].copy()
        R = preds[:, 3:].copy()
        for i in range(preds.shape[0]):
            E[i] = renorm3(np.clip(E[i], 0.0, None))
            R[i] = renorm3(np.clip(R[i], 0.0, None))
        preds = np.concatenate([E, R], axis=1)

    else:
        raise ValueError(f"Unknown backend={backend}")

    out = df_raw_test.copy()
    out["pred_Ex1"] = preds[:, 0]; out["pred_Ex2"] = preds[:, 1]; out["pred_Ex3"] = preds[:, 2]
    out["pred_Rx1"] = preds[:, 3]; out["pred_Rx2"] = preds[:, 4]; out["pred_Rx3"] = preds[:, 5]
    return out

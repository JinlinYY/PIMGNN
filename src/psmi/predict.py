# -*- coding: utf-8 -*-
"""Run pointwise PSMI inference on prepared LLE data frames."""

import os
import json
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.nn.parameter import is_lazy
from torch.utils.data import DataLoader

from . import config as C
from .data import FingerprintCache, GraphCache, MixGraphCache, FunctionalGroupCache, GraphLLEDataset, collate_graph_batch
from .utils import batch_to_device, temperature_scalar_value


def _build_fg_cache_from_model(model) -> Optional[FunctionalGroupCache]:
    """Build fg cache from model."""
    corpus = getattr(model, "fg_corpus", None)
    if not corpus:
        return None
    vocab_size = int(getattr(model, "fg_vocab_size", getattr(C, "FG_TOPK", len(corpus))))
    fg_cache = FunctionalGroupCache(corpus=corpus, vocab_size=vocab_size, min_freq=int(getattr(C, "FG_MIN_FREQ", 3)))
    fg_cache.set_corpus(list(corpus))
    return fg_cache


def _build_fg_cache_for_infer(model) -> Optional[FunctionalGroupCache]:
    """Build fg cache for infer."""
    fg_cache = _build_fg_cache_from_model(model)
    if fg_cache is not None:
        return fg_cache

    
    out_dir = getattr(C, "OUT_DIR", "")
    if out_dir:
        p = os.path.join(out_dir, "fg_corpus.json")
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    corpus = json.load(f)
                vocab_size = int(getattr(C, "FG_TOPK", len(corpus)))
                fg_cache = FunctionalGroupCache(corpus=corpus, vocab_size=vocab_size, min_freq=int(getattr(C, "FG_MIN_FREQ", 3)))
                fg_cache.set_corpus(list(corpus))
                return fg_cache
            except Exception:
                return None
    return None


def _infer_model_device(model: torch.nn.Module) -> str:
    """Return the device of initialized model parameters."""
    for parameter in model.parameters():
        if not is_lazy(parameter):
            return str(parameter.device)
    return str(getattr(C, "DEVICE", "cpu"))


@torch.no_grad()
def predict_pointwise_df_raw(
    model: torch.nn.Module,
    T_scaler,
    df_raw_test: pd.DataFrame,
    device: Optional[str] = None,
    P_scaler=None,
) -> pd.DataFrame:
    """Predict pointwise df raw."""
    model.eval()
    device = str(device or _infer_model_device(model))
    use_graph = getattr(C, "USE_GRAPH", False)

    
    fg_cache = _build_fg_cache_for_infer(model) if getattr(C, "USE_FG", False) else None

    if use_graph:
        
        g_cache = GraphCache(
            add_hs=getattr(C, "GRAPH_ADD_HS", False),
            add_3d=getattr(C, "GRAPH_ADD_3D", False),
            use_gasteiger=getattr(C, "GRAPH_USE_GASTEIGER", True),
            max_atoms=getattr(C, "GRAPH_MAX_ATOMS", 256),
        )
        smiles_all = df_raw_test[["smiles1", "smiles2", "smiles3"]].values.reshape(-1).tolist()
        g_cache.build_from_smiles(smiles_all)

        
        ds = GraphLLEDataset(
            df_raw_test,
            T_scaler,
            g_cache,
            P_scaler=P_scaler,
            fg_cache=fg_cache,
            use_fg=getattr(C, "USE_FG", False),
            scalar_dim=int(getattr(model, "scalar_dim", getattr(C, "SCALAR_DIM", 3))),
            precompute_scalars=True,
        )
        loader = DataLoader(
            ds,
            batch_size=getattr(C, "PRED_BATCH_SIZE_GRAPH", 64),
            shuffle=False,
            num_workers=0,
            pin_memory=device.startswith("cuda"),
            collate_fn=collate_graph_batch,
        )

        
        preds = []
        for x, _y in tqdm(loader, desc="Pointwise predict"):
            x = batch_to_device(x, device)
            pred = model(x).detach().cpu().numpy()
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)
    else:
        
        fp_cache = FingerprintCache(radius=getattr(C, "FP_RADIUS"), n_bits=getattr(C, "FP_BITS"))
        preds = []
        
        for i in tqdm(range(len(df_raw_test)), desc="Pointwise predict"):
            r = df_raw_test.iloc[i]
            
            fp1 = fp_cache.get(r["smiles1"])
            fp2 = fp_cache.get(r["smiles2"])
            fp3 = fp_cache.get(r["smiles3"])
            
            T_feature = temperature_scalar_value(
                np.array([r["T"]], dtype=np.float32),
                mode=getattr(C, "TEMPERATURE_ENCODING", "linear_quadratic"),
                reference_k=getattr(C, "TEMPERATURE_REFERENCE_K", 500.0),
            )
            Tn = T_scaler.transform(T_feature)[0].astype(np.float32)
            t = float(r["t"])
            
            parts = [fp1, fp2, fp3]
            
            if fg_cache is not None:
                parts.extend([fg_cache.get(r["smiles1"]), fg_cache.get(r["smiles2"]), fg_cache.get(r["smiles3"])])
            
            parts.append(np.array([Tn, t], dtype=np.float32))
            x = np.concatenate(parts, axis=0).astype(np.float32)
            
            x = torch.from_numpy(x[None, :]).to(device)
            y = model(x).detach().cpu().numpy().reshape(-1)
            preds.append(y)
        preds = np.stack(preds, axis=0)

    
    out = df_raw_test.copy()
    out["pred_Ex1"] = preds[:, 0]; out["pred_Ex2"] = preds[:, 1]; out["pred_Ex3"] = preds[:, 2]
    out["pred_Rx1"] = preds[:, 3]; out["pred_Rx2"] = preds[:, 4]; out["pred_Rx3"] = preds[:, 5]
    return out

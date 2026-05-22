# -*- coding: utf-8 -*-
"""
predict.py - 模型推理和预测

功能：
- 使用训练好的模型对新数据进行点态预测
- 支持图模式和指纹模式的推理
- 返回预测的 Extract 和 Raffinate 相组成
"""

import os
import json
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

import config as C
from data import FingerprintCache, GraphCache, MixGraphCache, FunctionalGroupCache, GraphLLEDataset, collate_graph_batch
from utils import batch_to_device


def _build_fg_cache_from_model(model) -> Optional[FunctionalGroupCache]:
    """
    从模型的 fg_corpus 属性构建功能基团缓存
    
    Args:
        model: 包含 fg_corpus 的神经网络模型
    
    Returns:
        FunctionalGroupCache 实例或 None
    """
    corpus = getattr(model, "fg_corpus", None)
    if not corpus:
        return None
    vocab_size = int(getattr(model, "fg_vocab_size", getattr(C, "FG_TOPK", len(corpus))))
    fg_cache = FunctionalGroupCache(corpus=corpus, vocab_size=vocab_size, min_freq=int(getattr(C, "FG_MIN_FREQ", 3)))
    fg_cache.set_corpus(list(corpus))
    return fg_cache


def _build_fg_cache_for_infer(model) -> Optional[FunctionalGroupCache]:
    """
    为推理构建功能基团缓存，优先级：
      1) 从模型的 fg_corpus
      2) 从 OUT_DIR/fg_corpus.json 加载
    """
    fg_cache = _build_fg_cache_from_model(model)
    if fg_cache is not None:
        return fg_cache

    # 降级方案：从保存的语料文件加载
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


@torch.no_grad()
def predict_pointwise_df_raw(model: torch.nn.Module, T_scaler, df_raw_test: pd.DataFrame) -> pd.DataFrame:
    """
    对测试集进行点态预测，使用其真实的 (T, t) 值
    
    Args:
        model: 训练好的预测模型
        T_scaler: 温度的标准化器
        df_raw_test: 测试数据的原始 DataFrame
    
    Returns:
        pd.DataFrame: 添加了预测列的 DataFrame
            - pred_Ex1, pred_Ex2, pred_Ex3: Extract 相预测组成
            - pred_Rx1, pred_Rx2, pred_Rx3: Raffinate 相预测组成
    """
    model.eval()
    device = getattr(C, "DEVICE")
    use_graph = getattr(C, "USE_GRAPH", False)

    # 构建功能基团缓存（适用于图模式和指纹模式）
    fg_cache = _build_fg_cache_for_infer(model) if getattr(C, "USE_FG", False) else None

    if use_graph:
        # 图模式推理
        g_cache = GraphCache(
            add_hs=getattr(C, "GRAPH_ADD_HS", False),
            add_3d=getattr(C, "GRAPH_ADD_3D", False),
            use_gasteiger=getattr(C, "GRAPH_USE_GASTEIGER", True),
            max_atoms=getattr(C, "GRAPH_MAX_ATOMS", 256),
        )
        smiles_all = df_raw_test[["smiles1", "smiles2", "smiles3"]].values.reshape(-1).tolist()
        g_cache.build_from_smiles(smiles_all)

        # 构建图数据集和加载器
        ds = GraphLLEDataset(df_raw_test, T_scaler, g_cache, fg_cache=fg_cache, use_fg=getattr(C, "USE_FG", False), precompute_scalars=True)
        loader = DataLoader(
            ds,
            batch_size=getattr(C, "PRED_BATCH_SIZE_GRAPH", 64),
            shuffle=False,
            num_workers=0,
            pin_memory=device.startswith("cuda"),
            collate_fn=collate_graph_batch,
        )

        # 批量预测
        preds = []
        for x, _y in tqdm(loader, desc="Pointwise predict"):
            x = batch_to_device(x, device)
            pred = model(x).detach().cpu().numpy()
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)
    else:
        # 指纹模式推理
        fp_cache = FingerprintCache(radius=getattr(C, "FP_RADIUS"), n_bits=getattr(C, "FP_BITS"))
        preds = []
        # 逐行预测
        for i in tqdm(range(len(df_raw_test)), desc="Pointwise predict"):
            r = df_raw_test.iloc[i]
            # 获取指纹特征
            fp1 = fp_cache.get(r["smiles1"])
            fp2 = fp_cache.get(r["smiles2"])
            fp3 = fp_cache.get(r["smiles3"])
            # 标准化温度
            Tn = T_scaler.transform(np.array([r["T"]], dtype=np.float32))[0].astype(np.float32)
            t = float(r["t"])
            # 拼接特征
            parts = [fp1, fp2, fp3]
            # 如果启用了功能基团特征，添加到输入
            if fg_cache is not None:
                parts.extend([fg_cache.get(r["smiles1"]), fg_cache.get(r["smiles2"]), fg_cache.get(r["smiles3"])])
            # 添加标量特征（温度和滴定点）
            parts.append(np.array([Tn, t], dtype=np.float32))
            x = np.concatenate(parts, axis=0).astype(np.float32)
            # 模型前向传递
            x = torch.from_numpy(x[None, :]).to(device)
            y = model(x).detach().cpu().numpy().reshape(-1)
            preds.append(y)
        preds = np.stack(preds, axis=0)

    # 将预测结果添加到 DataFrame
    out = df_raw_test.copy()
    out["pred_Ex1"] = preds[:, 0]; out["pred_Ex2"] = preds[:, 1]; out["pred_Ex3"] = preds[:, 2]
    out["pred_Rx1"] = preds[:, 3]; out["pred_Rx2"] = preds[:, 4]; out["pred_Rx3"] = preds[:, 5]
    return out

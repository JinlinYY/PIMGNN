# -*- coding: utf-8 -*-
"""
train.py - 模型训练和加载

功能：
- 构建和配置神经网络模型
- 设置优化器和损失函数
- 完整的训练流程（前向、反向、更新）
- 验证/测试集评估
- 模型检查点保存和加载
- 物理约束损失集成

支持两种模式：
1. 指纹模式：Morgan 指纹 + MLP
2. 图模式：RDKit 分子图 + GNN（支持混合物图、功能基团等）
"""
import os
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
# 必须在 import pyplot 之前设置后端
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from utils import Scaler, batch_to_device
from loss import MechanisticNRTLLoss, NRTLParamStore
from data import (
    MixGraphCache,
    FingerprintCache, LLEDataset,
    FunctionalGroupCache,
    GraphCache, GraphLLEDataset, collate_graph_batch
)
from metrics import evaluate_loader, print_metrics, compute_physics_metrics
from model import LLECurveNet, LLEGraphNet


def plot_history(history: Dict[str, List[float]], out_dir: str) -> None:
    """
    绘制并保存训练历史曲线图
    
    参数：
        history: 字典，包含各epoch的 train/val/test 指标
        out_dir: 输出目录
    
    输出文件：
        - curve_loss_mse.png: MSE 曲线
        - curve_mae.png: MAE 曲线
        - curve_rmse.png: RMSE 曲线
        - curve_r2.png: R² 曲线
        - curve_rmse_ex_rx.png: Extract/Raffinate 分组 RMSE 曲线
    """
    os.makedirs(out_dir, exist_ok=True)

    # 绘制 MSE 曲线（训练/验证/测试）
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot(history["epoch"], history["train_mse"])
    ax.plot(history["epoch"], history["val_mse"])
    ax.plot(history["epoch"], history["test_mse"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Loss Curves (MSE)")
    ax.legend(["train", "val", "test"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_loss_mse.png"), dpi=200)
    plt.close(fig)

    # 绘制 MAE 曲线（如果存在）
    if "val_mae" in history and "test_mae" in history:
        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.plot(history["epoch"], history["val_mae"])
        ax.plot(history["epoch"], history["test_mae"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.set_title("MAE Curves")
        ax.legend(["val", "test"], loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_mae.png"), dpi=200)
        plt.close(fig)

    # 绘制 RMSE 曲线
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot(history["epoch"], history["val_rmse"])
    ax.plot(history["epoch"], history["test_rmse"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Curves")
    ax.legend(["val", "test"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_rmse.png"), dpi=200)
    plt.close(fig)

    # 绘制 R² 曲线
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot(history["epoch"], history["val_r2"])
    ax.plot(history["epoch"], history["test_r2"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R2")
    ax.set_title("R2 Curves")
    ax.legend(["val", "test"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_r2.png"), dpi=200)
    plt.close(fig)

    # 绘制 Extract 和 Raffinate 分组的 RMSE 曲线
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot(history["epoch"], history["val_rmse_E"])
    ax.plot(history["epoch"], history["val_rmse_R"])
    ax.plot(history["epoch"], history["test_rmse_E"])
    ax.plot(history["epoch"], history["test_rmse_R"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE (Ex vs Rx)")
    ax.legend(["val_E", "val_R", "test_E", "test_R"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_rmse_ex_rx.png"), dpi=200)
    plt.close(fig)

    # 绘制物理指标曲线（如果存在）
    if "val_mu_res_mae" in history and len(history["val_mu_res_mae"]) > 0:
        # 检查是否有有效数据（非nan）
        val_data = [v for v in history["val_mu_res_mae"] if not np.isnan(v)]
        if len(val_data) > 0:
            # 化学势残差 (mu_res_mae)
            fig = plt.figure(figsize=(7, 5))
            ax = plt.gca()
            ax.plot(history["epoch"], history["val_mu_res_mae"])
            ax.plot(history["epoch"], history["test_mu_res_mae"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Chemical Potential Residual MAE")
            ax.set_title("Physics: mu_res_mae")
            ax.legend(["val", "test"], loc="upper right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "curve_physics_mu_res_mae.png"), dpi=200)
            plt.close(fig)

    if "val_mu_res_max" in history and len(history["val_mu_res_max"]) > 0:
        val_data = [v for v in history["val_mu_res_max"] if not np.isnan(v)]
        if len(val_data) > 0:
            # 化学势残差最大值 (mu_res_max)
            fig = plt.figure(figsize=(7, 5))
            ax = plt.gca()
            ax.plot(history["epoch"], history["val_mu_res_max"])
            ax.plot(history["epoch"], history["test_mu_res_max"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Chemical Potential Residual Max")
            ax.set_title("Physics: mu_res_max")
            ax.legend(["val", "test"], loc="upper right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "curve_physics_mu_res_max.png"), dpi=200)
            plt.close(fig)

    if "val_tpd_viol_rate" in history and len(history["val_tpd_viol_rate"]) > 0:
        # TPD违例率 (tpd_viol_rate)
        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.plot(history["epoch"], history["val_tpd_viol_rate"])
        ax.plot(history["epoch"], history["test_tpd_viol_rate"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("TPD Violation Rate")
        ax.set_title("Physics: TPD Violation Rate")
        ax.legend(["val", "test"], loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_physics_tpd_viol_rate.png"), dpi=200)
        plt.close(fig)

    if "val_gd_res_mae" in history and len(history["val_gd_res_mae"]) > 0:
        val_data = [v for v in history["val_gd_res_mae"] if not np.isnan(v)]
        if len(val_data) > 0:
        # Gibbs-Duhem残差 (gd_res_mae)
            fig = plt.figure(figsize=(7, 5))
            ax = plt.gca()
            ax.plot(history["epoch"], history["val_gd_res_mae"])
            ax.plot(history["epoch"], history["test_gd_res_mae"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Gibbs-Duhem Residual MAE")
            ax.set_title("Physics: gd_res_mae")
            ax.legend(["val", "test"], loc="upper right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "curve_physics_gd_res_mae.png"), dpi=200)
            plt.close(fig)


def build_model():
    """
    根据配置构建神经网络模型
    
    返回：
        - USE_GRAPH=True: LLEGraphNet（支持图模式、混合物图、FG 等）
        - USE_GRAPH=False: LLECurveNet（指纹 + MLP 基线）
    """
    use_graph = getattr(C, "USE_GRAPH", False)
    if use_graph:
        return LLEGraphNet(
            gnn_hidden=getattr(C, "GNN_HIDDEN", 256),
            gnn_layers=getattr(C, "GNN_LAYERS", 4),
            mlp_hidden=getattr(C, "GNN_HEAD_HIDDEN", 512),
            dropout=getattr(C, "DROPOUT", 0.15),
            pool=getattr(C, "GNN_POOL", "mean"),
            use_interaction=getattr(C, "GNN_INTERACTION", True),
            use_mix_graph=getattr(C, "USE_MIX_GRAPH", False),
            mix_layers=getattr(C, "MIX_LAYERS", 2),
            mix_hidden=getattr(C, "MIX_HIDDEN", getattr(C, "GNN_HIDDEN", 256)),
            mix_dropout=getattr(C, "MIX_DROPOUT", 0.10),
            # FG options (optional)
            use_fg=getattr(C, "USE_FG", False),
            fg_vocab_size=int(getattr(C, "FG_TOPK", 0)),
            fg_hidden=int(getattr(C, "FG_MLP_HIDDEN", 256)),
            fg_dropout=float(getattr(C, "FG_DROPOUT", 0.10)),
            fg_token_mode=bool(getattr(C, "FG_TOKEN_MODE", False)),
            fg_max_tokens=int(getattr(C, "FG_MAX_TOKENS", 32)),
            fg_cross_attn=bool(getattr(C, "FG_CROSS_ATTN", False)),
            fg_attn_heads=int(getattr(C, "FG_ATTN_HEADS", 8)),
            s3_equivariant=bool(getattr(C, "S3_EQUIVARIANT", False)),
            # Transformer fusion (optional): concat -> token transformer
            fusion_mode=getattr(C, "FUSION_MODE", "concat"),
            tf_dim=int(getattr(C, "TF_DIM", getattr(C, "GNN_HIDDEN", 256))),
            tf_layers=int(getattr(C, "TF_LAYERS", 2)),
            tf_heads=int(getattr(C, "TF_HEADS", 8)),
            tf_ff=int(getattr(C, "TF_FF", 1024)),
            tf_dropout=float(getattr(C, "TF_DROPOUT", 0.10)),
            tf_pool=str(getattr(C, "TF_POOL", "cls")),
            tf_max_len=int(getattr(C, "TF_MAX_LEN", 32)),
            tf_type_vocab=int(getattr(C, "TF_TYPE_VOCAB", 16)),
        )
    in_dim = 3 * getattr(C, "FP_BITS") + 2
    if getattr(C, "USE_FG", False):
        in_dim += 3 * int(getattr(C, "FG_TOPK", 0))
    return LLECurveNet(in_dim=in_dim, hidden=getattr(C, "HIDDEN"), dropout=getattr(C, "DROPOUT"))


def _make_loader(ds, batch_size: int, shuffle: bool, device: str, collate_fn=None) -> DataLoader:
    # Windows 也可用，多进程能显著提升吞吐
    use_graph = getattr(C, "USE_GRAPH", False)
    num_workers = getattr(C, "NUM_WORKERS_GRAPH", 0) if use_graph else getattr(C, "NUM_WORKERS", min(8, os.cpu_count() or 4))
    pin = device.startswith("cuda")

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = getattr(C, "PREFETCH_FACTOR", 8)  # 提高prefetch，减少IO等待

    return DataLoader(ds, **kwargs)


def train_or_load(train_df, val_df, test_df):
    out_dir = getattr(C, "OUT_DIR")
    device = getattr(C, "DEVICE")

    eval_every = getattr(C, "EVAL_EVERY", 1)
    plot_every = getattr(C, "PLOT_EVERY", 5)

    use_amp = getattr(C, "USE_AMP", True) and device.startswith("cuda")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    T_scaler = Scaler.fit(train_df["T"].to_numpy(dtype="float32"))

    model = build_model().to(device)

    history = {k: [] for k in [
        "epoch",
        "train_mse",

        "val_mse", "test_mse",
        "val_mae", "test_mae",
        "val_rmse", "test_rmse",
        "val_r2", "test_r2",

        "val_mae_E", "val_mae_R",
        "val_rmse_E", "val_rmse_R",
        "val_r2_E", "val_r2_R",

        "test_mae_E", "test_mae_R",
        "test_rmse_E", "test_rmse_R",
        "test_r2_E", "test_r2_R",
    ]}

    ckpt_path = getattr(C, "LOAD_CKPT_PATH", "")
    loaded_from_ckpt = False
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            ckpt = {"state_dict": ckpt}

        # 兼容不同保存格式：{"state_dict":...} / {"model":...} / 纯 state_dict 字典
        state_dict = None
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            # best effort: 直接把 ckpt 当作 state_dict
            state_dict = ckpt

        if not isinstance(state_dict, dict):
            raise KeyError(
                f"[CKPT] 无法解析模型参数: {ckpt_path}，可用键: {list(ckpt.keys())}"
            )

        # keep runtime config consistent with checkpoint (important when USE_FG / FG_TOPK differs)
        if "use_fg" in ckpt:
            setattr(C, "USE_FG", bool(ckpt["use_fg"]))
        if "fg_topk" in ckpt:
            setattr(C, "FG_TOPK", int(ckpt["fg_topk"]))
        # Rebuild model with potentially updated config
        model = build_model().to(device)
        model.load_state_dict(state_dict)
        setattr(model, "fg_corpus", ckpt.get("fg_corpus", None))
        if ("T_mean" in ckpt) and ("T_std" in ckpt):
            T_scaler = Scaler(mean=float(ckpt["T_mean"]), std=float(ckpt["T_std"]))
        else:
            print("[WARN] Checkpoint missing T_mean/T_std, fallback to scaler fitted from current train split")
        loaded_from_ckpt = True
        print(f"[OK] Loaded checkpoint: {ckpt_path}")
        print(f"  - T_scaler: mean={T_scaler.mean:.2f}, std={T_scaler.std:.2f}")
        print(f"  - Best epoch from ckpt: {ckpt.get('best_epoch', ckpt.get('epoch', 'N/A'))}")
        print(f"  - Will continue training from loaded weights...")

    os.makedirs(out_dir, exist_ok=True)

    # Build FG corpus/cache (train-only) if enabled
    fg_cache = None
    if getattr(C, "USE_FG", False):
        fg_cache = FunctionalGroupCache(
            corpus=None,
            vocab_size=int(getattr(C, "FG_TOPK", 512)),
            min_freq=int(getattr(C, "FG_MIN_FREQ", 3)),
        )
        # If loaded from checkpoint and has fg_corpus, use it; otherwise build new
        if loaded_from_ckpt and hasattr(model, "fg_corpus") and model.fg_corpus:
            _corpus = model.fg_corpus
            print(f"  - Using FG corpus from checkpoint ({len(_corpus)} groups)")
        else:
            _smiles_train = []
            for col in ["smiles1", "smiles2", "smiles3"]:
                if col in train_df.columns:
                    _smiles_train.extend(train_df[col].astype(str).tolist())
            _smiles_train = sorted(set(_smiles_train))
            _corpus = fg_cache.build_corpus_from_smiles(_smiles_train)
            print(f"  - Built new FG corpus ({len(_corpus)} groups)")
        
        fg_cache.set_corpus(_corpus)
        try:
            with open(os.path.join(out_dir, "fg_corpus.json"), "w", encoding="utf-8") as f:
                json.dump(_corpus, f, ensure_ascii=False)
        except Exception:
            pass

    if fg_cache is not None:
        setattr(model, "fg_corpus", list(fg_cache.corpus))


    if getattr(C, "USE_GRAPH", False):
        g_cache = GraphCache(
            add_hs=getattr(C, "GRAPH_ADD_HS", False),
            add_3d=getattr(C, "GRAPH_ADD_3D", False),
            use_gasteiger=getattr(C, "GRAPH_USE_GASTEIGER", True),
            max_atoms=getattr(C, "GRAPH_MAX_ATOMS", 256),
        )
        smiles_all = pd.concat([train_df[["smiles1","smiles2","smiles3"]],
                                val_df[["smiles1","smiles2","smiles3"]],
                                test_df[["smiles1","smiles2","smiles3"]]], axis=0)
        g_cache.build_from_smiles(smiles_all.values.reshape(-1).tolist())

        mix_cache = MixGraphCache(C) if getattr(C, "USE_MIX_GRAPH", False) else None
        train_ds = GraphLLEDataset(train_df, T_scaler, g_cache, mix_cache=mix_cache, fg_cache=fg_cache, use_fg=getattr(C, "USE_FG", False), precompute_scalars=getattr(C, "PRECOMPUTE_SCALARS", True))
        val_ds = GraphLLEDataset(val_df, T_scaler, g_cache, mix_cache=mix_cache, fg_cache=fg_cache, use_fg=getattr(C, "USE_FG", False), precompute_scalars=getattr(C, "PRECOMPUTE_SCALARS", True))
        test_ds = GraphLLEDataset(test_df, T_scaler, g_cache, mix_cache=mix_cache, fg_cache=fg_cache, use_fg=getattr(C, "USE_FG", False), precompute_scalars=getattr(C, "PRECOMPUTE_SCALARS", True))

        batch_size = getattr(C, "BATCH_SIZE_GRAPH", 64)
        train_loader = _make_loader(train_ds, batch_size, shuffle=True, device=device, collate_fn=collate_graph_batch)
        val_loader = _make_loader(val_ds, batch_size, shuffle=False, device=device, collate_fn=collate_graph_batch)
        test_loader = _make_loader(test_ds, batch_size, shuffle=False, device=device, collate_fn=collate_graph_batch)
    else:
        fp_cache = FingerprintCache()
        precompute = getattr(C, "PRECOMPUTE_FEATURES", True)
        train_ds = LLEDataset(train_df, T_scaler, fp_cache, fg_cache=fg_cache, use_fg=getattr(C, "USE_FG", False), precompute=precompute)
        val_ds = LLEDataset(val_df, T_scaler, fp_cache, fg_cache=fg_cache, use_fg=getattr(C, "USE_FG", False), precompute=precompute)
        test_ds = LLEDataset(test_df, T_scaler, fp_cache, fg_cache=fg_cache, use_fg=getattr(C, "USE_FG", False), precompute=precompute)

        batch_size = getattr(C, "BATCH_SIZE", 1024)
        train_loader = _make_loader(train_ds, batch_size, shuffle=True, device=device, collate_fn=None)
        val_loader = _make_loader(val_ds, batch_size, shuffle=False, device=device, collate_fn=None)
        test_loader = _make_loader(test_ds, batch_size, shuffle=False, device=device, collate_fn=None)

    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_mse = float("inf")
    best_state = None
    best_epoch = -1
    best_val_metrics = None
    best_test_metrics = None
    
    # 早停机制 - 支持数据指标和物理指标
    use_early_stop = bool(getattr(C, "USE_EARLY_STOP", False))
    use_physics_finetune = bool(getattr(C, "USE_PHYSICS_FINETUNE", False))
    
    if use_physics_finetune:
        # 物理微调模式
        early_stop_patience = int(getattr(C, "FINETUNE_PATIENCE", 30))
        early_stop_metric = str(getattr(C, "FINETUNE_EARLY_STOP_METRIC", "mu_res_mae")).lower()
        print(f"[OK] Physics finetune mode enabled")
    else:
        # 标准预训练模式
        early_stop_patience = int(getattr(C, "EARLY_STOP_PATIENCE", 30))
        early_stop_metric = str(getattr(C, "EARLY_STOP_METRIC", "mse")).lower()
    
    early_stop_min_delta = float(getattr(C, "EARLY_STOP_MIN_DELTA", 0.0))
    early_stop_counter = 0  # 记录验证指标没有改善的epoch数
    best_monitor_value = float("inf") if early_stop_metric not in ["r2"] else float("-inf")
    
    if use_early_stop:
        print(f"[OK] Early stopping enabled: metric={early_stop_metric}, patience={early_stop_patience}, min_delta={early_stop_min_delta}")

    log_path = os.path.join(out_dir, "train_metrics_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_mse,"
            "val_mae,val_rmse,val_r2,val_mae_E,val_rmse_E,val_r2_E,val_mae_R,val_rmse_R,val_r2_R,"
            "test_mae,test_rmse,test_r2,test_mae_E,test_rmse_E,test_r2_E,test_mae_R,test_rmse_R,test_r2_R\n"
        )

    def _fmt8(v) -> str:
        try:
            v = float(v)
        except Exception:
            return "nan"
        if np.isnan(v) or np.isinf(v):
            return "nan"
        return f"{v:.8f}"

    # Initialize NRTL param store for physics metrics (even if USE_MECH_LOSS=False)
    nrtl_store = None
    nrtl_path = getattr(C, "NRTL_PARAMS_PATH", "")
    
    if not nrtl_path or not os.path.isfile(nrtl_path):
        # Fallback to search for params file
        for candidate in [
            r"D:\GGNN\YXFL\src\nrtl_params_all.json",
            r"D:\GGNN\YXFL\nrtl_param\nrtl_params_all.json",
            r"D:\GGNN\YXFL\src\nrtl_params_train.json",
        ]:
            if os.path.isfile(candidate):
                nrtl_path = candidate
                break
    
    if nrtl_path and os.path.isfile(nrtl_path):
        print(f"[OK] Loading NRTL params from: {nrtl_path}")
        nrtl_store = NRTLParamStore(nrtl_path, device=device)
    else:
        print("[WARN] NRTL params not found, physics metrics will be limited")
    
    use_mech_loss = bool(getattr(C, "USE_MECH_LOSS", True))
    if use_mech_loss:
        if nrtl_store is None:
            raise FileNotFoundError(
                "NRTL params file not found. Please run fit_nrtl_params.py first:\n"
                "  python src/fit_nrtl_params.py --out_dir nrtl_param\n"
                "Then update NRTL_PARAMS_PATH in config.py"
            )
        
        loss_fn = MechanisticNRTLLoss(
            T_mean=T_scaler.mean,
            T_std=T_scaler.std,
            nrtl_params_path=nrtl_path,
            lambda_phy=C.LAMBDA_PHY,
            warmup_epochs=C.WARMUP_EPOCHS,
            ramp_epochs=C.RAMP_EPOCHS,
            robust_delta=C.ROBUST_DELTA,
            device=device,
            tau_clip=C.TAU_CLIP,
            ln_gamma_clip=C.LN_GAMMA_CLIP,
            use_kelvin=getattr(C, "MECH_USE_KELVIN", None),
            w_eq=getattr(C, "MECH_W_EQ", 1.0),
            w_gd=getattr(C, "MECH_W_GD", 0.10),
            w_stab=getattr(C, "MECH_W_STAB", 0.10),
            gd_n_dir=getattr(C, "MECH_GD_N_DIR", 2),
            gd_eps=getattr(C, "MECH_GD_EPS", 1e-4),
            stab_n_trial=getattr(C, "MECH_STAB_N_TRIAL", 4),
            stab_sigma=getattr(C, "MECH_STAB_SIGMA", 0.05),
            stab_margin=getattr(C, "MECH_STAB_MARGIN", 0.0),
        )
    else:
        loss_fn = nn.MSELoss(reduction="mean")
    
    # Freeze backbone if configured (for second stage fine-tuning)
    freeze_backbone = bool(getattr(C, "FREEZE_BACKBONE", False))
    if freeze_backbone and loaded_from_ckpt:
        print("[OK] Freezing backbone (only training output heads)")
        frozen_count = 0
        trainable_count = 0
        
        for name, param in model.named_parameters():
            # Keep output heads trainable
            if "head_E" in name or "head_R" in name:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        
        print(f"  - Frozen parameters: {frozen_count}")
        print(f"  - Trainable parameters: {trainable_count}")
        
        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params,
            lr=getattr(C, "LR"),
            weight_decay=getattr(C, "WEIGHT_DECAY")
        )
    else:
        # Full model training
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=getattr(C, "LR"),
            weight_decay=getattr(C, "WEIGHT_DECAY")
        )
    
    grad_clip = float(getattr(C, "GRAD_CLIP", 1.0) or 0.0)

    epochs = getattr(C, "EPOCHS", 300)
    for epoch in range(1, epochs + 1):
        model.train()
        if use_mech_loss:
            loss_fn.set_epoch(epoch)
        running_sup = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x = batch_to_device(x, device)
            y = batch_to_device(y, device)

            opt.zero_grad(set_to_none=True)

            try:
                _ac = torch.amp.autocast('cuda', enabled=use_amp)
            except Exception:
                _ac = torch.cuda.amp.autocast(enabled=use_amp)
            with _ac:
                pred = model(x)
                if use_mech_loss:
                    d = loss_fn(pred, y, x)
                    loss = d["loss"]
                else:
                    loss = loss_fn(pred, y)
                    d = {
                        "loss": loss,
                        "sup": loss.detach(),
                        "phy": torch.zeros_like(loss.detach()),
                        "lambda": torch.tensor(0.0, device=loss.device),
                    }

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            bs = y.shape[0]
            running_sup += float(d["sup"].detach().cpu().item()) * bs
            n += bs
            pbar.set_postfix({
                "mse": running_sup / max(1, n),
                "phy": float(d["phy"].detach().cpu()),
                "lam": float(d["lambda"].detach().cpu()),
            })

        train_mse = running_sup / max(1, n)

        if epoch % eval_every == 0:
            val_m = evaluate_loader(model, val_loader, device)
            test_m = evaluate_loader(model, test_loader, device)
            if use_physics_finetune:
                val_phy = compute_physics_metrics(
                    model, val_loader, device,
                    nrtl_store=nrtl_store,
                    T_mean=T_scaler.mean,
                    T_std=T_scaler.std,
                    use_kelvin=getattr(C, "MECH_USE_KELVIN", None),
                )
                test_phy = compute_physics_metrics(
                    model, test_loader, device,
                    nrtl_store=nrtl_store,
                    T_mean=T_scaler.mean,
                    T_std=T_scaler.std,
                    use_kelvin=getattr(C, "MECH_USE_KELVIN", None),
                )
                val_m.update(val_phy)
                test_m.update(test_phy)

            history["epoch"].append(epoch)
            history["train_mse"].append(train_mse)

            history["val_mse"].append(val_m["mse"])
            history["test_mse"].append(test_m["mse"])

            history["val_mae"].append(val_m["mae"])
            history["test_mae"].append(test_m["mae"])

            history["val_rmse"].append(val_m["rmse"])
            history["test_rmse"].append(test_m["rmse"])

            history["val_r2"].append(val_m["r2"])
            history["test_r2"].append(test_m["r2"])

            for k in ["mae_E", "mae_R", "rmse_E", "rmse_R", "r2_E", "r2_R"]:
                history["val_" + k].append(val_m.get(k, float("nan")))
                history["test_" + k].append(test_m.get(k, float("nan")))

            # 收集物理指标到历史记录
            if use_physics_finetune:
                for phy_key in ["mu_res_mae", "mu_res_max", "gd_res_mae", "tpd_viol_rate"]:
                    if "val_" + phy_key not in history:
                        history["val_" + phy_key] = []
                    if "test_" + phy_key not in history:
                        history["test_" + phy_key] = []
                    history["val_" + phy_key].append(val_m.get(phy_key, float("nan")))
                    history["test_" + phy_key].append(test_m.get(phy_key, float("nan")))

            print_metrics(f"[Epoch {epoch:03d}] Val :", val_m)
            print_metrics(f"[Epoch {epoch:03d}] Test:", test_m)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{_fmt8(train_mse)},"
                    f"{_fmt8(val_m.get('mae'))},{_fmt8(val_m.get('rmse'))},{_fmt8(val_m.get('r2'))},"
                    f"{_fmt8(val_m.get('mae_E'))},{_fmt8(val_m.get('rmse_E'))},{_fmt8(val_m.get('r2_E'))},"
                    f"{_fmt8(val_m.get('mae_R'))},{_fmt8(val_m.get('rmse_R'))},{_fmt8(val_m.get('r2_R'))},"
                    f"{_fmt8(test_m.get('mae'))},{_fmt8(test_m.get('rmse'))},{_fmt8(test_m.get('r2'))},"
                    f"{_fmt8(test_m.get('mae_E'))},{_fmt8(test_m.get('rmse_E'))},{_fmt8(test_m.get('r2_E'))},"
                    f"{_fmt8(test_m.get('mae_R'))},{_fmt8(test_m.get('rmse_R'))},{_fmt8(test_m.get('r2_R'))}\n"
                )

            # ---- 关键：best checkpoint 与 early stopping 共用同一份“是否改善”判定 ----
            monitor_key = early_stop_metric
            current_monitor = float(val_m.get(monitor_key, float("nan")))

            improved = False
            if not np.isnan(current_monitor):
                # 用进入本轮前的 best 值做比较，避免同一 epoch 内先更新后比较导致误判
                prev_best_monitor = float(best_monitor_value)
                if early_stop_metric == "r2":  # 越大越好
                    improved = (current_monitor - prev_best_monitor) > early_stop_min_delta
                else:  # 越小越好
                    improved = (prev_best_monitor - current_monitor) > early_stop_min_delta

                if improved:
                    best_monitor_value = current_monitor
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    best_val_metrics = val_m
                    best_test_metrics = test_m
                    best_val_mse = float(val_m.get("mse", best_val_mse))

                    ckpt = {
                        "model": best_state,
                        "epoch": int(best_epoch),
                        "val_metrics": val_m,
                    }
                    torch.save(ckpt, os.path.join(out_dir, "best_model.pt"))

            # 早停判断 - 支持物理指标
            if use_early_stop:
                if np.isnan(current_monitor):
                    print(f"  [早停] {early_stop_metric} 尚未计算（值为nan）")
                elif improved:
                    early_stop_counter = 0  # 重置计数器
                else:
                    early_stop_counter += 1
                    print(f"  [早停] {early_stop_metric}未改善 ({early_stop_counter}/{early_stop_patience})")

                # 触发早停
                if early_stop_counter >= early_stop_patience:
                    print(f"\n[STOP] Early stopping triggered! {early_stop_metric} not improved for {early_stop_patience} epochs")
                    print(f"  Best epoch: {best_epoch}, best {early_stop_metric}: {best_monitor_value:.6f}")
                    break  # 跳出训练循环

        if epoch % plot_every == 0 and len(history["epoch"]) > 0:
            plot_history(history, out_dir)

    # Always save final (last-epoch) checkpoint before restoring best weights
    final_ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "T_mean": float(T_scaler.mean),
        "T_std": float(T_scaler.std),
        "epoch": int(epochs),
        "config_use_graph": bool(getattr(C, "USE_GRAPH", False)),
    }
    torch.save(final_ckpt, os.path.join(out_dir, "last_model.pt"))

    if best_state is not None:
        model.load_state_dict(best_state)
        
        # Compute physics metrics ONLY ONCE on the final best model
        print("\n" + "=" * 60)
        print("Computing physics consistency metrics on best model...")
        print("=" * 60)
        test_physics = compute_physics_metrics(
            model, test_loader, device,
            nrtl_store=nrtl_store,
            T_mean=T_scaler.mean,
            T_std=T_scaler.std,
            use_kelvin=getattr(C, "MECH_USE_KELVIN", None),
        )
        if best_test_metrics is not None:
            best_test_metrics.update(test_physics)
        print("Physics metrics computation completed.\n")

    if best_val_metrics is not None:
        summary = {
            "best_epoch": int(best_epoch),
            "best_val": best_val_metrics,
            "best_test": best_test_metrics,
            "use_graph": bool(getattr(C, "USE_GRAPH", False)),
        }
        with open(os.path.join(out_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_dir, "best_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"Best epoch: {best_epoch}\n\n")
            
            # Validation metrics
            f.write("=" * 60 + "\n")
            f.write("Validation Metrics:\n")
            f.write("=" * 60 + "\n")
            for k, v in best_val_metrics.items():
                f.write(f"  {k}: {v:.6f}\n")
            
            # Test metrics
            f.write("\n" + "=" * 60 + "\n")
            f.write("Test Metrics:\n")
            f.write("=" * 60 + "\n")
            
            # Separate standard metrics and physics metrics
            standard_keys = ['mse', 'mae', 'rmse', 'r2', 'mae_E', 'mae_R', 'rmse_E', 'rmse_R', 'r2_E', 'r2_R']
            physics_keys = ['sum_err_E', 'sum_err_R', 'sum_err_95', 'neg_frac', 'param_cov',
                          'mu_res_mae', 'mu_res_rmse', 'gd_penalty_mean', 'gd_penalty_p95',
                          'tpd_viol_rate', 'tpd_viol_mean']
            
            f.write("\nStandard Metrics:\n")
            for k in standard_keys:
                if k in (best_test_metrics or {}):
                    f.write(f"  {k}: {best_test_metrics[k]:.6f}\n")
            
            f.write("\nPhysics Consistency Metrics:\n")
            has_physics = False
            for k in physics_keys:
                if k in (best_test_metrics or {}):
                    has_physics = True
                    val = best_test_metrics[k]
                    if not np.isnan(val):
                        f.write(f"  {k}: {val:.6f}\n")
                    else:
                        f.write(f"  {k}: N/A\n")
            
            if not has_physics:
                f.write("  (No NRTL parameters available for physics metrics)\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("\nMetric Descriptions:\n")
            f.write("  sum_err_E/R: Sum-to-one constraint error (Extract/Raffinate)\n")
            f.write("  sum_err_95: 95th percentile of sum-to-one error\n")
            f.write("  neg_frac: Fraction of negative predictions\n")
            f.write("  param_cov: Coverage rate of NRTL parameters\n")
            f.write("  mu_res_*: Chemical potential equilibrium residual\n")
            f.write("  gd_penalty_*: Gibbs-Duhem consistency violation\n")
            f.write("  tpd_viol_*: Tangent-plane distance stability violation\n")

    return model, T_scaler, history

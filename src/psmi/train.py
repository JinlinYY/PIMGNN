# -*- coding: utf-8 -*-
"""Build, train, validate, and checkpoint PSMI models."""
import os
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .utils import Scaler, batch_to_device, temperature_scalar_value
from .loss import MechanisticNRTLLoss, NRTLParamStore
from .data import (
    MixGraphCache,
    FingerprintCache, LLEDataset,
    FunctionalGroupCache,
    GraphCache, GraphLLEDataset, collate_graph_batch
)
from .metrics import evaluate_loader, print_metrics, compute_physics_metrics
from .model import LLECurveNet, LLEGraphNet
from .checkpoints import build_checkpoint_provenance, load_state_dict_compat


def should_restore_pressure_scaler(adaptations) -> bool:
    """Return false when pressure is a newly initialized transfer feature."""
    return not any("appended zero input column" in str(item) for item in adaptations)
from .nrtl_isolation import (
    validate_evaluation_parameter_file,
    validate_training_parameter_file,
    write_usage_manifest,
)


def plot_history(history: Dict[str, List[float]], out_dir: str) -> None:
    """Plot history."""
    os.makedirs(out_dir, exist_ok=True)

    
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

    
    if "val_mu_res_mae" in history and len(history["val_mu_res_mae"]) > 0:
        
        val_data = [v for v in history["val_mu_res_mae"] if not np.isnan(v)]
        if len(val_data) > 0:
            
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
    """Build model."""
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
            mixture_node_layout=getattr(C, "MIXTURE_NODE_LAYOUT", "sample_major"),
            scalar_dim=int(getattr(C, "SCALAR_DIM", 3)),
            # FG options (optional)
            use_fg=getattr(C, "USE_FG", False),
            fg_vocab_size=int(getattr(C, "FG_TOPK", 0)),
            fg_hidden=int(getattr(C, "FG_MLP_HIDDEN", 256)),
            fg_dropout=float(getattr(C, "FG_DROPOUT", 0.10)),
            fg_token_mode=bool(getattr(C, "FG_TOKEN_MODE", False)),
            fg_max_tokens=int(getattr(C, "FG_MAX_TOKENS", 32)),
            fg_cross_attn=bool(getattr(C, "FG_CROSS_ATTN", False)),
            fg_attn_heads=int(getattr(C, "FG_ATTN_HEADS", 8)),
            s3_equivariant=bool(
                getattr(C, "USE_S3_COMPONENT_EMBEDDING", None)
                if getattr(C, "USE_S3_COMPONENT_EMBEDDING", None) is not None
                else getattr(C, "S3_EQUIVARIANT", False)
            ),
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
        kwargs["prefetch_factor"] = getattr(C, "PREFETCH_FACTOR", 8)  

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

    temperature_encoding = str(getattr(C, "TEMPERATURE_ENCODING", "linear_quadratic"))
    temperature_reference_k = float(getattr(C, "TEMPERATURE_REFERENCE_K", 500.0))
    train_temperature_feature = temperature_scalar_value(
        train_df["T"].to_numpy(dtype="float32"),
        mode=temperature_encoding,
        reference_k=temperature_reference_k,
    )
    T_scaler = Scaler.fit(train_temperature_feature)
    # P_scaler (handle potentially missing or constant P)
    if "P" in train_df.columns:
        P_scaler = Scaler.fit(train_df["P"].to_numpy(dtype="float32"))
    else:
        # Fallback if somehow P is missing (should not happen with updated data.py)
        P_scaler = Scaler(mean=101.325, std=1.0)

    model = build_model().to(device)
    training_contract = {
        "seed": int(getattr(C, "SEED", 42)),
        "optimizer": "AdamW",
        "learning_rate": float(getattr(C, "LR", 0.0)),
        "weight_decay": float(getattr(C, "WEIGHT_DECAY", 0.0)),
        "maximum_epochs": int(getattr(C, "EPOCHS", 0)),
        "batch_size": int(
            getattr(C, "BATCH_SIZE_GRAPH", 0)
            if getattr(C, "USE_GRAPH", False)
            else getattr(C, "BATCH_SIZE", 0)
        ),
        "component_23_augmentation": bool(getattr(C, "PERMUTE_23_AUG", False)),
        "augmentation_scope": "training_only",
        "use_mechanistic_loss": bool(getattr(C, "USE_MECH_LOSS", False)),
        "freeze_backbone": bool(getattr(C, "FREEZE_BACKBONE", False)),
        "checkpoint_selection_partition": "validation",
        "test_evaluated_during_training": False,
    }
    checkpoint_provenance = build_checkpoint_provenance(
        model,
        dataset_path=getattr(C, "EXCEL_PATH", None),
        split_manifest_path=(
            getattr(C, "SPLIT_MANIFEST_PATH", None)
            if str(getattr(C, "SPLIT_STRATEGY", "")).lower() == "manifest"
            else None
        ),
        source_checkpoint_path=getattr(C, "LOAD_CKPT_PATH", None),
        training_contract=training_contract,
    )

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
        # keep runtime config consistent with checkpoint (important when USE_FG / FG_TOPK differs)
        if "use_fg" in ckpt:
            setattr(C, "USE_FG", bool(ckpt["use_fg"]))
        if "fg_topk" in ckpt:
            setattr(C, "FG_TOPK", int(ckpt["fg_topk"]))
        # Rebuild model with potentially updated config
        model = build_model().to(device)
        checkpoint_provenance = build_checkpoint_provenance(
            model,
            dataset_path=getattr(C, "EXCEL_PATH", None),
            split_manifest_path=(
                getattr(C, "SPLIT_MANIFEST_PATH", None)
                if str(getattr(C, "SPLIT_STRATEGY", "")).lower() == "manifest"
                else None
            ),
            source_checkpoint_path=getattr(C, "LOAD_CKPT_PATH", None),
            training_contract=training_contract,
        )
        
        
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            
            state_dict = ckpt
            
        adaptations = load_state_dict_compat(model, state_dict)
        for adaptation in adaptations:
            print(f"[INFO] Checkpoint compatibility: {adaptation}")

        setattr(model, "fg_corpus", ckpt.get("fg_corpus", None))
        
        
        if "T_mean" in ckpt and "T_std" in ckpt:
            T_scaler = Scaler(mean=float(ckpt["T_mean"]), std=float(ckpt["T_std"]))
        else:
            print("[WARN] T_scaler params not found in checkpoint, keeping computed ones")
            
        
        if (
            "P_mean" in ckpt
            and "P_std" in ckpt
            and should_restore_pressure_scaler(adaptations)
        ):
             P_scaler = Scaler(mean=float(ckpt["P_mean"]), std=float(ckpt["P_std"]))
        elif not should_restore_pressure_scaler(adaptations):
             print(
                 "[INFO] Keeping the target training-set pressure scaler because "
                 "pressure is a newly initialized transfer feature"
             )
        else:
             print("[WARN] P_scaler params not found in checkpoint, keeping computed ones (fine for fine-tuning)")

        loaded_from_ckpt = True
        print(f"[OK] Loaded checkpoint: {ckpt_path}")
        print(f"  - T_scaler: mean={T_scaler.mean:.2f}, std={T_scaler.std:.2f}")
        print(f"  - P_scaler: mean={P_scaler.mean:.2f}, std={P_scaler.std:.2f}")
        print(f"  - Best epoch from ckpt: {ckpt.get('epoch', ckpt.get('best_epoch', 'N/A'))}")
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
        graph_dataset_kwargs = {
            "P_scaler": P_scaler,
            "mix_cache": mix_cache,
            "fg_cache": fg_cache,
            "use_fg": getattr(C, "USE_FG", False),
            "scalar_dim": int(getattr(C, "SCALAR_DIM", 3)),
            "precompute_scalars": getattr(C, "PRECOMPUTE_SCALARS", True),
        }
        train_ds = GraphLLEDataset(train_df, T_scaler, g_cache, **graph_dataset_kwargs)
        val_ds = GraphLLEDataset(val_df, T_scaler, g_cache, **graph_dataset_kwargs)
        test_ds = GraphLLEDataset(test_df, T_scaler, g_cache, **graph_dataset_kwargs)

        batch_size = getattr(C, "BATCH_SIZE_GRAPH", 64)
        train_loader = _make_loader(train_ds, batch_size, shuffle=True, device=device, collate_fn=collate_graph_batch)
        val_loader = _make_loader(val_ds, batch_size, shuffle=False, device=device, collate_fn=collate_graph_batch)
        test_loader = _make_loader(test_ds, batch_size, shuffle=False, device=device, collate_fn=collate_graph_batch)
    else:
        fp_cache = FingerprintCache()
        precompute = getattr(C, "PRECOMPUTE_FEATURES", True)
        train_ds = LLEDataset(train_df, T_scaler, fp_cache, P_scaler=P_scaler, fg_cache=fg_cache, use_fg=getattr(C, "USE_FG", False), precompute=precompute)
        val_ds = LLEDataset(val_df, T_scaler, fp_cache, P_scaler=P_scaler, fg_cache=fg_cache, use_fg=getattr(C, "USE_FG", False), precompute=precompute)
        test_ds = LLEDataset(test_df, T_scaler, fp_cache, P_scaler=P_scaler, fg_cache=fg_cache, use_fg=getattr(C, "USE_FG", False), precompute=precompute)

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
    
    
    use_early_stop = bool(getattr(C, "USE_EARLY_STOP", False))
    use_physics_finetune = bool(getattr(C, "USE_PHYSICS_FINETUNE", False))
    
    if use_physics_finetune:
        
        early_stop_patience = int(getattr(C, "FINETUNE_PATIENCE", 30))
        early_stop_metric = str(getattr(C, "FINETUNE_EARLY_STOP_METRIC", "mu_res_mae")).lower()
        print(f"[OK] Physics finetune mode enabled")
    else:
        
        early_stop_patience = int(getattr(C, "EARLY_STOP_PATIENCE", 30))
        early_stop_metric = str(getattr(C, "EARLY_STOP_METRIC", "mse")).lower()

    supervised_monitor_metrics = {"mse", "mae", "rmse", "r2"}
    if early_stop_metric not in supervised_monitor_metrics:
        raise ValueError(
            "Checkpoint selection and early stopping may use only validation-set "
            f"predictive metrics {sorted(supervised_monitor_metrics)}; got "
            f"{early_stop_metric!r}. Thermodynamic metrics are post-hoc only."
        )
    
    early_stop_min_delta = float(getattr(C, "EARLY_STOP_MIN_DELTA", 0.0))
    early_stop_counter = 0  
    best_monitor_value_ckpt = float("inf") if early_stop_metric not in ["r2"] else float("-inf")
    best_monitor_value_es = best_monitor_value_ckpt
    best_epoch_es = -1
    
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

    training_nrtl_audit = None
    evaluation_nrtl_audit = None
    use_mech_loss = bool(getattr(C, "USE_MECH_LOSS", True))
    if use_mech_loss:
        training_nrtl_path = str(getattr(C, "NRTL_TRAIN_PARAMS_PATH", ""))
        if not training_nrtl_path or not os.path.isfile(training_nrtl_path):
            raise FileNotFoundError(
                "Training-only NRTL params file not found. Run:\n"
                "  python scripts/fit_nrtl.py --out_dir datasets/parameters\n"
                "Then update NRTL_TRAIN_PARAMS_PATH in config.py"
            )
        training_nrtl_audit = validate_training_parameter_file(
            training_nrtl_path,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            dataset_path=getattr(C, "EXCEL_PATH", None),
        )
        print(f"[OK] Loading training-only NRTL params: {training_nrtl_path}")
        print(
            "  - held-out overlap: validation=0, test=0 | "
            f"parameter systems={training_nrtl_audit['parameter_system_count']}"
        )
        loss_fn = MechanisticNRTLLoss(
            T_mean=T_scaler.mean,
            T_std=T_scaler.std,
            nrtl_params_path=training_nrtl_path,
            ge_model=getattr(C, "GE_MODEL", "nrtl"),
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
    last_epoch_completed = 0
    for epoch in range(1, epochs + 1):
        last_epoch_completed = epoch
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
                    if isinstance(x, dict) and "sample_weight" in x:
                        row_mse = (pred - y).square().mean(dim=-1)
                        sample_weight = x["sample_weight"].reshape(-1).to(row_mse)
                        loss = (row_mse * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
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
            # Test predictions and all-system NRTL parameters remain untouched
            # until validation-only checkpoint selection has finished.
            test_m = {}

            history["epoch"].append(epoch)
            history["train_mse"].append(train_mse)

            history["val_mse"].append(val_m["mse"])
            history["test_mse"].append(float("nan"))

            history["val_mae"].append(val_m["mae"])
            history["test_mae"].append(float("nan"))

            history["val_rmse"].append(val_m["rmse"])
            history["test_rmse"].append(float("nan"))

            history["val_r2"].append(val_m["r2"])
            history["test_r2"].append(float("nan"))

            for k in ["mae_E", "mae_R", "rmse_E", "rmse_R", "r2_E", "r2_R"]:
                history["val_" + k].append(val_m.get(k, float("nan")))
                history["test_" + k].append(test_m.get(k, float("nan")))

            
            print_metrics(f"[Epoch {epoch:03d}] Val :", val_m)

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

            
            monitor_key = early_stop_metric  
            monitor_val = val_m.get(monitor_key, float("nan"))

            
            if not np.isnan(monitor_val):
                improved = False
                if early_stop_metric == "r2":
                    improved = monitor_val > best_monitor_value_ckpt
                else:
                    improved = monitor_val < best_monitor_value_ckpt

                if improved:
                    best_monitor_value_ckpt = float(monitor_val)
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    best_val_metrics = val_m
                    best_val_mse = float(val_m.get("mse", best_val_mse))

                    ckpt = {
                        "model": best_state,
                        "epoch": int(best_epoch),
                        "val_metrics": val_m,
                        "T_mean": float(T_scaler.mean),
                        "T_std": float(T_scaler.std),
                        "P_mean": float(P_scaler.mean),
                        "P_std": float(P_scaler.std),
                        "nrtl_training_audit": training_nrtl_audit,
                        "test_evaluated_during_training": False,
                        "posthoc_nrtl_loaded_during_training": False,
                        "provenance": checkpoint_provenance,
                    }
                    torch.save(ckpt, os.path.join(out_dir, "best_model.pt"))
            
            
            if use_early_stop:
                current_monitor = val_m.get(early_stop_metric, float("nan"))
                
                if not np.isnan(current_monitor):
                    
                    improved = False
                    if early_stop_metric == "r2":  
                        if current_monitor - best_monitor_value_es > early_stop_min_delta:
                            improved = True
                            best_monitor_value_es = current_monitor
                    else:  
                        if best_monitor_value_es - current_monitor > early_stop_min_delta:
                            improved = True
                            best_monitor_value_es = current_monitor
                    
                    if improved:
                        early_stop_counter = 0
                        best_epoch_es = epoch
                    else:
                        early_stop_counter += 1
                        print(f"  [EarlyStop] {early_stop_metric} not improved ({early_stop_counter}/{early_stop_patience})")
                else:
                    print(f"  [EarlyStop] {early_stop_metric} not computed (nan)")
                
                
                if early_stop_counter >= early_stop_patience:
                    stop_epoch = best_epoch_es if best_epoch_es >= 0 else best_epoch
                    stop_val = best_monitor_value_es
                    print(f"\n[STOP] Early stopping triggered! {early_stop_metric} not improved for {early_stop_patience} epochs")
                    print(f"  Best epoch: {stop_epoch}, best {early_stop_metric}: {stop_val:.6f}")
                    break  

        if epoch % plot_every == 0 and len(history["epoch"]) > 0:
            plot_history(history, out_dir)

    # Always save final (last-epoch) checkpoint before restoring best weights
    final_ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "T_mean": float(T_scaler.mean),
        "T_std": float(T_scaler.std),
        "P_mean": float(P_scaler.mean), # Save P scaler
        "P_std": float(P_scaler.std),
        "epoch": int(last_epoch_completed),
        "config_use_graph": bool(getattr(C, "USE_GRAPH", False)),
        "nrtl_training_audit": training_nrtl_audit,
        "test_evaluated_during_training": False,
        "posthoc_nrtl_loaded_during_training": False,
        "provenance": checkpoint_provenance,
    }
    torch.save(final_ckpt, os.path.join(out_dir, "last_model.pt"))

    if best_state is not None:
        model.load_state_dict(best_state)

        # Predictive validation/test metrics are computed once using the model
        # selected exclusively from validation performance.
        best_val_metrics = evaluate_loader(model, val_loader, device)
        best_test_metrics = evaluate_loader(model, test_loader, device)

        # Physics diagnostics are useful for the main study, but controlled
        # supervised ablations may disable them to avoid repeating an
        # unrelated and comparatively expensive NRTL evaluation for every run.
        if bool(getattr(C, "COMPUTE_FINAL_PHYSICS_METRICS", True)):
            evaluation_nrtl_path = str(getattr(C, "NRTL_EVAL_PARAMS_PATH", ""))
            if not evaluation_nrtl_path or not os.path.isfile(evaluation_nrtl_path):
                raise FileNotFoundError(
                    "Post-hoc NRTL evaluation file not found. Run:\n"
                    "  python scripts/fit_nrtl.py --scope both --out_dir datasets/parameters"
                )
            evaluation_nrtl_audit = validate_evaluation_parameter_file(
                evaluation_nrtl_path,
                val_df=val_df,
                test_df=test_df,
                dataset_path=getattr(C, "EXCEL_PATH", None),
            )
            nrtl_store = NRTLParamStore(evaluation_nrtl_path, device=device)
            print("\n" + "=" * 60)
            print("Computing post-hoc physics metrics on the selected best model...")
            print("=" * 60)
            val_physics = compute_physics_metrics(
                model, val_loader, device,
                nrtl_store=nrtl_store,
                T_mean=T_scaler.mean,
                T_std=T_scaler.std,
                use_kelvin=getattr(C, "MECH_USE_KELVIN", None),
            )
            test_physics = compute_physics_metrics(
                model, test_loader, device,
                nrtl_store=nrtl_store,
                T_mean=T_scaler.mean,
                T_std=T_scaler.std,
                use_kelvin=getattr(C, "MECH_USE_KELVIN", None),
            )
            best_val_metrics.update(val_physics)
            best_test_metrics.update(test_physics)
            print("Physics metrics computation completed.\n")

    write_usage_manifest(
        os.path.join(out_dir, "nrtl_usage_manifest.json"),
        {
            "training_loss_parameter_store": training_nrtl_audit,
            "posthoc_evaluation_parameter_store": evaluation_nrtl_audit,
            "test_evaluated_during_training": False,
            "posthoc_nrtl_loaded_after_checkpoint_selection": bool(
                evaluation_nrtl_audit is not None
            ),
            "checkpoint_selection_partition": "validation",
            "checkpoint_selection_metric": early_stop_metric,
            "best_epoch": int(best_epoch),
        },
    )

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

    
    return model, T_scaler, P_scaler, history




# -*- coding: utf-8 -*-
"""Run the configured PSMI training, evaluation, prediction, and plotting workflow."""
import os
import json

os.environ.setdefault("MPLBACKEND", "Agg")


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from torch.utils.data import DataLoader

from . import config as C
from .utils import set_seed
from .data import (
    augment_component_23, load_and_prepare_excel, split_by_system,
    stratified_split_by_system, split_by_manifest,
    FingerprintCache, LLEDataset,
    FunctionalGroupCache,
    GraphCache, GraphLLEDataset, collate_graph_batch
)
from .train import train_or_load
from .metrics import evaluate_loader, print_metrics
from .predict import predict_pointwise_df_raw
from .viz import parity_plots, visualize_all_test_groups


def main():
    """Run the command-line workflow."""
    
    set_seed(C.SEED)
    
    
    if getattr(C, "USE_FINE_TUNE", False):
        print("\n" + "="*40)
        print("[INFO] FINE-TUNE MODE ENABLED")
        print("="*40)
        
        
        C.EXCEL_PATH = C.FINE_TUNE_EXCEL_PATH
        C.LOAD_CKPT_PATH = C.PRETRAINED_MODEL_PATH
        C.LR = C.FINE_TUNE_LR
        if hasattr(C, "FINE_TUNE_EPOCHS"):
            C.EPOCHS = C.FINE_TUNE_EPOCHS
            
        
        if not C.OUT_DIR.endswith("_finetuned"):
            C.OUT_DIR = C.OUT_DIR + "_finetuned"

        print(f"  - Dataset: {C.EXCEL_PATH}")
        print(f"  - Pretrained Model: {C.LOAD_CKPT_PATH}")
        print(f"  - Learning Rate: {C.LR}")
        print(f"  - Epochs: {C.EPOCHS}")
        print(f"  - Output Dir: {C.OUT_DIR}")
        print("-" * 40 + "\n")

    
    os.makedirs(C.OUT_DIR, exist_ok=True)

    
    print("1) Load & prepare Excel ...")
    
    # Split experimental records first; augmentation is applied to training only.
    df_raw, _ = load_and_prepare_excel(C.EXCEL_PATH, C.MIN_POINTS_PER_GROUP, False)
    print("df_raw:", len(df_raw), "rows | systems:", df_raw["system_id"].nunique())

    
    print("2) Split experimental records by system_id ...")
    
    split_strategy = str(getattr(C, "SPLIT_STRATEGY", "random")).lower()
    split_kwargs = {
        "train_ratio": float(getattr(C, "TRAIN_RATIO", 0.8)),
        "val_ratio": float(getattr(C, "VAL_RATIO", 0.1)),
        "seed": int(C.SEED),
    }
    if split_strategy == "random":
        train_df, val_df, test_df = split_by_system(df_raw, **split_kwargs)
    elif split_strategy == "stratified":
        train_df, val_df, test_df = stratified_split_by_system(
            df_raw,
            **split_kwargs,
            n_bins=int(getattr(C, "STRATIFIED_N_BINS", 3)),
            min_bin_size=int(getattr(C, "STRATIFIED_MIN_BIN_SIZE", 5)),
        )
    elif split_strategy == "manifest":
        split_manifest_path = str(getattr(C, "SPLIT_MANIFEST_PATH", "")).strip()
        if not split_manifest_path:
            raise ValueError("SPLIT_MANIFEST_PATH is required for manifest splitting")
        train_df, val_df, test_df = split_by_manifest(df_raw, split_manifest_path)
    else:
        raise ValueError(f"Unsupported SPLIT_STRATEGY: {split_strategy!r}")

    train_df = augment_component_23(
        train_df,
        enabled=bool(getattr(C, "PERMUTE_23_AUG", False)),
    )
    val_df = augment_component_23(val_df, enabled=False)
    test_df = augment_component_23(test_df, enabled=False)
    print(
        "training rows after component-2/3 augmentation:",
        len(train_df),
    )

    train_system_ids = set(train_df["system_id"].unique().tolist())
    val_system_ids = set(val_df["system_id"].unique().tolist())
    test_system_ids = set(test_df["system_id"].unique().tolist())
    print(
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)} | "
        f"train systems={len(train_system_ids)} val systems={len(val_system_ids)} test systems={len(test_system_ids)}"
    )

    
    print("3) Train / load model ...")
    model, T_scaler, P_scaler, _history = train_or_load(train_df, val_df, test_df)

    
    fg_cache = None
    if getattr(C, 'USE_FG', False):
        
        corpus = getattr(model, 'fg_corpus', None)
        if (corpus is None) or (isinstance(corpus, list) and len(corpus) == 0):
            
            fg_path = os.path.join(C.OUT_DIR, 'fg_corpus.json')
            if os.path.exists(fg_path) and os.path.getsize(fg_path) > 0:
                try:
                    with open(fg_path, 'r', encoding='utf-8') as f:
                        corpus = json.load(f)
                except Exception:
                    corpus = None
        if corpus is not None:
            fg_cache = FunctionalGroupCache(corpus=corpus)

    
    if getattr(C, "USE_GRAPH", False):
        
        print("  Using GRAPH mode")
        g_cache = GraphCache(
            add_hs=getattr(C, "GRAPH_ADD_HS", False),
            add_3d=getattr(C, "GRAPH_ADD_3D", False),
            use_gasteiger=getattr(C, "GRAPH_USE_GASTEIGER", True),
            max_atoms=getattr(C, "GRAPH_MAX_ATOMS", 256),
        )
        
        smiles_all = pd.concat([val_df[["smiles1","smiles2","smiles3"]],
                                test_df[["smiles1","smiles2","smiles3"]]], axis=0).values.reshape(-1).tolist()
        g_cache.build_from_smiles(smiles_all)

        
        val_loader = DataLoader(
            GraphLLEDataset(val_df, T_scaler, g_cache, P_scaler=P_scaler, mix_cache=None, fg_cache=fg_cache, use_fg=getattr(C,'USE_FG',False), scalar_dim=int(getattr(model, 'scalar_dim', getattr(C, 'SCALAR_DIM', 3))), precompute_scalars=True),
            batch_size=getattr(C, "BATCH_SIZE_GRAPH", 64),
            shuffle=False,
            num_workers=0,
            pin_memory=C.DEVICE.startswith("cuda"),
            collate_fn=collate_graph_batch,
        )
        
        test_loader = DataLoader(
            GraphLLEDataset(test_df, T_scaler, g_cache, P_scaler=P_scaler, mix_cache=None, fg_cache=fg_cache, use_fg=getattr(C,'USE_FG',False), scalar_dim=int(getattr(model, 'scalar_dim', getattr(C, 'SCALAR_DIM', 3))), precompute_scalars=True),
            batch_size=getattr(C, "BATCH_SIZE_GRAPH", 64),
            shuffle=False,
            num_workers=0,
            pin_memory=C.DEVICE.startswith("cuda"),
            collate_fn=collate_graph_batch,
        )
    else:
        
        print("  Using Fingerprint mode")
        fp_cache = FingerprintCache()
        val_loader = DataLoader(
            LLEDataset(val_df, T_scaler, fp_cache, P_scaler=P_scaler, fg_cache=fg_cache, use_fg=getattr(C,'USE_FG',False)),
            batch_size=C.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            LLEDataset(test_df, T_scaler, fp_cache, P_scaler=P_scaler, fg_cache=fg_cache, use_fg=getattr(C,'USE_FG',False)),
            batch_size=C.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

    
    val_m = evaluate_loader(model, val_loader, C.DEVICE)
    test_m = evaluate_loader(model, test_loader, C.DEVICE)
    print("\nFinal metrics (best-by-val model):")
    print_metrics("  Val :", val_m)
    print_metrics("  Test:", test_m)

    
    print("\n4) Test pointwise predictions on df_raw (no augmentation) ...")
    
    df_raw_test = df_raw[df_raw["system_id"].isin(test_system_ids)].copy()
    df_pred = predict_pointwise_df_raw(
        model,
        T_scaler,
        df_raw_test,
        P_scaler=P_scaler,
    )

    
    pred_csv = os.path.join(C.OUT_DIR, "test_df_raw_pointwise_predictions.csv")
    df_pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print("Saved test predictions CSV:", pred_csv)

    
    print("5) Parity plots ...")
    parity_plots(df_pred, C.OUT_DIR)
    print("Saved parity plots: parity_E.png / parity_R.png")

    
    if getattr(C, "GENERATE_PHASE_DIAGRAMS", True):
        print("6) Visualize ALL test groups ternary + PDF ...")
        visualize_all_test_groups(
            model,
            T_scaler,
            df_raw,
            test_system_ids,
            df_pred,
            C.OUT_DIR,
            P_scaler=P_scaler,
        )
    else:
        print("6) Phase-diagram rendering skipped by configuration.")

    print("\nDONE. Everything is in:", C.OUT_DIR)


if __name__ == "__main__":
    
    main()

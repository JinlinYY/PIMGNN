# -*- coding: utf-8 -*-
"""
Entry point.

Run:
  python main.py

Outputs:
  - curves (loss/metrics)
  - test_df_raw_pointwise_predictions.csv
  - parity plots
  - per-group ternary PNGs + one multi-page PDF
"""

import os
import pandas as pd
from torch.utils.data import DataLoader

from . import config as C
from .utils import set_seed
from .data import (
    FingerprintCache,
    GraphCache,
    LLEDataset,
    LLEGNNDataset,
    SmilesRNNDataset,
    build_smiles_vocab,
    augment_component_23,
    gnn_collate_fn,
    load_and_prepare_excel,
    split_by_manifest,
)
from .train import train_or_load
from .metrics import evaluate_loader, print_metrics
from .predict import predict_pointwise_df_raw
from .viz import parity_plots, visualize_all_test_groups

def main():
    set_seed(C.SEED)
    os.makedirs(C.OUT_DIR, exist_ok=True)

    print("1) Load & prepare Excel ...")
    df_raw, _ = load_and_prepare_excel(C.EXCEL_PATH, C.MIN_POINTS_PER_GROUP, False)
    print("df_raw:", len(df_raw), "rows | systems:", df_raw["system_id"].nunique())

    print("2) Split experimental records by system_id ...")
    train_df, val_df, test_df = split_by_manifest(df_raw, C.SPLIT_MANIFEST_PATH)
    train_df = augment_component_23(train_df, enabled=C.PERMUTE_23_AUG)
    val_df = augment_component_23(val_df, enabled=False)
    test_df = augment_component_23(test_df, enabled=False)
    test_system_ids = set(test_df["system_id"].unique().tolist())
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} | test systems={len(test_system_ids)}")

    print("3) Train / load model ...")
    model, T_scaler, _history = train_or_load(train_df, val_df, test_df)

    # Final val/test metrics for best model
    model_name = getattr(C, "MODEL_NAME", "mlp").lower()
    if model_name == "smiles_rnn":
        vocabulary = build_smiles_vocab([train_df])
        use_t = bool(getattr(C, "SMILES_USE_TIE_T", True))
        val_dataset = SmilesRNNDataset(val_df, vocabulary, T_scaler, use_t=use_t)
        test_dataset = SmilesRNNDataset(test_df, vocabulary, T_scaler, use_t=use_t)
        val_loader = DataLoader(val_dataset, batch_size=C.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=C.BATCH_SIZE, shuffle=False)
    elif model_name == "gnn":
        graph_cache = GraphCache()
        val_loader = DataLoader(
            LLEGNNDataset(val_df, T_scaler, graph_cache),
            batch_size=C.BATCH_SIZE,
            shuffle=False,
            collate_fn=gnn_collate_fn,
        )
        test_loader = DataLoader(
            LLEGNNDataset(test_df, T_scaler, graph_cache),
            batch_size=C.BATCH_SIZE,
            shuffle=False,
            collate_fn=gnn_collate_fn,
        )
    else:
        fp_cache = FingerprintCache()
        val_loader = DataLoader(
            LLEDataset(val_df, T_scaler, fp_cache),
            batch_size=C.BATCH_SIZE,
            shuffle=False,
        )
        test_loader = DataLoader(
            LLEDataset(test_df, T_scaler, fp_cache),
            batch_size=C.BATCH_SIZE,
            shuffle=False,
        )
    val_m = evaluate_loader(model, val_loader, C.DEVICE)
    test_m = evaluate_loader(model, test_loader, C.DEVICE)
    print("\nFinal metrics (best-by-val model):")
    print_metrics("  Val :", val_m)
    print_metrics("  Test:", test_m)

    print("\n4) Test pointwise predictions on df_raw (no augmentation) ...")
    df_raw_test = df_raw[df_raw["system_id"].isin(test_system_ids)].copy()
    df_pred = predict_pointwise_df_raw(model, T_scaler, df_raw_test)

    pred_csv = os.path.join(C.OUT_DIR, "test_df_raw_pointwise_predictions.csv")
    df_pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print("Saved test predictions CSV:", pred_csv)

    print("5) Parity plots ...")
    parity_plots(df_pred, C.OUT_DIR)
    print("Saved parity plots: parity_E.png / parity_R.png")

    print("6) Visualize ALL test groups ternary + PDF ...")
    visualize_all_test_groups(model, T_scaler, df_raw, test_system_ids, df_pred, C.OUT_DIR)

    print("\nDONE. Everything is in:", C.OUT_DIR)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Baseline comparison entry point.

Run:
  python main_compare.py

It will:
  1) Load + prepare Excel (same as main.py)
  2) Split by system_id
  3) Train/Eval each model in config.MODELS_TO_RUN
  4) Save:
     - baseline_compare_metrics.csv
     - per-model df_raw pointwise predictions CSV
     - per-model parity plots
     - (optional) per-model ternary plots (slow)

Notes:
- Torch models are trained via train.train_or_load()
- Sklearn models use baselines_sklearn.py
"""

import json
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from . import config as C
from .utils import set_seed, Scaler
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
from .metrics import evaluate_loader, print_metrics, compute_metrics
from .predict import predict_pointwise_df_raw_generic
from .viz import parity_plots, visualize_all_test_groups

from .baselines_sklearn import fit_sklearn_model, predict_sklearn, SKLEARN_MODEL_ALIASES


TORCH_MODELS = {
    "mlp", "lle_curve_net", "ann", "lstm", "transformer", "tabknet",
    "tabkan", "tabkanet", "smiles_rnn", "gnn",
}
SKLEARN_MODELS = set(SKLEARN_MODEL_ALIASES.values())


def _is_torch_model(name: str) -> bool:
    return name.lower() in TORCH_MODELS


def _is_sklearn_model(name: str) -> bool:
    return SKLEARN_MODEL_ALIASES.get(name.lower(), name.lower()) in SKLEARN_MODELS


def main():
    set_seed(C.SEED)
    os.makedirs(C.OUT_DIR, exist_ok=True)

    print("1) Load & prepare Excel ...")
    df_raw, _ = load_and_prepare_excel(C.EXCEL_PATH, permute_23_aug=False)

    print("2) Split experimental records by system_id ...")
    train_df, val_df, test_df = split_by_manifest(df_raw, C.SPLIT_MANIFEST_PATH)
    train_df = augment_component_23(train_df, enabled=C.PERMUTE_23_AUG)
    val_df = augment_component_23(val_df, enabled=False)
    test_df = augment_component_23(test_df, enabled=False)
    test_system_ids = set(test_df["system_id"].unique().tolist())
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} | test systems={len(test_system_ids)}")

    # df_raw for prediction/visualization uses the original orientation (no augmentation)
    df_raw_test = df_raw[df_raw["system_id"].isin(test_system_ids)].copy()

    models = getattr(C, "MODELS_TO_RUN", ["mlp"])
    compare_draw_ternary = bool(getattr(C, "COMPARE_DRAW_TERNARY", False))
    save_pdf = bool(getattr(C, "SAVE_TERNARY_PDF", False))

    results = []

    for model_name in models:
        name = model_name.lower()
        out_dir = os.path.join(C.OUT_DIR, f"baseline_{name}")
        os.makedirs(out_dir, exist_ok=True)

        print("\n" + "=" * 90)
        print(f"MODEL: {model_name} -> {out_dir}")
        print("=" * 90)

        if _is_torch_model(name):
            # Train / load torch model
            model, T_scaler, _hist = train_or_load(train_df, val_df, test_df, model_name=name, out_dir=out_dir)

            # Metrics use unaugmented validation and test records.
            if name == "smiles_rnn":
                smiles_vocab = build_smiles_vocab([train_df])
                max_len = getattr(C, "SMILES_MAX_LEN", 256)
                scalar_dim = 2 if getattr(C, "SMILES_USE_TIE_T", True) else 1
                val_loader = DataLoader(SmilesRNNDataset(val_df, smiles_vocab, T_scaler, max_len=max_len, use_t=scalar_dim==2),
                                        batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
                test_loader = DataLoader(SmilesRNNDataset(test_df, smiles_vocab, T_scaler, max_len=max_len, use_t=scalar_dim==2),
                                         batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
            elif name == "gnn":
                graph_cache = GraphCache()
                val_loader = DataLoader(
                    LLEGNNDataset(val_df, T_scaler, graph_cache),
                    batch_size=C.BATCH_SIZE,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=gnn_collate_fn,
                )
                test_loader = DataLoader(
                    LLEGNNDataset(test_df, T_scaler, graph_cache),
                    batch_size=C.BATCH_SIZE,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=gnn_collate_fn,
                )
            else:
                fp_cache = FingerprintCache()
                val_loader = DataLoader(LLEDataset(val_df, T_scaler, fp_cache),
                                        batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
                test_loader = DataLoader(LLEDataset(test_df, T_scaler, fp_cache),
                                         batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
            val_m = evaluate_loader(model, val_loader, C.DEVICE)
            test_m = evaluate_loader(model, test_loader, C.DEVICE)

            # Pointwise prediction on df_raw (no augmentation)
            df_pred = predict_pointwise_df_raw_generic(model, T_scaler, df_raw_test, backend="torch")

            # Optional ternary plots
            if compare_draw_ternary:
                visualize_all_test_groups(model, T_scaler, df_raw, test_system_ids, df_pred, out_dir,
                                          backend="torch", save_pdf=save_pdf)

        elif _is_sklearn_model(name):
            # Fit scaler on train (same as train.py)
            T_scaler = Scaler.fit(train_df["T"].values.astype(np.float32))
            fp_cache = FingerprintCache()
            precompute = True

            # Build cached X/Y using your dataset feature pipeline
            train_ds = LLEDataset(train_df, T_scaler, fp_cache, precompute=precompute)
            val_ds   = LLEDataset(val_df,   T_scaler, fp_cache, precompute=precompute)
            test_ds  = LLEDataset(test_df,  T_scaler, fp_cache, precompute=precompute)

            X_train = train_ds._X.cpu().numpy()
            y_train = train_ds._Y.cpu().numpy()
            X_val   = val_ds._X.cpu().numpy()
            y_val   = val_ds._Y.cpu().numpy()
            X_test  = test_ds._X.cpu().numpy()
            y_test  = test_ds._Y.cpu().numpy()

            # Train baseline
            try:
                est = fit_sklearn_model(name, X_train, y_train, X_val, y_val, out_dir=out_dir, seed=C.SEED)
            except Exception as e:
                print(f"[SKIP] {name}: {type(e).__name__}: {e}")
                continue

            # Evaluate
            y_val_pred = predict_sklearn(est, X_val)
            y_test_pred = predict_sklearn(est, X_test)
            val_m = compute_metrics(y_val, y_val_pred)
            test_m = compute_metrics(y_test, y_test_pred)

            # Save best_metrics for sklearn models (for consistency with torch)
            try:
                metrics_summary = {
                    "val_metrics": val_m,
                    "test_metrics": test_m,
                }
                metrics_json = os.path.join(out_dir, "best_metrics.json")
                with open(metrics_json, "w", encoding="utf-8") as f:
                    json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
                
                metrics_txt = os.path.join(out_dir, "best_metrics.txt")
                with open(metrics_txt, "w", encoding="utf-8") as f:
                    f.write("val_metrics:\n")
                    for k, v in sorted(val_m.items()):
                        f.write(f"  {k}: {v:.6f}\n")
                    f.write("\ntest_metrics:\n")
                    for k, v in sorted(test_m.items()):
                        f.write(f"  {k}: {v:.6f}\n")
            except Exception as e:
                print(f"Warning: failed to write best_metrics: {e}")

            # Pointwise prediction on df_raw (no augmentation)
            df_pred = predict_pointwise_df_raw_generic(est, T_scaler, df_raw_test, backend="sklearn")

            if compare_draw_ternary:
                # Ternary sweep uses model.predict(X); works for sklearn baseline
                visualize_all_test_groups(est, T_scaler, df_raw, test_system_ids, df_pred, out_dir,
                                          backend="sklearn", save_pdf=save_pdf)

        else:
            print(f"[SKIP] Unknown model_name={model_name}. "
                  f"torch={sorted(TORCH_MODELS)} sklearn={sorted(SKLEARN_MODELS)}")
            continue

        # Persist test metrics only after validation-based checkpoint selection.
        # The multi-seed aggregator treats this file as the authoritative record.
        metrics_json = os.path.join(out_dir, "best_metrics.json")
        metrics_summary = {}
        if os.path.isfile(metrics_json):
            with open(metrics_json, "r", encoding="utf-8") as stream:
                metrics_summary = json.load(stream)
        metrics_summary["test_metrics"] = {
            key: float(value) for key, value in test_m.items()
        }
        metrics_summary["test_evaluation_policy"] = (
            "evaluated once after validation-based model selection"
        )
        with open(metrics_json, "w", encoding="utf-8") as stream:
            json.dump(metrics_summary, stream, ensure_ascii=False, indent=2)

        # Save per-model predictions + parity
        pred_csv = os.path.join(out_dir, "test_df_raw_pointwise_predictions.csv")
        df_pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")
        print("Saved test predictions CSV:", pred_csv)

        parity_plots(df_pred, out_dir)
        print("Saved parity plots:", os.path.join(out_dir, "parity_E.png"), "/", os.path.join(out_dir, "parity_R.png"))

        # Summarize metrics
        print("\nFinal metrics:")
        print_metrics("  Val :", val_m)
        print_metrics("  Test:", test_m)

        row = {"model": name}
        row.update({f"val_{k}": v for k, v in val_m.items()})
        row.update({f"test_{k}": v for k, v in test_m.items()})
        results.append(row)

    if len(results) == 0:
        print("\nNo models were successfully run.")
        return

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(C.OUT_DIR, "baseline_compare_metrics.csv")
    df_res.sort_values("test_mse", ascending=True).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("Comparison saved:", out_csv)
    print(df_res.sort_values("test_mse", ascending=True)[["model", "test_mse", "test_rmse", "test_r2"]].head(20))
    print("=" * 90)


if __name__ == "__main__":
    main()

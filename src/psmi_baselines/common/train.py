# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .utils import Scaler
from .data import (
    FingerprintCache,
    GraphCache,
    LLEDataset,
    LLEGNNDataset,
    SmilesRNNDataset,
    build_smiles_vocab,
    gnn_collate_fn,
)
from .metrics import evaluate_loader, print_metrics
from .model import build_torch_model


def plot_history(history: Dict[str, List[float]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Loss (MSE)
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot(history["epoch"], history["train_mse"])
    ax.plot(history["epoch"], history["val_mse"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Loss Curves (MSE)")
    ax.legend(["train", "val"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_loss_mse.png"), dpi=200)
    plt.close(fig)

    # MAE
    if "val_mae" in history:
        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.plot(history["epoch"], history["val_mae"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.set_title("MAE Curves")
        ax.legend(["val"], loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_mae.png"), dpi=200)
        plt.close(fig)

    # RMSE
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot(history["epoch"], history["val_rmse"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Curves")
    ax.legend(["val"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_rmse.png"), dpi=200)
    plt.close(fig)

    # R2
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot(history["epoch"], history["val_r2"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R2")
    ax.set_title("R2 Curves")
    ax.legend(["val"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_r2.png"), dpi=200)
    plt.close(fig)

    # Ex/Rx RMSE
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot(history["epoch"], history["val_rmse_E"])
    ax.plot(history["epoch"], history["val_rmse_R"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE (Ex vs Rx)")
    ax.legend(["val_E", "val_R"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_rmse_ex_rx.png"), dpi=200)
    plt.close(fig)


def build_model(model_name: Optional[str] = None, **extra_kwargs) -> nn.Module:
    """
    Build a torch model by name.

    model_name:
      - "mlp" (default): original LLECurveNet
      - "ann", "lstm", "transformer", "tabknet", "smiles_rnn", "gnn"
    """
    name = (model_name or getattr(C, "MODEL_NAME", "mlp")).lower()
    fp_bits = getattr(C, "FP_BITS")
    in_dim = 3 * fp_bits + 2

    # forward all UPPERCASE config vars as kwargs (safe + convenient)
    cfg_kwargs = {k: getattr(C, k) for k in dir(C) if k.isupper()}
    return build_torch_model(
        name,
        in_dim=in_dim,
        fp_bits=fp_bits,
        hidden=getattr(C, "HIDDEN"),
        dropout=getattr(C, "DROPOUT"),
        **cfg_kwargs,
        **extra_kwargs,
    )
def _make_loader(
    ds,
    batch_size: int,
    shuffle: bool,
    device: str,
    collate_fn=None,
) -> DataLoader:
    num_workers = getattr(C, "NUM_WORKERS", min(8, os.cpu_count() or 4))
    pin = device.startswith("cuda")

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = getattr(C, "PREFETCH_FACTOR", 4)

    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn

    return DataLoader(ds, **kwargs)


def train_or_load(train_df, val_df, test_df, model_name: Optional[str] = None, out_dir: Optional[str] = None, load_ckpt_path: Optional[str] = None) -> Tuple[nn.Module, Scaler, Dict[str, List[float]]]:
    name = (model_name or getattr(C, "MODEL_NAME", "mlp")).lower()
    out_dir = out_dir or getattr(C, "OUT_DIR")
    os.makedirs(out_dir, exist_ok=True)
    device = getattr(C, "DEVICE")
    is_smiles_rnn = name == "smiles_rnn"
    is_gnn = name == "gnn"

    # ----- perf knobs -----
    eval_every = getattr(C, "EVAL_EVERY", 1)
    plot_every = getattr(C, "PLOT_EVERY", 5)

    # AMP/TF32
    use_amp = getattr(C, "USE_AMP", True) and device.startswith("cuda")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Fit scaler on train only
    T_scaler = Scaler.fit(train_df["T"].to_numpy(dtype="float32"))

    # Build vocab for SMILES RNN
    smiles_vocab = None
    scalar_dim = 2 if getattr(C, "SMILES_USE_TIE_T", True) else 1
    if is_smiles_rnn:
        smiles_vocab = build_smiles_vocab([train_df])

    model = build_model(
        name,
        vocab_size=len(smiles_vocab) if smiles_vocab else None,
        pad_idx=0,
        scalar_dim=scalar_dim,
    ).to(device)

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

    # Load ckpt
    ckpt_path = load_ckpt_path or getattr(C, "LOAD_CKPT_PATH", "")
    # Auto-load per-model ckpt if it exists
    if (not ckpt_path) and getattr(C, "AUTO_LOAD_IF_EXISTS", True):
        auto_path = os.path.join(out_dir, f"{name}.pt")
        auto_path_old = os.path.join(out_dir, "lle_curve_net.pt") if name in {"mlp", "lle_curve_net"} else ""
        if (not os.path.isfile(auto_path)) and auto_path_old and os.path.isfile(auto_path_old):
            auto_path = auto_path_old
        if os.path.isfile(auto_path):
            ckpt_path = auto_path
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict") or ckpt.get("model")
        if state_dict:
            model.load_state_dict(state_dict)
        if "T_scaler" in ckpt:
            T_scaler = Scaler.from_state_dict(ckpt["T_scaler"])
        elif "T_mean" in ckpt and "T_std" in ckpt:
            T_scaler = Scaler(mean=float(ckpt["T_mean"]), std=float(ckpt["T_std"]))
        print(f"Loaded ckpt: {ckpt_path}")
        return model, T_scaler, history

    batch_size = getattr(C, "BATCH_SIZE")
    if is_smiles_rnn:
        max_len = getattr(C, "SMILES_MAX_LEN", 256)
        train_ds = SmilesRNNDataset(train_df, smiles_vocab, T_scaler, max_len=max_len, use_t=scalar_dim == 2)
        val_ds = SmilesRNNDataset(val_df, smiles_vocab, T_scaler, max_len=max_len, use_t=scalar_dim == 2)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    elif is_gnn:
        graph_cache = GraphCache()
        train_ds = LLEGNNDataset(train_df, T_scaler, graph_cache)
        val_ds = LLEGNNDataset(val_df, T_scaler, graph_cache)
        train_loader = _make_loader(
            train_ds, batch_size, shuffle=True, device=device, collate_fn=gnn_collate_fn
        )
        val_loader = _make_loader(
            val_ds, batch_size, shuffle=False, device=device, collate_fn=gnn_collate_fn
        )
    else:
        fp_cache = FingerprintCache()
        precompute = getattr(C, "PRECOMPUTE_FEATURES", True)

        train_ds = LLEDataset(train_df, T_scaler, fp_cache, precompute=precompute)
        val_ds = LLEDataset(val_df, T_scaler, fp_cache, precompute=precompute)

        train_loader = _make_loader(train_ds, batch_size, shuffle=True, device=device)
        val_loader = _make_loader(val_ds, batch_size, shuffle=False, device=device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(C, "LR"),
        weight_decay=getattr(C, "WEIGHT_DECAY")
    )

    # Use new torch.amp API instead of deprecated torch.cuda.amp
    if use_amp:
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    else:
        scaler = torch.amp.GradScaler('cuda', enabled=False)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    best_val_metrics = None

    os.makedirs(out_dir, exist_ok=True)

    # Per-epoch metric log
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

    # ---- training ----
    epochs = getattr(C, "EPOCHS")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) == 5:
                graph1, graph2, graph3, scalars, y = batch
                graph1 = tuple(value.to(device, non_blocking=True) for value in graph1)
                graph2 = tuple(value.to(device, non_blocking=True) for value in graph2)
                graph3 = tuple(value.to(device, non_blocking=True) for value in graph3)
                scalars = scalars.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                inputs = (graph1, graph2, graph3, scalars)
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                tokens, scalars, y = batch
                tokens = tokens.to(device, non_blocking=True)
                scalars = scalars.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                inputs = (tokens, scalars)
            else:
                x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                inputs = x

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                if isinstance(inputs, tuple):
                    pred = model(*inputs)
                else:
                    pred = model(inputs)
                loss = torch.mean((pred - y) ** 2)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            batch_size = y.shape[0]
            running += float(loss.item()) * batch_size
            n += batch_size
            pbar.set_postfix({"batch_mse": float(loss.item())})

        train_mse = running / max(n, 1)

        # ---- eval (can be less frequent to increase avg GPU util) ----
        do_eval = (epoch % eval_every == 0) or (epoch == epochs)
        if do_eval:
            val_m = evaluate_loader(model, val_loader, device)
            # Keep the test partition sealed until validation-based selection is complete.
            test_m = {"mse": np.nan, "rmse": np.nan, "r2": np.nan,
                      "rmse_E": np.nan, "rmse_R": np.nan, "r2_E": np.nan, "r2_R": np.nan,
                      "mae": np.nan, "mae_E": np.nan, "mae_R": np.nan}
        else:
            # keep last values as NaN placeholders
            val_m = {"mse": np.nan, "rmse": np.nan, "r2": np.nan,
                     "rmse_E": np.nan, "rmse_R": np.nan, "r2_E": np.nan, "r2_R": np.nan,
                     "mae": np.nan, "mae_E": np.nan, "mae_R": np.nan}
            test_m = dict(val_m)

        if do_eval and (val_m.get("mse", np.inf) < best_val):
            best_val = float(val_m["mse"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_metrics = dict(val_m)

        # Log history
        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)

        history["val_mse"].append(val_m.get("mse", np.nan))
        history["test_mse"].append(test_m.get("mse", np.nan))

        history["val_mae"].append(val_m.get("mae", np.nan))
        history["test_mae"].append(test_m.get("mae", np.nan))
        history["val_rmse"].append(val_m.get("rmse", np.nan))
        history["test_rmse"].append(test_m.get("rmse", np.nan))
        history["val_r2"].append(val_m.get("r2", np.nan))
        history["test_r2"].append(test_m.get("r2", np.nan))

        history["val_mae_E"].append(val_m.get("mae_E", np.nan))
        history["val_mae_R"].append(val_m.get("mae_R", np.nan))
        history["val_rmse_E"].append(val_m.get("rmse_E", np.nan))
        history["val_rmse_R"].append(val_m.get("rmse_R", np.nan))
        history["val_r2_E"].append(val_m.get("r2_E", np.nan))
        history["val_r2_R"].append(val_m.get("r2_R", np.nan))

        history["test_mae_E"].append(test_m.get("mae_E", np.nan))
        history["test_mae_R"].append(test_m.get("mae_R", np.nan))
        history["test_rmse_E"].append(test_m.get("rmse_E", np.nan))
        history["test_rmse_R"].append(test_m.get("rmse_R", np.nan))
        history["test_r2_E"].append(test_m.get("r2_E", np.nan))
        history["test_r2_R"].append(test_m.get("r2_R", np.nan))

        print(f"[Epoch {epoch:03d}] train_MSE={train_mse:.6f}")
        if do_eval:
            print_metrics("  Val :", val_m)
        else:
            print("  (skip val/test eval this epoch to keep GPU busy)")

        # Append to per-epoch log file
        vals = [
            epoch,
            train_mse,

            val_m.get("mae", np.nan),  val_m.get("rmse", np.nan),  val_m.get("r2", np.nan),
            val_m.get("mae_E", np.nan), val_m.get("rmse_E", np.nan), val_m.get("r2_E", np.nan),
            val_m.get("mae_R", np.nan), val_m.get("rmse_R", np.nan), val_m.get("r2_R", np.nan),

            test_m.get("mae", np.nan), test_m.get("rmse", np.nan), test_m.get("r2", np.nan),
            test_m.get("mae_E", np.nan), test_m.get("rmse_E", np.nan), test_m.get("r2_E", np.nan),
            test_m.get("mae_R", np.nan), test_m.get("rmse_R", np.nan), test_m.get("r2_R", np.nan),
        ]
        line = ",".join([str(int(vals[0])), _fmt8(vals[1])] + [_fmt8(v) for v in vals[2:]]) + "\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

        # Save/update curves (reduce CPU overhead)
        if (epoch % plot_every == 0) or (epoch == epochs):
            plot_history(history, out_dir)

        # Save history CSV
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "train_history.csv"),
                                     index=False, encoding="utf-8-sig")

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model by val_mse={best_val:.6f}")

        # Write a compact validation-only selection record.
        try:
            import json

            def _round4(x):
                if isinstance(x, float):
                    # Preserve NaN for metrics that are undefined on degenerate subsets.
                    return round(x, 4)
                if isinstance(x, (int, str)) or x is None:
                    return x
                if isinstance(x, dict):
                    return {k: _round4(v) for k, v in x.items()}
                if isinstance(x, (list, tuple)):
                    return [_round4(v) for v in x]
                return x

            best_val_metrics_4 = _round4(best_val_metrics or {})
            summary = {
                "best_epoch": int(best_epoch),
                "best_val_mse": round(float(best_val), 4),
                "best_val_metrics": best_val_metrics_4,
                "selection_policy": "validation_only; test evaluated after model selection",
            }

            with open(os.path.join(out_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            def _fmt4(v):
                try:
                    v = float(v)
                except Exception:
                    return str(v)
                if np.isnan(v) or np.isinf(v):
                    return str(v)
                return f"{v:.4f}"

            with open(os.path.join(out_dir, "best_metrics.txt"), "w", encoding="utf-8") as f:
                f.write(f"best_epoch: {best_epoch}\n")
                f.write(f"best_val_mse: {_fmt4(best_val)}\n\n")

                f.write("best_val_metrics:\n")
                for k, v in (best_val_metrics_4 or {}).items():
                    if isinstance(v, float):
                        f.write(f"  {k}: {_fmt4(v)}\n")
                    else:
                        f.write(f"  {k}: {v}\n")

                f.write("\nselection_policy: validation_only; test evaluated after model selection\n")

        except Exception as e:
            print("Warning: failed to write best_metrics files:", e)


    # Save ckpt
    # Preserve the established filename used by the MLP evaluation scripts.
    ckpt_fname = "lle_curve_net.pt" if name in {"mlp", "lle_curve_net"} else f"{name}.pt"
    ckpt_out = os.path.join(out_dir, ckpt_fname)
    # also save an alias file <name>.pt for convenience
    ckpt_out_alias = os.path.join(out_dir, f"{name}.pt")
    ckpt_payload = {
        "state_dict": model.state_dict(),
        "T_mean": float(T_scaler.mean),
        "T_std": float(T_scaler.std),
        "epoch": epoch,
        "val_mse": float(best_val),
        "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
    }
    torch.save(ckpt_payload, ckpt_out)
    if ckpt_out_alias != ckpt_out:
        torch.save(ckpt_payload, ckpt_out_alias)
    print("Saved ckpt:", ckpt_out)

    return model, T_scaler, history

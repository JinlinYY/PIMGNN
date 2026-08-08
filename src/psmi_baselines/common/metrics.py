# -*- coding: utf-8 -*-
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1).astype(np.float64)
    y_pred = y_pred.reshape(-1).astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def calc_mae_rmse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    y_true/y_pred: (N, D) or (D,)
    Return: mae, rmse, r2 (r2 on flattened values)
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)
    return mae, rmse, r2


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    overall / Ex(3) / Rx(3)
    Returns: mse/rmse/r2 and mae for overall, Ex, Rx
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # overall
    mae_all, rmse_all, r2_all = calc_mae_rmse_r2(y_true, y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))

    # Ex / Rx
    y_true_E, y_pred_E = y_true[:, :3], y_pred[:, :3]
    y_true_R, y_pred_R = y_true[:, 3:], y_pred[:, 3:]

    mae_E, rmse_E, r2_E = calc_mae_rmse_r2(y_true_E, y_pred_E)
    mae_R, rmse_R, r2_R = calc_mae_rmse_r2(y_true_R, y_pred_R)

    return {
        "mse": float(mse),
        "rmse": float(rmse_all),
        "r2": float(r2_all),
        "mae": float(mae_all),

        "rmse_E": float(rmse_E),
        "rmse_R": float(rmse_R),
        "r2_E": float(r2_E),
        "r2_R": float(r2_R),
        "mae_E": float(mae_E),
        "mae_R": float(mae_R),
    }


@torch.no_grad()
def collect_preds(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        # Support fingerprint, sequence, and ternary graph batches.
        if isinstance(batch, (list, tuple)) and len(batch) == 5:
            graph1, graph2, graph3, scalars, y = batch
            graph1 = tuple(value.to(device) for value in graph1)
            graph2 = tuple(value.to(device) for value in graph2)
            graph3 = tuple(value.to(device) for value in graph3)
            pred = model(graph1, graph2, graph3, scalars.to(device)).cpu().numpy()
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            tokens, scalars, y = batch
            tokens = tokens.to(device)
            scalars = scalars.to(device)
            pred = model(tokens, scalars).cpu().numpy()
        else:
            x, y = batch
            x = x.to(device)
            pred = model(x).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    y_true, y_pred = collect_preds(model, loader, device)
    return compute_metrics(y_true, y_pred)


def print_metrics(prefix: str, m: Dict[str, float]) -> None:
    print(
        f"{prefix} "
        f"MAE={m.get('mae', float('nan')):.6f} "
        f"MSE={m['mse']:.6f} RMSE={m['rmse']:.6f} R2={m['r2']:.4f} | "
        f"Ex: MAE={m.get('mae_E', float('nan')):.6f} RMSE={m['rmse_E']:.6f} R2={m['r2_E']:.4f} | "
        f"Rx: MAE={m.get('mae_R', float('nan')):.6f} RMSE={m['rmse_R']:.6f} R2={m['r2_R']:.4f}"
    )

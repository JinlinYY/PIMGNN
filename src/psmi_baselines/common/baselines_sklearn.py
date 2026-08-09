# -*- coding: utf-8 -*-
"""
Sklearn / xgboost / (optional) TabNet baselines for the LLE point prediction task.

Conventions:
- Input X: same as torch models (3 fingerprints + [Tn, t])
- Output y: 6 dims [Ex1,Ex2,Ex3,Rx1,Rx2,Rx3]
- Predictions are projected onto the non-negative, sum-to-one phase simplex.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import joblib

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from pytorch_tabnet.tab_model import TabNetRegressor  # type: ignore
    _HAS_TABNET = True
except Exception:
    _HAS_TABNET = False

from .utils import renorm3


SKLEARN_MODEL_ALIASES: Dict[str, str] = {
    "rf": "random_forest",
    "randomforest": "random_forest",
    "random_forest": "random_forest",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "tabnet": "tabnet",
}


def _postprocess_preds(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim != 2 or pred.shape[1] != 6:
        raise ValueError(f"pred must be (N,6), got {pred.shape}")
    E = pred[:, :3].copy()
    R = pred[:, 3:].copy()
    for i in range(pred.shape[0]):
        E[i] = renorm3(np.clip(E[i], 0.0, None))
        R[i] = renorm3(np.clip(R[i], 0.0, None))
    return np.concatenate([E, R], axis=1)


def build_sklearn_model(model_name: str, seed: int = 42):
    name = SKLEARN_MODEL_ALIASES.get(model_name.lower(), model_name.lower())

    if name == "random_forest":
        # Multioutput supported natively
        return RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )

    if name == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed; run `pip install xgboost`.")
        # Use GPU if available (XGBoost 2.0+ API)
        import torch
        device = "cpu"
        
        if torch.cuda.is_available():
            device = "cuda"
            print(f"XGBoost: Using GPU acceleration (device={device})")
        else:
            print(f"XGBoost: Using CPU (device={device})")
        
        base = XGBRegressor(
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            device=device,
            random_state=seed,
            n_jobs=-1 if device == "cpu" else 1,
        )
        return MultiOutputRegressor(base, n_jobs=-1)

    if name == "tabnet":
        if not _HAS_TABNET:
            raise ImportError("pytorch-tabnet is not installed; run `pip install pytorch-tabnet`.")
        # TabNetRegressor supports multi-target regression directly
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TabNetRegressor(
            seed=seed,
            verbose=0,
            device_name=device,
        )

    raise ValueError(f"Unknown sklearn baseline: {model_name}. Available: {sorted(set(SKLEARN_MODEL_ALIASES.values()))}")


def fit_sklearn_model(model_name: str,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                      out_dir: str | None = None,
                      seed: int = 42):
    """
    Fit and optionally save model.

    Returns: fitted estimator
    """
    name = SKLEARN_MODEL_ALIASES.get(model_name.lower(), model_name.lower())
    model = build_sklearn_model(name, seed=seed)

    if name == "tabnet":
        # TabNet needs float32
        Xtr = X_train.astype(np.float32)
        ytr = y_train.astype(np.float32)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.astype(np.float32), y_val.astype(np.float32))]
        model.fit(
            Xtr, ytr,
            eval_set=eval_set,
            max_epochs=300,
            patience=50,
            batch_size=4096,
            virtual_batch_size=1024,
            num_workers=0,
            drop_last=False,
        )
    else:
        model.fit(X_train, y_train)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(model, os.path.join(out_dir, f"{name}.joblib"))

    return model


def predict_sklearn(model, X: np.ndarray) -> np.ndarray:
    pred = np.asarray(model.predict(X), dtype=np.float32)
    return _postprocess_preds(pred)


def quick_metric_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))

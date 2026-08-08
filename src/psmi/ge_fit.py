"""Fit system-specific GE interaction energies from training tie-lines only."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .ge_models import GAS_CONSTANT, SUPPORTED_GE_MODELS, ge_mu_residual


COMPOSITION_COLUMNS = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3", "T"]


def _parameter_transform(model: str, raw: torch.Tensor, maximum: float) -> torch.Tensor:
    if model == "van_laar":
        transformed = float(maximum) * torch.sigmoid(raw)
    else:
        transformed = float(maximum) * torch.tanh(raw)
    identity = torch.eye(3, dtype=raw.dtype, device=raw.device)
    return transformed * (1.0 - identity)


def fit_one_system(
    data: pd.DataFrame,
    *,
    model: str,
    steps: int = 3000,
    learning_rate: float = 5e-2,
    maximum_energy: float = 8000.0,
    alpha: float = 0.3,
    device: str = "cpu",
) -> np.ndarray:
    model = str(model).lower()
    if model not in SUPPORTED_GE_MODELS:
        raise ValueError(f"Unsupported GE model {model!r}")
    clean = data.dropna(subset=COMPOSITION_COLUMNS)
    if len(clean) < 3:
        raise ValueError("At least three tie-lines are required to fit GE parameters")

    dtype = torch.float32
    extract = torch.tensor(clean[["Ex1", "Ex2", "Ex3"]].to_numpy(), dtype=dtype, device=device)
    raffinate = torch.tensor(clean[["Rx1", "Rx2", "Rx3"]].to_numpy(), dtype=dtype, device=device)
    temperature = torch.tensor(clean["T"].to_numpy(), dtype=dtype, device=device)
    initial = -2.0 if model == "van_laar" else 0.0
    raw = torch.full((3, 3), initial, dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=float(learning_rate))

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        parameters = _parameter_transform(model, raw, maximum_energy)
        batch_parameters = parameters.unsqueeze(0).expand(len(clean), -1, -1)
        residual = ge_mu_residual(
            model,
            extract,
            raffinate,
            temperature,
            batch_parameters,
            nrtl_alpha=alpha,
        )
        loss = residual.square().mean() + 1e-4 * raw.square().mean()
        loss.backward()
        optimizer.step()

    fitted = _parameter_transform(model, raw, maximum_energy).detach().cpu().numpy()
    np.fill_diagonal(fitted, 0.0)
    return fitted.astype(np.float32)


def fit_parameter_store(
    data: pd.DataFrame,
    *,
    model: str,
    training_system_ids: Iterable[int],
    steps: int = 3000,
    learning_rate: float = 5e-2,
    maximum_energy: float = 8000.0,
    alpha: float = 0.3,
    device: str = "cpu",
    vectorized: bool = False,
) -> dict[str, Any]:
    """Fit only explicitly declared training systems; no implicit all-data fallback."""
    model = str(model).lower()
    declared_ids = sorted({int(value) for value in training_system_ids})
    eligible_ids = [
        system_id
        for system_id in declared_ids
        if len(data[data["system_id"] == system_id].dropna(subset=COMPOSITION_COLUMNS)) >= 3
    ]
    parameters: dict[str, Any] = {}
    if vectorized and eligible_ids:
        selected = data[data["system_id"].isin(eligible_ids)].dropna(
            subset=COMPOSITION_COLUMNS
        )
        index_by_id = {system_id: index for index, system_id in enumerate(eligible_ids)}
        row_indices = torch.tensor(
            [index_by_id[int(value)] for value in selected["system_id"]],
            dtype=torch.long,
            device=device,
        )
        dtype = torch.float32
        extract = torch.tensor(
            selected[["Ex1", "Ex2", "Ex3"]].to_numpy(), dtype=dtype, device=device
        )
        raffinate = torch.tensor(
            selected[["Rx1", "Rx2", "Rx3"]].to_numpy(), dtype=dtype, device=device
        )
        temperature = torch.tensor(selected["T"].to_numpy(), dtype=dtype, device=device)
        initial = -2.0 if model == "van_laar" else 0.0
        raw = torch.full(
            (len(eligible_ids), 3, 3),
            initial,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([raw], lr=float(learning_rate))
        counts = torch.bincount(row_indices, minlength=len(eligible_ids)).clamp_min(1)
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            all_parameters = _parameter_transform(model, raw, maximum_energy)
            row_parameters = all_parameters[row_indices]
            residual = ge_mu_residual(
                model,
                extract,
                raffinate,
                temperature,
                row_parameters,
                nrtl_alpha=alpha,
            )
            row_loss = residual.square().mean(dim=-1)
            system_loss_sum = torch.zeros(
                len(eligible_ids), dtype=dtype, device=device
            ).scatter_add_(0, row_indices, row_loss)
            loss = (system_loss_sum / counts).mean() + 1e-4 * raw.square().mean()
            loss.backward()
            optimizer.step()
        fitted_all = _parameter_transform(model, raw, maximum_energy).detach().cpu().numpy()
        for system_id, fitted in zip(eligible_ids, fitted_all):
            np.fill_diagonal(fitted, 0.0)
            parameters[str(system_id)] = fitted.astype(np.float32).tolist()
    else:
        for system_id in eligible_ids:
            system_data = data[data["system_id"] == system_id]
            fitted = fit_one_system(
                system_data,
                model=model,
                steps=steps,
                learning_rate=learning_rate,
                maximum_energy=maximum_energy,
                alpha=alpha,
                device=device,
            )
            parameters[str(system_id)] = fitted.tolist()

    return {
        "meta": {
            "schema_version": 2,
            "role": "training_loss",
            "fitted_independently_by_system": True,
            "model": model,
            "alpha": float(alpha),
            "R": GAS_CONSTANT,
            "maximum_energy_J_per_mol": float(maximum_energy),
            "fit_scope": "training_systems_only",
            "declared_training_systems": len(declared_ids),
            "successfully_fitted_systems": len(parameters),
            "optimizer_layout": "vectorized" if vectorized else "per_system",
        },
        "params": parameters,
    }

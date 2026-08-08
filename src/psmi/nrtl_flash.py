"""Numerical ternary NRTL flash and tangent-plane stability utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np
from scipy.optimize import minimize


GAS_CONSTANT = 8.314462618


def _simplex_grid(grid_size: int, floor: float = 1e-3) -> np.ndarray:
    values = np.linspace(float(floor), 1.0 - 2.0 * float(floor), int(grid_size))
    points = []
    for first in values:
        for second in values:
            third = 1.0 - first - second
            if third >= float(floor):
                points.append([first, second, third])
    return np.asarray(points, dtype=float)


def fit_stable_nrtl_parameters(
    data,
    *,
    steps: int = 3000,
    learning_rate: float = 2e-2,
    maximum_energy: float = 8000.0,
    alpha: float = 0.3,
    stability_weight: float = 5.0,
    stability_grid_size: int = 11,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit NRTL energies with chemical-equilibrium and global TPD penalties."""
    import torch

    from .loss import nrtl_ln_gamma, nrtl_mu_residual, renorm3_torch

    required = ["T", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
    clean = data[required].apply(lambda column: column.astype(float)).dropna()
    if len(clean) < 3:
        raise ValueError("At least three valid tie-lines are required")
    torch.manual_seed(int(seed))
    dtype = torch.float32
    extract = renorm3_torch(
        torch.tensor(clean[["Ex1", "Ex2", "Ex3"]].to_numpy(), dtype=dtype, device=device)
    )
    raffinate = renorm3_torch(
        torch.tensor(clean[["Rx1", "Rx2", "Rx3"]].to_numpy(), dtype=dtype, device=device)
    )
    temperature = torch.tensor(clean["T"].to_numpy(), dtype=dtype, device=device)
    trial_grid = torch.tensor(
        _simplex_grid(stability_grid_size), dtype=dtype, device=device
    )
    batch_size = len(clean)
    n_trials = trial_grid.shape[0]
    trials = trial_grid.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
    trial_temperature = temperature.repeat_interleave(n_trials)

    raw = (0.05 * torch.randn((3, 3), dtype=dtype, device=device)).requires_grad_(True)
    off_diagonal = 1.0 - torch.eye(3, dtype=dtype, device=device)
    optimizer = torch.optim.Adam([raw], lr=float(learning_rate))
    best_loss = float("inf")
    best_matrix = None
    best_audit: dict[str, float] = {}
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        matrix = float(maximum_energy) * torch.tanh(raw) * off_diagonal
        row_matrix = matrix.unsqueeze(0).expand(batch_size, -1, -1)
        residual = nrtl_mu_residual(
            extract,
            raffinate,
            temperature,
            row_matrix,
            alpha=alpha,
            R=GAS_CONSTANT,
            ln_gamma_clip=None,
        )
        residual_loss = residual.square().mean()

        trial_matrix = matrix.unsqueeze(0).expand(batch_size * n_trials, -1, -1)
        trial_ln_gamma = nrtl_ln_gamma(
            trials,
            trial_temperature,
            trial_matrix,
            alpha=alpha,
            R=GAS_CONSTANT,
            ln_gamma_clip=None,
        )
        trial_mu = (torch.log(trials.clamp_min(1e-12)) + trial_ln_gamma).reshape(
            batch_size, n_trials, 3
        )
        extract_mu = torch.log(extract.clamp_min(1e-12)) + nrtl_ln_gamma(
            extract,
            temperature,
            row_matrix,
            alpha=alpha,
            R=GAS_CONSTANT,
            ln_gamma_clip=None,
        )
        raffinate_mu = torch.log(raffinate.clamp_min(1e-12)) + nrtl_ln_gamma(
            raffinate,
            temperature,
            row_matrix,
            alpha=alpha,
            R=GAS_CONSTANT,
            ln_gamma_clip=None,
        )
        tpd_extract = (
            trials.reshape(batch_size, n_trials, 3)
            * (trial_mu - extract_mu[:, None, :])
        ).sum(dim=-1)
        tpd_raffinate = (
            trials.reshape(batch_size, n_trials, 3)
            * (trial_mu - raffinate_mu[:, None, :])
        ).sum(dim=-1)
        stability_loss = (
            torch.relu(-tpd_extract).square().mean()
            + torch.relu(-tpd_raffinate).square().mean()
        )
        regularization = 1e-4 * raw.square().mean()
        loss = residual_loss + float(stability_weight) * stability_loss + regularization
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_matrix = matrix.detach().cpu().numpy().copy()
            best_audit = {
                "objective": loss_value,
                "mu_residual_rmse": float(torch.sqrt(residual_loss).detach().cpu()),
                "tpd_penalty": float(stability_loss.detach().cpu()),
                "grid_min_tpd": float(
                    torch.minimum(tpd_extract.min(), tpd_raffinate.min()).detach().cpu()
                ),
            }
    if best_matrix is None:
        raise RuntimeError("stable NRTL optimization did not produce finite parameters")
    np.fill_diagonal(best_matrix, 0.0)
    return best_matrix.astype(np.float32), best_audit


def _composition(first_two: Iterable[float], eps: float = 1e-10) -> np.ndarray:
    values = np.asarray(list(first_two), dtype=float)
    result = np.asarray([values[0], values[1], 1.0 - values.sum()], dtype=float)
    result = np.clip(result, eps, None)
    return result / result.sum()


def _normalize(values: Iterable[float], eps: float = 1e-10) -> np.ndarray:
    result = np.clip(np.asarray(list(values), dtype=float), eps, None)
    if result.shape != (3,):
        raise ValueError(f"ternary composition must have shape (3,), got {result.shape}")
    return result / result.sum()


def nrtl_ln_gamma_numpy(
    composition: Iterable[float],
    temperature: float,
    interaction: np.ndarray,
    *,
    alpha: float = 0.3,
    gas_constant: float = GAS_CONSTANT,
    tau_clip: float | None = 10.0,
) -> np.ndarray:
    """NumPy equivalent of :func:`psmi.loss.nrtl_ln_gamma` for one state."""
    x = _normalize(composition)
    matrix = np.asarray(interaction, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"interaction must have shape (3, 3), got {matrix.shape}")
    tau = matrix / (float(gas_constant) * max(float(temperature), 1.0))
    if tau_clip is not None and tau_clip > 0:
        tau = np.clip(tau, -float(tau_clip), float(tau_clip))
    weights = np.exp(-float(alpha) * tau)
    denominator = np.maximum((x[:, None] * weights).sum(axis=0), 1e-12)
    weighted_tau = (x[:, None] * tau * weights).sum(axis=0)
    first = (x[None, :] * (tau.T * weights.T)).sum(axis=1) / denominator
    normalized = x[None, :] * weights / denominator[None, :]
    second = (
        normalized * (tau - (weighted_tau / denominator)[None, :])
    ).sum(axis=1)
    return first + second


def mixing_gibbs_rt(
    composition: Iterable[float],
    temperature: float,
    interaction: np.ndarray,
    *,
    alpha: float = 0.3,
) -> float:
    """Dimensionless molar Gibbs energy of mixing, ``g_mix / (R T)``."""
    x = _normalize(composition)
    ln_gamma = nrtl_ln_gamma_numpy(x, temperature, interaction, alpha=alpha)
    return float(np.sum(x * (np.log(x) + ln_gamma)))


def chemical_potential_residual(
    extract: Iterable[float],
    raffinate: Iterable[float],
    temperature: float,
    interaction: np.ndarray,
    *,
    alpha: float = 0.3,
) -> np.ndarray:
    extract_array = _normalize(extract)
    raffinate_array = _normalize(raffinate)
    mu_extract = np.log(extract_array) + nrtl_ln_gamma_numpy(
        extract_array, temperature, interaction, alpha=alpha
    )
    mu_raffinate = np.log(raffinate_array) + nrtl_ln_gamma_numpy(
        raffinate_array, temperature, interaction, alpha=alpha
    )
    return mu_extract - mu_raffinate


def tangent_plane_distance(
    trial: Iterable[float],
    reference: Iterable[float],
    temperature: float,
    interaction: np.ndarray,
    *,
    alpha: float = 0.3,
) -> float:
    trial_array = _normalize(trial)
    reference_array = _normalize(reference)
    mu_trial = np.log(trial_array) + nrtl_ln_gamma_numpy(
        trial_array, temperature, interaction, alpha=alpha
    )
    mu_reference = np.log(reference_array) + nrtl_ln_gamma_numpy(
        reference_array, temperature, interaction, alpha=alpha
    )
    return float(np.sum(trial_array * (mu_trial - mu_reference)))


def minimum_tpd(
    reference: Iterable[float],
    temperature: float,
    interaction: np.ndarray,
    *,
    alpha: float = 0.3,
    grid_size: int = 21,
    extra_starts: Iterable[Iterable[float]] = (),
) -> float:
    """Approximate the global minimum tangent-plane distance on the simplex."""
    reference_array = _normalize(reference)
    eps = 1e-8
    starts: list[np.ndarray] = [reference_array]
    starts.extend(_normalize(value) for value in extra_starts)
    grid_values = np.linspace(eps, 1.0 - eps, int(grid_size))
    grid_candidates = []
    for first in grid_values:
        for second in grid_values:
            if first + second < 1.0 - eps:
                candidate = np.asarray([first, second, 1.0 - first - second])
                value = tangent_plane_distance(
                    candidate,
                    reference_array,
                    temperature,
                    interaction,
                    alpha=alpha,
                )
                grid_candidates.append((value, candidate))
    starts.extend(candidate for _, candidate in sorted(grid_candidates, key=lambda item: item[0])[:6])

    def objective(first_two: np.ndarray) -> float:
        return tangent_plane_distance(
            _composition(first_two),
            reference_array,
            temperature,
            interaction,
            alpha=alpha,
        )

    constraints = [
        {"type": "ineq", "fun": lambda values: 1.0 - eps - values[0] - values[1]}
    ]
    bounds = [(eps, 1.0 - eps), (eps, 1.0 - eps)]
    best = min(
        tangent_plane_distance(
            candidate,
            reference_array,
            temperature,
            interaction,
            alpha=alpha,
        )
        for candidate in starts
    )
    for start in starts:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Values in x were outside bounds", category=RuntimeWarning
            )
            fitted = minimize(
                objective,
                start[:2],
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-11},
            )
        if np.isfinite(fitted.fun):
            best = min(best, float(fitted.fun))
    return float(best)


@dataclass(frozen=True)
class FlashResult:
    success: bool
    status: str
    extract: np.ndarray
    raffinate: np.ndarray
    phase_fraction: float
    objective: float
    single_phase_objective: float
    gibbs_gain: float
    mass_balance_max: float
    mu_residual_max: float
    phase_separation: float
    minimum_tpd: float
    optimizer_success: bool


def _phase_fraction(
    overall: np.ndarray, extract: np.ndarray, raffinate: np.ndarray
) -> float:
    direction = extract - raffinate
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-14:
        return 0.5
    return float(np.clip(np.dot(overall - raffinate, direction) / denominator, 0.02, 0.98))


def flash_nrtl(
    overall: Iterable[float],
    temperature: float,
    interaction: np.ndarray,
    *,
    initial_extract: Iterable[float],
    initial_raffinate: Iterable[float],
    alpha: float = 0.3,
    random_starts: int = 4,
    seed: int = 42,
    minimum_phase_separation: float = 0.02,
    maximum_mu_residual: float = 5e-3,
    maximum_mass_balance_error: float = 2e-6,
    minimum_gibbs_gain: float = 1e-8,
    tpd_tolerance: float = 2e-4,
) -> FlashResult:
    """Solve an isothermal ternary two-liquid flash by Gibbs minimization."""
    z = _normalize(overall)
    initial_e = _normalize(initial_extract)
    initial_r = _normalize(initial_raffinate)
    matrix = np.asarray(interaction, dtype=float)
    eps = 1e-8
    single_objective = mixing_gibbs_rt(z, temperature, matrix, alpha=alpha)

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        return _composition(values[:2]), _composition(values[2:4]), float(values[4])

    def objective(values: np.ndarray) -> float:
        extract, raffinate, fraction = unpack(values)
        return float(
            fraction * mixing_gibbs_rt(extract, temperature, matrix, alpha=alpha)
            + (1.0 - fraction)
            * mixing_gibbs_rt(raffinate, temperature, matrix, alpha=alpha)
        )

    def mass_balance(values: np.ndarray) -> np.ndarray:
        extract, raffinate, fraction = unpack(values)
        reconstructed = fraction * extract + (1.0 - fraction) * raffinate
        return reconstructed[:2] - z[:2]

    constraints = [
        {"type": "eq", "fun": mass_balance},
        {"type": "ineq", "fun": lambda values: 1.0 - eps - values[0] - values[1]},
        {"type": "ineq", "fun": lambda values: 1.0 - eps - values[2] - values[3]},
    ]
    bounds = [
        (eps, 1.0 - eps),
        (eps, 1.0 - eps),
        (eps, 1.0 - eps),
        (eps, 1.0 - eps),
        (0.02, 0.98),
    ]
    starts = [
        np.r_[initial_e[:2], initial_r[:2], _phase_fraction(z, initial_e, initial_r)],
        np.r_[initial_r[:2], initial_e[:2], 1.0 - _phase_fraction(z, initial_e, initial_r)],
        np.r_[z[:2], z[:2], 0.5],
    ]
    rng = np.random.RandomState(int(seed))
    for _ in range(int(random_starts)):
        direction = rng.normal(size=3)
        direction -= direction.mean()
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            continue
        direction /= norm
        positive = direction > 0
        negative = direction < 0
        limits = []
        if positive.any():
            limits.append(np.min((1.0 - eps - z[positive]) / direction[positive]))
        if negative.any():
            limits.append(np.min((z[negative] - eps) / (-direction[negative])))
        scale_limit = max(eps, min(limits))
        scale = rng.uniform(0.1, 0.8) * scale_limit
        phase_a = _normalize(z + scale * direction)
        phase_b = _normalize(z - scale * direction)
        starts.append(np.r_[phase_a[:2], phase_b[:2], 0.5])

    candidates = []
    for start in starts:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Values in x were outside bounds", category=RuntimeWarning
            )
            fitted = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 800, "ftol": 1e-12},
            )
        if not np.isfinite(fitted.fun):
            continue
        extract, raffinate, fraction = unpack(fitted.x)
        reconstructed = fraction * extract + (1.0 - fraction) * raffinate
        balance_error = float(np.max(np.abs(reconstructed - z)))
        candidates.append(
            (float(fitted.fun), bool(fitted.success), extract, raffinate, fraction, balance_error)
        )

    if not candidates:
        return FlashResult(
            False,
            "optimization_failed",
            z.copy(),
            z.copy(),
            0.5,
            single_objective,
            single_objective,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            False,
        )

    objective_value, optimizer_success, extract, raffinate, fraction, balance_error = min(
        candidates, key=lambda item: item[0]
    )
    direct_distance = np.linalg.norm(extract - initial_e) + np.linalg.norm(
        raffinate - initial_r
    )
    swapped_distance = np.linalg.norm(raffinate - initial_e) + np.linalg.norm(
        extract - initial_r
    )
    if swapped_distance < direct_distance:
        extract, raffinate = raffinate, extract
        fraction = 1.0 - fraction

    mu_max = float(
        np.max(
            np.abs(
                chemical_potential_residual(
                    extract, raffinate, temperature, matrix, alpha=alpha
                )
            )
        )
    )
    separation = float(np.linalg.norm(extract - raffinate))
    gain = float(single_objective - objective_value)
    if separation < float(minimum_phase_separation) or gain <= float(minimum_gibbs_gain):
        return FlashResult(
            False,
            "single_phase",
            extract,
            raffinate,
            fraction,
            objective_value,
            single_objective,
            gain,
            balance_error,
            mu_max,
            separation,
            0.0,
            optimizer_success,
        )

    tpd_extract = minimum_tpd(
        extract,
        temperature,
        matrix,
        alpha=alpha,
        extra_starts=[raffinate],
    )
    tpd_raffinate = minimum_tpd(
        raffinate,
        temperature,
        matrix,
        alpha=alpha,
        extra_starts=[extract],
    )
    minimum_tpd_value = min(tpd_extract, tpd_raffinate)
    numerical_ok = (
        balance_error <= float(maximum_mass_balance_error)
        and mu_max <= float(maximum_mu_residual)
        and minimum_tpd_value >= -float(tpd_tolerance)
    )
    return FlashResult(
        numerical_ok,
        "two_phase" if numerical_ok else "no_stable_split",
        extract,
        raffinate,
        fraction,
        objective_value,
        single_objective,
        gain,
        balance_error,
        mu_max,
        separation,
        float(minimum_tpd_value),
        optimizer_success,
    )

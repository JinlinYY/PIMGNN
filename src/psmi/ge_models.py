"""Differentiable excess-Gibbs-energy activity-coefficient models.

All public functions use interaction energies in J/mol.  Temperature dependence is
introduced consistently through ``A_ij = g_ij / (R T)`` so the alternative models can
be compared with the existing NRTL implementation under the same parameter units.
"""

from __future__ import annotations

from typing import Final

import torch


GAS_CONSTANT: Final[float] = 8.314462618
SUPPORTED_GE_MODELS: Final[tuple[str, ...]] = ("nrtl", "margules", "van_laar")


def _validate_inputs(
    composition: torch.Tensor,
    temperature: torch.Tensor,
    interaction_energies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if composition.ndim != 2 or composition.shape[1] != 3:
        raise ValueError(f"composition must have shape (B, 3), got {tuple(composition.shape)}")
    temperature = temperature.reshape(-1)
    if temperature.shape[0] != composition.shape[0]:
        raise ValueError("temperature and composition batch sizes differ")
    if interaction_energies.ndim == 2:
        interaction_energies = interaction_energies.unsqueeze(0).expand(
            composition.shape[0], -1, -1
        )
    if interaction_energies.shape != (composition.shape[0], 3, 3):
        raise ValueError(
            "interaction_energies must have shape (3, 3) or (B, 3, 3), "
            f"got {tuple(interaction_energies.shape)}"
        )
    return composition, temperature, interaction_energies


def _dimensionless_parameters(
    temperature: torch.Tensor,
    interaction_energies: torch.Tensor,
    gas_constant: float,
) -> torch.Tensor:
    return interaction_energies / (
        float(gas_constant) * temperature.clamp_min(1.0).view(-1, 1, 1)
    )


def _pairwise_margules_ge(x: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    ge = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        xi = x[:, i]
        xj = x[:, j]
        a_ij = parameters[:, i, j]
        a_ji = parameters[:, j, i]
        ge = ge + xi * xj * (a_ji * xi + a_ij * xj)
    return ge


def _pairwise_van_laar_ge(
    x: torch.Tensor,
    parameters: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    ge = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        xi = x[:, i]
        xj = x[:, j]
        a_ij = parameters[:, i, j]
        a_ji = parameters[:, j, i]
        denominator = a_ij * xi + a_ji * xj
        safe_denominator = torch.where(
            denominator.abs() > eps,
            denominator,
            torch.full_like(denominator, eps),
        )
        ge = ge + xi * xj * a_ij * a_ji / safe_denominator
    return ge


def _ln_gamma_from_molar_ge(
    composition: torch.Tensor,
    molar_ge,
    *,
    preserve_graph: bool,
) -> torch.Tensor:
    """Return partial molar excess Gibbs energies from a scalar molar GE/RT."""
    with torch.enable_grad():
        amounts = composition
        if not amounts.requires_grad:
            amounts = amounts.detach().clone().requires_grad_(True)
        total = amounts.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        x = amounts / total
        extensive_ge = total.squeeze(-1) * molar_ge(x)
        ln_gamma = torch.autograd.grad(
            extensive_ge.sum(),
            amounts,
            create_graph=preserve_graph,
            retain_graph=preserve_graph,
        )[0]
    return ln_gamma if preserve_graph else ln_gamma.detach()


def ge_ln_gamma(
    model: str,
    composition: torch.Tensor,
    temperature: torch.Tensor,
    interaction_energies: torch.Tensor,
    *,
    gas_constant: float = GAS_CONSTANT,
    nrtl_alpha: float = 0.3,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute ``ln(gamma)`` using a supported GE model.

    ``model`` is one of ``nrtl``, ``margules``, or ``van_laar``.
    """
    model = str(model).strip().lower()
    if model not in SUPPORTED_GE_MODELS:
        raise ValueError(f"Unsupported GE model {model!r}; choose from {SUPPORTED_GE_MODELS}")
    composition, temperature, interaction_energies = _validate_inputs(
        composition, temperature, interaction_energies
    )

    if model == "nrtl":
        # Lazy import avoids a module cycle while the legacy loss module migrates to
        # this public activity-model interface.
        from .loss import nrtl_ln_gamma

        return nrtl_ln_gamma(
            composition,
            temperature,
            interaction_energies,
            alpha=nrtl_alpha,
            R=gas_constant,
            eps=eps,
        )

    parameters = _dimensionless_parameters(
        temperature, interaction_energies, gas_constant
    )
    if model == "margules":
        return _ln_gamma_from_molar_ge(
            composition,
            lambda x: _pairwise_margules_ge(x, parameters),
            preserve_graph=(composition.requires_grad or interaction_energies.requires_grad),
        )
    return _ln_gamma_from_molar_ge(
        composition,
        lambda x: _pairwise_van_laar_ge(x, parameters, eps),
        preserve_graph=(composition.requires_grad or interaction_energies.requires_grad),
    )


def ge_mu_residual(
    model: str,
    extract_composition: torch.Tensor,
    raffinate_composition: torch.Tensor,
    temperature: torch.Tensor,
    interaction_energies: torch.Tensor,
    *,
    gas_constant: float = GAS_CONSTANT,
    nrtl_alpha: float = 0.3,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return interphase logarithmic-activity residuals for all components."""
    extract = extract_composition.clamp_min(0.0)
    raffinate = raffinate_composition.clamp_min(0.0)
    extract = extract / extract.sum(dim=-1, keepdim=True).clamp_min(eps)
    raffinate = raffinate / raffinate.sum(dim=-1, keepdim=True).clamp_min(eps)
    ln_gamma_extract = ge_ln_gamma(
        model,
        extract,
        temperature,
        interaction_energies,
        gas_constant=gas_constant,
        nrtl_alpha=nrtl_alpha,
        eps=eps,
    )
    ln_gamma_raffinate = ge_ln_gamma(
        model,
        raffinate,
        temperature,
        interaction_energies,
        gas_constant=gas_constant,
        nrtl_alpha=nrtl_alpha,
        eps=eps,
    )
    return (
        torch.log(extract.clamp_min(eps))
        + ln_gamma_extract
        - torch.log(raffinate.clamp_min(eps))
        - ln_gamma_raffinate
    )


# -*- coding: utf-8 -*-
"""
loss.py
NRTL-based mechanistic (physics) constraint for ternary LLE.

This file is designed to be "additive": you can keep your existing training logic
and simply replace the supervised loss with MechanisticNRTLLoss.

Core idea (from your Word):
For each component k in {1,2,3}, at equilibrium:
    ln(xE_k * gammaE_k) == ln(xR_k * gammaR_k)
Define residual:
    r_k = ln(xE_k) + ln(gammaE_k) - ln(xR_k) - ln(gammaR_k)
Physics loss:
    L_phy = mean_k,b ( r_k^2 )

gamma is computed by NRTL with fitted g_ij (J/mol) per system.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.nn.functional as F_nn


# -------------------------
# Small utils
# -------------------------

def renorm3_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Make a 3-component composition vector feasible:
      - clamp to non-negative
      - normalize so sum=1
    x: (..., 3)
    """
    x = torch.clamp(x, min=0.0)
    s = x.sum(dim=-1, keepdim=True).clamp_min(eps)
    return x / s


def _to_sid_str(sid: Union[int, str, torch.Tensor]) -> str:
    if isinstance(sid, torch.Tensor):
        sid = sid.detach().cpu()
        if sid.numel() == 1:
            sid = sid.item()
        else:
            raise ValueError("system_id tensor must be scalar per sample.")
    # many datasets use numeric system_id; json keys are strings
    return str(int(sid)) if isinstance(sid, (int, float)) or (isinstance(sid, str) and sid.isdigit()) else str(sid)


def _permute_23_g(g: torch.Tensor) -> torch.Tensor:
    """
    Swap component-2 and component-3 in a 3x3 g matrix:
      g' = P^T g P, where P swaps indices 1 and 2 (0-based: 1<->2)
    g: (..., 3, 3)
    """
    idx = torch.tensor([0, 2, 1], device=g.device, dtype=torch.long)
    return g.index_select(-2, idx).index_select(-1, idx)


# -------------------------
# NRTL core
# -------------------------

def nrtl_ln_gamma(
    x: torch.Tensor,
    T: torch.Tensor,
    g: torch.Tensor,
    alpha: float = 0.3,
    R: float = 8.314462618,
    eps: float = 1e-12,
    tau_clip: Optional[float] = 10.0,
    ln_gamma_clip: Optional[float] = 20.0,
) -> torch.Tensor:
    """
    Compute ln(gamma) for a ternary mixture using NRTL.

    x: (B,3) composition (must be >=0, sum=1 ideally)
    T: (B,) temperature (must be consistent with fitting; typically Kelvin)
    g: (B,3,3) interaction energy parameters g_ij (J/mol), diag can be 0.

    Returns:
      ln_gamma: (B,3)
    """
    # Ensure shapes
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"x must be (B,3), got {tuple(x.shape)}")
    if T.ndim != 1:
        T = T.view(-1)
    B = x.shape[0]
    if g.ndim == 2:
        g = g.unsqueeze(0).expand(B, -1, -1)
    if g.ndim != 3 or g.shape[-2:] != (3, 3):
        raise ValueError(f"g must be (B,3,3) or (3,3), got {tuple(g.shape)}")

    x = torch.clamp(x, min=0.0)
    T = T.clamp_min(1.0)  # avoid division by 0

    # tau_ij = g_ij / (R*T)
    tau = g / (R * T.view(B, 1, 1).clamp_min(eps))
    if tau_clip is not None and tau_clip > 0:
        tau = torch.clamp(tau, -float(tau_clip), float(tau_clip))

    # G_ij = exp(-alpha * tau_ij)
    G = torch.exp(-float(alpha) * tau)

    # denom_j = sum_k x_k * G_kj  (B,3)
    denom = (x.unsqueeze(2) * G).sum(dim=1).clamp_min(eps)  # (B,3)

    # A_j = sum_k x_k * tau_kj * G_kj (B,3)
    A = (x.unsqueeze(2) * tau * G).sum(dim=1)  # (B,3)

    # term1_i = sum_j x_j * tau_ji * G_ji / denom_i
    tau_T = tau.transpose(1, 2)
    G_T = G.transpose(1, 2)
    num1 = (x.unsqueeze(1) * (tau_T * G_T)).sum(dim=2)  # (B,3)
    term1 = num1 / denom  # (B,3)

    # term2_i = sum_j x_j * G_ij / denom_j * (tau_ij - A_j/denom_j)
    W = x.unsqueeze(1) * G / denom.unsqueeze(1)  # (B,3,3)
    inside = tau - (A / denom).unsqueeze(1)       # (B,3,3)
    term2 = (W * inside).sum(dim=2)               # (B,3)

    ln_gamma = term1 + term2

    if ln_gamma_clip is not None and ln_gamma_clip > 0:
        ln_gamma = torch.clamp(ln_gamma, -float(ln_gamma_clip), float(ln_gamma_clip))
    return ln_gamma


def nrtl_mu_residual(
    xE: torch.Tensor,
    xR: torch.Tensor,
    T: torch.Tensor,
    g: torch.Tensor,
    alpha: float = 0.3,
    R: float = 8.314462618,
    eps: float = 1e-12,
    tau_clip: Optional[float] = 10.0,
    ln_gamma_clip: Optional[float] = 20.0,
) -> torch.Tensor:
    """
    Chemical potential residual r_k = ln(xE_k*gammaE_k) - ln(xR_k*gammaR_k).

    Returns:
      r: (B,3)
    """
    xE = renorm3_torch(xE, eps=eps)
    xR = renorm3_torch(xR, eps=eps)

    ln_gE = nrtl_ln_gamma(xE, T, g, alpha=alpha, R=R, eps=eps,
                         tau_clip=tau_clip, ln_gamma_clip=ln_gamma_clip)
    ln_gR = nrtl_ln_gamma(xR, T, g, alpha=alpha, R=R, eps=eps,
                         tau_clip=tau_clip, ln_gamma_clip=ln_gamma_clip)

    r = (torch.log(xE.clamp_min(eps)) + ln_gE) - (torch.log(xR.clamp_min(eps)) + ln_gR)
    return r


def _sample_simplex_directions(x: torch.Tensor, n_dir: int) -> torch.Tensor:
    if n_dir <= 0:
        return torch.zeros((0, x.shape[0], 3), device=x.device, dtype=x.dtype)
    v = torch.randn((n_dir, x.shape[0], 3), device=x.device, dtype=x.dtype)
    v = v - v.mean(dim=-1, keepdim=True)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v


def gibbs_duhem_penalty(
    x: torch.Tensor,
    T: torch.Tensor,
    g: torch.Tensor,
    alpha: float,
    R: float,
    tau_clip: Optional[float],
    ln_gamma_clip: Optional[float],
    n_dir: int = 2,
    eps_fd: float = 1e-4,
) -> torch.Tensor:
    """
    Gibbs-Duhem penalty: sum_i x_i * d(ln gamma_i)/ds == 0 along simplex directions.
    """
    if x is None or x.numel() == 0:
        return torch.tensor(0.0, device=T.device if T is not None else "cpu")
    x = renorm3_torch(x)
    eps_fd = float(max(eps_fd, 1e-8))

    dirs = _sample_simplex_directions(x, int(n_dir))
    if dirs.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    penalties = []
    for k in range(dirs.shape[0]):
        d = dirs[k]
        x_p = renorm3_torch(x + eps_fd * d)
        x_m = renorm3_torch(x - eps_fd * d)

        ln_p = nrtl_ln_gamma(x_p, T, g, alpha=alpha, R=R, tau_clip=tau_clip, ln_gamma_clip=ln_gamma_clip)
        ln_m = nrtl_ln_gamma(x_m, T, g, alpha=alpha, R=R, tau_clip=tau_clip, ln_gamma_clip=ln_gamma_clip)
        d_ln = (ln_p - ln_m) / (2.0 * eps_fd)
        gd = (x * d_ln).sum(dim=-1)
        penalties.append(gd * gd)

    return torch.stack(penalties, dim=0).mean()


def stability_tpd_penalty(
    x: torch.Tensor,
    T: torch.Tensor,
    g: torch.Tensor,
    alpha: float,
    R: float,
    tau_clip: Optional[float],
    ln_gamma_clip: Optional[float],
    n_trial: int = 4,
    sigma: float = 0.05,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Tangent-plane distance (TPD) stability penalty: enforce TPD >= margin for random trials.
    
    论文中的TPD定义：
    TPD(w|x) = Σ w_i[ln w_i + ln γ_i(w)] - Σ w_i[ln x_i + ln γ_i(x)] ≥ 0
    
    注意：第二项求和权重也是w_i（试探组成），不是x_i
    稳定相要求所有试探点的TPD >= 0（或margin）
    损失函数：L_TPD = E_w[ReLU(-TPD)]，惩罚出现负TPD的不稳定方向
    """
    if x is None or x.numel() == 0 or n_trial <= 0:
        return torch.tensor(0.0, device=T.device if T is not None else "cpu")

    x = renorm3_torch(x)
    B = x.shape[0]
    eps = 1e-12

    # 计算当前组成x的活度系数
    ln_gx = nrtl_ln_gamma(x, T, g, alpha=alpha, R=R, tau_clip=tau_clip, ln_gamma_clip=ln_gamma_clip)

    # 生成随机试探组成w
    noise = torch.randn((n_trial, B, 3), device=x.device, dtype=x.dtype) * float(sigma)
    x_rep = x.unsqueeze(0).expand(n_trial, -1, -1)
    w = renorm3_torch(x_rep + noise)

    # 批量计算试探组成的活度系数
    w_flat = w.reshape(n_trial * B, 3)
    T_flat = T.repeat(n_trial)
    if g.ndim == 2:
        g_flat = g.unsqueeze(0).expand(n_trial * B, -1, -1)
    else:
        g_flat = g.repeat(n_trial, 1, 1)

    ln_gw = nrtl_ln_gamma(w_flat, T_flat, g_flat, alpha=alpha, R=R, tau_clip=tau_clip, ln_gamma_clip=ln_gamma_clip)
    ln_gw = ln_gw.view(n_trial, B, 3)

    # 计算TPD：论文公式
    # TPD = Σ w_i[ln w_i + ln γ_i(w)] - Σ w_i[ln x_i + ln γ_i(x)]
    log_w = torch.log(w.clamp_min(eps))
    log_x = torch.log(x.clamp_min(eps)).unsqueeze(0)  # (1, B, 3)
    ln_gx = ln_gx.unsqueeze(0)  # (1, B, 3)

    tpd = (w * (log_w + ln_gw - log_x - ln_gx)).sum(dim=-1)  # (n_trial, B)
    
    # 稳定条件: TPD >= margin
    # 违例: TPD < margin，惩罚 max(0, margin - TPD)
    viol = torch.relu(margin - tpd)
    
    # 返回平均违例（所有trial和batch的平均）
    return viol.mean()


# -------------------------
# Parameter store
# -------------------------

class NRTLParamStore:
    """
    Load per-system NRTL g_ij matrices from a json file like:
      {"meta": {"alpha":..., "R":..., "g_max":...}, "params": {"<system_id>": [[...],[...],[...]], ...}}
    """
    def __init__(self, json_path: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.json_path = json_path
        self.device = device
        self.dtype = dtype

        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        self.meta: Dict[str, Any] = obj.get("meta", {})
        raw: Dict[str, Any] = obj.get("params", {})

        # store as CPU tensors first, move on demand
        self._g_cpu: Dict[str, torch.Tensor] = {}
        for sid, mat in raw.items():
            t = torch.tensor(mat, dtype=dtype, device="cpu")
            if t.shape != (3, 3):
                continue
            self._g_cpu[str(sid)] = t

        self.alpha = float(self.meta.get("alpha", 0.3))
        self.R = float(self.meta.get("R", 8.314462618))

    def has(self, sid: Union[int, str, torch.Tensor]) -> bool:
        return _to_sid_str(sid) in self._g_cpu

    def get_g_one(self, sid: Union[int, str, torch.Tensor], device: Optional[str] = None) -> Optional[torch.Tensor]:
        key = _to_sid_str(sid)
        if key not in self._g_cpu:
            return None
        g = self._g_cpu[key]
        if device is None:
            device = self.device
        return g.to(device=device, dtype=self.dtype)

    def get_g_batch(
        self,
        sids: torch.Tensor,
        swap23: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        sids: (B,) torch.long (recommended)
        swap23: (B,) 0/1, optional

        Returns:
          g_batch: (B,3,3) or None if none found
          mask: (B,) bool, True where param exists
        """
        if device is None:
            device = self.device
        if sids.ndim != 1:
            sids = sids.view(-1)

        B = int(sids.shape[0])
        gs = []
        mask = torch.zeros((B,), dtype=torch.bool, device=device)
        for i in range(B):
            key = _to_sid_str(sids[i])
            g = self._g_cpu.get(key, None)
            if g is None:
                # placeholder
                gs.append(torch.zeros((3, 3), dtype=self.dtype, device=device))
            else:
                mask[i] = True
                gs.append(g.to(device=device, dtype=self.dtype))
        g_batch = torch.stack(gs, dim=0)  # (B,3,3)

        if swap23 is not None:
            if swap23.ndim != 1:
                swap23 = swap23.view(-1)
            swap23 = swap23.to(device=device)
            if swap23.numel() == B:
                # apply permutation only where swap23==1
                idx = (swap23 != 0).nonzero(as_tuple=False).view(-1)
                if idx.numel() > 0:
                    g_batch[idx] = _permute_23_g(g_batch[idx])

        if mask.any():
            return g_batch, mask
        return None, mask


# -------------------------
# Combined loss
# -------------------------

class MechanisticNRTLLoss(nn.Module):
    """
    Total loss:
      L = L_sup + lambda * L_phy
    where L_sup is MSE on (E+R) compositions, and L_phy is NRTL chemical-potential residual loss.

    Expected batch format (graph mode after small patch):
      x["scalars"]   : (B,2) where scalars[:,0] is T_norm, scalars[:,1] is t
      x["system_id"] : (B,) torch.long  (or something convertible)
      x["aug_swap23"]: (B,) torch.long  0/1

    If system_id is missing, physics term is skipped (L_phy=0).
    """
    def __init__(
        self,
        T_mean: float,
        T_std: float,
        nrtl_params_path: str,
        lambda_phy: float = 1e-3,
        warmup_epochs: int = 10,
        ramp_epochs: int = 10,
        robust_delta: float = 5.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        tau_clip: float = 10.0,
        ln_gamma_clip: float = 20.0,
        use_kelvin: Optional[bool] = None,
        w_eq: float = 1.0,
        w_gd: float = 0.10,
        w_stab: float = 0.10,
        gd_n_dir: int = 2,
        gd_eps: float = 1e-4,
        stab_n_trial: int = 4,
        stab_sigma: float = 0.05,
        stab_margin: float = 0.0,
    ):
        super().__init__()
        self.T_mean = float(T_mean)
        self.T_std = float(T_std)
        self.lambda_phy_target = float(lambda_phy)
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.ramp_epochs = int(max(1, ramp_epochs))
        self.robust_delta = float(max(1e-6, robust_delta))
        self.device = device
        self.dtype = dtype
        self.tau_clip = float(tau_clip)
        self.ln_gamma_clip = float(ln_gamma_clip)
        self.use_kelvin = use_kelvin  # None -> auto: if mean(T)<200 then +273.15
        self.w_eq = float(w_eq)
        self.w_gd = float(w_gd)
        self.w_stab = float(w_stab)
        self.gd_n_dir = int(gd_n_dir)
        self.gd_eps = float(gd_eps)
        self.stab_n_trial = int(stab_n_trial)
        self.stab_sigma = float(stab_sigma)
        self.stab_margin = float(stab_margin)

        self.store = NRTLParamStore(nrtl_params_path, device=device, dtype=dtype)
        self.alpha = float(self.store.alpha)
        self.R = float(self.store.R)

        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _lambda_now(self) -> float:
        e = self._epoch
        if e <= self.warmup_epochs:
            return 0.0
        # linear ramp
        t = min(1.0, (e - self.warmup_epochs) / float(self.ramp_epochs))
        return self.lambda_phy_target * t

    def compute_equilibrium_loss(
        self,
        xE: torch.Tensor,
        xR: torch.Tensor,
        T: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        r = nrtl_mu_residual(
            xE, xR, T, g,
            alpha=self.alpha, R=self.R,
            tau_clip=self.tau_clip, ln_gamma_clip=self.ln_gamma_clip
        )
        d = self.robust_delta
        abs_r = torch.abs(r)
        quad = torch.minimum(abs_r, torch.tensor(d, device=r.device, dtype=r.dtype))
        lin = abs_r - quad
        huber = 0.5 * quad * quad + d * lin
        return huber.mean()

    def forward(self, pred: torch.Tensor, y: torch.Tensor, x: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Returns a dict:
          {"loss": total, "sup": L_sup, "phy": L_phy, "lambda": current_lambda}
        """
        pred = pred.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)

        # supervised (MSE)
        L_sup = F_nn.mse_loss(pred, y, reduction="mean")

        lam = float(self._lambda_now())
        if (x is None) or (lam <= 0.0):
            return {"loss": L_sup, "sup": L_sup.detach(), "phy": torch.zeros_like(L_sup), "lambda": torch.tensor(lam, device=pred.device)}

        # need T, system_id for physics
        scalars = x.get("scalars", None) if isinstance(x, dict) else None
        sys_id = x.get("system_id", None) if isinstance(x, dict) else None

        if (scalars is None) or (sys_id is None):
            return {"loss": L_sup, "sup": L_sup.detach(), "phy": torch.zeros_like(L_sup), "lambda": torch.tensor(lam, device=pred.device)}

        scalars = scalars.to(device=pred.device, dtype=self.dtype)
        T = scalars[:, 0] * self.T_std + self.T_mean  # (B,)

        # optional kelvin conversion
        if self.use_kelvin is None:
            # heuristic: if temperature values look like Celsius
            if float(T.detach().mean().cpu().item()) < 200.0:
                T = T + 273.15
        elif bool(self.use_kelvin):
            # assume input is Celsius, convert to Kelvin
            T = T + 273.15

        swap23 = x.get("aug_swap23", None) if isinstance(x, dict) else None
        if swap23 is not None:
            swap23 = swap23.to(device=pred.device)

        g_batch, mask = self.store.get_g_batch(sys_id.to(device=pred.device), swap23=swap23, device=pred.device)
        if g_batch is None or (not mask.any()):
            return {"loss": L_sup, "sup": L_sup.detach(), "phy": torch.zeros_like(L_sup), "lambda": torch.tensor(lam, device=pred.device)}

        # predicted compositions -> (B,3) each for E and R
        xE = pred[:, 0:3]
        xR = pred[:, 3:6]
        xE = renorm3_torch(xE)
        xR = renorm3_torch(xR)

        # apply mask (only systems with params)
        xE = xE[mask]
        xR = xR[mask]
        T = T[mask]
        g_batch = g_batch[mask]

        if xE.numel() == 0:
            return {"loss": L_sup, "sup": L_sup.detach(), "phy": torch.zeros_like(L_sup), "lambda": torch.tensor(lam, device=pred.device)}

        L_eq = self.compute_equilibrium_loss(xE, xR, T, g_batch)

        L_gd = torch.tensor(0.0, device=xE.device, dtype=xE.dtype)
        if self.w_gd > 0:
            gd_E = gibbs_duhem_penalty(
                xE, T, g_batch, self.alpha, self.R,
                self.tau_clip, self.ln_gamma_clip,
                n_dir=self.gd_n_dir, eps_fd=self.gd_eps
            )
            gd_R = gibbs_duhem_penalty(
                xR, T, g_batch, self.alpha, self.R,
                self.tau_clip, self.ln_gamma_clip,
                n_dir=self.gd_n_dir, eps_fd=self.gd_eps
            )
            L_gd = 0.5 * (gd_E + gd_R)

        L_stab = torch.tensor(0.0, device=xE.device, dtype=xE.dtype)
        if self.w_stab > 0:
            st_E = stability_tpd_penalty(
                xE, T, g_batch, self.alpha, self.R,
                self.tau_clip, self.ln_gamma_clip,
                n_trial=self.stab_n_trial, sigma=self.stab_sigma, margin=self.stab_margin
            )
            st_R = stability_tpd_penalty(
                xR, T, g_batch, self.alpha, self.R,
                self.tau_clip, self.ln_gamma_clip,
                n_trial=self.stab_n_trial, sigma=self.stab_sigma, margin=self.stab_margin
            )
            L_stab = 0.5 * (st_E + st_R)

        L_phy = (self.w_eq * L_eq) + (self.w_gd * L_gd) + (self.w_stab * L_stab)

        total = L_sup + (lam * L_phy)
        return {"loss": total, "sup": L_sup.detach(), "phy": L_phy.detach(), "lambda": torch.tensor(lam, device=pred.device)}

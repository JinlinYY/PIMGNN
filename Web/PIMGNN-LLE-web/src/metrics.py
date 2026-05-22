# -*- coding: utf-8 -*-
"""
metrics.py - 模型评估指标计算

功能：
- 计算 MAE（平均绝对误差）、RMSE（均方根误差）、R²（确定系数）
- 支持全体和分组（Extract、Raffinate）评估
- 为训练过程中的模型评估提供指标计算
- 物理一致性指标：sum-to-one error, chemical potential residual, GD penalty, TPD stability
"""
from typing import Dict, Tuple, Any, Optional
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import batch_to_device


def _compute_gd_per_sample(
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
    Compute Gibbs-Duhem penalty per sample (not batch-averaged).
    Returns (B,) tensor where each element is the GD penalty for that sample.
    """
    from loss import renorm3_torch, nrtl_ln_gamma, _sample_simplex_directions
    
    if x is None or x.numel() == 0:
        return torch.tensor([], dtype=x.dtype if x is not None else torch.float32)
    
    x = renorm3_torch(x)
    B = x.shape[0]
    eps_fd = float(max(eps_fd, 1e-8))
    
    sample_penalties = torch.zeros(B, device=x.device, dtype=x.dtype)
    
    dirs = _sample_simplex_directions(x, int(n_dir))
    if dirs.numel() == 0:
        return sample_penalties
    
    for k in range(dirs.shape[0]):
        d = dirs[k]  # (B, 3)
        x_p = renorm3_torch(x + eps_fd * d)
        x_m = renorm3_torch(x - eps_fd * d)
        
        ln_p = nrtl_ln_gamma(x_p, T, g, alpha=alpha, R=R, tau_clip=tau_clip, ln_gamma_clip=ln_gamma_clip)
        ln_m = nrtl_ln_gamma(x_m, T, g, alpha=alpha, R=R, tau_clip=tau_clip, ln_gamma_clip=ln_gamma_clip)
        d_ln = (ln_p - ln_m) / (2.0 * eps_fd)  # (B, 3)
        gd = (x * d_ln).sum(dim=-1)  # (B,)
        sample_penalties = sample_penalties + gd * gd
    
    return sample_penalties / dirs.shape[0]


def _compute_tpd_per_sample(
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
    Compute TPD stability penalty per sample (not batch-averaged).
    Returns (B,) tensor where each element is the TPD violation for that sample.
    """
    from loss import renorm3_torch, nrtl_ln_gamma
    
    if x is None or x.numel() == 0 or n_trial <= 0:
        return torch.tensor([], dtype=x.dtype if x is not None else torch.float32)
    
    x = renorm3_torch(x)
    B = x.shape[0]
    eps = 1e-12
    
    ln_gx = nrtl_ln_gamma(x, T, g, alpha=alpha, R=R, tau_clip=tau_clip, ln_gamma_clip=ln_gamma_clip)
    
    sample_viols = torch.zeros(B, device=x.device, dtype=x.dtype)
    
    for trial in range(n_trial):
        noise = torch.randn((B, 3), device=x.device, dtype=x.dtype) * float(sigma)
        w = renorm3_torch(x + noise)
        
        ln_gw = nrtl_ln_gamma(w, T, g, alpha=alpha, R=R, tau_clip=tau_clip, ln_gamma_clip=ln_gamma_clip)
        
        log_w = torch.log(w.clamp_min(eps))
        log_x = torch.log(x.clamp_min(eps))
        
        tpd = (w * (log_w + ln_gw - log_x - ln_gx)).sum(dim=-1)  # (B,)
        viol = torch.relu(float(margin) - tpd)  # (B,)
        sample_viols = sample_viols + viol
    
    return sample_viols / n_trial



def calc_mae_rmse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    计算单个指标：MAE、RMSE、R²
    
    Args:
        y_true: 真实值数组，任意形状
        y_pred: 预测值数组，与 y_true 相同形状
    
    Returns:
        tuple: (mae, rmse, r2) 三个标量指标
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    # 计算 MAE（平均绝对误差）
    mae = float(np.mean(np.abs(yt - yp)))
    # 计算 RMSE（均方根误差）
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    # 计算 R²（决定系数）：1 - (残差平方和 / 总平方和)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)
    return mae, rmse, r2


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算完整的评估指标集合
    
    包括全体指标和分组指标（Extract、Raffinate 相分离）
    
    Args:
        y_true: 真实值，形状 (N, 6)，前3列为 Extract，后3列为 Raffinate
        y_pred: 预测值，形状 (N, 6)
    
    Returns:
        dict: 包含所有指标的字典
            - mse: 平均平方误差
            - rmse: 均方根误差（全体）
            - r2: 确定系数（全体）
            - mae: 平均绝对误差（全体）
            - rmse_E/rmse_R: Extract/Raffinate 的 RMSE
            - r2_E/r2_R: Extract/Raffinate 的 R²
            - mae_E/mae_R: Extract/Raffinate 的 MAE
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # 全体数据的指标
    mae_all, rmse_all, r2_all = calc_mae_rmse_r2(y_true, y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))

    # Extract 相的指标（前3列）
    y_true_E, y_pred_E = y_true[:, :3], y_pred[:, :3]
    # Raffinate 相的指标（后3列）
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
    """
    从 DataLoader 中收集模型的预测和真实标签
    
    Args:
        model: 评估模式下的神经网络模型
        loader: PyTorch DataLoader
        device: 计算设备（'cpu' 或 'cuda'）
    
    Returns:
        tuple: (y_true, y_pred) 两个 numpy 数组
    """
    model.eval()
    ys, ps = [], []
    # 遍历 DataLoader，逐批进行预测
    for x, y in loader:
        # 将数据移到指定设备
        x = batch_to_device(x, device)
        y = batch_to_device(y, device)
        # 获取模型预测
        pred = model(x)
        # 收集真实值和预测值
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    # 沿批维度拼接所有结果
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """
    在整个 DataLoader 上评估模型，计算各项指标
    
    Args:
        model: 要评估的神经网络模型
        loader: 评估数据加载器
        device: 计算设备
    
    Returns:
        dict: 评估指标字典（包含 mae、rmse、r2 等）
    """
    y_true, y_pred = collect_preds(model, loader, device)
    return compute_metrics(y_true, y_pred)


@torch.no_grad()
def compute_physics_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    nrtl_store: Optional[Any] = None,
    T_mean: float = 0.0,
    T_std: float = 1.0,
    use_kelvin: Optional[bool] = None,
) -> Dict[str, float]:
    """
    计算物理一致性指标（需要 NRTL 参数）
    
    Args:
        model: 评估模型
        loader: 数据加载器
        device: 计算设备
        nrtl_store: NRTLParamStore 实例（若为 None 则只计算基础物理指标）
        T_mean/T_std: 温度标准化参数
        use_kelvin: 温度单位处理（None自动/True强制转K/False不转）
    
    Returns:
        dict: 物理一致性指标
            - sum_err_E/R: Extract/Raffinate 相的 sum-to-one 平均绝对误差
            - sum_err_95: sum-to-one 误差的 95 分位数
            - neg_frac: 预测为负的分数比例
            - param_cov: 有 NRTL 参数的样本覆盖率
            - mu_res_mae/rmse: 化学势残差的 MAE/RMSE（需要 NRTL）
            - gd_penalty_mean/p95: Gibbs-Duhem 约束违例（需要 NRTL）
            - tpd_viol_rate/mean: TPD 稳定性违例率和平均违例（需要 NRTL）
    """
    from loss import (
        nrtl_mu_residual, gibbs_duhem_penalty, stability_tpd_penalty,
        renorm3_torch, NRTLParamStore, _permute_23_g
    )
    
    model.eval()
    
    sum_errs_E, sum_errs_R = [], []
    neg_counts = []
    all_mu_res = []
    all_gd = []
    all_tpd = []
    total_samples = 0
    param_samples = 0
    
    for x, y in loader:
        x = batch_to_device(x, device)
        y = batch_to_device(y, device)
        pred = model(x)
        
        B = pred.shape[0]
        total_samples += B
        
        # 1. Sum-to-one error
        pred_E = pred[:, :3]
        pred_R = pred[:, 3:6]
        err_E = torch.abs(pred_E.sum(dim=1) - 1.0).detach().cpu().numpy()
        err_R = torch.abs(pred_R.sum(dim=1) - 1.0).detach().cpu().numpy()
        sum_errs_E.append(err_E)
        sum_errs_R.append(err_R)
        
        # 2. Negative fraction
        neg_count = (pred < 0).sum().item()
        neg_counts.append(neg_count)
        
        # 3. Physics metrics (need NRTL params)
        if nrtl_store is not None and isinstance(x, dict):
            scalars = x.get("scalars")
            sys_id = x.get("system_id")
            swap23 = x.get("aug_swap23")
            
            if scalars is not None and sys_id is not None:
                T = scalars[:, 0] * T_std + T_mean
                
                # Temperature unit handling
                if use_kelvin is None:
                    if float(T.mean().cpu().item()) < 200.0:
                        T = T + 273.15
                elif bool(use_kelvin):
                    T = T + 273.15
                
                # Get NRTL params
                g_batch, mask = nrtl_store.get_g_batch(sys_id, swap23=swap23, device=device)
                
                if g_batch is not None and mask.any():
                    param_samples += mask.sum().item()
                    
                    xE = renorm3_torch(pred[:, :3])
                    xR = renorm3_torch(pred[:, 3:6])
                    
                    xE_m = xE[mask]
                    xR_m = xR[mask]
                    T_m = T[mask]
                    g_m = g_batch[mask]
                    
                    # Chemical potential residual
                    r = nrtl_mu_residual(
                        xE_m, xR_m, T_m, g_m,
                        alpha=nrtl_store.alpha, R=nrtl_store.R,
                        tau_clip=10.0, ln_gamma_clip=20.0
                    )
                    all_mu_res.append(r.detach().cpu().numpy())
                    
                    # Gibbs-Duhem penalty (per-sample calculation)
                    # Calculate GD for each sample separately (vectorized)
                    gd_E_samples = _compute_gd_per_sample(
                        xE_m, T_m, g_m, nrtl_store.alpha, nrtl_store.R,
                        tau_clip=10.0, ln_gamma_clip=20.0, n_dir=2, eps_fd=1e-4
                    )
                    gd_R_samples = _compute_gd_per_sample(
                        xR_m, T_m, g_m, nrtl_store.alpha, nrtl_store.R,
                        tau_clip=10.0, ln_gamma_clip=20.0, n_dir=2, eps_fd=1e-4
                    )
                    gd_samples = 0.5 * (gd_E_samples + gd_R_samples)
                    all_gd.append(gd_samples.detach().cpu().numpy())
                    
                    # TPD stability (per-sample calculation)
                    tpd_E_samples = _compute_tpd_per_sample(
                        xE_m, T_m, g_m, nrtl_store.alpha, nrtl_store.R,
                        tau_clip=10.0, ln_gamma_clip=20.0,
                        n_trial=4, sigma=0.05, margin=0.0
                    )
                    tpd_R_samples = _compute_tpd_per_sample(
                        xR_m, T_m, g_m, nrtl_store.alpha, nrtl_store.R,
                        tau_clip=10.0, ln_gamma_clip=20.0,
                        n_trial=4, sigma=0.05, margin=0.0
                    )
                    tpd_samples = 0.5 * (tpd_E_samples + tpd_R_samples)
                    all_tpd.append(tpd_samples.detach().cpu().numpy())
    
    # Aggregate results
    sum_errs_E = np.concatenate(sum_errs_E)
    sum_errs_R = np.concatenate(sum_errs_R)
    sum_errs_all = np.concatenate([sum_errs_E, sum_errs_R])
    
    metrics = {
        "sum_err_E": float(np.mean(sum_errs_E)),
        "sum_err_R": float(np.mean(sum_errs_R)),
        "sum_err_95": float(np.percentile(sum_errs_all, 95)),
        "neg_frac": float(sum(neg_counts) / (total_samples * 6)) if total_samples > 0 else 0.0,
        "param_cov": float(param_samples / total_samples) if total_samples > 0 else 0.0,
    }
    
    # Physics metrics (only if we have NRTL params)
    if len(all_mu_res) > 0:
        mu_res = np.concatenate(all_mu_res).reshape(-1)
        metrics["mu_res_mae"] = float(np.mean(np.abs(mu_res)))
        metrics["mu_res_rmse"] = float(np.sqrt(np.mean(mu_res ** 2)))
        # 新增：化学势残差最大值（绝对值最大）
        metrics["mu_res_max"] = float(np.max(np.abs(mu_res)))
    else:
        metrics["mu_res_mae"] = float("nan")
        metrics["mu_res_rmse"] = float("nan")
        metrics["mu_res_max"] = float("nan")
    
    if len(all_gd) > 0:
        gd_arr = np.concatenate(all_gd)
        metrics["gd_penalty_mean"] = float(np.mean(gd_arr))
        metrics["gd_penalty_p95"] = float(np.percentile(gd_arr, 95))
        # 新增：GD残差MAE（与绘图键gd_res_mae匹配）
        metrics["gd_res_mae"] = float(np.mean(np.abs(gd_arr)))
    else:
        metrics["gd_penalty_mean"] = float("nan")
        metrics["gd_penalty_p95"] = float("nan")
        metrics["gd_res_mae"] = float("nan")
    
    if len(all_tpd) > 0:
        tpd_arr = np.concatenate(all_tpd)
        metrics["tpd_viol_rate"] = float(np.mean(tpd_arr > 0))
        metrics["tpd_viol_mean"] = float(np.mean(tpd_arr[tpd_arr > 0])) if np.any(tpd_arr > 0) else 0.0
    else:
        metrics["tpd_viol_rate"] = float("nan")
        metrics["tpd_viol_mean"] = float("nan")
    
    return metrics


def print_metrics(prefix: str, m: Dict[str, float]) -> None:
    """
    打印格式化的评估指标
    
    Args:
        prefix: 输出前缀（通常为 'Train', 'Val', 'Test' 等）
        m: 指标字典（来自 compute_metrics）
    """
    print(
        f"{prefix} "
        # 全体数据指标
        f"MAE={m.get('mae', float('nan')):.4f} "
        f"MSE={m['mse']:.4f} RMSE={m['rmse']:.4f} R2={m['r2']:.4f} | "
        # Extract 相指标
        f"Ex: MAE={m.get('mae_E', float('nan')):.4f} RMSE={m['rmse_E']:.4f} R2={m['r2_E']:.4f} | "
        # Raffinate 相指标
        f"Rx: MAE={m.get('mae_R', float('nan')):.4f} RMSE={m['rmse_R']:.4f} R2={m['r2_R']:.4f}"
    )

# -*- coding: utf-8 -*-
"""Implement the mmgnn train baseline module."""

import os
import json
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Load the input data.
# Baseline workflow step.
warnings.filterwarnings('ignore', message='.*torch_geometric.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pyg-lib.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch-scatter.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch-cluster.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch-spline-conv.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch-sparse.*', category=UserWarning)

from psmi_baselines.common.config import DEVICE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, SEED
from psmi_baselines.common.utils import set_seed, Scaler
from psmi_baselines.common.data import load_and_prepare_excel, split_by_system
from psmi_baselines.common.metrics import compute_metrics, print_metrics

from .model import MMGNN
from .dataset import MMGNNDataset, collate_fn
from psmi_baselines.paths import DATA_DIR, EXPERIMENT_ROOT


class EarlyStopping:
    """Represent the EarlyStopping baseline component."""
    def __init__(self, patience: int = 100, min_delta: float = 0.0, mode: str = 'min'):
        """Run the init baseline operation."""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Run the call baseline operation."""
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


def train_epoch(model, train_loader, optimizer, criterion, device, use_amp=True):
    """Run the train epoch baseline operation."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.startswith('cuda'))
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_graphs, T_norm, t, y in pbar:
        batch_graphs = batch_graphs.to(device)
        T_norm = T_norm.to(device)
        t = t.to(device)
        y = y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp and device.startswith('cuda')):
            pred = model(batch_graphs, T_norm, t)
            loss = criterion(pred, y)
        
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
        
        pbar.set_postfix({'loss': loss.item()})
        
        # Baseline workflow step.
        del pred, loss
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Run the evaluate baseline operation."""
    model.eval()
    all_preds = []
    all_targets = []
    
    for batch_graphs, T_norm, t, y in loader:
        batch_graphs = batch_graphs.to(device)
        T_norm = T_norm.to(device)
        t = t.to(device)
        y = y.to(device)
        
        pred = model(batch_graphs, T_norm, t)
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        
        # Baseline workflow step.
        del pred
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_metrics(all_targets, all_preds)
    return metrics, all_preds, all_targets


@torch.no_grad()
def predict_test_set(model, test_dataset, device):
    """Run the predict test set baseline operation."""
    model.eval()
    all_preds = []
    all_targets = []
    
    from MMGNN.dataset import collate_fn
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,  # Generate model predictions.
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    for batch_graphs, T_norm, t, y in test_loader:
        batch_graphs = batch_graphs.to(device)
        T_norm = T_norm.to(device)
        t = t.to(device)
        y = y.to(device)
        
        pred = model(batch_graphs, T_norm, t)
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return all_preds, all_targets


def compute_per_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Run the compute per sample metrics baseline operation."""
    n = y_true.shape[0]
    metrics = np.zeros((n, 12), dtype=np.float32)
    
    # Compute evaluation metrics.
    diff_all = y_true - y_pred
    metrics[:, 0] = np.mean(diff_all ** 2, axis=1)  # MSE_all
    metrics[:, 1] = np.mean(np.abs(diff_all), axis=1)  # MAE_all
    metrics[:, 3] = np.sqrt(metrics[:, 0])  # RMSE_all
    
    # Baseline workflow step.
    y_true_mean = np.mean(y_true, axis=1, keepdims=True)
    ss_res = np.sum(diff_all ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true_mean) ** 2, axis=1)
    metrics[:, 2] = 1.0 - ss_res / (ss_tot + 1e-12)  # R2_all
    
    # Compute evaluation metrics.
    diff_E = y_true[:, :3] - y_pred[:, :3]
    metrics[:, 4] = np.mean(diff_E ** 2, axis=1)  # MSE_E
    metrics[:, 5] = np.mean(np.abs(diff_E), axis=1)  # MAE_E
    metrics[:, 7] = np.sqrt(metrics[:, 4])  # RMSE_E
    
    y_true_E_mean = np.mean(y_true[:, :3], axis=1, keepdims=True)
    ss_res_E = np.sum(diff_E ** 2, axis=1)
    ss_tot_E = np.sum((y_true[:, :3] - y_true_E_mean) ** 2, axis=1)
    metrics[:, 6] = 1.0 - ss_res_E / (ss_tot_E + 1e-12)  # R2_E
    
    # Compute evaluation metrics.
    diff_R = y_true[:, 3:] - y_pred[:, 3:]
    metrics[:, 8] = np.mean(diff_R ** 2, axis=1)  # MSE_R
    metrics[:, 9] = np.mean(np.abs(diff_R), axis=1)  # MAE_R
    metrics[:, 11] = np.sqrt(metrics[:, 8])  # RMSE_R
    
    y_true_R_mean = np.mean(y_true[:, 3:], axis=1, keepdims=True)
    ss_res_R = np.sum(diff_R ** 2, axis=1)
    ss_tot_R = np.sum((y_true[:, 3:] - y_true_R_mean) ** 2, axis=1)
    metrics[:, 10] = 1.0 - ss_res_R / (ss_tot_R + 1e-12)  # R2_R
    
    return metrics


def format_metrics_detailed(metrics: Dict[str, float], prefix: str = "") -> str:
    """Run the format metrics detailed baseline operation."""
    lines = [
        f"{prefix}【平均指标】 MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}  R²={metrics['r2']:.6f}",
        f"{prefix}【E相指标】  MAE={metrics['mae_E']:.6f}  RMSE={metrics['rmse_E']:.6f}  R²={metrics['r2_E']:.6f}",
        f"{prefix}【R相指标】  MAE={metrics['mae_R']:.6f}  RMSE={metrics['rmse_R']:.6f}  R²={metrics['r2_R']:.6f}",
    ]
    return "\n".join(lines)


def train_mmgnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    hidden_dim: int = 256,
    num_layers: int = 3,
    set2set_steps: int = 3,  # Baseline workflow step.
    post_explain_layers: int = 2,  # Baseline workflow step.
    beta: float = 0.2,
    explainer_method: str = 'local_mask',
    dropout: float = 0.15,
    patience: int = 100,  # Apply early stopping.
    min_delta: float = 0.0,  # Apply early stopping.
    resume_from: str = None,  # Handle model checkpoints.
    save_checkpoint_every: int = 10,  # Save the generated artifacts.
    rest_interval: int = 60,  # Baseline workflow step.
    rest_duration: float = 600.0,  # Baseline workflow step.
) -> Tuple[nn.Module, Scaler, Dict]:
    """Run the train mmgnn baseline operation."""
    
    # Configure the output artifacts.
    print("\n" + "=" * 100)
    print("【训练配置参数】")
    print("=" * 100)
    
    # Process the experiment data.
    print("\n【数据集信息】")
    print(f"  训练集样本数: {len(train_df)}")
    print(f"  验证集样本数: {len(val_df)}")
    print(f"  测试集样本数: {len(test_df)}")
    print(f"  训练集体系数: {train_df['system_id'].nunique()}")
    print(f"  验证集体系数: {val_df['system_id'].nunique()}")
    print(f"  测试集体系数: {test_df['system_id'].nunique()}")
    
    # Configure the runtime device.
    print("\n【设备配置】")
    print(f"  设备: {device}")
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("  警告: CUDA不可用，将使用CPU")
    
    # Run the training step.
    print("\n【训练参数】")
    print(f"  随机种子: 2024")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {lr}")
    print(f"  权重衰减: {weight_decay}")
    print(f"  早停耐心值: {patience}")
    print(f"  早停最小改善: {min_delta}")
    print(f"  检查点保存频率: 每 {save_checkpoint_every} 个epoch")
    print(f"  休息策略: 每 {rest_interval} 个epoch休息 {rest_duration}秒（{rest_duration/60:.1f}分钟）")
    if resume_from:
        print(f"  断点续训: {resume_from}")
    else:
        print(f"  断点续训: 否（从头开始）")
    
    # Configure the baseline model.
    print("\n【模型超参数】")
    print(f"  隐藏层维度: {hidden_dim}")
    print(f"  图神经网络层数: {num_layers}")
    print(f"  Set2Set步骤数: {set2set_steps}")
    print(f"  解释后层数: {post_explain_layers}")
    print(f"  Beta参数: {beta}")
    print(f"  解释器方法: {explainer_method}")
    print(f"  Dropout率: {dropout}")
    print(f"  输出维度: 6 (LLE任务: Ex1, Ex2, Ex3, Rx1, Rx2, Rx3)")
    
    # Configure repository paths.
    print("\n【路径信息】")
    print(f"  输出目录: {out_dir}")
    print(f"  结果目录: MMGNN/results")
    
    print("\n" + "=" * 100)
    print("配置参数输出完成，开始初始化训练...")
    print("=" * 100 + "\n")
    # ====================================
    
    # Configure repository paths.
    # Configure repository paths.
    results_dir = "MMGNN/results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    # Set the random seed.
    seed_value = 2024  # Baseline workflow step.
    set_seed(seed_value)
    
    # Baseline workflow step.
    T_scaler = Scaler.fit(train_df["T"].values.astype(np.float32))
    
    # Process the experiment data.
    from MMGNN.dataset import MMGNNDataset
    
    train_dataset = MMGNNDataset(train_df, T_scaler, precompute=True)
    val_dataset = MMGNNDataset(val_df, T_scaler, precompute=True)
    test_dataset = MMGNNDataset(test_df, T_scaler, precompute=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Baseline workflow step.
        pin_memory=device.startswith('cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.startswith('cuda')
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.startswith('cuda')
    )
    
    # Handle model checkpoints.
    checkpoint_config = None
    checkpoint = None
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            print("=" * 100)
            print(f"检查点中找到配置信息，将使用检查点的配置:")
            print(f"  hidden_dim: {checkpoint_config.get('hidden_dim', hidden_dim)}")
            print(f"  num_layers: {checkpoint_config.get('num_layers', num_layers)}")
            print(f"  set2set_steps: {checkpoint_config.get('set2set_steps', set2set_steps)}")
            print(f"  post_explain_layers: {checkpoint_config.get('post_explain_layers', post_explain_layers)}")
            print("=" * 100)
            # Handle model checkpoints.
            hidden_dim = checkpoint_config.get('hidden_dim', hidden_dim)
            num_layers = checkpoint_config.get('num_layers', num_layers)
            set2set_steps = checkpoint_config.get('set2set_steps', set2set_steps)
            post_explain_layers = checkpoint_config.get('post_explain_layers', post_explain_layers)
            beta = checkpoint_config.get('beta', beta)
            explainer_method = checkpoint_config.get('explainer_method', explainer_method)
            dropout = checkpoint_config.get('dropout', dropout)
            # Handle model checkpoints.
            if 'lr' in checkpoint_config:
                lr = checkpoint_config.get('lr', lr)
            if 'weight_decay' in checkpoint_config:
                weight_decay = checkpoint_config.get('weight_decay', weight_decay)
    
    # Configure the baseline model.
    model = MMGNN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        set2set_steps=set2set_steps,
        post_explain_layers=post_explain_layers,
        beta=beta,
        explainer_method=explainer_method,
        dropout=dropout,
        output_dim=6,  # Configure the output artifacts.
    ).to(device)
    
    # Compute the training loss.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Run the training step.
    history = {
        'epoch': [],
        'train_loss': [],
        'epoch_time': [],  # Baseline workflow step.
        # Run the training step.
        'train_mse': [], 'train_rmse': [], 'train_mae': [], 'train_r2': [],
        'train_mae_E': [], 'train_rmse_E': [], 'train_r2_E': [],
        'train_mae_R': [], 'train_rmse_R': [], 'train_r2_R': [],
        # Evaluate the validation subset.
        'val_mse': [], 'val_rmse': [], 'val_mae': [], 'val_r2': [],
        'val_mae_E': [], 'val_rmse_E': [], 'val_r2_E': [],
        'val_mae_R': [], 'val_rmse_R': [], 'val_r2_R': [],
    }
    
    best_val_mse = float('inf')
    best_model_state = None
    best_epoch = -1
    start_epoch = 1
    
    # Apply early stopping.
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='min')
    
    # Load the input data.
    if resume_from and os.path.exists(resume_from):
        # Load the input data.
        if checkpoint is None:
            checkpoint = torch.load(resume_from, map_location=device)
        
        print("=" * 100)
        print(f"从检查点恢复训练: {resume_from}")
        print("=" * 100)
        print("=" * 100)
        print(f"从检查点恢复训练: {resume_from}")
        print("=" * 100)
        
        # Load the input data.
        if 'state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                print("✓ 模型状态加载成功（完全匹配）")
            except RuntimeError as e:
                print(f"警告: 完全加载失败，尝试部分加载: {e}")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
                if missing_keys:
                    print(f"  缺失的键: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  缺失的键: {missing_keys}")
                if unexpected_keys:
                    print(f"  多余的键: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  多余的键: {unexpected_keys}")
                print("✓ 模型状态加载成功（部分匹配）")
        elif 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                print("✓ 模型状态加载成功（完全匹配）")
            except RuntimeError as e:
                print(f"警告: 完全加载失败，尝试部分加载: {e}")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing_keys:
                    print(f"  缺失的键: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  缺失的键: {missing_keys}")
                if unexpected_keys:
                    print(f"  多余的键: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  多余的键: {unexpected_keys}")
                print("✓ 模型状态加载成功（部分匹配）")
        
        # Load the input data.
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ 优化器状态加载成功")
            except (ValueError, RuntimeError) as e:
                print(f"警告: 优化器状态加载失败（模型结构可能已改变），将使用新的优化器: {e}")
                # Baseline workflow step.
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                print("✓ 已创建新的优化器")
        
        # Load the input data.
        if 'history' in checkpoint:
            history = checkpoint['history']
        
        # Load the input data.
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        elif 'current_epoch' in checkpoint:
            start_epoch = checkpoint['current_epoch'] + 1
        
        # Load the input data.
        if 'best_val_mse' in checkpoint:
            best_val_mse = checkpoint['best_val_mse']
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']
        if 'best_model_state' in checkpoint and checkpoint['best_model_state'] is not None:
            # Baseline workflow step.
            best_model_state = checkpoint['best_model_state']
            # Configure the runtime device.
            if isinstance(best_model_state, dict):
                best_model_state = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in best_model_state.items()}
        
        # Load the input data.
        if 'T_scaler' in checkpoint:
            T_scaler = Scaler.from_state_dict(checkpoint['T_scaler'])
        
        # Load the input data.
        if 'early_stopping' in checkpoint:
            early_stopping.counter = checkpoint['early_stopping'].get('counter', 0)
            early_stopping.best_score = checkpoint['early_stopping'].get('best_score', float('inf'))
            early_stopping.early_stop = checkpoint['early_stopping'].get('early_stop', False)
        
        print(f"已恢复训练状态:")
        print(f"  - 从epoch {start_epoch} 继续训练")
        print(f"  - 最佳epoch: {best_epoch}, 最佳验证集MSE: {best_val_mse:.6f}")
        print(f"  - 已训练轮数: {len(history.get('epoch', []))}")
        print("=" * 100)
    elif resume_from:
        print(f"警告: 指定的检查点文件不存在: {resume_from}")
        print("将从头开始训练...")
    
    print("=" * 100)
    print("训练配置信息")
    print("=" * 100)
    print(f"设备: {device}")
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("  警告: CUDA不可用，将使用CPU")
    print(f"随机种子: 2024")
    print(f"训练轮数: {epochs}")
    if start_epoch > 1:
        print(f"从epoch {start_epoch} 继续训练（剩余 {epochs - start_epoch + 1} 轮）")
    else:
        print(f"从头开始训练（共 {epochs} 轮）")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"权重衰减: {weight_decay}")
    print(f"早停耐心值: {patience}")
    print(f"检查点保存频率: 每 {save_checkpoint_every} 个epoch")
    print("=" * 100)
    print("开始训练...")
    print("=" * 100)
    
    # Run the training step.
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        
        # Run the training step.
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Run the training step.
        train_metrics, train_preds, train_targets = evaluate(model, train_loader, device)
        val_metrics, val_preds, val_targets = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Baseline workflow step.
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['epoch_time'].append(epoch_time)
        # Run the training step.
        history['train_mse'].append(train_metrics['mse'])
        history['train_rmse'].append(train_metrics['rmse'])
        history['train_mae'].append(train_metrics['mae'])
        history['train_r2'].append(train_metrics['r2'])
        history['train_mae_E'].append(train_metrics['mae_E'])
        history['train_rmse_E'].append(train_metrics['rmse_E'])
        history['train_r2_E'].append(train_metrics['r2_E'])
        history['train_mae_R'].append(train_metrics['mae_R'])
        history['train_rmse_R'].append(train_metrics['rmse_R'])
        history['train_r2_R'].append(train_metrics['r2_R'])
        # Evaluate the validation subset.
        history['val_mse'].append(val_metrics['mse'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['val_mae_E'].append(val_metrics['mae_E'])
        history['val_rmse_E'].append(val_metrics['rmse_E'])
        history['val_r2_E'].append(val_metrics['r2_E'])
        history['val_mae_R'].append(val_metrics['mae_R'])
        history['val_rmse_R'].append(val_metrics['rmse_R'])
        history['val_r2_R'].append(val_metrics['r2_R'])
        
        # Save the generated artifacts.
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        
        # Baseline workflow step.
        print(f"\n{'='*100}")
        print(f"Epoch {epoch}/{epochs} | 训练时间: {epoch_time:.2f}秒 | Train Loss: {train_loss:.6f}")
        print(f"Best Loss: {best_val_mse:.6f} (at epoch {best_epoch})" if best_epoch > 0 else f"Best Loss: {best_val_mse:.6f} (initial)")
        print(f"{'='*100}")
        print("【训练集指标】")
        print(format_metrics_detailed(train_metrics, "  "))
        print("\n【验证集指标】")
        print(format_metrics_detailed(val_metrics, "  "))
        print(f"{'='*100}\n")
        
        # Save the generated artifacts.
        # Process the experiment data.
        data_cols = ['system_id', 'T', 'smiles1', 'smiles2', 'smiles3', 
                     'Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3', 't']
        
        # Run the training step.
        available_train_cols = [col for col in data_cols if col in train_df.columns]
        if len(available_train_cols) < len(data_cols):
            other_cols = [col for col in train_df.columns 
                         if col not in ['aug_swap23'] and col not in available_train_cols
                         and not col.startswith('pred_')]
            available_train_cols.extend(other_cols)
        
        train_df_with_preds = train_df[available_train_cols].copy()
        
        # Run the training step.
        train_df_with_preds['pred_Ex1'] = train_preds[:, 0]
        train_df_with_preds['pred_Ex2'] = train_preds[:, 1]
        train_df_with_preds['pred_Ex3'] = train_preds[:, 2]
        train_df_with_preds['pred_Rx1'] = train_preds[:, 3]
        train_df_with_preds['pred_Rx2'] = train_preds[:, 4]
        train_df_with_preds['pred_Rx3'] = train_preds[:, 5]
        
        # Run the training step.
        train_per_sample_metrics = compute_per_sample_metrics(train_targets, train_preds)
        
        # Run the training step.
        train_df_with_preds['train_MSE_E'] = train_per_sample_metrics[:, 4]
        train_df_with_preds['train_MAE_E'] = train_per_sample_metrics[:, 5]
        train_df_with_preds['train_R2_E'] = train_per_sample_metrics[:, 6]
        train_df_with_preds['train_RMSE_E'] = train_per_sample_metrics[:, 7]
        train_df_with_preds['train_MSE_R'] = train_per_sample_metrics[:, 8]
        train_df_with_preds['train_MAE_R'] = train_per_sample_metrics[:, 9]
        train_df_with_preds['train_R2_R'] = train_per_sample_metrics[:, 10]
        train_df_with_preds['train_RMSE_R'] = train_per_sample_metrics[:, 11]
        train_df_with_preds['train_MSE_all'] = train_per_sample_metrics[:, 0]
        train_df_with_preds['train_MAE_all'] = train_per_sample_metrics[:, 1]
        train_df_with_preds['train_R2_all'] = train_per_sample_metrics[:, 2]
        train_df_with_preds['train_RMSE_all'] = train_per_sample_metrics[:, 3]
        
        # Evaluate the validation subset.
        available_val_cols = [col for col in data_cols if col in val_df.columns]
        if len(available_val_cols) < len(data_cols):
            other_cols = [col for col in val_df.columns 
                         if col not in ['aug_swap23'] and col not in available_val_cols
                         and not col.startswith('pred_')]
            available_val_cols.extend(other_cols)
        
        val_df_with_preds = val_df[available_val_cols].copy()
        
        # Evaluate the validation subset.
        val_df_with_preds['pred_Ex1'] = val_preds[:, 0]
        val_df_with_preds['pred_Ex2'] = val_preds[:, 1]
        val_df_with_preds['pred_Ex3'] = val_preds[:, 2]
        val_df_with_preds['pred_Rx1'] = val_preds[:, 3]
        val_df_with_preds['pred_Rx2'] = val_preds[:, 4]
        val_df_with_preds['pred_Rx3'] = val_preds[:, 5]
        
        # Evaluate the validation subset.
        val_per_sample_metrics = compute_per_sample_metrics(val_targets, val_preds)
        
        # Evaluate the validation subset.
        val_df_with_preds['valid_MSE_E'] = val_per_sample_metrics[:, 4]
        val_df_with_preds['valid_MAE_E'] = val_per_sample_metrics[:, 5]
        val_df_with_preds['valid_R2_E'] = val_per_sample_metrics[:, 6]
        val_df_with_preds['valid_RMSE_E'] = val_per_sample_metrics[:, 7]
        val_df_with_preds['valid_MSE_R'] = val_per_sample_metrics[:, 8]
        val_df_with_preds['valid_MAE_R'] = val_per_sample_metrics[:, 9]
        val_df_with_preds['valid_R2_R'] = val_per_sample_metrics[:, 10]
        val_df_with_preds['valid_RMSE_R'] = val_per_sample_metrics[:, 11]
        val_df_with_preds['valid_MSE_all'] = val_per_sample_metrics[:, 0]
        val_df_with_preds['valid_MAE_all'] = val_per_sample_metrics[:, 1]
        val_df_with_preds['valid_R2_all'] = val_per_sample_metrics[:, 2]
        val_df_with_preds['valid_RMSE_all'] = val_per_sample_metrics[:, 3]
        
        # Save the generated artifacts.
        combined_df = pd.concat([train_df_with_preds, val_df_with_preds], ignore_index=True)
        
        # Save the generated artifacts.
        results_csv = os.path.join(results_dir, "training_results.csv")
        if epoch == 1:
            combined_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
        else:
            combined_df.to_csv(results_csv, mode='a', header=False, index=False, encoding='utf-8-sig')
        
        # Save the generated artifacts.
        if epoch % save_checkpoint_every == 0 or epoch == epochs:
            checkpoint_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'T_scaler': T_scaler.state_dict(),
                'history': history,
                'best_val_mse': best_val_mse,
                'best_epoch': best_epoch,
                'best_model_state': best_model_state,
                'early_stopping': {
                    'counter': early_stopping.counter,
                    'best_score': early_stopping.best_score,
                    'early_stop': early_stopping.early_stop,
                },
                'config': {
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'set2set_steps': set2set_steps,
                    'post_explain_layers': post_explain_layers,
                    'beta': beta,
                    'explainer_method': explainer_method,
                    'dropout': dropout,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size,
                    'patience': patience,
                    'min_delta': min_delta,
                }
            }, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
        
        # Baseline workflow step.
        if rest_interval > 0 and rest_duration > 0 and epoch % rest_interval == 0:
            print(f"\n{'='*100}")
            print(f"已完成 {epoch} 个epoch，休息 {rest_duration}秒（{rest_duration/60:.1f}分钟）让CPU/GPU有时间休息...")
            print(f"{'='*100}\n")
            time.sleep(rest_duration)
        
        # Apply early stopping.
        if early_stopping(val_metrics['mse']):
            print(f"\n早停触发！在epoch {epoch}停止训练。")
            print(f"最佳模型在epoch {best_epoch}，验证集MSE: {best_val_mse:.6f}")
            print(f"已等待 {early_stopping.counter}/{early_stopping.patience} 个epoch无改善")
            break
    
    # Load the input data.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch} (val_mse={best_val_mse:.6f})")
    
    # Save the generated artifacts.
    ckpt_path = os.path.join(out_dir, "mmgnn.pt")
    torch.save({
        'state_dict': model.state_dict(),
        'T_scaler': T_scaler.state_dict(),
        'best_epoch': best_epoch,
        'best_val_mse': best_val_mse,
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'set2set_steps': set2set_steps,
            'post_explain_layers': post_explain_layers,
            'beta': beta,
            'explainer_method': explainer_method,
            'dropout': dropout,
        }
    }, ckpt_path)
    print(f"Saved model to {ckpt_path}")
    
    # Save the generated artifacts.
    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, "train_history.csv"),
        index=False,
        encoding='utf-8-sig'
    )
    
    # Configure the baseline model.
    final_val_metrics, _, _ = evaluate(model, val_loader, device)
    
    # Run the training step.
    print("\n" + "="*100)
    print("训练完成，开始计算测试集指标...")
    print("="*100)
    final_test_metrics, test_preds, test_targets = evaluate(model, test_loader, device)
    
    # Save the generated artifacts.
    data_cols = ['system_id', 'T', 'smiles1', 'smiles2', 'smiles3', 
                 'Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3', 't']
    available_test_cols = [col for col in data_cols if col in test_df.columns]
    if len(available_test_cols) < len(data_cols):
        other_cols = [col for col in test_df.columns 
                     if col not in ['aug_swap23'] and col not in available_test_cols
                     and not col.startswith('pred_')]
        available_test_cols.extend(other_cols)
    
    test_df_with_preds = test_df[available_test_cols].copy()
    
    # Evaluate the test subset.
    test_df_with_preds['pred_Ex1'] = test_preds[:, 0]
    test_df_with_preds['pred_Ex2'] = test_preds[:, 1]
    test_df_with_preds['pred_Ex3'] = test_preds[:, 2]
    test_df_with_preds['pred_Rx1'] = test_preds[:, 3]
    test_df_with_preds['pred_Rx2'] = test_preds[:, 4]
    test_df_with_preds['pred_Rx3'] = test_preds[:, 5]
    
    # Evaluate the test subset.
    test_per_sample_metrics = compute_per_sample_metrics(test_targets, test_preds)
    
    # Evaluate the test subset.
    test_df_with_preds['test_MSE_E'] = test_per_sample_metrics[:, 4]
    test_df_with_preds['test_MAE_E'] = test_per_sample_metrics[:, 5]
    test_df_with_preds['test_R2_E'] = test_per_sample_metrics[:, 6]
    test_df_with_preds['test_RMSE_E'] = test_per_sample_metrics[:, 7]
    test_df_with_preds['test_MSE_R'] = test_per_sample_metrics[:, 8]
    test_df_with_preds['test_MAE_R'] = test_per_sample_metrics[:, 9]
    test_df_with_preds['test_R2_R'] = test_per_sample_metrics[:, 10]
    test_df_with_preds['test_RMSE_R'] = test_per_sample_metrics[:, 11]
    test_df_with_preds['test_MSE_all'] = test_per_sample_metrics[:, 0]
    test_df_with_preds['test_MAE_all'] = test_per_sample_metrics[:, 1]
    test_df_with_preds['test_R2_all'] = test_per_sample_metrics[:, 2]
    test_df_with_preds['test_RMSE_all'] = test_per_sample_metrics[:, 3]
    
    # Save the generated artifacts.
    test_results_csv = os.path.join(results_dir, "test_results.csv")
    test_df_with_preds.to_csv(test_results_csv, index=False, encoding='utf-8-sig')
    print(f"测试集结果已保存到: {test_results_csv}")
    
    # Run the training step.
    total_time = sum(history['epoch_time'])
    avg_time_per_epoch = np.mean(history['epoch_time'])
    
    best_metrics = {
        'best_epoch': best_epoch,
        'best_val_mse': float(best_val_mse),
        'total_training_time_seconds': float(total_time),
        'avg_time_per_epoch_seconds': float(avg_time_per_epoch),
        'total_epochs': len(history['epoch']),
        'best_val_metrics': {k: float(v) for k, v in final_val_metrics.items()},
        'best_test_metrics': {k: float(v) for k, v in final_test_metrics.items()},
    }
    
    with open(os.path.join(out_dir, "best_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)
    
    # Save the generated artifacts.
    metrics_txt_path = os.path.join(results_dir, "training_metrics.txt")
    with open(metrics_txt_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("MMGNN训练结果指标\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"最佳epoch: {best_epoch}\n")
        f.write(f"最佳验证集MSE: {best_val_mse:.6f}\n")
        f.write(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)\n")
        f.write(f"平均每轮时间: {avg_time_per_epoch:.2f}秒\n")
        f.write(f"总训练轮数: {len(history['epoch'])}\n\n")
        
        f.write("="*100 + "\n")
        f.write("每轮训练指标汇总\n")
        f.write("="*100 + "\n\n")
        
        # Compute evaluation metrics.
        for i, epoch_num in enumerate(history['epoch']):
            f.write(f"\nEpoch {epoch_num}:\n")
            f.write(f"  训练损失: {history['train_loss'][i]:.6f}\n")
            f.write(f"  训练时间: {history['epoch_time'][i]:.2f}秒\n")
            
            # Run the training step.
            train_metrics_dict = {
                'mae': history['train_mae'][i],
                'rmse': history['train_rmse'][i],
                'r2': history['train_r2'][i],
                'mae_E': history['train_mae_E'][i],
                'rmse_E': history['train_rmse_E'][i],
                'r2_E': history['train_r2_E'][i],
                'mae_R': history['train_mae_R'][i],
                'rmse_R': history['train_rmse_R'][i],
                'r2_R': history['train_r2_R'][i],
            }
            f.write(f"  训练集 - {format_metrics_detailed(train_metrics_dict, '    ')}\n")
            
            # Evaluate the validation subset.
            val_metrics_dict = {
                'mae': history['val_mae'][i],
                'rmse': history['val_rmse'][i],
                'r2': history['val_r2'][i],
                'mae_E': history['val_mae_E'][i],
                'rmse_E': history['val_rmse_E'][i],
                'r2_E': history['val_r2_E'][i],
                'mae_R': history['val_mae_R'][i],
                'rmse_R': history['val_rmse_R'][i],
                'r2_R': history['val_r2_R'][i],
            }
            f.write(f"  验证集 - {format_metrics_detailed(val_metrics_dict, '    ')}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("最佳模型指标（基于验证集MSE）\n")
        f.write("="*100 + "\n\n")
        
        f.write("最佳验证集指标:\n")
        f.write(format_metrics_detailed(final_val_metrics))
        f.write("\n\n")
        
        f.write("最佳测试集指标:\n")
        f.write(format_metrics_detailed(final_test_metrics))
        f.write("\n")
    
    # Save the generated artifacts.
    with open(os.path.join(out_dir, "best_metrics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"最佳epoch: {best_epoch}\n")
        f.write(f"最佳验证集MSE: {best_val_mse:.6f}\n")
        f.write(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)\n")
        f.write(f"平均每轮时间: {avg_time_per_epoch:.2f}秒\n")
        f.write(f"总训练轮数: {len(history['epoch'])}\n\n")
        
        f.write("="*80 + "\n")
        f.write("最佳验证集指标:\n")
        f.write("="*80 + "\n")
        f.write(format_metrics_detailed(final_val_metrics))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("最佳测试集指标:\n")
        f.write("="*80 + "\n")
        f.write(format_metrics_detailed(final_test_metrics))
        f.write("\n")
    
    print("\n" + "="*100)
    print("训练完成！")
    print(f"最佳模型在epoch {best_epoch}，验证集MSE: {best_val_mse:.6f}")
    print(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"平均每轮时间: {avg_time_per_epoch:.2f}秒")
    print(f"\n结果文件已保存:")
    print(f"  - 训练历史CSV: {os.path.join(out_dir, 'train_history.csv')}")
    print(f"  - 训练/验证结果CSV: {os.path.join(results_dir, 'training_results.csv')}")
    print(f"  - 测试集结果CSV: {os.path.join(results_dir, 'test_results.csv')}")
    print(f"  - 训练指标TXT: {os.path.join(results_dir, 'training_metrics.txt')}")
    print(f"  - 模型权重: {os.path.join(out_dir, 'mmgnn.pt')}")
    print("="*100)
    
    return model, T_scaler, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MMGNN训练脚本')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练（检查点文件路径）')
    parser.add_argument('--auto-resume', action='store_true',
                        help='自动查找并恢复最新的检查点（如果存在）')
    parser.add_argument('--checkpoint-every', type=int, default=10,
                        help='每N个epoch保存一次检查点（默认10）')
    args = parser.parse_args()
    
    # Configure the output artifacts.
    out_dir = str(EXPERIMENT_ROOT / "runs" / "mmgnn")
    os.makedirs(out_dir, exist_ok=True)
    
    # Handle model checkpoints.
    # Handle model checkpoints.
    # Handle model checkpoints.
    # Set the random seed.
    
    # Handle model checkpoints.
    args.auto_resume = True
    # ====================================================
    
    # Handle model checkpoints.
    if args.auto_resume and not args.resume:
        import glob
        checkpoint_pattern = os.path.join(out_dir, "checkpoint_epoch_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        if checkpoint_files:
            # Baseline workflow step.
            def extract_epoch(fpath):
                try:
                    return int(os.path.basename(fpath).replace("checkpoint_epoch_", "").replace(".pt", ""))
                except:
                    return -1
            checkpoint_files.sort(key=extract_epoch, reverse=True)
            args.resume = checkpoint_files[0]
            print(f"\n自动找到最新检查点: {args.resume}")
            print(f"  (epoch {extract_epoch(args.resume)})")
    
    # Run the training step.
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 90)
    print("MMGNN Training for LLE Prediction")
    print("=" * 90)
    
    # Load the input data.
    print("\n1) Loading datasets from specified CSV files...")
    from .data_loader import load_csv_data
    from psmi_baselines.common import config as C
    
    # Run the training step.
    train_csv_path = str(DATA_DIR / "train.csv")
    val_csv_path = str(DATA_DIR / "validation.csv")
    
    # Baseline workflow step.
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"训练集文件不存在: {train_csv_path}")
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"验证集文件不存在: {val_csv_path}")
    
    # Load the input data.
    print(f"  加载训练集: {train_csv_path}")
    _, train_df = load_csv_data(
        train_csv_path,
        min_points_per_group=C.MIN_POINTS_PER_GROUP,
        permute_23_aug=C.PERMUTE_23_AUG
    )
    print(f"  加载验证集: {val_csv_path}")
    _, val_df = load_csv_data(
        val_csv_path,
        min_points_per_group=C.MIN_POINTS_PER_GROUP,
        permute_23_aug=C.PERMUTE_23_AUG
    )
    
    # Evaluate the validation subset.
    test_csv_path = os.path.join(os.path.dirname(train_csv_path), "test.csv")
    if os.path.exists(test_csv_path):
        print(f"  加载测试集: {test_csv_path}")
        _, test_df = load_csv_data(
            test_csv_path,
            min_points_per_group=C.MIN_POINTS_PER_GROUP,
            permute_23_aug=C.PERMUTE_23_AUG
        )
    else:
        print(f"  警告: 测试集文件不存在 ({test_csv_path})，使用验证集作为测试集")
        test_df = val_df.copy()
    
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_df)} rows | systems: {train_df['system_id'].nunique()}")
    print(f"  验证集: {len(val_df)} rows | systems: {val_df['system_id'].nunique()}")
    print(f"  测试集: {len(test_df)} rows | systems: {test_df['system_id'].nunique()}")
    
    # Baseline workflow step.
    # Configure the baseline model.
    effective_batch_size = C.BATCH_SIZE
    if torch.cuda.is_available() and C.DEVICE.startswith('cuda'):
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb < 8:
            # Baseline workflow step.
            effective_batch_size = min(32, C.BATCH_SIZE)
            print(f"\n警告: GPU内存较小 ({gpu_memory_gb:.2f} GB)，将batch size从 {C.BATCH_SIZE} 调整为 {effective_batch_size}")
        elif gpu_memory_gb < 16:
            # Baseline workflow step.
            effective_batch_size = min(64, C.BATCH_SIZE)
            print(f"\n提示: GPU内存为 {gpu_memory_gb:.2f} GB，将batch size从 {C.BATCH_SIZE} 调整为 {effective_batch_size}")
        else:
            # Baseline workflow step.
            effective_batch_size = min(128, C.BATCH_SIZE)
            print(f"\n提示: GPU内存为 {gpu_memory_gb:.2f} GB，将batch size从 {C.BATCH_SIZE} 调整为 {effective_batch_size}")
    
    # Run the training step.
    print("\n3) Training MMGNN...")
    if args.resume:
        print(f"  断点续训模式: 从 {args.resume} 恢复训练")
    else:
        print("  从头开始训练")
    
    # Run the training step.
    # Baseline workflow step.
    # Configure the baseline model.
    # Baseline workflow step.
    # Run the training step.
    
    training_mode = "balanced"  # Baseline workflow step.
    
    if training_mode == "full":
        # Configure the baseline model.
        hidden_dim = 256
        num_layers = 3
        set2set_steps = 3
        post_explain_layers = 2
        print(f"\n使用完整模型配置（最佳性能）:")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  num_layers: {num_layers}")
        print(f"  set2set_steps: {set2set_steps}")
        print(f"  post_explain_layers: {post_explain_layers}")
    elif training_mode == "balanced":
        # Baseline workflow step.
        # Configure experiment parameters.
        hidden_dim = 256  # Baseline workflow step.
        num_layers = 3  # Baseline workflow step.
        set2set_steps = 2  # Baseline workflow step.
        post_explain_layers = 1  # Baseline workflow step.
        print(f"\n使用平衡配置（性能优先，适度加速）:")
        print(f"  hidden_dim: {hidden_dim} (保持)")
        print(f"  num_layers: {num_layers} (保持)")
        print(f"  set2set_steps: {set2set_steps} (从3减少到2，约1.2-1.5倍加速)")
        print(f"  post_explain_layers: {post_explain_layers} (从2减少到1，约1.2-1.5倍加速)")
        print(f"  预计总加速: 约1.5-2倍，性能损失<5%")
    else:  # fast
        # Run the training step.
        hidden_dim = 192  # Baseline workflow step.
        num_layers = 2  # Baseline workflow step.
        set2set_steps = 2  # Baseline workflow step.
        post_explain_layers = 1  # Baseline workflow step.
        print(f"\n使用快速训练配置（约2-3倍加速）:")
        print(f"  hidden_dim: {hidden_dim} (从256减少)")
        print(f"  num_layers: {num_layers} (从3减少)")
        print(f"  set2set_steps: {set2set_steps} (从3减少到2)")
        print(f"  post_explain_layers: {post_explain_layers} (从2减少到1)")
        print(f"  注意: 性能可能下降5-10%")
    
    model, T_scaler, history = train_mmgnn(
        train_df, val_df, test_df,
        out_dir=out_dir,
        device=C.DEVICE,
        batch_size=effective_batch_size,
        epochs=C.EPOCHS,
        lr=C.LR,
        weight_decay=C.WEIGHT_DECAY,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        set2set_steps=set2set_steps,
        post_explain_layers=post_explain_layers,
        beta=0.2,
        explainer_method='local_mask',
        dropout=C.DROPOUT,
        patience=100,  # Apply early stopping.
        min_delta=0.0,  # Baseline workflow step.
        resume_from=args.resume,  # Baseline workflow step.
        save_checkpoint_every=args.checkpoint_every,  # Save the generated artifacts.
        rest_interval=60,  # Baseline workflow step.
        rest_duration=600.0,  # Baseline workflow step.
    )
    
    print("\n" + "=" * 90)
    print("Training completed!")
    print(f"Results saved to: {out_dir}")
    print("=" * 90)


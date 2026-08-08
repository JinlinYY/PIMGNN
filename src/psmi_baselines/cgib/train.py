"""Implement the cgib train baseline module."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

from .models import CGIB
from .utils.data_loader import MolecularDataset, create_batch
from psmi_baselines.paths import EXPERIMENT_ROOT, TOTAL_CSV
from psmi_baselines.protocol import canonical_split_indices


def set_seed(seed):
    """Run the set seed baseline operation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(m, output_dim=6):
    """Run the init weights baseline operation."""
    if isinstance(m, nn.Linear):
        # Configure the output artifacts.
        # Configure the output artifacts.
        if m.out_features == output_dim:
            # Configure the output artifacts.
            # Baseline workflow step.
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                # Configure the output artifacts.
                nn.init.constant_(m.bias, 0.0)
        else:
            # Baseline workflow step.
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # Configure the output artifacts.
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def compute_metrics_batch(y_true, y_pred):
    """Run the compute metrics batch baseline operation."""
    # Configure the output artifacts.
    mae_all = mean_absolute_error(y_true, y_pred)
    rmse_all = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Compute evaluation metrics.
    e_phase_true = y_true[:, :3]  # Ex1, Ex2, Ex3
    e_phase_pred = y_pred[:, :3]
    mae_e = mean_absolute_error(e_phase_true, e_phase_pred)
    rmse_e = np.sqrt(mean_squared_error(e_phase_true, e_phase_pred))
    
    # Compute evaluation metrics.
    r_phase_true = y_true[:, 3:]  # Rx1, Rx2, Rx3
    r_phase_pred = y_pred[:, 3:]
    mae_r = mean_absolute_error(r_phase_true, r_phase_pred)
    rmse_r = np.sqrt(mean_squared_error(r_phase_true, r_phase_pred))
    
    return {
        'all': {'mae': mae_all, 'rmse': rmse_all},
        'e_phase': {'mae': mae_e, 'rmse': rmse_e},
        'r_phase': {'mae': mae_r, 'rmse': rmse_r}
    }


def safe_r2_score(y_true, y_pred):
    """Run the safe r2 score baseline operation."""
    # Baseline workflow step.
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Baseline workflow step.
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]
    
    if len(y_true_flat) == 0:
        return 0.0
    
    # Baseline workflow step.
    y_mean = np.mean(y_true_flat)
    
    # Baseline workflow step.
    ss_tot = np.sum((y_true_flat - y_mean) ** 2)
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    
    # Baseline workflow step.
    if ss_tot < 1e-10:
        # Baseline workflow step.
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    
    # Baseline workflow step.
    r2 = np.clip(r2, -10.0, 1.0)
    
    return r2


def compute_metrics(y_true, y_pred):
    """Run the compute metrics baseline operation."""
    # Baseline workflow step.
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(mask):
        print(f"警告：发现 {np.sum(~mask)} 个非有限值，将被忽略")
        y_true = y_true[mask.all(axis=1)]
        y_pred = y_pred[mask.all(axis=1)]
    
    # Process the experiment data.
    if len(y_true) == 0:
        print("错误：有效数据为空")
        return {
            'all': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
            'e_phase': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
            'r_phase': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
        }
    
    # Generate model predictions.
    pred_std = np.std(y_pred, axis=0)
    if np.any(pred_std < 0.01):
        print(f"[警告] 预测值方差过小，可能存在饱和问题。预测值标准差: {pred_std}")
    
    # Configure the output artifacts.
    mae_all = mean_absolute_error(y_true, y_pred)
    rmse_all = np.sqrt(mean_squared_error(y_true, y_pred))
    # Baseline workflow step.
    r2_all = safe_r2_score(y_true, y_pred)
    
    # Compute evaluation metrics.
    e_phase_true = y_true[:, :3]  # Ex1, Ex2, Ex3
    e_phase_pred = y_pred[:, :3]
    mae_e = mean_absolute_error(e_phase_true, e_phase_pred)
    rmse_e = np.sqrt(mean_squared_error(e_phase_true, e_phase_pred))
    r2_e = safe_r2_score(e_phase_true, e_phase_pred)
    
    # Compute evaluation metrics.
    r_phase_true = y_true[:, 3:]  # Rx1, Rx2, Rx3
    r_phase_pred = y_pred[:, 3:]
    mae_r = mean_absolute_error(r_phase_true, r_phase_pred)
    rmse_r = np.sqrt(mean_squared_error(r_phase_true, r_phase_pred))
    r2_r = safe_r2_score(r_phase_true, r_phase_pred)
    
    return {
        'all': {'mae': mae_all, 'rmse': rmse_all, 'r2': r2_all},
        'e_phase': {'mae': mae_e, 'rmse': rmse_e, 'r2': r2_e},
        'r_phase': {'mae': mae_r, 'rmse': rmse_r, 'r2': r2_r}
    }


def train_epoch(model, dataloader, optimizer, device, beta):
    """Run the train epoch baseline operation."""
    model.train()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    # Compute evaluation metrics.
    batch_metrics = {
        'all': {'mae': [], 'rmse': []},
        'e_phase': {'mae': [], 'rmse': []},
        'r_phase': {'mae': [], 'rmse': []}
    }
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        graphs1, graphs2, targets = batch_data
        
        # Configure the runtime device.
        graphs1 = graphs1.to(device)
        graphs2 = graphs2.to(device)
        targets = targets.to(device)
        
        # Baseline workflow step.
        pred, loss_components = model(graphs1, graphs2, return_loss_components=True)
        
        # Compute the training loss.
        pred_loss = nn.functional.mse_loss(pred, targets)
        mi1_loss = loss_components['mi1']
        mi2_loss = loss_components['mi2']
        
        total_loss_batch = pred_loss + beta * (mi1_loss + mi2_loss)
        
        # Baseline workflow step.
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        # Generate model predictions.
        total_loss += total_loss_batch.item()
        
        # Compute evaluation metrics.
        pred_np = pred.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # Generate model predictions.
        if batch_idx == 0 and len(all_predictions) == 0:
            print(f"\n[调试] 第一个batch的预测值范围: [{pred_np.min():.6f}, {pred_np.max():.6f}]")
            print(f"[调试] 第一个batch的真实值范围: [{targets_np.min():.6f}, {targets_np.max():.6f}]")
            print(f"[调试] 第一个batch的预测值均值: {pred_np.mean(axis=0)}")
            print(f"[调试] 第一个batch的真实值均值: {targets_np.mean(axis=0)}")
        
        batch_metric = compute_metrics_batch(targets_np, pred_np)
        
        # Compute evaluation metrics.
        batch_metrics['all']['mae'].append(batch_metric['all']['mae'])
        batch_metrics['all']['rmse'].append(batch_metric['all']['rmse'])
        batch_metrics['e_phase']['mae'].append(batch_metric['e_phase']['mae'])
        batch_metrics['e_phase']['rmse'].append(batch_metric['e_phase']['rmse'])
        batch_metrics['r_phase']['mae'].append(batch_metric['r_phase']['mae'])
        batch_metrics['r_phase']['rmse'].append(batch_metric['r_phase']['rmse'])
        
        all_predictions.append(pred_np)
        all_targets.append(targets_np)
    
    # Baseline workflow step.
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Generate model predictions.
    if len(all_predictions) > 0:
        pred_min, pred_max = all_predictions.min(), all_predictions.max()
        pred_mean = all_predictions.mean(axis=0)
        pred_std = all_predictions.std(axis=0)
        target_min, target_max = all_targets.min(), all_targets.max()
        target_mean = all_targets.mean(axis=0)
        target_std = all_targets.std(axis=0)
        
        # Baseline workflow step.
        if pred_max > 2.0 or pred_min < -1.0:
            print(f"\n[警告] 预测值范围异常:")
            print(f"  预测值: [{pred_min:.6f}, {pred_max:.6f}], 均值: {pred_mean}")
            print(f"  真实值: [{target_min:.6f}, {target_max:.6f}], 均值: {target_mean}")
            print(f"  建议：确保启用输出约束")
        
        # Generate model predictions.
        if np.any(pred_std < 0.01):
            print(f"\n[警告] 预测值可能饱和（标准差过小）:")
            print(f"  预测值标准差: {pred_std}")
            print(f"  真实值标准差: {target_std}")
            print(f"  预测值均值: {pred_mean}")
            print(f"  真实值均值: {target_mean}")
            print(f"  建议：检查模型初始化或降低学习率")
    
    # Process the experiment data.
    full_metrics = compute_metrics(all_targets, all_predictions)
    
    # Compute evaluation metrics.
    metrics = {
        'all': {
            'mae_mean': np.mean(batch_metrics['all']['mae']),
            'mae_std': np.std(batch_metrics['all']['mae']),
            'rmse_mean': np.mean(batch_metrics['all']['rmse']),
            'rmse_std': np.std(batch_metrics['all']['rmse']),
            'r2': full_metrics['all']['r2']
        },
        'e_phase': {
            'mae_mean': np.mean(batch_metrics['e_phase']['mae']),
            'mae_std': np.std(batch_metrics['e_phase']['mae']),
            'rmse_mean': np.mean(batch_metrics['e_phase']['rmse']),
            'rmse_std': np.std(batch_metrics['e_phase']['rmse']),
            'r2': full_metrics['e_phase']['r2']
        },
        'r_phase': {
            'mae_mean': np.mean(batch_metrics['r_phase']['mae']),
            'mae_std': np.std(batch_metrics['r_phase']['mae']),
            'rmse_mean': np.mean(batch_metrics['r_phase']['rmse']),
            'rmse_std': np.std(batch_metrics['r_phase']['rmse']),
            'r2': full_metrics['r_phase']['r2']
        }
    }
    
    return {
        'loss': total_loss / len(dataloader),
        'metrics': metrics,
        'predictions': all_predictions,
        'targets': all_targets
    }


def evaluate(model, dataloader, device):
    """Run the evaluate baseline operation."""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    # Compute evaluation metrics.
    batch_metrics = {
        'all': {'mae': [], 'rmse': []},
        'e_phase': {'mae': [], 'rmse': []},
        'r_phase': {'mae': [], 'rmse': []}
    }
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            graphs1, graphs2, targets = batch_data
            
            graphs1 = graphs1.to(device)
            graphs2 = graphs2.to(device)
            targets = targets.to(device)
            
            pred = model(graphs1, graphs2)
            loss = nn.functional.mse_loss(pred, targets)
            
            total_loss += loss.item()
            
            # Compute evaluation metrics.
            pred_np = pred.cpu().numpy()
            targets_np = targets.cpu().numpy()
            batch_metric = compute_metrics_batch(targets_np, pred_np)
            
            # Compute evaluation metrics.
            batch_metrics['all']['mae'].append(batch_metric['all']['mae'])
            batch_metrics['all']['rmse'].append(batch_metric['all']['rmse'])
            batch_metrics['e_phase']['mae'].append(batch_metric['e_phase']['mae'])
            batch_metrics['e_phase']['rmse'].append(batch_metric['e_phase']['rmse'])
            batch_metrics['r_phase']['mae'].append(batch_metric['r_phase']['mae'])
            batch_metrics['r_phase']['rmse'].append(batch_metric['r_phase']['rmse'])
            
            all_predictions.append(pred_np)
            all_targets.append(targets_np)
    
    # Baseline workflow step.
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Generate model predictions.
    if len(all_predictions) > 0:
        pred_min, pred_max = all_predictions.min(), all_predictions.max()
        pred_mean = all_predictions.mean()
        target_min, target_max = all_targets.min(), all_targets.max()
        target_mean = all_targets.mean()
        
        # Baseline workflow step.
        if pred_max > 2.0 or pred_min < -1.0:
            print(f"\n[警告] 预测值范围异常:")
            print(f"  预测值: [{pred_min:.6f}, {pred_max:.6f}], 均值: {pred_mean:.6f}")
            print(f"  真实值: [{target_min:.6f}, {target_max:.6f}], 均值: {target_mean:.6f}")
            print(f"  建议：确保启用输出约束")
    
    # Process the experiment data.
    full_metrics = compute_metrics(all_targets, all_predictions)
    
    # Compute evaluation metrics.
    metrics = {
        'all': {
            'mae_mean': np.mean(batch_metrics['all']['mae']),
            'mae_std': np.std(batch_metrics['all']['mae']),
            'rmse_mean': np.mean(batch_metrics['all']['rmse']),
            'rmse_std': np.std(batch_metrics['all']['rmse']),
            'r2': full_metrics['all']['r2']
        },
        'e_phase': {
            'mae_mean': np.mean(batch_metrics['e_phase']['mae']),
            'mae_std': np.std(batch_metrics['e_phase']['mae']),
            'rmse_mean': np.mean(batch_metrics['e_phase']['rmse']),
            'rmse_std': np.std(batch_metrics['e_phase']['rmse']),
            'r2': full_metrics['e_phase']['r2']
        },
        'r_phase': {
            'mae_mean': np.mean(batch_metrics['r_phase']['mae']),
            'mae_std': np.std(batch_metrics['r_phase']['mae']),
            'rmse_mean': np.mean(batch_metrics['r_phase']['rmse']),
            'rmse_std': np.std(batch_metrics['r_phase']['rmse']),
            'r2': full_metrics['r_phase']['r2']
        }
    }
    
    return {
        'loss': total_loss / len(dataloader),
        'metrics': metrics,
        'predictions': all_predictions,
        'targets': all_targets
    }


def print_metrics(metrics, prefix=""):
    """Run the print metrics baseline operation."""
    print(f"{prefix}【Overall】 MAE: {metrics['all']['mae_mean']:.6f}±{metrics['all']['mae_std']:.6f}, "
          f"RMSE: {metrics['all']['rmse_mean']:.6f}±{metrics['all']['rmse_std']:.6f}, "
          f"R²: {metrics['all']['r2']:.6f}")
    print(f"{prefix}【E相】 MAE: {metrics['e_phase']['mae_mean']:.6f}±{metrics['e_phase']['mae_std']:.6f}, "
          f"RMSE: {metrics['e_phase']['rmse_mean']:.6f}±{metrics['e_phase']['rmse_std']:.6f}, "
          f"R²: {metrics['e_phase']['r2']:.6f}")
    print(f"{prefix}【R相】 MAE: {metrics['r_phase']['mae_mean']:.6f}±{metrics['r_phase']['mae_std']:.6f}, "
          f"RMSE: {metrics['r_phase']['rmse_mean']:.6f}±{metrics['r_phase']['rmse_std']:.6f}, "
          f"R²: {metrics['r_phase']['r2']:.6f}")


def save_results(predictions, targets, split, output_dir):
    """Run the save results baseline operation."""
    results = pd.DataFrame({
        'Ex1_true': targets[:, 0],
        'Ex2_true': targets[:, 1],
        'Ex3_true': targets[:, 2],
        'Rx1_true': targets[:, 3],
        'Rx2_true': targets[:, 4],
        'Rx3_true': targets[:, 5],
        'pred_Ex1': predictions[:, 0],
        'pred_Ex2': predictions[:, 1],
        'pred_Ex3': predictions[:, 2],
        'pred_Rx1': predictions[:, 3],
        'pred_Rx2': predictions[:, 4],
        'pred_Rx3': predictions[:, 5],
        'split': split
    })
    
    if split == 'test':
        results.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
    else:
        # Save the generated artifacts.
        file_path = os.path.join(output_dir, 'training_results.csv')
        if os.path.exists(file_path):
            existing = pd.read_csv(file_path)
            results = pd.concat([existing, results], ignore_index=True)
        results.to_csv(file_path, index=False)


def save_metrics_txt(best_epoch, best_val_rmse, total_epochs, total_time, avg_time_per_epoch,
                     val_metrics, test_metrics=None, output_dir=None):
    """Run the save metrics txt baseline operation."""
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the generated artifacts.
    with open(os.path.join(results_dir, 'best_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("最佳模型指标\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【训练信息】\n")
        f.write(f"  最佳epoch: {best_epoch}\n")
        f.write(f"  最佳验证RMSE: {best_val_rmse:.6f}\n")
        f.write(f"  总训练轮数: {total_epochs}\n")
        f.write(f"  总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)\n")
        f.write(f"  平均每轮时间: {avg_time_per_epoch:.2f}秒\n\n")
        
        f.write("【验证集指标】\n")
        f.write(f"  【Overall】 MAE: {val_metrics['all']['mae_mean']:.6f}±{val_metrics['all']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['all']['rmse_mean']:.6f}±{val_metrics['all']['rmse_std']:.6f}, "
                f"R²: {val_metrics['all']['r2']:.6f}\n")
        f.write(f"  【E相】 MAE: {val_metrics['e_phase']['mae_mean']:.6f}±{val_metrics['e_phase']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['e_phase']['rmse_mean']:.6f}±{val_metrics['e_phase']['rmse_std']:.6f}, "
                f"R²: {val_metrics['e_phase']['r2']:.6f}\n")
        f.write(f"  【R相】 MAE: {val_metrics['r_phase']['mae_mean']:.6f}±{val_metrics['r_phase']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['r_phase']['rmse_mean']:.6f}±{val_metrics['r_phase']['rmse_std']:.6f}, "
                f"R²: {val_metrics['r_phase']['r2']:.6f}\n\n")
        
        if test_metrics is not None:
            f.write("【测试集指标】\n")
            f.write(f"  【Overall】 MAE: {test_metrics['all']['mae_mean']:.6f}±{test_metrics['all']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['all']['rmse_mean']:.6f}±{test_metrics['all']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['all']['r2']:.6f}\n")
            f.write(f"  【E相】 MAE: {test_metrics['e_phase']['mae_mean']:.6f}±{test_metrics['e_phase']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['e_phase']['rmse_mean']:.6f}±{test_metrics['e_phase']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['e_phase']['r2']:.6f}\n")
            f.write(f"  【R相】 MAE: {test_metrics['r_phase']['mae_mean']:.6f}±{test_metrics['r_phase']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['r_phase']['rmse_mean']:.6f}±{test_metrics['r_phase']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['r_phase']['r2']:.6f}\n")
    
    # Save the generated artifacts.
    with open(os.path.join(results_dir, 'training_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("训练指标总结\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"最佳模型在epoch {best_epoch}\n")
        f.write(f"最佳验证集RMSE: {best_val_rmse:.6f}\n\n")
        
        f.write("【最佳验证集指标】\n")
        f.write(f"  【Overall】 MAE: {val_metrics['all']['mae_mean']:.6f}±{val_metrics['all']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['all']['rmse_mean']:.6f}±{val_metrics['all']['rmse_std']:.6f}, "
                f"R²: {val_metrics['all']['r2']:.6f}\n")
        f.write(f"  【E相】 MAE: {val_metrics['e_phase']['mae_mean']:.6f}±{val_metrics['e_phase']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['e_phase']['rmse_mean']:.6f}±{val_metrics['e_phase']['rmse_std']:.6f}, "
                f"R²: {val_metrics['e_phase']['r2']:.6f}\n")
        f.write(f"  【R相】 MAE: {val_metrics['r_phase']['mae_mean']:.6f}±{val_metrics['r_phase']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['r_phase']['rmse_mean']:.6f}±{val_metrics['r_phase']['rmse_std']:.6f}, "
                f"R²: {val_metrics['r_phase']['r2']:.6f}\n\n")
        
        if test_metrics is not None:
            f.write("【测试集指标】\n")
            f.write(f"  【Overall】 MAE: {test_metrics['all']['mae_mean']:.6f}±{test_metrics['all']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['all']['rmse_mean']:.6f}±{test_metrics['all']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['all']['r2']:.6f}\n")
            f.write(f"  【E相】 MAE: {test_metrics['e_phase']['mae_mean']:.6f}±{test_metrics['e_phase']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['e_phase']['rmse_mean']:.6f}±{test_metrics['e_phase']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['e_phase']['r2']:.6f}\n")
            f.write(f"  【R相】 MAE: {test_metrics['r_phase']['mae_mean']:.6f}±{test_metrics['r_phase']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['r_phase']['rmse_mean']:.6f}±{test_metrics['r_phase']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['r_phase']['r2']:.6f}\n\n")
        
        f.write(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)\n")
        f.write(f"平均每轮时间: {avg_time_per_epoch:.2f}秒\n")


def train_single_seed(args):
    """Run the train single seed baseline operation."""
    # Set the random seed.
    set_seed(args.seed)
    
    # Configure the output artifacts.
    output_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoint'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    
    # Baseline workflow step.
    print("=" * 80)
    print("CGIB模型训练配置")
    print("=" * 80)
    print("\n【数据集信息】")
    
    # Load the input data.
    df = pd.read_csv(args.data_path)
    
    # Process the experiment data.
    if 'IL (Component 1) full name SMILES' in df.columns:
        # Baseline workflow step.
        print("  检测到total.csv格式，正在转换...")
        smiles1_list = []
        smiles2_list = []
        
        for idx, row in df.iterrows():
            il_smiles = str(row['IL (Component 1) full name SMILES']).strip()
            comp2_smiles = str(row['Component 2 SMILES']).strip()
            comp3_smiles = str(row['Component 3 SMILES']).strip()
            
            # Baseline workflow step.
            if pd.notna(comp3_smiles) and comp3_smiles != '' and comp3_smiles != 'nan':
                combined_smiles = f"{comp2_smiles}.{comp3_smiles}"
            else:
                combined_smiles = comp2_smiles
            
            smiles1_list.append(il_smiles)
            smiles2_list.append(combined_smiles)
        
        # Baseline workflow step.
        targets = df[['Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3']].values
    elif 'smiles1' in df.columns and 'smiles2' in df.columns:
        # Baseline workflow step.
        smiles1_list = df['smiles1'].tolist()
        smiles2_list = df['smiles2'].tolist()
        
        # Baseline workflow step.
        if 'Ex1' in df.columns:
            targets = df[['Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3']].values
        elif 'target' in df.columns:
            # Baseline workflow step.
            targets = np.array([eval(x) if isinstance(x, str) else x for x in df['target']])
            if targets.ndim == 1:
                targets = targets.reshape(-1, 1)
        else:
            raise ValueError("数据文件中必须包含Ex1-Ex3, Rx1-Rx3列或target列")
    else:
        raise ValueError("数据文件格式不支持。需要包含smiles1/smiles2列或IL/Component列")
    
    # Process the experiment data.
    dataset = MolecularDataset(smiles1_list, smiles2_list, targets)

    # Prefer the canonical, system-exclusive partitions exported in total.csv.
    total_size = len(dataset)
    canonical_indices = canonical_split_indices(df)
    if canonical_indices is not None:
        train_dataset = torch.utils.data.Subset(dataset, canonical_indices["train"])
        val_dataset = torch.utils.data.Subset(dataset, canonical_indices["validation"])
        test_dataset = torch.utils.data.Subset(dataset, canonical_indices["test"])
    elif args.allow_random_row_split:
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        raise ValueError(
            "The input CSV must contain the canonical split column. "
            "Use --allow_random_row_split only for explicitly labeled legacy reproduction."
        )
    
    print(f"  总样本数: {total_size}")
    print(f"  训练集样本数: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"  验证集样本数: {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    print(f"  测试集样本数: {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")
    
    print("\n【设备配置】")
    print(f"  设备类型: {args.device}")
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU内存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n【训练参数】")
    print(f"  随机种子: {args.seed}")
    print(f"  训练轮数（epochs）: {args.epochs}")
    print(f"  批次大小（batch_size）: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  早停耐心值: {args.patience}")
    print(f"  早停最小改善: {args.min_delta}")
    print(f"  检查点保存频率: 每{args.checkpoint_freq}个epoch")
    print(f"  休息策略: 每{args.rest_interval/3600:.1f}小时休息{args.rest_duration/60:.1f}分钟")
    
    print("\n【模型超参数】")
    print(f"  隐藏层维度（hidden_dim）: {args.hidden_dim}")
    print(f"  图神经网络层数（num_layers）: {args.num_layers}")
    print(f"  Set2Set步骤数: {args.set2set_steps}")
    print(f"  Dropout率: 0.0")
    print(f"  输出维度: 6 (LLE任务: Ex1, Ex2, Ex3, Rx1, Rx2, Rx3)")
    print(f"  使用Set2Set: 是")
    print(f"  使用对比学习: {'是' if args.use_contrastive else '否'}")
    
    print("\n【路径信息】")
    print(f"  输出目录: {output_dir}")
    print(f"  结果目录: {output_dir}/results")
    
    # Load the input data.
    def collate_fn(batch):
        graphs1 = [item[0] for item in batch]
        graphs2 = [item[1] for item in batch]
        targets = [item[2] for item in batch]
        return create_batch(graphs1, graphs2, targets)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Baseline workflow step.
    sample_graph = dataset.graphs1[0]
    input_dim = sample_graph.x.size(1)
    output_dim = 6
    
    # Configure the baseline model.
    # Configure the output artifacts.
    constrain_output = getattr(args, 'constrain_output', True)
    if not hasattr(args, 'constrain_output') or args.constrain_output is None:
        constrain_output = True
    print(f"  输出约束: {'启用' if constrain_output else '禁用'}")
    
    model = CGIB(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        beta=args.beta,
        gnn_type=args.gnn_type,
        use_contrastive=args.use_contrastive,
        set2set_steps=args.set2set_steps,
        constrain_output=constrain_output
    ).to(args.device)
    
    # Generate model predictions.
    def init_with_output_dim(m):
        init_weights(m, output_dim=output_dim)
    model.apply(init_with_output_dim)
    print("  模型权重已初始化（输出层：小权重初始化，其他层：Xavier初始化，gain=0.1）")
    
    # Baseline workflow step.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Run the training step.
    history = []
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    start_epoch = 0
    
    # Baseline workflow step.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_rmse = checkpoint.get('best_val_rmse', checkpoint.get('best_val_mse', float('inf')))
        best_epoch = checkpoint['best_epoch']
        patience_counter = checkpoint.get('patience_counter', 0)
        history = checkpoint.get('history', [])
        print(f"\n从检查点恢复训练: epoch {start_epoch}, 最佳RMSE: {best_val_rmse:.6f}")
    
    # Run the training step.
    training_start_time = time.time()
    last_rest_time = time.time()
    
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80 + "\n")
    
    # Run the training step.
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Run the training step.
        train_results = train_epoch(model, train_loader, optimizer, args.device, args.beta)
        
        # Evaluate the validation subset.
        val_results = evaluate(model, val_loader, args.device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Baseline workflow step.
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_results['loss'],
            'train_mae_mean': train_results['metrics']['all']['mae_mean'],
            'train_mae_std': train_results['metrics']['all']['mae_std'],
            'train_rmse_mean': train_results['metrics']['all']['rmse_mean'],
            'train_rmse_std': train_results['metrics']['all']['rmse_std'],
            'train_r2': train_results['metrics']['all']['r2'],
            'train_e_mae_mean': train_results['metrics']['e_phase']['mae_mean'],
            'train_e_mae_std': train_results['metrics']['e_phase']['mae_std'],
            'train_e_rmse_mean': train_results['metrics']['e_phase']['rmse_mean'],
            'train_e_rmse_std': train_results['metrics']['e_phase']['rmse_std'],
            'train_e_r2': train_results['metrics']['e_phase']['r2'],
            'train_r_mae_mean': train_results['metrics']['r_phase']['mae_mean'],
            'train_r_mae_std': train_results['metrics']['r_phase']['mae_std'],
            'train_r_rmse_mean': train_results['metrics']['r_phase']['rmse_mean'],
            'train_r_rmse_std': train_results['metrics']['r_phase']['rmse_std'],
            'train_r_r2': train_results['metrics']['r_phase']['r2'],
            'val_mae_mean': val_results['metrics']['all']['mae_mean'],
            'val_mae_std': val_results['metrics']['all']['mae_std'],
            'val_rmse_mean': val_results['metrics']['all']['rmse_mean'],
            'val_rmse_std': val_results['metrics']['all']['rmse_std'],
            'val_r2': val_results['metrics']['all']['r2'],
            'val_e_mae_mean': val_results['metrics']['e_phase']['mae_mean'],
            'val_e_mae_std': val_results['metrics']['e_phase']['mae_std'],
            'val_e_rmse_mean': val_results['metrics']['e_phase']['rmse_mean'],
            'val_e_rmse_std': val_results['metrics']['e_phase']['rmse_std'],
            'val_e_r2': val_results['metrics']['e_phase']['r2'],
            'val_r_mae_mean': val_results['metrics']['r_phase']['mae_mean'],
            'val_r_mae_std': val_results['metrics']['r_phase']['mae_std'],
            'val_r_rmse_mean': val_results['metrics']['r_phase']['rmse_mean'],
            'val_r_rmse_std': val_results['metrics']['r_phase']['rmse_std'],
            'val_r_r2': val_results['metrics']['r_phase']['r2'],
        })
        
        # Baseline workflow step.
        print(f"\nEpoch {epoch+1}/{args.epochs} | 训练时间: {epoch_time:.2f}秒 | Train Loss: {train_results['loss']:.6f}")
        print(f"Best RMSE: {best_val_rmse:.6f} (epoch {best_epoch+1})")
        
        print("\n【训练集指标】")
        print_metrics(train_results['metrics'])
        
        print("\n【验证集指标】")
        print_metrics(val_results['metrics'])
        
        # Baseline workflow step.
        current_val_rmse = val_results['metrics']['all']['rmse_mean']
        improved = current_val_rmse < (best_val_rmse - args.min_delta)
        
        if improved:
            best_val_rmse = current_val_rmse
            best_epoch = epoch
            patience_counter = 0
            
            # Save the generated artifacts.
            best_model_path = os.path.join(output_dir, f'seed_{args.seed}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_rmse': best_val_rmse,
                'val_metrics': val_results['metrics'],
                'args': vars(args)
            }, best_model_path)
        else:
            patience_counter += 1
        
        # Save the generated artifacts.
        if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(output_dir, 'checkpoint', f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': current_val_rmse,
                'best_val_rmse': best_val_rmse,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'history': history,
                'args': vars(args)
            }, checkpoint_path)
            print(f"\n检查点已保存: {checkpoint_path}")
        
        # Apply early stopping.
        if patience_counter >= args.patience:
            print(f"\n早停触发！在epoch {epoch+1}停止训练。")
            print(f"最佳模型在epoch {best_epoch+1}，验证集RMSE: {best_val_rmse:.6f}")
            print(f"已等待 {patience_counter}/{args.patience} 个epoch无改善")
            break
        
        # Baseline workflow step.
        current_time = time.time()
        elapsed_since_rest = current_time - last_rest_time
        if elapsed_since_rest >= args.rest_interval:
            elapsed_hours = elapsed_since_rest / 3600
            rest_minutes = args.rest_duration / 60
            print(f"\n已运行 {elapsed_hours:.2f} 小时（{elapsed_since_rest:.0f} 秒），当前epoch已完成")
            print(f"休息 {rest_minutes:.1f} 分钟（{args.rest_duration:.0f} 秒）让CPU/GPU有时间休息...")
            time.sleep(args.rest_duration)
            last_rest_time = time.time()
    
    # Run the training step.
    total_time = time.time() - training_start_time
    avg_time_per_epoch = total_time / (epoch + 1 - start_epoch) if epoch + 1 > start_epoch else 0
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"最佳模型在epoch {best_epoch+1}，验证集RMSE: {best_val_rmse:.6f}")
    print(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"平均每轮时间: {avg_time_per_epoch:.2f}秒")
    
    # Load the input data.
    best_model_path = os.path.join(output_dir, f'seed_{args.seed}_best.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Baseline workflow step.
    print("\n【最终验证集指标】")
    final_val_results = evaluate(model, val_loader, args.device)
    print_metrics(final_val_results['metrics'])
    
    # Evaluate the test subset.
    print("\n" + "=" * 80)
    print("开始测试集评估")
    print("=" * 80)
    print(f"测试集样本数: {len(test_dataset)}")
    
    test_results = evaluate(model, test_loader, args.device)
    
    print("\n【测试集指标】")
    print_metrics(test_results['metrics'])
    
    # Save the generated artifacts.
    save_results(test_results['predictions'], test_results['targets'], 'test', 
                 os.path.join(output_dir, 'results'))
    
    # Save the generated artifacts.
    print("\n保存结果文件...")
    
    # Save the generated artifacts.
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'results', 'train_history.csv'), index=False)
    
    # Save the generated artifacts.
    train_results_final = evaluate(model, train_loader, args.device)
    save_results(train_results_final['predictions'], train_results_final['targets'], 'train', 
                 os.path.join(output_dir, 'results'))
    save_results(final_val_results['predictions'], final_val_results['targets'], 'val', 
                 os.path.join(output_dir, 'results'))
    
    # Save the generated artifacts.
    save_metrics_txt(best_epoch, best_val_rmse, epoch + 1, total_time, avg_time_per_epoch,
                     final_val_results['metrics'], test_results['metrics'] if test_results else None, output_dir)
    
    print("所有文件已保存完成！")


def main():
    parser = argparse.ArgumentParser(description='CGIB Training - 支持单个seed或批量训练所有seed')
    parser.add_argument('--data_path', type=str, default=str(TOTAL_CSV), help='Input comparison CSV.')
    parser.add_argument(
        '--allow_random_row_split',
        action='store_true',
        help='Allow the legacy random row split when the CSV has no split column.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(EXPERIMENT_ROOT / 'runs' / 'cgib'),
        help='Directory for seed-specific CGIB run artifacts.',
    )
    parser.add_argument('--seed', type=int, default=None, choices=[42, 123, 456, 789, 2024, None], help='随机种子（如果指定则只训练该seed）')
    parser.add_argument('--all_seeds', action='store_true', help='按顺序训练所有5个seed（42, 123, 456, 789, 2024）')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='GNN层数')
    parser.add_argument('--set2set_steps', type=int, default=3, help='Set2Set步骤数')
    parser.add_argument('--beta', type=float, default=1e-3, help='信息瓶颈平衡参数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率（默认: 5e-4，已降低以防止预测值饱和）')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=400, help='训练轮数')
    parser.add_argument('--patience', type=int, default=80, help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=0.0, help='早停最小改善')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='检查点保存频率（每N个epoch）')
    parser.add_argument('--rest_interval', type=float, default=7200, help='休息间隔（秒），默认2小时')
    parser.add_argument('--rest_duration', type=float, default=300, help='休息时长（秒），默认5分钟')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--gnn_type', type=str, default='mpnn', choices=['mpnn', 'gin'], help='GNN类型')
    parser.add_argument('--use_contrastive', action='store_true', help='使用对比学习')
    parser.add_argument('--no_constrain_output', dest='constrain_output', action='store_false', default=True, help='禁用输出约束（默认启用输出约束）')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # Baseline workflow step.
    if not hasattr(args, 'constrain_output') or args.constrain_output is None:
        args.constrain_output = True
    
    # Run the training step.
    if args.all_seeds:
        # Run the training step.
        seeds = [42, 123, 456, 789, 2024]
        print("=" * 80)
        print("开始运行所有seed的训练")
        print("=" * 80)
        print(f"数据路径: {args.data_path}")
        print(f"Seeds: {seeds}")
        print("=" * 80 + "\n")
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n{'='*80}")
            print(f"训练 Seed {seed} ({i}/{len(seeds)})")
            print(f"{'='*80}\n")
            
            # Baseline workflow step.
            seed_args = argparse.Namespace(**vars(args))
            seed_args.seed = seed
            
            try:
                train_single_seed(seed_args)
                print(f"\nSeed {seed} 训练完成！\n")
            except Exception as e:
                print(f"\nSeed {seed} 训练失败，错误: {str(e)}\n")
                print("是否继续下一个seed？(y/n): ", end='')
                try:
                    response = input().strip().lower()
                    if response != 'y':
                        print("终止训练")
                        return
                except:
                    print("终止训练")
                    return
        
        print("\n" + "=" * 80)
        print("所有seed的训练已完成！")
        print("=" * 80)
    else:
        # Run the training step.
        if args.seed is None:
            args.seed = 42  # Baseline workflow step.
        train_single_seed(args)


if __name__ == '__main__':
    main()

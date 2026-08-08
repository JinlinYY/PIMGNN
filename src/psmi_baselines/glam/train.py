"""Implement the glam train baseline module."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from .dataset.data_loader import load_LLE_dataset, collate_fn
from .model.glam import GLAM_LLE
from .config import Config, default_config
from psmi_baselines.paths import EXPERIMENT_ROOT


class LLEDataset(Dataset):
    """Represent the LLEDataset baseline component."""
    def __init__(self, data_dict):
        self.data = data_dict['data']
        self.labels = data_dict['labels']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def set_seed(seed):
    """Run the set seed baseline operation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info(device):
    """Run the get device info baseline operation."""
    info = {'device': str(device)}
    if device.type == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['gpu_memory_gb'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}"
    else:
        info['gpu_name'] = "N/A"
        info['cuda_version'] = "N/A"
        info['gpu_memory_gb'] = "N/A"
    return info


def count_unique_systems(data_list):
    """Run the count unique systems baseline operation."""
    system_nos = [item.get('system_no', idx) for idx, item in enumerate(data_list)]
    return len(set(system_nos))


def calculate_metrics_E_R(predictions, labels):
    """Run the calculate metrics E R baseline operation."""
    # Baseline workflow step.
    pred_E = predictions[:, :3]
    true_E = labels[:, :3]
    
    # Baseline workflow step.
    pred_R = predictions[:, 3:]
    true_R = labels[:, 3:]
    
    # Compute evaluation metrics.
    mae_E = mean_absolute_error(true_E, pred_E)
    rmse_E = np.sqrt(mean_squared_error(true_E, pred_E))
    
    # Baseline workflow step.
    r2_E_list = []
    for dim in range(3):
        y_true_dim = true_E[:, dim]
        y_pred_dim = pred_E[:, dim]
        ss_res = np.sum((y_true_dim - y_pred_dim) ** 2)
        ss_tot = np.sum((y_true_dim - np.mean(y_true_dim)) ** 2)
        if ss_tot < 1e-10:  # Baseline workflow step.
            r2_dim = 0.0
        else:
            r2_dim = 1 - (ss_res / ss_tot)
        r2_E_list.append(r2_dim)
    r2_E = np.mean(r2_E_list)
    
    # Compute evaluation metrics.
    mae_R = mean_absolute_error(true_R, pred_R)
    rmse_R = np.sqrt(mean_squared_error(true_R, pred_R))
    
    # Baseline workflow step.
    r2_R_list = []
    for dim in range(3):
        y_true_dim = true_R[:, dim]
        y_pred_dim = pred_R[:, dim]
        ss_res = np.sum((y_true_dim - y_pred_dim) ** 2)
        ss_tot = np.sum((y_true_dim - np.mean(y_true_dim)) ** 2)
        if ss_tot < 1e-10:  # Baseline workflow step.
            r2_dim = 0.0
        else:
            r2_dim = 1 - (ss_res / ss_tot)
        r2_R_list.append(r2_dim)
    r2_R = np.mean(r2_R_list)
    
    return {
        'mae_E': mae_E, 'rmse_E': rmse_E, 'r2_E': r2_E,
        'mae_R': mae_R, 'rmse_R': rmse_R, 'r2_R': r2_R
    }


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Run the train epoch baseline operation."""
    model.train()
    total_loss = 0.0
    
    for batch_data, batch_labels in tqdm(dataloader, desc="Training", leave=False):
        # Process the experiment data.
        il_graph = batch_data['il_graph'].to(device)
        comp2_graph = batch_data['comp2_graph'].to(device)
        comp3_graph = batch_data['comp3_graph'].to(device)
        temperature = batch_data['temperature'].to(device)
        
        # Baseline workflow step.
        labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
        
        # Baseline workflow step.
        optimizer.zero_grad()
        outputs = model(il_graph, comp2_graph, comp3_graph, temperature)
        
        # Compute the training loss.
        loss = criterion(outputs, labels)
        
        # Baseline workflow step.
        loss.backward()
        
        # Update model gradients.
        if config.training.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Run the evaluate baseline operation."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(dataloader, desc="Evaluating", leave=False):
            # Process the experiment data.
            il_graph = batch_data['il_graph'].to(device)
            comp2_graph = batch_data['comp2_graph'].to(device)
            comp3_graph = batch_data['comp3_graph'].to(device)
            temperature = batch_data['temperature'].to(device)
            
            # Baseline workflow step.
            labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            
            # Baseline workflow step.
            outputs = model(il_graph, comp2_graph, comp3_graph, temperature)
            
            # Compute the training loss.
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Generate model predictions.
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Generate model predictions.
    if not hasattr(evaluate, '_debug_printed'):
        print(f"\n[调试] 预测值范围: min={all_preds.min():.6f}, max={all_preds.max():.6f}, mean={all_preds.mean():.6f}")
        print(f"[调试] 真实值范围: min={all_labels.min():.6f}, max={all_labels.max():.6f}, mean={all_labels.mean():.6f}")
        print(f"[调试] 预测值各维度均值: {all_preds.mean(axis=0)}")
        print(f"[调试] 真实值各维度均值: {all_labels.mean(axis=0)}")
        print(f"[调试] 预测值各维度标准差: {all_preds.std(axis=0)}")
        print(f"[调试] 真实值各维度标准差: {all_labels.std(axis=0)}")
        # Baseline workflow step.
        print(f"[调试] R相真实值方差: {np.var(all_labels[:, 3:], axis=0)}")
        print(f"[调试] R相预测值方差: {np.var(all_preds[:, 3:], axis=0)}")
        evaluate._debug_printed = True
    
    # Compute evaluation metrics.
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    
    # Baseline workflow step.
    r2_list = []
    for dim in range(all_labels.shape[1]):
        y_true_dim = all_labels[:, dim]
        y_pred_dim = all_preds[:, dim]
        ss_res = np.sum((y_true_dim - y_pred_dim) ** 2)
        ss_tot = np.sum((y_true_dim - np.mean(y_true_dim)) ** 2)
        if ss_tot < 1e-10:  # Baseline workflow step.
            r2_dim = 0.0
        else:
            r2_dim = 1 - (ss_res / ss_tot)
        r2_list.append(r2_dim)
    r2 = np.mean(r2_list)
    
    # Baseline workflow step.
    # Configure the output artifacts.
    mae_per_dim = []
    rmse_per_dim = []
    for dim in range(all_labels.shape[1]):
        mae_dim = mean_absolute_error(all_labels[:, dim], all_preds[:, dim])
        rmse_dim = np.sqrt(mean_squared_error(all_labels[:, dim], all_preds[:, dim]))
        mae_per_dim.append(mae_dim)
        rmse_per_dim.append(rmse_dim)
    
    mae_per_dim = np.array(mae_per_dim)
    rmse_per_dim = np.array(rmse_per_dim)
    mae_mean = np.mean(mae_per_dim)
    mae_std = np.std(mae_per_dim)
    rmse_mean = np.mean(rmse_per_dim)
    rmse_std = np.std(rmse_per_dim)
    
    # Compute evaluation metrics.
    metrics_E_R = calculate_metrics_E_R(all_preds, all_labels)
    
    # Compute evaluation metrics.
    # Baseline workflow step.
    mae_E_per_dim = []
    rmse_E_per_dim = []
    for dim in range(3):
        mae_dim = mean_absolute_error(all_labels[:, dim], all_preds[:, dim])
        rmse_dim = np.sqrt(mean_squared_error(all_labels[:, dim], all_preds[:, dim]))
        mae_E_per_dim.append(mae_dim)
        rmse_E_per_dim.append(rmse_dim)
    mae_E_per_dim = np.array(mae_E_per_dim)
    rmse_E_per_dim = np.array(rmse_E_per_dim)
    mae_E_mean = np.mean(mae_E_per_dim)
    mae_E_std = np.std(mae_E_per_dim)
    rmse_E_mean = np.mean(rmse_E_per_dim)
    rmse_E_std = np.std(rmse_E_per_dim)
    
    # Baseline workflow step.
    mae_R_per_dim = []
    rmse_R_per_dim = []
    for dim in range(3, 6):
        mae_dim = mean_absolute_error(all_labels[:, dim], all_preds[:, dim])
        rmse_dim = np.sqrt(mean_squared_error(all_labels[:, dim], all_preds[:, dim]))
        mae_R_per_dim.append(mae_dim)
        rmse_R_per_dim.append(rmse_dim)
    mae_R_per_dim = np.array(mae_R_per_dim)
    rmse_R_per_dim = np.array(rmse_R_per_dim)
    mae_R_mean = np.mean(mae_R_per_dim)
    mae_R_std = np.std(mae_R_per_dim)
    rmse_R_mean = np.mean(rmse_R_per_dim)
    rmse_R_std = np.std(rmse_R_per_dim)
    
    return {
        'loss': total_loss / len(dataloader),
        'mse': mse,
        'mae': mae,
        'mae_mean': mae_mean,
        'mae_std': mae_std,
        'rmse': rmse,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'r2': r2,
        'mae_E': metrics_E_R['mae_E'],
        'mae_E_mean': mae_E_mean,
        'mae_E_std': mae_E_std,
        'rmse_E': metrics_E_R['rmse_E'],
        'rmse_E_mean': rmse_E_mean,
        'rmse_E_std': rmse_E_std,
        'r2_E': metrics_E_R['r2_E'],
        'mae_R': metrics_E_R['mae_R'],
        'mae_R_mean': mae_R_mean,
        'mae_R_std': mae_R_std,
        'rmse_R': metrics_E_R['rmse_R'],
        'rmse_R_mean': rmse_R_mean,
        'rmse_R_std': rmse_R_std,
        'r2_R': metrics_E_R['r2_R'],
        'predictions': all_preds,
        'labels': all_labels
    }


def get_optimizer(model, config):
    """Run the get optimizer baseline operation."""
    if config.training.optimizer.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def get_scheduler(optimizer, config):
    """Run the get scheduler baseline operation."""
    if config.training.scheduler.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience
            # Configure experiment parameters.
        )
    elif config.training.scheduler.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.scheduler_patience,
            gamma=config.training.scheduler_factor
        )
    elif config.training.scheduler.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs
        )
    else:
        return None


def find_latest_checkpoint(checkpoint_dir):
    """Run the find latest checkpoint baseline operation."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt') and '_with_history' not in filename:
            try:
                epoch_num = int(filename.replace('checkpoint_epoch_', '').replace('.pt', ''))
                filepath = os.path.join(checkpoint_dir, filename)
                checkpoint_files.append((epoch_num, filepath))
            except ValueError:
                continue
    
    if not checkpoint_files:
        return None
    
    # Baseline workflow step.
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_files[0][1]  # Configure repository paths.


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Run the load checkpoint baseline operation."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the input data.
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the input data.
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load the input data.
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("警告: 无法加载调度器状态，将使用当前调度器状态")
    
    # Run the training step.
    resume_info = {
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss', float('inf')),
        'config': checkpoint.get('config', None)
    }
    
    # Load the input data.
    history_path = checkpoint_path.replace('.pt', '_with_history.pt')
    if os.path.exists(history_path):
        history_checkpoint = torch.load(history_path, map_location=device)
        resume_info['train_history'] = history_checkpoint.get('train_history', [])
    else:
        resume_info['train_history'] = []
    
    return resume_info


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, filepath):
    """Run the save checkpoint baseline operation."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config.to_dict()
    }
    # Save the generated artifacts.
    if scheduler is not None:
        try:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        except:
            pass
    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath}")


def save_results_csv(train_history, train_preds, train_labels, val_preds, val_labels, 
                     test_preds, test_labels, result_dir):
    """Run the save results csv baseline operation."""
    # Run the training step.
    df_history = pd.DataFrame(train_history)
    df_history.to_csv(os.path.join(result_dir, 'train_history.csv'), index=False)
    
    # Run the training step.
    train_val_data = []
    for i in range(len(train_preds)):
        train_val_data.append({
            'Ex1': train_labels[i, 0], 'Ex2': train_labels[i, 1], 'Ex3': train_labels[i, 2],
            'Rx1': train_labels[i, 3], 'Rx2': train_labels[i, 4], 'Rx3': train_labels[i, 5],
            'pred_Ex1': train_preds[i, 0], 'pred_Ex2': train_preds[i, 1], 'pred_Ex3': train_preds[i, 2],
            'pred_Rx1': train_preds[i, 3], 'pred_Rx2': train_preds[i, 4], 'pred_Rx3': train_preds[i, 5],
            'split': 'train'
        })
    for i in range(len(val_preds)):
        train_val_data.append({
            'Ex1': val_labels[i, 0], 'Ex2': val_labels[i, 1], 'Ex3': val_labels[i, 2],
            'Rx1': val_labels[i, 3], 'Rx2': val_labels[i, 4], 'Rx3': val_labels[i, 5],
            'pred_Ex1': val_preds[i, 0], 'pred_Ex2': val_preds[i, 1], 'pred_Ex3': val_preds[i, 2],
            'pred_Rx1': val_preds[i, 3], 'pred_Rx2': val_preds[i, 4], 'pred_Rx3': val_preds[i, 5],
            'split': 'val'
        })
    df_train_val = pd.DataFrame(train_val_data)
    df_train_val.to_csv(os.path.join(result_dir, 'training_results.csv'), index=False)
    
    # Evaluate the test subset.
    test_data = []
    for i in range(len(test_preds)):
        test_data.append({
            'Ex1': test_labels[i, 0], 'Ex2': test_labels[i, 1], 'Ex3': test_labels[i, 2],
            'Rx1': test_labels[i, 3], 'Rx2': test_labels[i, 4], 'Rx3': test_labels[i, 5],
            'pred_Ex1': test_preds[i, 0], 'pred_Ex2': test_preds[i, 1], 'pred_Ex3': test_preds[i, 2],
            'pred_Rx1': test_preds[i, 3], 'pred_Rx2': test_preds[i, 4], 'pred_Rx3': test_preds[i, 5]
        })
    df_test = pd.DataFrame(test_data)
    df_test.to_csv(os.path.join(result_dir, 'test_results.csv'), index=False)


def save_metrics_txt(best_epoch, best_val_mse, best_val_metrics, best_test_metrics,
                     total_time, avg_time_per_epoch, result_dir):
    """Run the save metrics txt baseline operation."""
    with open(os.path.join(result_dir, 'training_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("训练指标总结\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"最佳模型在epoch {best_epoch}，验证集MSE: {best_val_mse:.6f}\n\n")
        f.write("最佳验证集指标:\n")
        f.write(f"  【平均指标】 MAE={best_val_metrics['mae_mean']:.6f}±{best_val_metrics['mae_std']:.6f}  RMSE={best_val_metrics['rmse_mean']:.6f}±{best_val_metrics['rmse_std']:.6f}  R²={best_val_metrics['r2']:.6f}\n")
        f.write(f"  【E相指标】  MAE={best_val_metrics['mae_E_mean']:.6f}±{best_val_metrics['mae_E_std']:.6f}  RMSE={best_val_metrics['rmse_E_mean']:.6f}±{best_val_metrics['rmse_E_std']:.6f}  R²={best_val_metrics['r2_E']:.6f}\n")
        f.write(f"  【R相指标】  MAE={best_val_metrics['mae_R_mean']:.6f}±{best_val_metrics['mae_R_std']:.6f}  RMSE={best_val_metrics['rmse_R_mean']:.6f}±{best_val_metrics['rmse_R_std']:.6f}  R²={best_val_metrics['r2_R']:.6f}\n\n")
        f.write("测试集指标:\n")
        f.write(f"  【平均指标】 MAE={best_test_metrics['mae_mean']:.6f}±{best_test_metrics['mae_std']:.6f}  RMSE={best_test_metrics['rmse_mean']:.6f}±{best_test_metrics['rmse_std']:.6f}  R²={best_test_metrics['r2']:.6f}\n")
        f.write(f"  【E相指标】  MAE={best_test_metrics['mae_E_mean']:.6f}±{best_test_metrics['mae_E_std']:.6f}  RMSE={best_test_metrics['rmse_E_mean']:.6f}±{best_test_metrics['rmse_E_std']:.6f}  R²={best_test_metrics['r2_E']:.6f}\n")
        f.write(f"  【R相指标】  MAE={best_test_metrics['mae_R_mean']:.6f}±{best_test_metrics['mae_R_std']:.6f}  RMSE={best_test_metrics['rmse_R_mean']:.6f}±{best_test_metrics['rmse_R_std']:.6f}  R²={best_test_metrics['r2_R']:.6f}\n\n")
        f.write(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)\n")
        f.write(f"平均每轮时间: {avg_time_per_epoch:.2f}秒\n")


def main(config=None, seed=None, base_output_dir=None, resume_checkpoint=None, auto_resume=False):
    """Run the main baseline operation."""
    if config is None:
        config = default_config
    
    # Baseline workflow step.
    if seed is not None:
        config.seed = seed
    
    # Save the generated artifacts.
    if base_output_dir is None:
        base_output_dir = str(EXPERIMENT_ROOT / "runs" / "glam")
    
    # Configure repository paths.
    seed_dir = os.path.join(base_output_dir, f'seed_{config.seed}')
    config.model_save_dir = os.path.join(seed_dir, 'checkpoint')
    config.result_dir = os.path.join(seed_dir, 'results')
    # Save the generated artifacts.
    best_model_dir = seed_dir
    
    # Configure repository paths.
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Baseline workflow step.
    resume_from_epoch = 0
    resume_train_history = []
    resume_best_val_mse = float('inf')
    resume_best_val_loss = float('inf')
    is_resuming = False
    
    if resume_checkpoint:
        # Handle model checkpoints.
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f"指定的检查点文件不存在: {resume_checkpoint}")
        checkpoint_path = resume_checkpoint
        is_resuming = True
    elif auto_resume:
        # Handle model checkpoints.
        checkpoint_path = find_latest_checkpoint(config.model_save_dir)
        if checkpoint_path:
            is_resuming = True
            print(f"\n自动找到最新检查点: {checkpoint_path}")
        else:
            print("\n未找到检查点，从头开始训练")
    
    # Set the random seed.
    if not is_resuming:
        set_seed(config.seed)
    
    # Configure the runtime device.
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(config.device)
    
    device_info = get_device_info(device)
    
    print("=" * 100)
    print("【训练配置参数】")
    print("=" * 100)
    
    # Load the input data.
    print("\n【数据集信息】")
    datasets = load_LLE_dataset(
        config.data.csv_path,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_state=config.data.random_state
    )
    
    train_data_list = datasets['train']['data']
    val_data_list = datasets['val']['data']
    test_data_list = datasets['test']['data']
    
    train_systems = count_unique_systems(train_data_list)
    val_systems = count_unique_systems(val_data_list)
    test_systems = count_unique_systems(test_data_list)
    
    print(f"  训练集样本数: {len(train_data_list)}")
    print(f"  验证集样本数: {len(val_data_list)}")
    print(f"  测试集样本数: {len(test_data_list)}")
    print(f"  训练集体系数: {train_systems}")
    print(f"  验证集体系数: {val_systems}")
    print(f"  测试集体系数: {test_systems}")
    
    # Configure the runtime device.
    print("\n【设备配置】")
    print(f"  设备: {device_info['device']}")
    print(f"  GPU名称: {device_info['gpu_name']}")
    print(f"  CUDA版本: {device_info['cuda_version']}")
    print(f"  GPU内存: {device_info['gpu_memory_gb']} GB")
    
    # Run the training step.
    print("\n【训练参数】")
    print(f"  随机种子: {config.seed}")
    print(f"  训练轮数: {config.training.num_epochs}")
    print(f"  批次大小: {config.data.batch_size}")
    print(f"  学习率: {config.training.learning_rate}")
    print(f"  权重衰减: {config.training.weight_decay}")
    print(f"  早停耐心值: {config.training.early_stop_patience}")
    print(f"  早停最小改善: {config.training.early_stop_min_delta}")
    print(f"  检查点保存频率: 每 {config.training.checkpoint_save_freq} 个epoch")
    print(f"  休息策略: 每 {config.training.rest_interval_hours} 小时休息 {config.training.rest_duration}秒（{config.training.rest_duration/60:.1f}分钟）")
    if is_resuming:
        print(f"  断点续训: 是（从检查点恢复）")
    else:
        print(f"  断点续训: 否（从头开始）")
    
    # Configure the baseline model.
    print("\n【模型超参数】")
    print(f"  隐藏层维度: {config.model.hidden_dim}")
    print(f"  图神经网络层数: {config.model.num_mp_layers}")
    print(f"  消息传递类型: {config.model.mp_type}")
    print(f"  Dropout率: {config.model.dropout}")
    print(f"  输出维度: {config.model.out_dim} (LLE任务: Ex1, Ex2, Ex3, Rx1, Rx2, Rx3)")
    
    # Configure repository paths.
    print("\n【路径信息】")
    print(f"  基础输出目录: {base_output_dir}")
    print(f"  Seed目录: {seed_dir}")
    print(f"  检查点保存目录: {config.model_save_dir}")
    print(f"  结果保存目录: {config.result_dir}")
    print(f"  最佳模型保存目录: {best_model_dir}")
    
    # Save the generated artifacts.
    print("\n【文件保存路径】")
    print(f"  检查点文件: {os.path.join(config.model_save_dir, 'checkpoint_epoch_*.pt')}")
    print(f"  最佳模型权重: {os.path.join(best_model_dir, 'GLAM.pt')}")
    print(f"  训练历史CSV: {os.path.join(config.result_dir, 'train_history.csv')}")
    print(f"  训练/验证结果CSV: {os.path.join(config.result_dir, 'training_results.csv')}")
    print(f"  测试集结果CSV: {os.path.join(config.result_dir, 'test_results.csv')}")
    print(f"  训练指标TXT: {os.path.join(config.result_dir, 'training_metrics.txt')}")
    print(f"  最佳指标TXT: {os.path.join(config.result_dir, 'best_metrics.txt')}")
    print(f"  测试指标TXT: {os.path.join(config.result_dir, 'test_metrics.txt')}")
    
    print("=" * 100)
    
    # Load the input data.
    train_dataset = LLEDataset(datasets['train'])
    val_dataset = LLEDataset(datasets['val'])
    test_dataset = LLEDataset(datasets['test'])
    
    def custom_collate_fn(batch):
        """Run the custom collate fn baseline operation."""
        data_items = [item[0] for item in batch]
        labels = np.array([item[1] for item in batch])
        graph_batch = collate_fn(data_items)
        return graph_batch, labels
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=config.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.data.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.data.num_workers
    )
    
    # Baseline workflow step.
    sample_graph = datasets['train']['data'][0]['il_graph']
    node_dim = sample_graph.x.shape[1]
    config.model.node_dim = node_dim
    
    # Configure the baseline model.
    model_config = {
        'norm_type': config.model.norm_type,
        'activation': config.model.activation,
        'mp_type': config.model.mp_type,
        'pool_type': config.model.pool_type,
        'dropout': config.model.dropout,
        'num_mp_layers': config.model.num_mp_layers,
        'hidden_dim': config.model.hidden_dim,
        'fusion_type': config.model.fusion_type
    }
    
    # Configure the baseline model.
    model = GLAM_LLE(
        node_dim=node_dim,
        out_dim=config.model.out_dim,
        config=model_config
    ).to(device)
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # Compute the training loss.
    criterion = nn.MSELoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Run the training step.
    best_val_loss = float('inf')
    best_val_mse = float('inf')
    patience_counter = 0
    train_history = []
    best_train_preds = None
    best_train_labels = None
    best_val_preds = None
    best_val_labels = None
    
    # Load the input data.
    if is_resuming:
        resume_info = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        resume_from_epoch = resume_info['epoch']
        resume_train_history = resume_info.get('train_history', [])
        resume_best_val_mse = resume_info.get('val_loss', float('inf'))
        resume_best_val_loss = resume_from_epoch
        
        # Run the training step.
        train_history = resume_train_history
        
        # Compute evaluation metrics.
        best_val_mse = resume_best_val_mse
        best_val_loss = resume_best_val_loss
        
        print(f"\n从epoch {resume_from_epoch} 恢复训练")
        print(f"最佳验证损失: {best_val_mse:.6f}")
        print(f"已训练历史记录数: {len(train_history)}")
        print("=" * 100)
    else:
        resume_from_epoch = 0
    
    print("\n开始训练...")
    print("=" * 100)
    
    start_time = time.time()
    last_rest_time = start_time  # Baseline workflow step.
    
    # Run the training step.
    for epoch in range(resume_from_epoch, config.training.num_epochs):
        epoch_start = time.time()
        
        # Run the training step.
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Evaluate the validation subset.
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Baseline workflow step.
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Configure the output artifacts.
        train_metrics = evaluate(model, train_loader, criterion, device)
        
        # Baseline workflow step.
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'epoch_time': epoch_time,
            'train_mse': train_metrics['mse'],
            'train_rmse': train_metrics['rmse'],
            'train_rmse_mean': train_metrics['rmse_mean'],
            'train_rmse_std': train_metrics['rmse_std'],
            'train_mae': train_metrics['mae'],
            'train_mae_mean': train_metrics['mae_mean'],
            'train_mae_std': train_metrics['mae_std'],
            'train_r2': train_metrics['r2'],
            'train_mae_E': train_metrics['mae_E'],
            'train_mae_E_mean': train_metrics['mae_E_mean'],
            'train_mae_E_std': train_metrics['mae_E_std'],
            'train_rmse_E': train_metrics['rmse_E'],
            'train_rmse_E_mean': train_metrics['rmse_E_mean'],
            'train_rmse_E_std': train_metrics['rmse_E_std'],
            'train_r2_E': train_metrics['r2_E'],
            'train_mae_R': train_metrics['mae_R'],
            'train_mae_R_mean': train_metrics['mae_R_mean'],
            'train_mae_R_std': train_metrics['mae_R_std'],
            'train_rmse_R': train_metrics['rmse_R'],
            'train_rmse_R_mean': train_metrics['rmse_R_mean'],
            'train_rmse_R_std': train_metrics['rmse_R_std'],
            'train_r2_R': train_metrics['r2_R'],
            'val_mse': val_metrics['mse'],
            'val_rmse': val_metrics['rmse'],
            'val_rmse_mean': val_metrics['rmse_mean'],
            'val_rmse_std': val_metrics['rmse_std'],
            'val_mae': val_metrics['mae'],
            'val_mae_mean': val_metrics['mae_mean'],
            'val_mae_std': val_metrics['mae_std'],
            'val_r2': val_metrics['r2'],
            'val_mae_E': val_metrics['mae_E'],
            'val_mae_E_mean': val_metrics['mae_E_mean'],
            'val_mae_E_std': val_metrics['mae_E_std'],
            'val_rmse_E': val_metrics['rmse_E'],
            'val_rmse_E_mean': val_metrics['rmse_E_mean'],
            'val_rmse_E_std': val_metrics['rmse_E_std'],
            'val_r2_E': val_metrics['r2_E'],
            'val_mae_R': val_metrics['mae_R'],
            'val_mae_R_mean': val_metrics['mae_R_mean'],
            'val_mae_R_std': val_metrics['mae_R_std'],
            'val_rmse_R': val_metrics['rmse_R'],
            'val_rmse_R_mean': val_metrics['rmse_R_mean'],
            'val_rmse_R_std': val_metrics['rmse_R_std'],
            'val_r2_R': val_metrics['r2_R']
        }
        train_history.append(history_entry)
        
        # Baseline workflow step.
        best_loss_str = f"{best_val_mse:.6f} (epoch {best_val_loss})" if best_val_loss != float('inf') else "initial"
        print(f"Epoch {epoch+1}/{config.training.num_epochs} | 训练时间: {epoch_time:.2f}秒 | Train Loss: {train_loss:.6f}")
        print(f"Best Loss: {best_loss_str}")
        print("=" * 100)
        print("【训练集指标】")
        print(f"  【平均指标】 MAE={train_metrics['mae_mean']:.6f}±{train_metrics['mae_std']:.6f}  RMSE={train_metrics['rmse_mean']:.6f}±{train_metrics['rmse_std']:.6f}  R²={train_metrics['r2']:.6f}")
        print(f"  【E相指标】  MAE={train_metrics['mae_E_mean']:.6f}±{train_metrics['mae_E_std']:.6f}  RMSE={train_metrics['rmse_E_mean']:.6f}±{train_metrics['rmse_E_std']:.6f}  R²={train_metrics['r2_E']:.6f}")
        print(f"  【R相指标】  MAE={train_metrics['mae_R_mean']:.6f}±{train_metrics['mae_R_std']:.6f}  RMSE={train_metrics['rmse_R_mean']:.6f}±{train_metrics['rmse_R_std']:.6f}  R²={train_metrics['r2_R']:.6f}")
        print("\n【验证集指标】")
        print(f"  【平均指标】 MAE={val_metrics['mae_mean']:.6f}±{val_metrics['mae_std']:.6f}  RMSE={val_metrics['rmse_mean']:.6f}±{val_metrics['rmse_std']:.6f}  R²={val_metrics['r2']:.6f}")
        print(f"  【E相指标】  MAE={val_metrics['mae_E_mean']:.6f}±{val_metrics['mae_E_std']:.6f}  RMSE={val_metrics['rmse_E_mean']:.6f}±{val_metrics['rmse_E_std']:.6f}  R²={val_metrics['r2_E']:.6f}")
        print(f"  【R相指标】  MAE={val_metrics['mae_R_mean']:.6f}±{val_metrics['mae_R_std']:.6f}  RMSE={val_metrics['rmse_R_mean']:.6f}±{val_metrics['rmse_R_std']:.6f}  R²={val_metrics['r2_R']:.6f}")
        print("=" * 100)
        
        # Save the generated artifacts.
        if (epoch + 1) % config.training.checkpoint_save_freq == 0 or (epoch + 1) == config.training.num_epochs:
            checkpoint_path = os.path.join(config.model_save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_metrics['mse'], config, checkpoint_path)
            # Save the generated artifacts.
            checkpoint_path_with_history = checkpoint_path.replace('.pt', '_with_history.pt')
            checkpoint_data = torch.load(checkpoint_path)
            checkpoint_data['train_history'] = train_history
            torch.save(checkpoint_data, checkpoint_path_with_history)
            print(f"  检查点保存路径: {checkpoint_path}")
            print(f"  训练历史保存路径: {checkpoint_path_with_history}")
        
        # Baseline workflow step.
        current_time = time.time()
        elapsed_hours = (current_time - last_rest_time) / 3600.0  # Baseline workflow step.
        
        if elapsed_hours >= config.training.rest_interval_hours and (epoch + 1) < config.training.num_epochs:
            print("\n" + "=" * 100)
            print(f"已训练 {elapsed_hours:.2f} 小时，休息 {config.training.rest_duration}秒（{config.training.rest_duration/60:.1f}分钟）让CPU/GPU有时间休息...")
            print("=" * 100 + "\n")
            time.sleep(config.training.rest_duration)
            last_rest_time = time.time()  # Baseline workflow step.
        
        # Save the generated artifacts.
        if val_metrics['mse'] < best_val_mse - config.training.early_stop_min_delta:
            best_val_mse = val_metrics['mse']
            best_val_loss = epoch + 1
            patience_counter = 0
            # Save the generated artifacts.
            model_path = os.path.join(best_model_dir, 'GLAM.pt')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_metrics['mse'], config, model_path)
            print(f"  最佳模型保存路径: {model_path}")
            # Save the generated artifacts.
            best_train_preds = train_metrics['predictions']
            best_train_labels = train_metrics['labels']
            best_val_preds = val_metrics['predictions']
            best_val_labels = val_metrics['labels']
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stop_patience:
                print(f"\n早停触发！在epoch {epoch+1}停止训练。")
                print(f"最佳模型在epoch {best_val_loss}，验证集MSE: {best_val_mse:.6f}")
                print(f"已等待 {patience_counter}/{config.training.early_stop_patience} 个epoch无改善")
                break
    
    total_time = time.time() - start_time
    avg_time_per_epoch = total_time / len(train_history)
    
    print("\n" + "=" * 100)
    print("训练完成！")
    print(f"最佳模型在epoch {best_val_loss}，验证集MSE: {best_val_mse:.6f}")
    print(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"平均每轮时间: {avg_time_per_epoch:.2f}秒")
    
    # Load the input data.
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load(os.path.join(best_model_dir, 'GLAM.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # Save the generated artifacts.
    save_results_csv(
        train_history,
        best_train_preds, best_train_labels,
        best_val_preds, best_val_labels,
        test_metrics['predictions'], test_metrics['labels'],
        config.result_dir
    )
    
    # Compute evaluation metrics.
    if best_val_preds is not None:
        # Compute evaluation metrics.
        best_val_preds_np = best_val_preds if isinstance(best_val_preds, np.ndarray) else best_val_preds.cpu().numpy()
        best_val_labels_np = best_val_labels if isinstance(best_val_labels, np.ndarray) else best_val_labels.cpu().numpy()
        
        # Compute evaluation metrics.
        mse = mean_squared_error(best_val_labels_np, best_val_preds_np)
        mae = mean_absolute_error(best_val_labels_np, best_val_preds_np)
        rmse = np.sqrt(mse)
        
        # Baseline workflow step.
        r2_list = []
        for dim in range(best_val_labels_np.shape[1]):
            y_true_dim = best_val_labels_np[:, dim]
            y_pred_dim = best_val_preds_np[:, dim]
            ss_res = np.sum((y_true_dim - y_pred_dim) ** 2)
            ss_tot = np.sum((y_true_dim - np.mean(y_true_dim)) ** 2)
            if ss_tot < 1e-10:  # Baseline workflow step.
                r2_dim = 0.0
            else:
                r2_dim = 1 - (ss_res / ss_tot)
            r2_list.append(r2_dim)
        r2 = np.mean(r2_list)
        
        # Baseline workflow step.
        mae_per_dim = []
        rmse_per_dim = []
        for dim in range(best_val_labels_np.shape[1]):
            mae_dim = mean_absolute_error(best_val_labels_np[:, dim], best_val_preds_np[:, dim])
            rmse_dim = np.sqrt(mean_squared_error(best_val_labels_np[:, dim], best_val_preds_np[:, dim]))
            mae_per_dim.append(mae_dim)
            rmse_per_dim.append(rmse_dim)
        
        mae_per_dim = np.array(mae_per_dim)
        rmse_per_dim = np.array(rmse_per_dim)
        mae_mean = np.mean(mae_per_dim)
        mae_std = np.std(mae_per_dim)
        rmse_mean = np.mean(rmse_per_dim)
        rmse_std = np.std(rmse_per_dim)
        
        # Compute evaluation metrics.
        metrics_E_R = calculate_metrics_E_R(best_val_preds_np, best_val_labels_np)
        
        # Compute evaluation metrics.
        # Baseline workflow step.
        mae_E_per_dim = []
        rmse_E_per_dim = []
        for dim in range(3):
            mae_dim = mean_absolute_error(best_val_labels_np[:, dim], best_val_preds_np[:, dim])
            rmse_dim = np.sqrt(mean_squared_error(best_val_labels_np[:, dim], best_val_preds_np[:, dim]))
            mae_E_per_dim.append(mae_dim)
            rmse_E_per_dim.append(rmse_dim)
        mae_E_per_dim = np.array(mae_E_per_dim)
        rmse_E_per_dim = np.array(rmse_E_per_dim)
        mae_E_mean = np.mean(mae_E_per_dim)
        mae_E_std = np.std(mae_E_per_dim)
        rmse_E_mean = np.mean(rmse_E_per_dim)
        rmse_E_std = np.std(rmse_E_per_dim)
        
        # Baseline workflow step.
        mae_R_per_dim = []
        rmse_R_per_dim = []
        for dim in range(3, 6):
            mae_dim = mean_absolute_error(best_val_labels_np[:, dim], best_val_preds_np[:, dim])
            rmse_dim = np.sqrt(mean_squared_error(best_val_labels_np[:, dim], best_val_preds_np[:, dim]))
            mae_R_per_dim.append(mae_dim)
            rmse_R_per_dim.append(rmse_dim)
        mae_R_per_dim = np.array(mae_R_per_dim)
        rmse_R_per_dim = np.array(rmse_R_per_dim)
        mae_R_mean = np.mean(mae_R_per_dim)
        mae_R_std = np.std(mae_R_per_dim)
        rmse_R_mean = np.mean(rmse_R_per_dim)
        rmse_R_std = np.std(rmse_R_per_dim)
        
        best_val_metrics_full = {
            'mse': mse,
            'mae': mae,
            'mae_mean': mae_mean,
            'mae_std': mae_std,
            'rmse': rmse,
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'r2': r2,
            'mae_E': metrics_E_R['mae_E'],
            'mae_E_mean': mae_E_mean,
            'mae_E_std': mae_E_std,
            'rmse_E': metrics_E_R['rmse_E'],
            'rmse_E_mean': rmse_E_mean,
            'rmse_E_std': rmse_E_std,
            'r2_E': metrics_E_R['r2_E'],
            'mae_R': metrics_E_R['mae_R'],
            'mae_R_mean': mae_R_mean,
            'mae_R_std': mae_R_std,
            'rmse_R': metrics_E_R['rmse_R'],
            'rmse_R_mean': rmse_R_mean,
            'rmse_R_std': rmse_R_std,
            'r2_R': metrics_E_R['r2_R']
        }
    else:
        # Save the generated artifacts.
        best_val_metrics_full = val_metrics.copy()
    
    best_val_metrics = best_val_metrics_full
    
    best_test_metrics = {
        'mae': test_metrics['mae'],
        'mae_mean': test_metrics['mae_mean'],
        'mae_std': test_metrics['mae_std'],
        'rmse': test_metrics['rmse'],
        'rmse_mean': test_metrics['rmse_mean'],
        'rmse_std': test_metrics['rmse_std'],
        'r2': test_metrics['r2'],
        'mae_E': test_metrics['mae_E'],
        'mae_E_mean': test_metrics['mae_E_mean'],
        'mae_E_std': test_metrics['mae_E_std'],
        'rmse_E': test_metrics['rmse_E'],
        'rmse_E_mean': test_metrics['rmse_E_mean'],
        'rmse_E_std': test_metrics['rmse_E_std'],
        'r2_E': test_metrics['r2_E'],
        'mae_R': test_metrics['mae_R'],
        'mae_R_mean': test_metrics['mae_R_mean'],
        'mae_R_std': test_metrics['mae_R_std'],
        'rmse_R': test_metrics['rmse_R'],
        'rmse_R_mean': test_metrics['rmse_R_mean'],
        'rmse_R_std': test_metrics['rmse_R_std'],
        'r2_R': test_metrics['r2_R']
    }
    
    save_metrics_txt(
        best_val_loss, best_val_mse,
        best_val_metrics, best_test_metrics,
        total_time, avg_time_per_epoch,
        config.result_dir
    )
    
    # Save the generated artifacts.
    best_metrics_txt_path = os.path.join(config.result_dir, 'best_metrics.txt')
    with open(best_metrics_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("最佳模型指标\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"随机种子: {config.seed}\n")
        f.write(f"最佳epoch: {best_val_loss}\n")
        f.write(f"最佳验证集MSE: {best_val_mse:.6f}\n")
        f.write(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)\n")
        f.write(f"平均每轮时间: {avg_time_per_epoch:.2f}秒\n")
        f.write(f"总训练轮数: {len(train_history)}\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("最佳验证集指标\n")
        f.write("=" * 100 + "\n\n")
        f.write("【Overall指标】\n")
        f.write(f"  MAE: {best_val_metrics['mae_mean']:.6f}±{best_val_metrics['mae_std']:.6f}\n")
        f.write(f"  RMSE: {best_val_metrics['rmse_mean']:.6f}±{best_val_metrics['rmse_std']:.6f}\n")
        f.write(f"  R²: {best_val_metrics['r2']:.6f}\n\n")
        f.write("【E相指标】\n")
        f.write(f"  MAE: {best_val_metrics['mae_E_mean']:.6f}±{best_val_metrics['mae_E_std']:.6f}\n")
        f.write(f"  RMSE: {best_val_metrics['rmse_E_mean']:.6f}±{best_val_metrics['rmse_E_std']:.6f}\n")
        f.write(f"  R²: {best_val_metrics['r2_E']:.6f}\n\n")
        f.write("【R相指标】\n")
        f.write(f"  MAE: {best_val_metrics['mae_R_mean']:.6f}±{best_val_metrics['mae_R_std']:.6f}\n")
        f.write(f"  RMSE: {best_val_metrics['rmse_R_mean']:.6f}±{best_val_metrics['rmse_R_std']:.6f}\n")
        f.write(f"  R²: {best_val_metrics['r2_R']:.6f}\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("测试集指标\n")
        f.write("=" * 100 + "\n\n")
        f.write("【Overall指标】\n")
        f.write(f"  MAE: {best_test_metrics['mae_mean']:.6f}±{best_test_metrics['mae_std']:.6f}\n")
        f.write(f"  RMSE: {best_test_metrics['rmse_mean']:.6f}±{best_test_metrics['rmse_std']:.6f}\n")
        f.write(f"  R²: {best_test_metrics['r2']:.6f}\n\n")
        f.write("【E相指标】\n")
        f.write(f"  MAE: {best_test_metrics['mae_E_mean']:.6f}±{best_test_metrics['mae_E_std']:.6f}\n")
        f.write(f"  RMSE: {best_test_metrics['rmse_E_mean']:.6f}±{best_test_metrics['rmse_E_std']:.6f}\n")
        f.write(f"  R²: {best_test_metrics['r2_E']:.6f}\n\n")
        f.write("【R相指标】\n")
        f.write(f"  MAE: {best_test_metrics['mae_R_mean']:.6f}±{best_test_metrics['mae_R_std']:.6f}\n")
        f.write(f"  RMSE: {best_test_metrics['rmse_R_mean']:.6f}±{best_test_metrics['rmse_R_std']:.6f}\n")
        f.write(f"  R²: {best_test_metrics['r2_R']:.6f}\n")
    
    print("\n" + "=" * 100)
    print("训练结果文件保存路径:")
    print("=" * 100)
    print(f"  基础输出目录: {os.path.abspath(base_output_dir)}")
    print(f"  Seed目录: {os.path.abspath(seed_dir)}")
    print(f"  检查点保存目录: {os.path.abspath(config.model_save_dir)}")
    print(f"  结果保存目录: {os.path.abspath(config.result_dir)}")
    print(f"  最佳模型保存目录: {os.path.abspath(best_model_dir)}")
    print("\n  已保存的文件:")
    print(f"    ✓ 训练历史CSV: {os.path.abspath(os.path.join(config.result_dir, 'train_history.csv'))}")
    print(f"    ✓ 训练/验证结果CSV: {os.path.abspath(os.path.join(config.result_dir, 'training_results.csv'))}")
    print(f"    ✓ 测试集结果CSV: {os.path.abspath(os.path.join(config.result_dir, 'test_results.csv'))}")
    print(f"    ✓ 训练指标TXT: {os.path.abspath(os.path.join(config.result_dir, 'training_metrics.txt'))}")
    print(f"    ✓ 最佳指标TXT: {os.path.abspath(os.path.join(config.result_dir, 'best_metrics.txt'))}")
    print(f"    ✓ 测试指标TXT: {os.path.abspath(os.path.join(config.result_dir, 'test_metrics.txt'))}")
    print(f"    ✓ 最佳模型权重: {os.path.abspath(os.path.join(best_model_dir, 'GLAM.pt'))}")
    print("=" * 100)
    
    # Evaluate the test subset.
    return test_metrics


def calculate_std_metrics(all_test_metrics):
    """Run the calculate std metrics baseline operation."""
    # Compute evaluation metrics.
    metrics_list = {
        'mae': [],
        'rmse': [],
        'r2': [],
        'mae_E': [],
        'rmse_E': [],
        'r2_E': [],
        'mae_R': [],
        'rmse_R': [],
        'r2_R': []
    }
    
    for metrics in all_test_metrics:
        for key in metrics_list.keys():
            metrics_list[key].append(metrics[key])
    
    # Baseline workflow step.
    result = {}
    for key, values in metrics_list.items():
        result[f'{key}_mean'] = np.mean(values)
        result[f'{key}_std'] = np.std(values)
    
    return result


def format_metric_with_std(mean, std, decimals=6):
    """Run the format metric with std baseline operation."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def save_summary_txt(all_seeds, all_test_metrics, summary_dir):
    """Run the save summary txt baseline operation."""
    # Compute evaluation metrics.
    summary_metrics = calculate_std_metrics(all_test_metrics)
    
    # Baseline workflow step.
    content = "=" * 100 + "\n"
    content += "多Seed训练结果汇总\n"
    content += "=" * 100 + "\n\n"
    
    content += f"训练Seed列表: {all_seeds}\n"
    content += f"Seed数量: {len(all_seeds)}\n\n"
    
    content += "=" * 100 + "\n"
    content += "测试集指标汇总（均值 ± 标准差）\n"
    content += "=" * 100 + "\n\n"
    
    # Compute evaluation metrics.
    content += "【Overall指标】\n"
    content += f"  MAE: {format_metric_with_std(summary_metrics['mae_mean'], summary_metrics['mae_std'])}\n"
    content += f"  RMSE: {format_metric_with_std(summary_metrics['rmse_mean'], summary_metrics['rmse_std'])}\n"
    content += f"  R²: {format_metric_with_std(summary_metrics['r2_mean'], summary_metrics['r2_std'])}\n\n"
    
    # Compute evaluation metrics.
    content += "【E相指标】\n"
    content += f"  MAE: {format_metric_with_std(summary_metrics['mae_E_mean'], summary_metrics['mae_E_std'])}\n"
    content += f"  RMSE: {format_metric_with_std(summary_metrics['rmse_E_mean'], summary_metrics['rmse_E_std'])}\n"
    content += f"  R²: {format_metric_with_std(summary_metrics['r2_E_mean'], summary_metrics['r2_E_std'])}\n\n"
    
    # Compute evaluation metrics.
    content += "【R相指标】\n"
    content += f"  MAE: {format_metric_with_std(summary_metrics['mae_R_mean'], summary_metrics['mae_R_std'])}\n"
    content += f"  RMSE: {format_metric_with_std(summary_metrics['rmse_R_mean'], summary_metrics['rmse_R_std'])}\n"
    content += f"  R²: {format_metric_with_std(summary_metrics['r2_R_mean'], summary_metrics['r2_R_std'])}\n\n"
    
    content += "=" * 100 + "\n"
    content += "各Seed详细结果\n"
    content += "=" * 100 + "\n\n"
    
    for seed, metrics in zip(all_seeds, all_test_metrics):
        content += f"Seed {seed}:\n"
        content += f"  【Overall】 MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}  R²={metrics['r2']:.6f}\n"
        content += f"  【E相】     MAE={metrics['mae_E']:.6f}  RMSE={metrics['rmse_E']:.6f}  R²={metrics['r2_E']:.6f}\n"
        content += f"  【R相】     MAE={metrics['mae_R']:.6f}  RMSE={metrics['rmse_R']:.6f}  R²={metrics['r2_R']:.6f}\n\n"
    
    # Save the generated artifacts.
    summary_path = os.path.join(summary_dir, 'summary_results.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n汇总结果已保存到: {summary_path}")
    
    # Baseline workflow step.
    print("\n" + "=" * 100)
    print("测试集指标汇总（均值 ± 标准差）")
    print("=" * 100)
    print("\n【Overall指标】")
    print(f"  MAE: {format_metric_with_std(summary_metrics['mae_mean'], summary_metrics['mae_std'])}")
    print(f"  RMSE: {format_metric_with_std(summary_metrics['rmse_mean'], summary_metrics['rmse_std'])}")
    print(f"  R²: {format_metric_with_std(summary_metrics['r2_mean'], summary_metrics['r2_std'])}")
    print("\n【E相指标】")
    print(f"  MAE: {format_metric_with_std(summary_metrics['mae_E_mean'], summary_metrics['mae_E_std'])}")
    print(f"  RMSE: {format_metric_with_std(summary_metrics['rmse_E_mean'], summary_metrics['rmse_E_std'])}")
    print(f"  R²: {format_metric_with_std(summary_metrics['r2_E_mean'], summary_metrics['r2_E_std'])}")
    print("\n【R相指标】")
    print(f"  MAE: {format_metric_with_std(summary_metrics['mae_R_mean'], summary_metrics['mae_R_std'])}")
    print(f"  RMSE: {format_metric_with_std(summary_metrics['rmse_R_mean'], summary_metrics['rmse_R_std'])}")
    print(f"  R²: {format_metric_with_std(summary_metrics['r2_R_mean'], summary_metrics['r2_R_std'])}")
    print("=" * 100)


def main_multi_seed(seeds=[42, 123, 456, 789, 2024], base_output_dir=None):
    """Run the main multi seed baseline operation."""
    print("=" * 100)
    print("多Seed训练开始")
    print("=" * 100)
    print(f"Seed列表: {seeds}")
    if base_output_dir is None:
        base_output_dir = str(EXPERIMENT_ROOT / "runs" / "glam")
    print(f"输出目录: {base_output_dir}")
    print("=" * 100)
    
    # Configure the output artifacts.
    os.makedirs(base_output_dir, exist_ok=True)
    
    all_test_metrics = []
    
    # Run the training step.
    for i, seed in enumerate(seeds):
        print(f"\n{'='*100}")
        print(f"开始训练 Seed {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*100}\n")
        
        # Baseline workflow step.
        config = Config()
        
        # Run the training step.
        test_metrics = main(config=config, seed=seed, base_output_dir=base_output_dir)
        
        # Save the generated artifacts.
        all_test_metrics.append({
            'mae': test_metrics['mae'],
            'mae_mean': test_metrics['mae_mean'],
            'mae_std': test_metrics['mae_std'],
            'rmse': test_metrics['rmse'],
            'rmse_mean': test_metrics['rmse_mean'],
            'rmse_std': test_metrics['rmse_std'],
            'r2': test_metrics['r2'],
            'mae_E': test_metrics['mae_E'],
            'mae_E_mean': test_metrics['mae_E_mean'],
            'mae_E_std': test_metrics['mae_E_std'],
            'rmse_E': test_metrics['rmse_E'],
            'rmse_E_mean': test_metrics['rmse_E_mean'],
            'rmse_E_std': test_metrics['rmse_E_std'],
            'r2_E': test_metrics['r2_E'],
            'mae_R': test_metrics['mae_R'],
            'mae_R_mean': test_metrics['mae_R_mean'],
            'mae_R_std': test_metrics['mae_R_std'],
            'rmse_R': test_metrics['rmse_R'],
            'rmse_R_mean': test_metrics['rmse_R_mean'],
            'rmse_R_std': test_metrics['rmse_R_std'],
            'r2_R': test_metrics['r2_R']
        })
        
        # Save the generated artifacts.
        seed_dir = os.path.join(base_output_dir, f'seed_{seed}')
        results_dir = os.path.join(seed_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        test_metrics_path = os.path.join(results_dir, 'test_metrics.txt')
        with open(test_metrics_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"Seed {seed} - 测试集指标\n")
            f.write("=" * 100 + "\n\n")
            f.write("【Overall指标】\n")
            f.write(f"  MAE: {test_metrics['mae_mean']:.6f}±{test_metrics['mae_std']:.6f}\n")
            f.write(f"  RMSE: {test_metrics['rmse_mean']:.6f}±{test_metrics['rmse_std']:.6f}\n")
            f.write(f"  R²: {test_metrics['r2']:.6f}\n\n")
            f.write("【E相指标】\n")
            f.write(f"  MAE: {test_metrics['mae_E_mean']:.6f}±{test_metrics['mae_E_std']:.6f}\n")
            f.write(f"  RMSE: {test_metrics['rmse_E_mean']:.6f}±{test_metrics['rmse_E_std']:.6f}\n")
            f.write(f"  R²: {test_metrics['r2_E']:.6f}\n\n")
            f.write("【R相指标】\n")
            f.write(f"  MAE: {test_metrics['mae_R_mean']:.6f}±{test_metrics['mae_R_std']:.6f}\n")
            f.write(f"  RMSE: {test_metrics['rmse_R_mean']:.6f}±{test_metrics['rmse_R_std']:.6f}\n")
            f.write(f"  R²: {test_metrics['r2_R']:.6f}\n")
        
        print(f"\nSeed {seed} 训练完成！")
        print(f"测试指标已保存到: {test_metrics_path}")
    
    # Save the generated artifacts.
    save_summary_txt(seeds, all_test_metrics, base_output_dir)
    
    print("\n" + "=" * 100)
    print("所有Seed训练完成！")
    print("=" * 100)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GLAM模型训练脚本')
    parser.add_argument('--multi-seed', action='store_true', 
                        help='运行多seed训练')
    parser.add_argument('--single-seed', action='store_true', 
                        help='运行单seed训练')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 2024], 
                        help='Seed列表（仅在--multi-seed时使用）')
    parser.add_argument('--output-dir', type=str,
                        default=str(EXPERIMENT_ROOT / 'runs' / 'glam'),
                        help='输出目录（仅在--multi-seed时使用）')
    parser.add_argument('--seed', type=int, default=None, 
                        help='单个seed训练时的随机种子')
    parser.add_argument('--base-output-dir', type=str,
                        default=str(EXPERIMENT_ROOT / 'runs' / 'glam'),
                        help='基础输出目录（默认: outputs，会自动创建seed_{seed}子目录）')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定检查点恢复训练（检查点文件路径）')
    parser.add_argument('--auto-resume', action='store_true',
                        help='自动查找最新检查点并恢复训练')
    
    args = parser.parse_args()
    
    # Run the training step.
    if not args.multi_seed and not args.single_seed:
        # Run the training step.
        print("=" * 100)
        print("默认运行多Seed训练")
        print(f"Seed列表: {args.seeds}")
        print("=" * 100)
        main_multi_seed(seeds=args.seeds, base_output_dir=args.output_dir)
    elif args.single_seed:
        # Run the training step.
        config = default_config
        if args.seed is not None:
            main(config=config, seed=args.seed, base_output_dir=args.base_output_dir,
                 resume_checkpoint=args.resume, auto_resume=args.auto_resume)
        else:
            # Baseline workflow step.
            main(config=config, base_output_dir=args.base_output_dir,
                 resume_checkpoint=args.resume, auto_resume=args.auto_resume)
    else:
        # Run the training step.
        main_multi_seed(seeds=args.seeds, base_output_dir=args.output_dir)

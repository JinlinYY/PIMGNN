"""Implement the solvbert train baseline module."""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import time
import sys

# Configure the baseline model.
from .solvbert_model import SolvBERTForMLM, SolvBERT
from .data_utils import (
    SolvDataset, build_tokenizer, create_data_loader, 
    mask_tokens_for_mlm
)
from psmi_baselines.paths import EXPERIMENT_ROOT, TRAIN_CSV, VALIDATION_CSV, TEST_CSV

# Compute evaluation metrics.
def r2_score(y_true, y_pred):
    """Run the r2 score baseline operation."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def mean_squared_error(y_true, y_pred):
    """Run the mean squared error baseline operation."""
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def mean_absolute_error(y_true, y_pred):
    """Run the mean absolute error baseline operation."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def pretrain_epoch(model, dataloader, optimizer, scheduler, device, tokenizer, mlm_probability=0.15):
    """Run the pretrain epoch baseline operation."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="预训练", leave=False)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Baseline workflow step.
        masked_input_ids, labels = mask_tokens_for_mlm(
            input_ids, tokenizer, mlm_probability
        )
        labels = labels.to(device)
        
        # Baseline workflow step.
        loss, _ = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Baseline workflow step.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def pretrain_evaluate(model, dataloader, device, tokenizer, mlm_probability=0.15):
    """Run the pretrain evaluate baseline operation."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预训练验证", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            masked_input_ids, labels = mask_tokens_for_mlm(
                input_ids, tokenizer, mlm_probability
            )
            labels = labels.to(device)
            
            loss, _ = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def finetune_epoch(model, dataloader, optimizer, scheduler, device):
    """Run the finetune epoch baseline operation."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    criterion = nn.MSELoss()
    progress_bar = tqdm(dataloader, desc="微调", leave=False)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # [batch_size, 6]
        
        # Baseline workflow step.
        predictions = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )  # [batch_size, 6]
        
        loss = criterion(predictions, labels)
        
        # Baseline workflow step.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def finetune_evaluate(model, dataloader, device, return_predictions=False, dataset=None):
    """Run the finetune evaluate baseline operation."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="微调验证", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [batch_size, 6]
            
            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )  # [batch_size, 6]
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)  # [n_samples, 6]
    all_labels = np.array(all_labels)  # [n_samples, 6]
    
    # Configure the output artifacts.
    num_outputs = all_predictions.shape[1]
    per_output_metrics = []
    
    for i in range(num_outputs):
        pred_i = all_predictions[:, i]
        label_i = all_labels[:, i]
        
        r2_i = r2_score(label_i, pred_i)
        rmse_i = np.sqrt(mean_squared_error(label_i, pred_i))
        mae_i = mean_absolute_error(label_i, pred_i)
        mse_i = mean_squared_error(label_i, pred_i)
        residuals_i = pred_i - label_i
        std_i = np.std(residuals_i)
        
        per_output_metrics.append({
            'r2': r2_i,
            'rmse': rmse_i,
            'mae': mae_i,
            'mse': mse_i,
            'std': std_i
        })
    
    # Configure the output artifacts.
    avg_r2 = np.mean([m['r2'] for m in per_output_metrics])
    avg_rmse = np.mean([m['rmse'] for m in per_output_metrics])
    avg_mae = np.mean([m['mae'] for m in per_output_metrics])
    avg_mse = np.mean([m['mse'] for m in per_output_metrics])
    avg_std = np.mean([m['std'] for m in per_output_metrics])
    
    # Configure the output artifacts.
    all_predictions_flat = all_predictions.flatten()
    all_labels_flat = all_labels.flatten()
    
    overall_r2 = r2_score(all_labels_flat, all_predictions_flat)
    overall_rmse = np.sqrt(mean_squared_error(all_labels_flat, all_predictions_flat))
    overall_mae = mean_absolute_error(all_labels_flat, all_predictions_flat)
    overall_mse = mean_squared_error(all_labels_flat, all_predictions_flat)
    overall_residuals = all_predictions_flat - all_labels_flat
    overall_std = np.std(overall_residuals)
    
    result = {
        'r2': overall_r2,  # Baseline workflow step.
        'rmse': overall_rmse,  # Baseline workflow step.
        'mae': overall_mae,  # Baseline workflow step.
        'mse': overall_mse,  # Baseline workflow step.
        'std': overall_std,  # Baseline workflow step.
        'loss': overall_rmse,  # Configure the baseline model.
        'avg_r2': avg_r2,  # Baseline workflow step.
        'avg_rmse': avg_rmse,  # Baseline workflow step.
        'avg_mae': avg_mae,  # Baseline workflow step.
        'avg_mse': avg_mse,  # Baseline workflow step.
        'avg_std': avg_std,  # Baseline workflow step.
        'per_output_metrics': per_output_metrics  # Configure the output artifacts.
    }
    
    if return_predictions:
        result['predictions'] = all_predictions
        result['labels'] = all_labels
    
    return result


def print_config_info(args, train_size, val_size, test_size, device):
    """Run the print config info baseline operation."""
    print("\n" + "="*100)
    print("【训练配置参数】")
    print("="*100)
    
    print("\n【数据集信息】")
    print(f"  训练集样本数: {train_size}")
    print(f"  验证集样本数: {val_size}")
    if test_size > 0:
        print(f"  测试集样本数: {test_size}")
    
    print("\n【设备配置】")
    print(f"  设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU内存: {gpu_memory:.2f} GB")
    
    print("\n【训练参数】")
    print(f"  随机种子: {args.random_seed}")
    print(f"  训练轮数: {args.finetune_num_epochs}")
    print(f"  批次大小: {args.finetune_batch_size}")
    print(f"  学习率: {args.finetune_learning_rate}")
    print(f"  权重衰减: 0.0001")
    print(f"  早停耐心值: {args.early_stop_patience}")
    print(f"  早停最小改善: {args.early_stop_min_delta}")
    print(f"  检查点保存频率: 每 {args.checkpoint_save_freq} 个epoch")
    if args.rest_interval_hours > 0:
        rest_interval_seconds = args.rest_interval_hours * 3600
        print(f"  休息策略: 每 {args.rest_interval_hours:.1f}小时（{rest_interval_seconds:.0f}秒）休息 {args.rest_duration}秒（{args.rest_duration/60:.1f}分钟）")
    else:
        print(f"  休息策略: 无")
    print(f"  断点续训: {'是' if args.resume_from_checkpoint else '否'}（{'从检查点恢复' if args.resume_from_checkpoint else '从头开始'}）")
    
    print("\n【模型超参数】")
    print(f"  隐藏层维度: {args.hidden_size}")
    print(f"  Transformer层数: {args.num_layers}")
    print(f"  注意力头数: {args.num_heads}")
    print(f"  前馈网络中间层维度: {args.intermediate_size}")
    print(f"  Dropout率: {args.hidden_dropout_rate}")
    print(f"  输出维度: 6 (多元回归任务: Ex1, Ex2, Ex3, Rx1, Rx2, Rx3)")
    
    print("\n【路径信息】")
    print(f"  输出目录: {args.finetune_output_dir}")
    print(f"  结果目录: {args.finetune_output_dir}")
    if args.checkpoint_subdir:
        print(f"  检查点目录: {os.path.join(args.finetune_output_dir, args.checkpoint_subdir)}")
    
    print("="*100 + "\n")


def print_epoch_info(epoch, total_epochs, epoch_time, train_loss, train_result, val_result, best_val_mse, best_epoch):
    """Run the print epoch info baseline operation."""
    print("="*100)
    print(f"Epoch {epoch}/{total_epochs} | 训练时间: {epoch_time:.2f}秒 | Train Loss: {train_loss:.6f}")
    if best_epoch >= 0:
        print(f"Best Loss: {best_val_mse:.6f} (epoch {best_epoch})")
    else:
        print(f"Best Loss: {best_val_mse:.6f} (initial)")
    print("="*100)
    
    # Compute evaluation metrics.
    train_e_mae = np.mean([train_result['per_output_metrics'][i]['mae'] for i in [0, 1, 2]])
    train_e_rmse = np.mean([train_result['per_output_metrics'][i]['rmse'] for i in [0, 1, 2]])
    train_e_r2 = np.mean([train_result['per_output_metrics'][i]['r2'] for i in [0, 1, 2]])
    train_e_std = np.mean([train_result['per_output_metrics'][i]['std'] for i in [0, 1, 2]])
    
    train_r_mae = np.mean([train_result['per_output_metrics'][i]['mae'] for i in [3, 4, 5]])
    train_r_rmse = np.mean([train_result['per_output_metrics'][i]['rmse'] for i in [3, 4, 5]])
    train_r_r2 = np.mean([train_result['per_output_metrics'][i]['r2'] for i in [3, 4, 5]])
    train_r_std = np.mean([train_result['per_output_metrics'][i]['std'] for i in [3, 4, 5]])
    
    val_e_mae = np.mean([val_result['per_output_metrics'][i]['mae'] for i in [0, 1, 2]])
    val_e_rmse = np.mean([val_result['per_output_metrics'][i]['rmse'] for i in [0, 1, 2]])
    val_e_r2 = np.mean([val_result['per_output_metrics'][i]['r2'] for i in [0, 1, 2]])
    val_e_std = np.mean([val_result['per_output_metrics'][i]['std'] for i in [0, 1, 2]])
    
    val_r_mae = np.mean([val_result['per_output_metrics'][i]['mae'] for i in [3, 4, 5]])
    val_r_rmse = np.mean([val_result['per_output_metrics'][i]['rmse'] for i in [3, 4, 5]])
    val_r_r2 = np.mean([val_result['per_output_metrics'][i]['r2'] for i in [3, 4, 5]])
    val_r_std = np.mean([val_result['per_output_metrics'][i]['std'] for i in [3, 4, 5]])
    
    print("【训练集指标】")
    print(f"  Overall - MAE: {train_result['mae']:.6f} ± {train_result['std']:.6f}, RMSE: {train_result['rmse']:.6f} ± {train_result['std']:.6f}, R²: {train_result['r2']:.6f} ± {train_result['std']:.6f}")
    print(f"  E相 - MAE: {train_e_mae:.6f} ± {train_e_std:.6f}, RMSE: {train_e_rmse:.6f} ± {train_e_std:.6f}, R²: {train_e_r2:.6f} ± {train_e_std:.6f}")
    print(f"  R相 - MAE: {train_r_mae:.6f} ± {train_r_std:.6f}, RMSE: {train_r_rmse:.6f} ± {train_r_std:.6f}, R²: {train_r_r2:.6f} ± {train_r_std:.6f}")
    
    print("\n【验证集指标】")
    print(f"  Overall - MAE: {val_result['mae']:.6f} ± {val_result['std']:.6f}, RMSE: {val_result['rmse']:.6f} ± {val_result['std']:.6f}, R²: {val_result['r2']:.6f} ± {val_result['std']:.6f}")
    print(f"  E相 - MAE: {val_e_mae:.6f} ± {val_e_std:.6f}, RMSE: {val_e_rmse:.6f} ± {val_e_std:.6f}, R²: {val_e_r2:.6f} ± {val_e_std:.6f}")
    print(f"  R相 - MAE: {val_r_mae:.6f} ± {val_r_std:.6f}, RMSE: {val_r_rmse:.6f} ± {val_r_std:.6f}, R²: {val_r_r2:.6f} ± {val_r_std:.6f}")
    
    print("="*100 + "\n")


def save_training_history(history, output_dir):
    """Run the save training history baseline operation."""
    # Baseline workflow step.
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(history)
    csv_path = os.path.join(output_dir, 'train_history.csv')
    df.to_csv(csv_path, index=False)
    return csv_path


def save_predictions(predictions, labels, output_path, dataset_type='train'):
    """Run the save predictions baseline operation."""
    # Baseline workflow step.
    df = pd.DataFrame({
        'true_Ex1': labels[:, 0],
        'true_Ex2': labels[:, 1],
        'true_Ex3': labels[:, 2],
        'true_Rx1': labels[:, 3],
        'true_Rx2': labels[:, 4],
        'true_Rx3': labels[:, 5],
        'pred_Ex1': predictions[:, 0],
        'pred_Ex2': predictions[:, 1],
        'pred_Ex3': predictions[:, 2],
        'pred_Rx1': predictions[:, 3],
        'pred_Rx2': predictions[:, 4],
        'pred_Rx3': predictions[:, 5],
    })
    df.to_csv(output_path, index=False)
    return output_path


def convert_to_python_type(value):
    """Run the convert to python type baseline operation."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item() if hasattr(value, 'item') else float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, 'item'):  # torch tensor
        return value.item()
    elif isinstance(value, float) and (np.isinf(value) or np.isnan(value)):
        if np.isinf(value):
            return float('inf') if value > 0 else float('-inf')
        else:
            return None
    return value


def save_metrics(best_metrics, output_dir, total_time, avg_time_per_epoch, total_epochs):
    """Run the save metrics baseline operation."""
    # Baseline workflow step.
    results_dir = output_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Baseline workflow step.
    best_val_mse = convert_to_python_type(best_metrics.get('best_val_mse', float('inf')))
    best_val_rmse = convert_to_python_type(best_metrics.get('best_val_rmse', float('inf')))
    best_val_mae = convert_to_python_type(best_metrics.get('best_val_mae', float('inf')))
    best_val_r2 = convert_to_python_type(best_metrics.get('best_val_r2', float('-inf')))
    best_val_std = convert_to_python_type(best_metrics.get('best_val_std', 0))
    
    best_train_rmse = convert_to_python_type(best_metrics.get('best_train_rmse', float('inf')))
    best_train_mae = convert_to_python_type(best_metrics.get('best_train_mae', float('inf')))
    best_train_r2 = convert_to_python_type(best_metrics.get('best_train_r2', float('-inf')))
    
    # Compute evaluation metrics.
    best_train_e_mae = convert_to_python_type(best_metrics.get('best_train_e_phase_mae', float('inf')))
    best_train_e_rmse = convert_to_python_type(best_metrics.get('best_train_e_phase_rmse', float('inf')))
    best_train_r_mae = convert_to_python_type(best_metrics.get('best_train_r_phase_mae', float('inf')))
    best_train_r_rmse = convert_to_python_type(best_metrics.get('best_train_r_phase_rmse', float('inf')))
    best_val_e_mae = convert_to_python_type(best_metrics.get('best_val_e_phase_mae', float('inf')))
    best_val_e_rmse = convert_to_python_type(best_metrics.get('best_val_e_phase_rmse', float('inf')))
    best_val_r_mae = convert_to_python_type(best_metrics.get('best_val_r_phase_mae', float('inf')))
    best_val_r_rmse = convert_to_python_type(best_metrics.get('best_val_r_phase_rmse', float('inf')))
    
    # Save the generated artifacts.
    txt_path = os.path.join(results_dir, 'best_metrics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("最佳模型指标\n")
        f.write("="*100 + "\n\n")
        f.write(f"最佳epoch: {int(best_metrics.get('best_epoch', -1))}\n\n")
        
        f.write("【Overall指标】\n")
        f.write(f"训练集 - MAE: {best_train_mae:.6f}, RMSE: {best_train_rmse:.6f}, R²: {best_train_r2:.6f}\n")
        f.write(f"验证集 - MAE: {best_val_mae:.6f}, RMSE: {best_val_rmse:.6f}, R²: {best_val_r2:.6f}\n\n")
        
        f.write("【E相指标（Ex1, Ex2, Ex3）】\n")
        f.write(f"训练集 - MAE: {best_train_e_mae:.6f}, RMSE: {best_train_e_rmse:.6f}\n")
        f.write(f"验证集 - MAE: {best_val_e_mae:.6f}, RMSE: {best_val_e_rmse:.6f}\n\n")
        
        f.write("【R相指标（Rx1, Rx2, Rx3）】\n")
        f.write(f"训练集 - MAE: {best_train_r_mae:.6f}, RMSE: {best_train_r_rmse:.6f}\n")
        f.write(f"验证集 - MAE: {best_val_r_mae:.6f}, RMSE: {best_val_r_rmse:.6f}\n\n")
        
        f.write(f"总训练时间: {float(total_time):.2f}秒 ({float(total_time)/60:.2f}分钟)\n")
        f.write(f"平均每轮时间: {float(avg_time_per_epoch):.2f}秒\n")
        f.write(f"总训练轮数: {int(total_epochs)}\n")
    
    return txt_path


def train_single(args):
    """Run the train single baseline operation."""
    # Set the random seed.
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    
    # Configure the output artifacts.
    os.makedirs(args.pretrain_output_dir, exist_ok=True)
    os.makedirs(args.finetune_output_dir, exist_ok=True)
    
    # Configure repository paths.
    checkpoint_dir = args.finetune_output_dir
    pretrain_checkpoint_dir = args.pretrain_output_dir
    if args.checkpoint_subdir:
        checkpoint_dir = os.path.join(args.finetune_output_dir, args.checkpoint_subdir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the generated artifacts.
        pretrain_checkpoint_dir = os.path.join(args.pretrain_output_dir, args.checkpoint_subdir)
        os.makedirs(pretrain_checkpoint_dir, exist_ok=True)
    
    # Save the generated artifacts.
    results_dir = args.finetune_output_dir
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Load the input data.
    # Save the generated artifacts.
    tokenizer_path = args.pretrain_output_dir
    tokenizer = None
    
    if os.path.exists(tokenizer_path):
        tokenizer_config = os.path.join(tokenizer_path, 'tokenizer_config.json')
        vocab_file = os.path.join(tokenizer_path, 'vocab.txt')
        if os.path.exists(tokenizer_config) or os.path.exists(vocab_file):
            print(f"从 {tokenizer_path} 加载已保存的tokenizer")
            try:
                tokenizer = build_tokenizer(vocab_path=tokenizer_path, local_files_only=True)
            except Exception as e:
                print(f"无法从本地加载tokenizer: {e}，尝试其他方法...")
    
    # Load the input data.
    if tokenizer is None:
        try:
            print(f"尝试从HuggingFace下载tokenizer: {args.tokenizer_name}")
            tokenizer = build_tokenizer(model_name=args.tokenizer_name, local_files_only=False, vocab_size=args.vocab_size)
        except Exception as e:
            print(f"在线下载失败: {e}")
            print("尝试使用本地缓存...")
            try:
                tokenizer = build_tokenizer(model_name=args.tokenizer_name, local_files_only=True, vocab_size=args.vocab_size)
            except Exception as e2:
                print(f"本地缓存也不存在: {e2}")
                print("使用简单的字符级tokenizer作为后备方案...")
                # Baseline workflow step.
                tokenizer = build_tokenizer(model_name=args.tokenizer_name, local_files_only=True, vocab_size=args.vocab_size)
    
    # Save the generated artifacts.
    print(f"保存tokenizer到: {args.pretrain_output_dir}")
    tokenizer.save_pretrained(args.pretrain_output_dir)
    
    # Run the training step.
    pretrained_model_path = None
    if not args.skip_pretrain:
        # Run the training step.
        pretrain_train_dataset = SolvDataset(
            args.pretrain_train_data,
            tokenizer,
            max_length=args.max_length,
            is_pretrain=True
        )
        pretrain_train_loader = create_data_loader(
            pretrain_train_dataset,
            batch_size=args.pretrain_batch_size,
            shuffle=True
        )
        
        pretrain_val_loader = None
        if args.pretrain_val_data:
            pretrain_val_dataset = SolvDataset(
                args.pretrain_val_data,
                tokenizer,
                max_length=args.max_length,
                is_pretrain=True
            )
            pretrain_val_loader = create_data_loader(
                pretrain_val_dataset,
                batch_size=args.pretrain_batch_size,
                shuffle=False
            )
        
        # Run the training step.
        pretrain_model = SolvBERTForMLM(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            pad_token_id=tokenizer.pad_token_id,
            cls_token_id=tokenizer.cls_token_id,
            mask_token_id=tokenizer.mask_token_id,
        ).to(device)
        
        # Run the training step.
        pretrain_optimizer = AdamW(pretrain_model.parameters(), lr=args.pretrain_learning_rate)
        pretrain_total_steps = len(pretrain_train_loader) * args.pretrain_num_epochs
        pretrain_scheduler = get_linear_schedule_with_warmup(
            pretrain_optimizer,
            num_warmup_steps=args.pretrain_warmup_steps,
            num_training_steps=pretrain_total_steps
        )
        
        # Run the training step.
        best_pretrain_val_loss = float('inf')
        
        for epoch in range(args.pretrain_num_epochs):
            epoch_start_time = time.time()
            
            # Run the training step.
            train_loss = pretrain_epoch(
                pretrain_model, pretrain_train_loader, pretrain_optimizer,
                pretrain_scheduler, device, tokenizer, args.mlm_probability
            )
            
            # Evaluate the validation subset.
            val_loss = float('inf')
            if pretrain_val_loader:
                val_loss = pretrain_evaluate(
                    pretrain_model, pretrain_val_loader, device,
                    tokenizer, args.mlm_probability
                )
                
                # Save the generated artifacts.
                if val_loss < best_pretrain_val_loss:
                    best_pretrain_val_loss = val_loss
                    # Configure repository paths.
                    os.makedirs(args.pretrain_output_dir, exist_ok=True)
                    checkpoint_path = os.path.join(args.pretrain_output_dir, 'best_model.pt')
                    checkpoint_path = os.path.normpath(checkpoint_path)  # Configure repository paths.
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': pretrain_model.state_dict(),
                        'optimizer_state_dict': pretrain_optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, checkpoint_path)
            
            epoch_time = time.time() - epoch_start_time
            
            # Save the generated artifacts.
            if (epoch + 1) % args.checkpoint_save_freq == 0 or (epoch + 1) == args.pretrain_num_epochs:
                # Configure repository paths.
                os.makedirs(pretrain_checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(pretrain_checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                checkpoint_path = os.path.normpath(checkpoint_path)  # Configure repository paths.
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': pretrain_model.state_dict(),
                    'optimizer_state_dict': pretrain_optimizer.state_dict(),
                }, checkpoint_path)
        
        # Save the generated artifacts.
        os.makedirs(args.pretrain_output_dir, exist_ok=True)
        final_pretrain_path = os.path.join(args.pretrain_output_dir, 'final_model.pt')
        final_pretrain_path = os.path.normpath(final_pretrain_path)  # Configure repository paths.
        torch.save({
            'model_state_dict': pretrain_model.state_dict(),
            'config': {
                'vocab_size': len(tokenizer),
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'intermediate_size': args.intermediate_size,
            }
        }, final_pretrain_path)
        
        # Configure the baseline model.
        pretrained_model_path = os.path.join(args.pretrain_output_dir, 'best_model.pt')
        if not os.path.exists(pretrained_model_path):
            pretrained_model_path = final_pretrain_path
    
    # Baseline workflow step.
    # Process the experiment data.
    finetune_train_dataset = SolvDataset(
        args.finetune_train_data,
        tokenizer,
        max_length=args.max_length,
        is_pretrain=False
    )
    finetune_train_loader = create_data_loader(
        finetune_train_dataset,
        batch_size=args.finetune_batch_size,
        shuffle=True
    )
    # Baseline workflow step.
    finetune_train_eval_loader = create_data_loader(
        finetune_train_dataset,
        batch_size=args.finetune_batch_size,
        shuffle=False
    )
    
    finetune_val_dataset = SolvDataset(
        args.finetune_val_data,
        tokenizer,
        max_length=args.max_length,
        is_pretrain=False
    )
    finetune_val_loader = create_data_loader(
        finetune_val_dataset,
        batch_size=args.finetune_batch_size,
        shuffle=False
    )
    
    finetune_test_loader = None
    test_size = 0
    if args.finetune_test_data:
        finetune_test_dataset = SolvDataset(
            args.finetune_test_data,
            tokenizer,
            max_length=args.max_length,
            is_pretrain=False
        )
        finetune_test_loader = create_data_loader(
            finetune_test_dataset,
            batch_size=args.finetune_batch_size,
            shuffle=False
        )
        test_size = len(finetune_test_dataset)
    
    # Baseline workflow step.
    print_config_info(
        args,
        len(finetune_train_dataset),
        len(finetune_val_dataset),
        test_size,
        device
    )
    
    # Configure the output artifacts.
    finetune_model = SolvBERT(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_rate=args.hidden_dropout_rate,
        num_outputs=6,  # Baseline workflow step.
    ).to(device)
    
    # Load the input data.
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            pretrained_state = checkpoint['model_state_dict']
            model_state = finetune_model.state_dict()
            
            # Load the input data.
            loaded_count = 0
            for key in pretrained_state:
                if key.startswith('bert.'):
                    model_key = key
                    if model_key in model_state:
                        model_state[model_key] = pretrained_state[key]
                        loaded_count += 1
            
            finetune_model.load_state_dict(model_state, strict=False)
    
    # Baseline workflow step.
    finetune_optimizer = AdamW(finetune_model.parameters(), lr=args.finetune_learning_rate, weight_decay=0.0001)
    finetune_total_steps = len(finetune_train_loader) * args.finetune_num_epochs
    finetune_scheduler = get_linear_schedule_with_warmup(
        finetune_optimizer,
        num_warmup_steps=args.finetune_warmup_steps,
        num_training_steps=finetune_total_steps
    )
    
    # Run the training step.
    training_history = []
    best_val_mse = float('inf')
    best_val_rmse = float('inf')
    best_val_mae = float('inf')
    best_val_r2 = float('-inf')
    best_val_std = 0.0
    best_epoch = -1
    no_improve_count = 0
    
    # Handle model checkpoints.
    start_epoch = 0
    
    # Handle model checkpoints.
    checkpoint_path_to_load = None
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            checkpoint_path_to_load = args.resume_from_checkpoint
        else:
            print(f"警告: 指定的检查点路径不存在: {args.resume_from_checkpoint}")
    
    # Handle model checkpoints.
    if checkpoint_path_to_load is None:
        # Handle model checkpoints.
        checkpoint_files = []
        if os.path.exists(checkpoint_dir):
            for filename in os.listdir(checkpoint_dir):
                if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
                    filepath = os.path.join(checkpoint_dir, filename)
                    checkpoint_files.append(filepath)
        
        # Handle model checkpoints.
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            checkpoint_path_to_load = checkpoint_files[0]
            print(f"\n自动找到最新检查点: {checkpoint_path_to_load}")
    
    # Load the input data.
    if checkpoint_path_to_load and os.path.exists(checkpoint_path_to_load):
        checkpoint = torch.load(checkpoint_path_to_load, map_location=device, weights_only=False)
        finetune_model.load_state_dict(checkpoint['model_state_dict'])
        finetune_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_mse = checkpoint.get('best_val_mse', checkpoint.get('val_mse', float('inf')))
        best_val_rmse = checkpoint.get('best_val_rmse', checkpoint.get('val_rmse', float('inf')))
        best_val_mae = checkpoint.get('best_val_mae', checkpoint.get('val_mae', float('inf')))
        best_val_r2 = checkpoint.get('best_val_r2', checkpoint.get('val_r2', float('-inf')))
        best_val_std = checkpoint.get('best_val_std', checkpoint.get('val_std', 0.0))
        best_epoch = checkpoint.get('best_epoch', -1)
        no_improve_count = checkpoint.get('no_improve_count', 0)
        
        print(f"\n从检查点恢复训练: {checkpoint_path_to_load}")
        print(f"从epoch {start_epoch}继续训练")
        print(f"最佳epoch: {best_epoch}, 最佳验证集MSE: {best_val_mse:.6f}")
        print(f"已等待 {no_improve_count}/{args.early_stop_patience} 个epoch无改善")
    
    # Baseline workflow step.
    total_start_time = time.time()
    last_rest_time = total_start_time  # Baseline workflow step.
    rest_interval_seconds = args.rest_interval_hours * 3600 if args.rest_interval_hours > 0 else 0
    
    for epoch in range(start_epoch, args.finetune_num_epochs):
        epoch_start_time = time.time()
        
        # Run the training step.
        train_loss = finetune_epoch(
            finetune_model, finetune_train_loader, finetune_optimizer,
            finetune_scheduler, device
        )
        
        # Run the training step.
        train_result = finetune_evaluate(finetune_model, finetune_train_eval_loader, device, return_predictions=True)
        train_metrics = {
            'mae': train_result['mae'],
            'rmse': train_result['rmse'],
            'r2': train_result['r2'],
            'mse': train_result['mse'],
            'std': train_result['std']
        }
        
        # Evaluate the validation subset.
        val_result = finetune_evaluate(finetune_model, finetune_val_loader, device, return_predictions=True)
        val_metrics = {
            'mae': val_result['mae'],
            'rmse': val_result['rmse'],
            'r2': val_result['r2'],
            'mse': val_result['mse'],
            'std': val_result['std']
        }
        
        epoch_time = time.time() - epoch_start_time
        
        # Baseline workflow step.
        print_epoch_info(
            epoch + 1,
            args.finetune_num_epochs,
            epoch_time,
            train_loss,
            train_result,
            val_result,
            best_val_mse,
            best_epoch
        )
        
        # Run the training step.
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'epoch_time': epoch_time,
            'train_mse': train_metrics['mse'],
            'train_rmse': train_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'train_std': train_metrics['std'],
            'val_mse': val_metrics['mse'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2'],
            'val_std': val_metrics['std']
        }
        training_history.append(history_entry)
        
        # Baseline workflow step.
        improved = False
        if val_metrics['mse'] < (best_val_mse - args.early_stop_min_delta):
            improved = True
            best_val_mse = val_metrics['mse']
            best_val_rmse = val_metrics['rmse']
            best_val_mae = val_metrics['mae']
            best_val_r2 = val_metrics['r2']
            best_val_std = val_metrics['std']
            best_epoch = epoch + 1
            no_improve_count = 0
            
            # Save the generated artifacts.
            os.makedirs(args.finetune_output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.finetune_output_dir, 'best_model.pt')
            checkpoint_path = os.path.normpath(checkpoint_path)  # Configure repository paths.
            torch.save({
                'epoch': epoch,
                'model_state_dict': finetune_model.state_dict(),
                'optimizer_state_dict': finetune_optimizer.state_dict(),
                'val_mse': val_metrics['mse'],
                'val_rmse': val_metrics['rmse'],
                'val_mae': val_metrics['mae'],
                'val_r2': val_metrics['r2'],
                'val_std': val_metrics['std'],
                'best_epoch': best_epoch,
                'best_val_mse': best_val_mse,
                'best_val_rmse': best_val_rmse,
                'best_val_mae': best_val_mae,
                'best_val_r2': best_val_r2,
                'best_val_std': best_val_std,
                'no_improve_count': no_improve_count,
            }, checkpoint_path)
        else:
            no_improve_count += 1
        
        # Save the generated artifacts.
        if (epoch + 1) % args.checkpoint_save_freq == 0 or (epoch + 1) == args.finetune_num_epochs:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            checkpoint_path = os.path.normpath(checkpoint_path)  # Configure repository paths.
            torch.save({
                'epoch': epoch,
                'model_state_dict': finetune_model.state_dict(),
                'optimizer_state_dict': finetune_optimizer.state_dict(),
                'val_mse': val_metrics['mse'],
                'val_rmse': val_metrics['rmse'],
                'val_mae': val_metrics['mae'],
                'val_r2': val_metrics['r2'],
                'val_std': val_metrics['std'],
                'best_epoch': best_epoch,
                'best_val_mse': best_val_mse,
                'best_val_rmse': best_val_rmse,
                'best_val_mae': best_val_mae,
                'best_val_r2': best_val_r2,
                'best_val_std': best_val_std,
                'no_improve_count': no_improve_count,
            }, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
        
        # Baseline workflow step.
        current_time = time.time()
        elapsed_since_last_rest = current_time - last_rest_time
        
        if rest_interval_seconds > 0 and elapsed_since_last_rest >= rest_interval_seconds and (epoch + 1) < args.finetune_num_epochs:
            elapsed_hours = elapsed_since_last_rest / 3600
            print("="*100)
            print(f"已训练 {elapsed_hours:.2f}小时（{elapsed_since_last_rest:.0f}秒），休息 {args.rest_duration}秒（{args.rest_duration/60:.1f}分钟）让CPU/GPU有时间休息...")
            print("="*100 + "\n")
            time.sleep(args.rest_duration)
            last_rest_time = time.time()  # Baseline workflow step.
        
        # Apply early stopping.
        if no_improve_count >= args.early_stop_patience:
            print("="*100)
            print(f"早停触发！在epoch {epoch + 1}停止训练。")
            print(f"最佳模型在epoch {best_epoch}，验证集MSE: {best_val_mse:.6f}")
            print(f"已等待 {no_improve_count}/{args.early_stop_patience} 个epoch无改善")
            print("="*100 + "\n")
            break
    
    # Run the training step.
    total_time = time.time() - total_start_time
    avg_time_per_epoch = total_time / len(training_history) if training_history else 0
    
    # Save the generated artifacts.
    history_csv_path = save_training_history(training_history, args.finetune_output_dir)
    
    # Save the generated artifacts.
    train_result = finetune_evaluate(finetune_model, finetune_train_loader, device, return_predictions=True)
    val_result = finetune_evaluate(finetune_model, finetune_val_loader, device, return_predictions=True)
    
    train_pred_path = os.path.join(results_dir, 'training_results.csv')
    train_labels = train_result['labels']  # [n_samples, 6]
    train_predictions = train_result['predictions']  # [n_samples, 6]
    train_df = pd.DataFrame({
        'true_Ex1': train_labels[:, 0],
        'true_Ex2': train_labels[:, 1],
        'true_Ex3': train_labels[:, 2],
        'true_Rx1': train_labels[:, 3],
        'true_Rx2': train_labels[:, 4],
        'true_Rx3': train_labels[:, 5],
        'pred_Ex1': train_predictions[:, 0],
        'pred_Ex2': train_predictions[:, 1],
        'pred_Ex3': train_predictions[:, 2],
        'pred_Rx1': train_predictions[:, 3],
        'pred_Rx2': train_predictions[:, 4],
        'pred_Rx3': train_predictions[:, 5],
    })
    train_df.to_csv(train_pred_path, index=False)
    
    val_pred_path = os.path.join(results_dir, 'validation_results.csv')
    val_labels = val_result['labels']  # [n_samples, 6]
    val_predictions = val_result['predictions']  # [n_samples, 6]
    val_df = pd.DataFrame({
        'true_Ex1': val_labels[:, 0],
        'true_Ex2': val_labels[:, 1],
        'true_Ex3': val_labels[:, 2],
        'true_Rx1': val_labels[:, 3],
        'true_Rx2': val_labels[:, 4],
        'true_Rx3': val_labels[:, 5],
        'pred_Ex1': val_predictions[:, 0],
        'pred_Ex2': val_predictions[:, 1],
        'pred_Ex3': val_predictions[:, 2],
        'pred_Rx1': val_predictions[:, 3],
        'pred_Rx2': val_predictions[:, 4],
        'pred_Rx3': val_predictions[:, 5],
    })
    val_df.to_csv(val_pred_path, index=False)
    
    # Keep the test partition sealed until the best validation checkpoint is restored.
    test_pred_path = None
    test_metrics_path = None
    
    # Save the generated artifacts.
    os.makedirs(args.finetune_output_dir, exist_ok=True)
    final_model_path = os.path.join(args.finetune_output_dir, 'solvbert.pt')
    final_model_path = os.path.normpath(final_model_path)  # Configure repository paths.
    torch.save({
        'model_state_dict': finetune_model.state_dict(),
        'config': {
            'vocab_size': len(tokenizer),
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'intermediate_size': args.intermediate_size,
            'hidden_dropout_rate': args.hidden_dropout_rate,
        }
    }, final_model_path)
    
    # Run the training step.
    best_model_checkpoint = torch.load(os.path.join(args.finetune_output_dir, 'best_model.pt'), map_location=device, weights_only=False)
    finetune_model.load_state_dict(best_model_checkpoint['model_state_dict'])
    best_train_result = finetune_evaluate(finetune_model, finetune_train_eval_loader, device, return_predictions=False)
    
    # Evaluate the validation subset.
    best_val_result = finetune_evaluate(finetune_model, finetune_val_loader, device, return_predictions=False)
    
    # Save the generated artifacts.
    best_metrics = {
        'best_epoch': best_epoch,
        'best_val_mse': best_val_mse,
        'best_val_rmse': best_val_rmse,
        'best_val_mae': best_val_mae,
        'best_val_r2': best_val_r2,
        'best_val_std': best_val_std,
        'best_train_rmse': best_train_result['rmse'],
        'best_train_mae': best_train_result['mae'],
        'best_train_r2': best_train_result['r2'],
        'best_train_mse': best_train_result['mse'],
        'best_train_std': best_train_result['std'],
        # Run the training step.
        'best_train_e_phase_mae': np.mean([best_train_result['per_output_metrics'][i]['mae'] for i in [0, 1, 2]]),
        'best_train_e_phase_rmse': np.mean([best_train_result['per_output_metrics'][i]['rmse'] for i in [0, 1, 2]]),
        'best_train_r_phase_mae': np.mean([best_train_result['per_output_metrics'][i]['mae'] for i in [3, 4, 5]]),
        'best_train_r_phase_rmse': np.mean([best_train_result['per_output_metrics'][i]['rmse'] for i in [3, 4, 5]]),
        # Evaluate the validation subset.
        'best_val_e_phase_mae': np.mean([best_val_result['per_output_metrics'][i]['mae'] for i in [0, 1, 2]]),
        'best_val_e_phase_rmse': np.mean([best_val_result['per_output_metrics'][i]['rmse'] for i in [0, 1, 2]]),
        'best_val_r_phase_mae': np.mean([best_val_result['per_output_metrics'][i]['mae'] for i in [3, 4, 5]]),
        'best_val_r_phase_rmse': np.mean([best_val_result['per_output_metrics'][i]['rmse'] for i in [3, 4, 5]])
    }
    metrics_txt_path = save_metrics(
        best_metrics,
        args.finetune_output_dir,
        total_time,
        avg_time_per_epoch,
        len(training_history)
    )
    
    # Save the generated artifacts.
    training_metrics_path = os.path.join(results_dir, 'training_metrics.txt')
    with open(training_metrics_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("训练指标详情\n")
        f.write("="*100 + "\n\n")
        for entry in training_history:
            f.write(f"Epoch {entry['epoch']}:\n")
            f.write(f"  训练集 - MAE={entry['train_mae']:.6f}  RMSE={entry['train_rmse']:.6f}  R²={entry['train_r2']:.6f}  STD={entry['train_std']:.4f}\n")
            f.write(f"  验证集 - MAE={entry['val_mae']:.6f}  RMSE={entry['val_rmse']:.6f}  R²={entry['val_r2']:.6f}  STD={entry['val_std']:.4f}\n\n")
        f.write("="*100 + "\n")
        f.write("最佳模型指标\n")
        f.write("="*100 + "\n")
        f.write(f"最佳epoch: {best_epoch}\n")
        f.write(f"最佳验证集MSE: {best_val_mse:.6f}\n")
        f.write(f"最佳验证集RMSE: {best_val_rmse:.6f}\n")
        f.write(f"最佳验证集MAE: {best_val_mae:.6f}\n")
        f.write(f"最佳验证集R²: {best_val_r2:.6f}\n")
        f.write(f"最佳验证集STD: {best_val_std:.4f}\n")
    
    # Evaluate the test subset.
    if finetune_test_loader:
        print("\n" + "="*100)
        print("【测试集评估（使用最佳模型权重）】")
        print("="*100)
        test_result = finetune_evaluate(finetune_model, finetune_test_loader, device, return_predictions=True)
        
        # Compute evaluation metrics.
        test_e_mae = np.mean([test_result['per_output_metrics'][i]['mae'] for i in [0, 1, 2]])
        test_e_rmse = np.mean([test_result['per_output_metrics'][i]['rmse'] for i in [0, 1, 2]])
        test_e_r2 = np.mean([test_result['per_output_metrics'][i]['r2'] for i in [0, 1, 2]])
        test_e_std = np.mean([test_result['per_output_metrics'][i]['std'] for i in [0, 1, 2]])
        
        test_r_mae = np.mean([test_result['per_output_metrics'][i]['mae'] for i in [3, 4, 5]])
        test_r_rmse = np.mean([test_result['per_output_metrics'][i]['rmse'] for i in [3, 4, 5]])
        test_r_r2 = np.mean([test_result['per_output_metrics'][i]['r2'] for i in [3, 4, 5]])
        test_r_std = np.mean([test_result['per_output_metrics'][i]['std'] for i in [3, 4, 5]])
        
        print("【测试集指标】")
        print(f"  Overall - MAE: {test_result['mae']:.6f} ± {test_result['std']:.6f}, RMSE: {test_result['rmse']:.6f} ± {test_result['std']:.6f}")
        print(f"  E相 - MAE: {test_e_mae:.6f} ± {test_e_std:.6f}, RMSE: {test_e_rmse:.6f} ± {test_e_std:.6f}, R²: {test_e_r2:.6f} ± {test_e_std:.6f}")
        print(f"  R相 - MAE: {test_r_mae:.6f} ± {test_r_std:.6f}, RMSE: {test_r_rmse:.6f} ± {test_r_std:.6f}, R²: {test_r_r2:.6f} ± {test_r_std:.6f}")
        print("="*100 + "\n")
        
        # Save the generated artifacts.
        test_pred_path = os.path.join(results_dir, 'test_results.csv')
        test_labels = test_result['labels']  # [n_samples, 6]
        test_predictions = test_result['predictions']  # [n_samples, 6]
        test_df = pd.DataFrame({
            'true_Ex1': test_labels[:, 0],
            'true_Ex2': test_labels[:, 1],
            'true_Ex3': test_labels[:, 2],
            'true_Rx1': test_labels[:, 3],
            'true_Rx2': test_labels[:, 4],
            'true_Rx3': test_labels[:, 5],
            'pred_Ex1': test_predictions[:, 0],
            'pred_Ex2': test_predictions[:, 1],
            'pred_Ex3': test_predictions[:, 2],
            'pred_Rx1': test_predictions[:, 3],
            'pred_Rx2': test_predictions[:, 4],
            'pred_Rx3': test_predictions[:, 5],
        })
        test_df.to_csv(test_pred_path, index=False)
        
        # Save the generated artifacts.
        test_metrics_path = os.path.join(results_dir, 'test_metrics.txt')
        with open(test_metrics_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("测试集指标（使用最佳模型权重）\n")
            f.write("="*100 + "\n\n")
            f.write("【Overall指标】\n")
            f.write(f"MAE: {test_result['mae']:.6f} ± {test_result['std']:.6f}\n")
            f.write(f"RMSE: {test_result['rmse']:.6f} ± {test_result['std']:.6f}\n\n")
            f.write("【E相指标（Ex1, Ex2, Ex3）】\n")
            f.write(f"MAE: {test_e_mae:.6f} ± {test_e_std:.6f}\n")
            f.write(f"RMSE: {test_e_rmse:.6f} ± {test_e_std:.6f}\n")
            f.write(f"R²: {test_e_r2:.6f} ± {test_e_std:.6f}\n\n")
            f.write("【R相指标（Rx1, Rx2, Rx3）】\n")
            f.write(f"MAE: {test_r_mae:.6f} ± {test_r_std:.6f}\n")
            f.write(f"RMSE: {test_r_rmse:.6f} ± {test_r_std:.6f}\n")
            f.write(f"R²: {test_r_r2:.6f} ± {test_r_std:.6f}\n")
            f.write("="*100 + "\n")
    
    # Run the training step.
    print("="*100)
    print("训练完成！")
    print(f"最佳模型在epoch {best_epoch}，验证集MSE: {best_val_mse:.6f}")
    print(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"平均每轮时间: {avg_time_per_epoch:.2f}秒")
    print("\n结果文件已保存:")
    print(f"  - 训练历史CSV: {history_csv_path}")
    print(f"  - 训练集结果CSV: {train_pred_path}")
    print(f"  - 验证集结果CSV: {val_pred_path}")
    if test_pred_path:
        print(f"  - 测试集结果CSV: {test_pred_path}")
        print(f"  - 测试集指标TXT: {test_metrics_path}")
    print(f"  - 训练指标TXT: {training_metrics_path}")
    print(f"  - 最佳指标TXT: {metrics_txt_path}")
    print(f"  - 模型权重: {final_model_path}")
    print("="*100)


# Run the training step.

# Process the experiment data.
PRETRAIN_TRAIN_DATA = str(TRAIN_CSV)
PRETRAIN_VAL_DATA = str(VALIDATION_CSV)
FINETUNE_TRAIN_DATA = str(TRAIN_CSV)
FINETUNE_VAL_DATA = str(VALIDATION_CSV)
FINETUNE_TEST_DATA = str(TEST_CSV)

# Set the random seed.
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]

# Save the generated artifacts.
BASE_OUTPUT_DIR = str(EXPERIMENT_ROOT / 'runs' / 'solvbert')


def train_with_seed(seed, args_override=None):
    """Run the train with seed baseline operation."""
    print("\n" + "="*100)
    print(f"开始训练 - 随机种子: {seed}")
    print("="*100 + "\n")
    
    # Set the random seed.
    seed_output_dir = os.path.join(BASE_OUTPUT_DIR, f'seed_{seed}')
    pretrain_output_dir = os.path.join(seed_output_dir, 'pretrain')
    finetune_output_dir = seed_output_dir  # Save the generated artifacts.
    
    # Configure experiment parameters.
    class Args:
        pass
    
    args = Args()
    
    # Configure experiment parameters.
    args.pretrain_train_data = PRETRAIN_TRAIN_DATA
    args.pretrain_val_data = PRETRAIN_VAL_DATA
    args.finetune_train_data = FINETUNE_TRAIN_DATA
    args.finetune_val_data = FINETUNE_VAL_DATA
    args.finetune_test_data = FINETUNE_TEST_DATA if os.path.exists(FINETUNE_TEST_DATA) else None
    args.pretrain_output_dir = pretrain_output_dir
    args.finetune_output_dir = finetune_output_dir
    args.checkpoint_subdir = 'checkpoint'
    args.random_seed = seed
    args.pretrain_num_epochs = 10
    args.finetune_num_epochs = 200  # Apply early stopping.
    args.vocab_size = 1000
    args.hidden_size = 256
    args.num_layers = 6
    args.num_heads = 8
    args.intermediate_size = 1024
    args.max_length = 512
    args.pretrain_batch_size = 16
    args.pretrain_learning_rate = 2e-5
    args.pretrain_warmup_steps = 1000
    args.mlm_probability = 0.15
    args.finetune_batch_size = 16
    args.finetune_learning_rate = 8e-5
    args.finetune_warmup_steps = 500
    args.hidden_dropout_rate = 0.4
    args.early_stop_patience = 50  # Apply early stopping.
    args.early_stop_min_delta = 0.0
    args.checkpoint_save_freq = 10
    args.rest_interval_hours = 2.0
    args.rest_duration = 300
    args.resume_from_checkpoint = None
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.tokenizer_name = 'bert-base-uncased'
    args.skip_pretrain = False
    
    # Configure experiment parameters.
    if args_override:
        for key, value in args_override.items():
            # Configure the output artifacts.
            if key not in ['pretrain_output_dir', 'finetune_output_dir']:
                setattr(args, key, value)
            else:
                print(f"警告: 参数 '{key}' 被忽略，将使用seed特定的路径: {getattr(args, key)}")
    
    # Configure repository paths.
    args.pretrain_output_dir = pretrain_output_dir
    args.finetune_output_dir = finetune_output_dir
    
    try:
        train_single(args)
        print(f"\n种子 {seed} 训练完成！")
        print(f"结果保存在: {seed_output_dir}")
        
        # Read the input data.
        metrics_file = os.path.join(finetune_output_dir, 'best_metrics.txt')
        metrics = {'seed': seed}
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Compute evaluation metrics.
                    import re
                    overall_start = content.find('【Overall指标】')
                    e_start = content.find('【E相指标')
                    if overall_start >= 0 and e_start >= 0:
                        overall_section = content[overall_start:e_start]
                        train_overall_match = re.search(r'训练集 - MAE: ([\d.]+), RMSE: ([\d.]+)', overall_section)
                        val_overall_match = re.search(r'验证集 - MAE: ([\d.]+), RMSE: ([\d.]+)', overall_section)
                        if train_overall_match:
                            metrics['train_mae'] = float(train_overall_match.group(1))
                            metrics['train_rmse'] = float(train_overall_match.group(2))
                        if val_overall_match:
                            metrics['val_mae'] = float(val_overall_match.group(1))
                            metrics['val_rmse'] = float(val_overall_match.group(2))
                    
                    # Compute evaluation metrics.
                    r_start = content.find('【R相指标')
                    if e_start >= 0 and r_start >= 0:
                        e_section = content[e_start:r_start]
                        train_e_match = re.search(r'训练集 - MAE: ([\d.]+), RMSE: ([\d.]+)', e_section)
                        val_e_match = re.search(r'验证集 - MAE: ([\d.]+), RMSE: ([\d.]+)', e_section)
                        if train_e_match:
                            metrics['train_e_mae'] = float(train_e_match.group(1))
                            metrics['train_e_rmse'] = float(train_e_match.group(2))
                        if val_e_match:
                            metrics['val_e_mae'] = float(val_e_match.group(1))
                            metrics['val_e_rmse'] = float(val_e_match.group(2))
                    
                    # Compute evaluation metrics.
                    if r_start >= 0:
                        r_section = content[r_start:]
                        train_r_match = re.search(r'训练集 - MAE: ([\d.]+), RMSE: ([\d.]+)', r_section)
                        val_r_match = re.search(r'验证集 - MAE: ([\d.]+), RMSE: ([\d.]+)', r_section)
                        if train_r_match:
                            metrics['train_r_mae'] = float(train_r_match.group(1))
                            metrics['train_r_rmse'] = float(train_r_match.group(2))
                        if val_r_match:
                            metrics['val_r_mae'] = float(val_r_match.group(1))
                            metrics['val_r_rmse'] = float(val_r_match.group(2))
            except Exception as e:
                print(f"警告: 无法解析种子 {seed} 的指标文件: {e}")
        
        return metrics
    except Exception as e:
        print(f"\n种子 {seed} 训练失败！错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_multiple_seeds(seeds=None, args_override=None):
    """Run the train multiple seeds baseline operation."""
    if seeds is None:
        seeds = DEFAULT_SEEDS
    
    print("="*100)
    print("批量训练脚本 - 使用多个随机种子")
    print("="*100)
    print(f"\n将使用以下随机种子: {seeds}")
    print(f"每个种子的训练结果将保存在: {BASE_OUTPUT_DIR}/seed_{{seed}}/")
    print(f"检查点将保存在: {BASE_OUTPUT_DIR}/seed_{{seed}}/checkpoint/")
    print(f"最佳模型权重将保存在: {BASE_OUTPUT_DIR}/seed_{{seed}}/best_model.pt")
    print(f"所有训练结果文件将保存在: {BASE_OUTPUT_DIR}/seed_{{seed}}/")
    print("\n" + "="*100 + "\n")
    
    # Configure the output artifacts.
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Set the random seed.
    success_seeds = []
    failed_seeds = []
    all_metrics = []
    
    # Set the random seed.
    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] 处理种子: {seed}")
        
        result = train_with_seed(seed, args_override)
        if result:
            success_seeds.append(seed)
            if isinstance(result, dict):
                all_metrics.append(result)
        else:
            failed_seeds.append(seed)
        
        # Baseline workflow step.
        if i < len(seeds):
            print("\n" + "-"*100 + "\n")
    
    # Save the generated artifacts.
    if all_metrics:
        summary_path = os.path.join(BASE_OUTPUT_DIR, 'summary_metrics.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("多种子训练结果汇总（均值 ± 标准差）\n")
            f.write("="*100 + "\n\n")
            
            # Compute evaluation metrics.
            train_maes = [m['train_mae'] for m in all_metrics if 'train_mae' in m]
            train_rmses = [m['train_rmse'] for m in all_metrics if 'train_rmse' in m]
            val_maes = [m['val_mae'] for m in all_metrics if 'val_mae' in m]
            val_rmses = [m['val_rmse'] for m in all_metrics if 'val_rmse' in m]
            
            if train_maes:
                train_mae_mean = np.mean(train_maes)
                train_mae_std = np.std(train_maes)
                train_rmse_mean = np.mean(train_rmses)
                train_rmse_std = np.std(train_rmses)
                val_mae_mean = np.mean(val_maes)
                val_mae_std = np.std(val_maes)
                val_rmse_mean = np.mean(val_rmses)
                val_rmse_std = np.std(val_rmses)
                
                f.write("【Overall指标】\n")
                f.write(f"训练集 - MAE: {train_mae_mean:.4f} ± {train_mae_std:.4f}, RMSE: {train_rmse_mean:.4f} ± {train_rmse_std:.4f}\n")
                f.write(f"验证集 - MAE: {val_mae_mean:.4f} ± {val_mae_std:.4f}, RMSE: {val_rmse_mean:.4f} ± {val_rmse_std:.4f}\n\n")
            
            # Compute evaluation metrics.
            train_e_maes = [m['train_e_mae'] for m in all_metrics if 'train_e_mae' in m]
            train_e_rmses = [m['train_e_rmse'] for m in all_metrics if 'train_e_rmse' in m]
            val_e_maes = [m['val_e_mae'] for m in all_metrics if 'val_e_mae' in m]
            val_e_rmses = [m['val_e_rmse'] for m in all_metrics if 'val_e_rmse' in m]
            
            if train_e_maes:
                train_e_mae_mean = np.mean(train_e_maes)
                train_e_mae_std = np.std(train_e_maes)
                train_e_rmse_mean = np.mean(train_e_rmses)
                train_e_rmse_std = np.std(train_e_rmses)
                val_e_mae_mean = np.mean(val_e_maes)
                val_e_mae_std = np.std(val_e_maes)
                val_e_rmse_mean = np.mean(val_e_rmses)
                val_e_rmse_std = np.std(val_e_rmses)
                
                f.write("【E相指标（Ex1, Ex2, Ex3）】\n")
                f.write(f"训练集 - MAE: {train_e_mae_mean:.4f} ± {train_e_mae_std:.4f}, RMSE: {train_e_rmse_mean:.4f} ± {train_e_rmse_std:.4f}\n")
                f.write(f"验证集 - MAE: {val_e_mae_mean:.4f} ± {val_e_mae_std:.4f}, RMSE: {val_e_rmse_mean:.4f} ± {val_e_rmse_std:.4f}\n\n")
            
            # Compute evaluation metrics.
            train_r_maes = [m['train_r_mae'] for m in all_metrics if 'train_r_mae' in m]
            train_r_rmses = [m['train_r_rmse'] for m in all_metrics if 'train_r_rmse' in m]
            val_r_maes = [m['val_r_mae'] for m in all_metrics if 'val_r_mae' in m]
            val_r_rmses = [m['val_r_rmse'] for m in all_metrics if 'val_r_rmse' in m]
            
            if train_r_maes:
                train_r_mae_mean = np.mean(train_r_maes)
                train_r_mae_std = np.std(train_r_maes)
                train_r_rmse_mean = np.mean(train_r_rmses)
                train_r_rmse_std = np.std(train_r_rmses)
                val_r_mae_mean = np.mean(val_r_maes)
                val_r_mae_std = np.std(val_r_maes)
                val_r_rmse_mean = np.mean(val_r_rmses)
                val_r_rmse_std = np.std(val_r_rmses)
                
                f.write("【R相指标（Rx1, Rx2, Rx3）】\n")
                f.write(f"训练集 - MAE: {train_r_mae_mean:.4f} ± {train_r_mae_std:.4f}, RMSE: {train_r_rmse_mean:.4f} ± {train_r_rmse_std:.4f}\n")
                f.write(f"验证集 - MAE: {val_r_mae_mean:.4f} ± {val_r_mae_std:.4f}, RMSE: {val_r_rmse_mean:.4f} ± {val_r_rmse_std:.4f}\n\n")
            
            f.write(f"训练的种子数: {len(success_seeds)}\n")
            f.write(f"成功的种子: {success_seeds}\n")
            if failed_seeds:
                f.write(f"失败的种子: {failed_seeds}\n")
        
        print(f"\n汇总指标已保存到: {summary_path}")
    
    # Baseline workflow step.
    print("\n" + "="*100)
    print("批量训练完成！")
    print("="*100)
    print(f"\n成功训练的种子: {success_seeds}")
    if failed_seeds:
        print(f"失败的种子: {failed_seeds}")
    print(f"\n所有结果保存在: {BASE_OUTPUT_DIR}/")
    if all_metrics:
        print(f"汇总指标保存在: {os.path.join(BASE_OUTPUT_DIR, 'summary_metrics.txt')}")
    print("="*100)


def create_parser():
    """Run the create parser baseline operation."""
    parser = argparse.ArgumentParser(description='SolvBERT完整训练流程（预训练+微调），支持单次训练和批量多种子训练')
    
    # Run the training step.
    parser.add_argument('--mode', type=str, choices=['single', 'multiple'], default='multiple',
                        help='训练模式: single=单次训练, multiple=批量多种子训练')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='批量训练时的种子列表（仅在mode=multiple时有效）')
    
    # Run the training step.
    parser.add_argument('--pretrain_train_data', type=str, default=PRETRAIN_TRAIN_DATA, help='预训练训练数据路径')
    parser.add_argument('--pretrain_val_data', type=str, default=PRETRAIN_VAL_DATA, help='预训练验证数据路径')
    
    # Process the experiment data.
    parser.add_argument('--finetune_train_data', type=str, default=FINETUNE_TRAIN_DATA, help='微调训练数据路径')
    parser.add_argument('--finetune_val_data', type=str, default=FINETUNE_VAL_DATA, help='微调验证数据路径')
    parser.add_argument('--finetune_test_data', type=str, default=FINETUNE_TEST_DATA, help='微调测试数据路径（可选）')
    
    # Configure the output artifacts.
    parser.add_argument('--pretrain_output_dir', type=str,
                        default=str(EXPERIMENT_ROOT / 'runs' / 'solvbert' / 'pretrain'),
                        help='Pretraining output directory.')
    parser.add_argument('--finetune_output_dir', type=str,
                        default=str(EXPERIMENT_ROOT / 'runs' / 'solvbert' / 'finetune'),
                        help='Fine-tuning output directory.')
    parser.add_argument('--checkpoint_subdir', type=str, default=None, help='checkpoint子目录（如果指定，checkpoint将保存到此子目录）')
    
    # Configure the baseline model.
    parser.add_argument('--vocab_size', type=int, default=1000, help='词汇表大小')
    parser.add_argument('--hidden_size', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--intermediate_size', type=int, default=1024, help='前馈网络中间层维度')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    
    # Run the training step.
    parser.add_argument('--pretrain_batch_size', type=int, default=16, help='预训练批次大小')
    parser.add_argument('--pretrain_learning_rate', type=float, default=2e-5, help='预训练学习率')
    parser.add_argument('--pretrain_num_epochs', type=int, default=10, help='预训练轮数')
    parser.add_argument('--pretrain_warmup_steps', type=int, default=1000, help='预训练预热步数')
    parser.add_argument('--mlm_probability', type=float, default=0.15, help='掩码概率')
    
    # Baseline workflow step.
    parser.add_argument('--finetune_batch_size', type=int, default=16, help='微调批次大小')
    parser.add_argument('--finetune_learning_rate', type=float, default=8e-5, help='微调学习率')
    parser.add_argument('--finetune_num_epochs', type=int, default=200, help='微调轮数（增加到200以支持早停）')
    parser.add_argument('--finetune_warmup_steps', type=int, default=500, help='微调预热步数')
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.4, help='Dropout率')
    
    # Run the training step.
    parser.add_argument('--early_stop_patience', type=int, default=50, help='早停耐心值（设置为50）')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='早停最小改善')
    parser.add_argument('--checkpoint_save_freq', type=int, default=10, help='检查点保存频率（每N个epoch）')
    parser.add_argument('--rest_interval_hours', type=float, default=2.0, help='休息间隔（每N小时休息，0表示不休息）')
    parser.add_argument('--rest_duration', type=int, default=300, help='休息时长（秒）')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='从检查点恢复训练（检查点路径）')
    
    # Baseline workflow step.
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='tokenizer名称')
    parser.add_argument('--skip_pretrain', action='store_true', help='跳过预训练，直接微调')
    parser.add_argument('--random_seed', type=int, default=2024, help='随机种子（仅在mode=single时有效）')
    
    return parser


def main():
    """Run the main baseline operation."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.mode == 'multiple':
        # Run the training step.
        seeds = args.seeds if args.seeds else DEFAULT_SEEDS
        # Baseline workflow step.
        args_dict = vars(args)
        # Run the training step.
        args_dict.pop('mode')
        args_dict.pop('seeds')
        train_multiple_seeds(seeds, args_dict)
    else:
        # Run the training step.
        train_single(args)


if __name__ == '__main__':
    main()

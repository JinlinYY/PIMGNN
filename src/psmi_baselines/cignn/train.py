"""Implement the cignn train baseline module."""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from tqdm import tqdm
import time
from datetime import datetime

from .model import CIGIN
from .data_utils import smiles_to_graph, batch_graphs
from psmi_baselines.paths import EXPERIMENT_ROOT, TOTAL_CSV
from psmi_baselines.protocol import canonical_split_indices


class LLEDataset(Dataset):
    """Represent the LLEDataset baseline component."""
    def __init__(self, il_smiles_list, comp2_smiles_list, comp3_smiles_list, 
                 labels, temperatures=None):
        """Run the init baseline operation."""
        self.il_smiles_list = il_smiles_list
        self.comp2_smiles_list = comp2_smiles_list
        self.comp3_smiles_list = comp3_smiles_list
        self.labels = labels
        self.temperatures = temperatures if temperatures is not None else [None] * len(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'il_smiles': self.il_smiles_list[idx],
            'comp2_smiles': self.comp2_smiles_list[idx],
            'comp3_smiles': self.comp3_smiles_list[idx],
            'label': self.labels[idx],
            'temperature': self.temperatures[idx]
        }


def collate_fn(batch):
    """Run the collate fn baseline operation."""
    il_graphs = []
    comp2_graphs = []
    comp3_graphs = []
    labels = []
    temperatures = []
    
    for item in batch:
        il_graph = smiles_to_graph(item['il_smiles'])
        comp2_graph = smiles_to_graph(item['comp2_smiles'])
        comp3_graph = smiles_to_graph(item['comp3_smiles'])
        
        if il_graph is not None and comp2_graph is not None and comp3_graph is not None:
            il_graphs.append(il_graph)
            comp2_graphs.append(comp2_graph)
            comp3_graphs.append(comp3_graph)
            labels.append(item['label'])
            temperatures.append(item['temperature'])
    
    if len(il_graphs) == 0:
        return None, None, None, None, None
    
    # Baseline workflow step.
    il_batch = batch_graphs(il_graphs)
    comp2_batch = batch_graphs(comp2_graphs)
    comp3_batch = batch_graphs(comp3_graphs)
    
    # Baseline workflow step.
    labels = np.array(labels, dtype=np.float32)  # [batch_size, 6]
    labels = torch.from_numpy(labels)  # [batch_size, 6]
    
    # Baseline workflow step.
    if temperatures[0] is not None:
        temperatures = np.array(temperatures, dtype=np.float32)
        temperatures = torch.from_numpy(temperatures)
    else:
        temperatures = None
    
    return il_batch, comp2_batch, comp3_batch, labels, temperatures


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Run the train epoch baseline operation."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch[0] is None:
            continue
        
        il_batch, comp2_batch, comp3_batch, labels, temperatures = batch
        il_batch = il_batch.to(device)
        comp2_batch = comp2_batch.to(device)
        comp3_batch = comp3_batch.to(device)
        labels = labels.to(device)
        
        if temperatures is not None:
            temperatures = temperatures.to(device)
        
        optimizer.zero_grad()
        
        predictions, _ = model(il_batch, comp2_batch, comp3_batch, temperatures)
        # predictions shape: [batch_size, 6]
        
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def calculate_metrics(predictions, labels):
    """Run the calculate metrics baseline operation."""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Baseline workflow step.
    errors = predictions - labels
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Configure the output artifacts.
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)
    r2 = 1 - np.sum(squared_errors) / np.sum((labels - np.mean(labels, axis=0)) ** 2)
    
    # Baseline workflow step.
    sample_mae = np.mean(abs_errors, axis=1)  # Baseline workflow step.
    sample_rmse = np.sqrt(np.mean(squared_errors, axis=1))  # Baseline workflow step.
    mae_std = np.std(sample_mae)
    rmse_std = np.std(sample_rmse)
    
    # Compute evaluation metrics.
    e_predictions = predictions[:, :3]
    e_labels = labels[:, :3]
    e_errors = e_predictions - e_labels
    e_abs_errors = np.abs(e_errors)
    e_squared_errors = e_errors ** 2
    
    mse_e = np.mean(e_squared_errors)
    rmse_e = np.sqrt(mse_e)
    mae_e = np.mean(e_abs_errors)
    r2_e = 1 - np.sum(e_squared_errors) / np.sum((e_labels - np.mean(e_labels, axis=0)) ** 2)
    
    # Baseline workflow step.
    e_sample_mae = np.mean(e_abs_errors, axis=1)
    e_sample_rmse = np.sqrt(np.mean(e_squared_errors, axis=1))
    mae_e_std = np.std(e_sample_mae)
    rmse_e_std = np.std(e_sample_rmse)
    
    # Compute evaluation metrics.
    r_predictions = predictions[:, 3:]
    r_labels = labels[:, 3:]
    r_errors = r_predictions - r_labels
    r_abs_errors = np.abs(r_errors)
    r_squared_errors = r_errors ** 2
    
    mse_r = np.mean(r_squared_errors)
    rmse_r = np.sqrt(mse_r)
    mae_r = np.mean(r_abs_errors)
    r2_r = 1 - np.sum(r_squared_errors) / np.sum((r_labels - np.mean(r_labels, axis=0)) ** 2)
    
    # Baseline workflow step.
    r_sample_mae = np.mean(r_abs_errors, axis=1)
    r_sample_rmse = np.sqrt(np.mean(r_squared_errors, axis=1))
    mae_r_std = np.std(r_sample_mae)
    rmse_r_std = np.std(r_sample_rmse)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'rmse_std': rmse_std,
        'mae': mae,
        'mae_std': mae_std,
        'r2': r2,
        'mse_e': mse_e,
        'rmse_e': rmse_e,
        'rmse_e_std': rmse_e_std,
        'mae_e': mae_e,
        'mae_e_std': mae_e_std,
        'r2_e': r2_e,
        'mse_r': mse_r,
        'rmse_r': rmse_r,
        'rmse_r_std': rmse_r_std,
        'mae_r': mae_r,
        'mae_r_std': mae_r_std,
        'r2_r': r2_r
    }
    
    return metrics, predictions, labels


def evaluate(model, dataloader, criterion, device):
    """Run the evaluate baseline operation."""
    model.eval()
    total_loss = 0
    predictions_list = []
    labels_list = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if batch[0] is None:
                continue
            
            il_batch, comp2_batch, comp3_batch, labels, temperatures = batch
            il_batch = il_batch.to(device)
            comp2_batch = comp2_batch.to(device)
            comp3_batch = comp3_batch.to(device)
            labels = labels.to(device)
            
            if temperatures is not None:
                temperatures = temperatures.to(device)
            
            predictions, _ = model(il_batch, comp2_batch, comp3_batch, temperatures)
            # predictions shape: [batch_size, 6]
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # Compute evaluation metrics.
    metrics, predictions_array, labels_array = calculate_metrics(predictions_list, labels_list)
    
    return avg_loss, metrics, predictions_array, labels_array


def save_results(args, history, train_pred, train_true, val_pred, val_true,
                 test_pred, test_true, best_epoch, best_val_mse,
                 total_time, avg_time_per_epoch, val_metrics, test_metrics):
    """Run the save results baseline operation."""
    
    # Save the generated artifacts.
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.results_dir, 'train_history.csv'), index=False)
    
    # Save the generated artifacts.
    train_results = pd.DataFrame({
        'Ex1_true': train_true[:, 0],
        'Ex2_true': train_true[:, 1],
        'Ex3_true': train_true[:, 2],
        'Rx1_true': train_true[:, 3],
        'Rx2_true': train_true[:, 4],
        'Rx3_true': train_true[:, 5],
        'pred_Ex1': train_pred[:, 0],
        'pred_Ex2': train_pred[:, 1],
        'pred_Ex3': train_pred[:, 2],
        'pred_Rx1': train_pred[:, 3],
        'pred_Rx2': train_pred[:, 4],
        'pred_Rx3': train_pred[:, 5],
        'split': 'train'
    })
    
    val_results = pd.DataFrame({
        'Ex1_true': val_true[:, 0],
        'Ex2_true': val_true[:, 1],
        'Ex3_true': val_true[:, 2],
        'Rx1_true': val_true[:, 3],
        'Rx2_true': val_true[:, 4],
        'Rx3_true': val_true[:, 5],
        'pred_Ex1': val_pred[:, 0],
        'pred_Ex2': val_pred[:, 1],
        'pred_Ex3': val_pred[:, 2],
        'pred_Rx1': val_pred[:, 3],
        'pred_Rx2': val_pred[:, 4],
        'pred_Rx3': val_pred[:, 5],
        'split': 'val'
    })
    
    training_results = pd.concat([train_results, val_results], ignore_index=True)
    training_results.to_csv(os.path.join(args.results_dir, 'training_results.csv'), index=False)
    
    # Save the generated artifacts.
    if test_pred is not None:
        test_results = pd.DataFrame({
            'Ex1_true': test_true[:, 0],
            'Ex2_true': test_true[:, 1],
            'Ex3_true': test_true[:, 2],
            'Rx1_true': test_true[:, 3],
            'Rx2_true': test_true[:, 4],
            'Rx3_true': test_true[:, 5],
            'pred_Ex1': test_pred[:, 0],
            'pred_Ex2': test_pred[:, 1],
            'pred_Ex3': test_pred[:, 2],
            'pred_Rx1': test_pred[:, 3],
            'pred_Rx2': test_pred[:, 4],
            'pred_Rx3': test_pred[:, 5],
            'split': 'test'
        })
        test_results.to_csv(os.path.join(args.results_dir, 'test_results.csv'), index=False)
    
    # Save the generated artifacts.
    def format_metrics_txt(metrics_dict, prefix=""):
        """Run the format metrics txt baseline operation."""
        lines = []
        if "mse" in metrics_dict:
            lines.append(f"{prefix}[ mean metrics ]")
            lines.append(f"  MSE:  {metrics_dict['mse']:.6f}")
            lines.append(f"  RMSE: {metrics_dict['rmse']:.6f} ± {metrics_dict.get('rmse_std', 0):.4f}")
            lines.append(f"  MAE:  {metrics_dict['mae']:.6f} ± {metrics_dict.get('mae_std', 0):.4f}")
            lines.append(f"  R²:   {metrics_dict['r2']:.6f}")
            lines.append("")
        
        if "mse_e" in metrics_dict:
            lines.append(f"{prefix}[E phase metrics ]")
            lines.append(f"  MSE:  {metrics_dict['mse_e']:.6f}")
            lines.append(f"  RMSE: {metrics_dict['rmse_e']:.6f} ± {metrics_dict.get('rmse_e_std', 0):.4f}")
            lines.append(f"  MAE:  {metrics_dict['mae_e']:.6f} ± {metrics_dict.get('mae_e_std', 0):.4f}")
            lines.append(f"  R²:   {metrics_dict['r2_e']:.6f}")
            lines.append("")
        
        if "mse_r" in metrics_dict:
            lines.append(f"{prefix}[R phase metrics ]")
            lines.append(f"  MSE:  {metrics_dict['mse_r']:.6f}")
            lines.append(f"  RMSE: {metrics_dict['rmse_r']:.6f} ± {metrics_dict.get('rmse_r_std', 0):.4f}")
            lines.append(f"  MAE:  {metrics_dict['mae_r']:.6f} ± {metrics_dict.get('mae_r_std', 0):.4f}")
            lines.append(f"  R²:   {metrics_dict['r2_r']:.6f}")
            lines.append("")
        
        return "\n".join(lines)
    
    # Save the generated artifacts.
    txt_lines = []
    txt_lines.append("=" * 100)
    txt_lines.append(" best-model metrics ")
    txt_lines.append("=" * 100)
    txt_lines.append("")
    
    # Baseline workflow step.
    txt_lines.append("[ training information ]")
    txt_lines.append(f" best epoch: {best_epoch}")
    txt_lines.append(f" best validation MSE: {best_val_mse:.6f}")
    txt_lines.append(f" total training epochs : {len(history)}")
    txt_lines.append(f" total training time : {total_time:.2f} seconds ({total_time/60:.2f} minutes )")
    txt_lines.append(f" mean time per epoch : {avg_time_per_epoch:.2f} seconds ")
    txt_lines.append("")
    
    # Evaluate the validation subset.
    txt_lines.append("[ validation metrics ]")
    txt_lines.append(format_metrics_txt(val_metrics))
    
    # Evaluate the test subset.
    if test_metrics is not None:
        txt_lines.append("[ test metrics ]")
        txt_lines.append(format_metrics_txt(test_metrics))
    
    txt_lines.append("=" * 100)
    
    with open(os.path.join(args.results_dir, 'best_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(txt_lines))
    
    # Save the generated artifacts.
    with open(os.path.join(args.results_dir, 'training_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(" training-metric summary \n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f" best model at epoch {best_epoch}\n")
        f.write(f" best validation MSE: {best_val_mse:.6f}\n\n")
        
        f.write("[ best validation metrics ]\n")
        f.write(f" [ mean metrics ] MAE={val_metrics['mae']:.6f}±{val_metrics['mae_std']:.4f} RMSE={val_metrics['rmse']:.6f}±{val_metrics['rmse_std']:.4f} R²={val_metrics['r2']:.6f}\n")
        f.write(f" [E phase metrics ] MAE={val_metrics['mae_e']:.6f}±{val_metrics['mae_e_std']:.4f} RMSE={val_metrics['rmse_e']:.6f}±{val_metrics['rmse_e_std']:.4f} R²={val_metrics['r2_e']:.6f}\n")
        f.write(f" [R phase metrics ] MAE={val_metrics['mae_r']:.6f}±{val_metrics['mae_r_std']:.4f} RMSE={val_metrics['rmse_r']:.6f}±{val_metrics['rmse_r_std']:.4f} R²={val_metrics['r2_r']:.6f}\n\n")
        
        if test_metrics is not None:
            f.write("[ test metrics ]\n")
            f.write(f" [ mean metrics ] MAE={test_metrics['mae']:.6f}±{test_metrics['mae_std']:.4f} RMSE={test_metrics['rmse']:.6f}±{test_metrics['rmse_std']:.4f} R²={test_metrics['r2']:.6f}\n")
            f.write(f" [E phase metrics ] MAE={test_metrics['mae_e']:.6f}±{test_metrics['mae_e_std']:.4f} RMSE={test_metrics['rmse_e']:.6f}±{test_metrics['rmse_e_std']:.4f} R²={test_metrics['r2_e']:.6f}\n")
            f.write(f" [R phase metrics ] MAE={test_metrics['mae_r']:.6f}±{test_metrics['mae_r_std']:.4f} RMSE={test_metrics['rmse_r']:.6f}±{test_metrics['rmse_r_std']:.4f} R²={test_metrics['r2_r']:.6f}\n\n")
        
        f.write(f" total training time : {total_time:.2f} seconds ({total_time/60:.2f} minutes )\n")
        f.write(f" mean time per epoch : {avg_time_per_epoch:.2f} seconds \n")
        f.write("=" * 100 + "\n")


def load_csv_data(csv_path):
    """Run the load csv data baseline operation."""
    df = pd.read_csv(csv_path)
    
    il_smiles = df['IL (Component 1) full name SMILES'].tolist()
    comp2_smiles = df['Component 2 SMILES'].tolist()
    comp3_smiles = df['Component 3 SMILES'].tolist()
    
    # Baseline workflow step.
    labels = df[['Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3']].values.astype(np.float32)
    
    # Baseline workflow step.
    temperatures = df['T/K'].values.astype(np.float32)
    # Baseline workflow step.
    temperatures = (temperatures - 250.0) / 150.0  # Baseline workflow step.
    
    return il_smiles, comp2_smiles, comp3_smiles, labels, temperatures


def print_config_info(args, train_size, val_size, test_size, device_info):
    """Run the print config info baseline operation."""
    print("=" * 100)
    print("[ training configuration ]")
    print("=" * 100)
    print()
    print("[ dataset information ]")
    print(f" number of training samples : {train_size}")
    print(f" number of validation samples : {val_size}")
    print(f" number of test samples : {test_size}")
    print()
    print("[ Device configuration ]")
    print(f" device : {device_info['device']}")
    if device_info['device'] == 'cuda':
        print(f" GPU name : {device_info['gpu_name']}")
        print(f" CUDA version : {device_info['cuda_version']}")
        print(f" GPU memory : {device_info['gpu_memory']:.2f} GB")
    print()
    print("[ training parameters ]")
    print(f" random seed : {args.seed}")
    print(f" training epochs : {args.epochs}")
    print(f" batch size : {args.batch_size}")
    print(f" learning rate : {args.lr}")
    print(f" weight decay : {args.weight_decay}")
    print(f" early-stopping patience : {args.patience}")
    print(f" minimum early-stopping improvement : 0.0")
    print(f" checkpoint frequency : per {args.checkpoint_freq} epoch")
    if args.rest_freq > 0:
        print(f" rest policy : per {args.rest_freq} epoch Break {args.rest_duration} seconds ({args.rest_duration/60:.1f} minutes )")
    else:
        print(f" rest policy : Time-based ( per 2 hours before cooldown 5 minutes )")
    # Handle model checkpoints.
    # Configure repository paths.
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f" resume training : yes ( resume from checkpoint )")
    else:
        print(f" resume training : no ( start from scratch )")
    print()
    print("[ model hyperparameters ]")
    print(f" hidden-layer dimension : {args.hidden_dim}")
    print(f" number of graph-neural-network layers : {args.num_mp_layers}")
    print(f" Set2Set number of steps : 3")
    print(f" Dropout rate : 0.0")
    print(f" output dimension : 6 (LLE task : Ex1, Ex2, Ex3, Rx1, Rx2, Rx3)")
    print(f" use Set2Set: {' yes ' if args.use_set2set else ' no '}")
    print(f" use temperature : {' yes ' if args.use_temperature else ' no '}")
    print()
    print("[ path information ]")
    print(f" output directory : {args.output_dir}")
    print(f" result directory : {os.path.join(args.output_dir, 'results')}")
    print()
    print("=" * 100)
    print()


def get_device_info(device):
    """Run the get device info baseline operation."""
    info = {'device': device}
    if device == 'cuda' and torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return info


def find_latest_checkpoint(checkpoint_dir):
    """Run the find latest checkpoint baseline operation."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
            checkpoint_files.append(os.path.join(checkpoint_dir, filename))
    
    if not checkpoint_files:
        return None
    
    # Baseline workflow step.
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoint_files[0]


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Run the load checkpoint baseline operation."""
    print("=" * 100)
    print(f" Discovery checkpoint file : {checkpoint_path}")
    print(" True at load checkpoint in resume training ...")
    print("=" * 100)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load the input data.
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the input data.
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Run the training step.
    start_epoch = checkpoint.get('epoch', 0)
    history = checkpoint.get('history', [])
    
    # Save the generated artifacts.
    best_val_mse = checkpoint.get('best_val_mse', checkpoint.get('val_mse', float('inf')))
    best_epoch = checkpoint.get('best_epoch', start_epoch)
    patience_counter = checkpoint.get('patience_counter', 0)
    
    # Save the generated artifacts.
    if best_val_mse == float('inf') and history:
        best_epoch_info = min(history, key=lambda x: x.get('val_mse', float('inf')))
        best_epoch = best_epoch_info.get('epoch', start_epoch)
        best_val_mse = best_epoch_info.get('val_mse', best_val_mse)
    
    print(f" successful load checkpoint !")
    print(f" - start epoch: {start_epoch}")
    print(f" - best epoch: {best_epoch}")
    print(f" - best validation MSE: {best_val_mse:.6f}")
    print(f" - Patience Counter : {patience_counter}")
    print(f" - History count : {len(history)}")
    print("=" * 100)
    print()
    
    return {
        'start_epoch': start_epoch,
        'history': history,
        'best_val_mse': best_val_mse,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter
    }


def get_default_args():
    """Run the get default args baseline operation."""
    import argparse
    args = argparse.Namespace()
    # Process the experiment data.
    args.data_path = str(TOTAL_CSV)
    args.test_data_path = None  # Evaluate the test subset.
    
    # Save the generated artifacts.
    args.output_dir = str(EXPERIMENT_ROOT / 'runs' / 'cignn' / 'seed_2024')
    args.results_dir = None  # Baseline workflow step.
    
    # Run the training step.
    args.batch_size = 64
    args.epochs = 400
    args.lr = 0.001
    args.weight_decay = 0.0001
    
    # Configure the baseline model.
    args.hidden_dim = 256
    args.num_mp_layers = 3
    args.use_set2set = False
    args.use_temperature = True
    
    # Run the training step.
    args.patience = 80
    args.checkpoint_freq = 10
    args.rest_freq = 0  # Baseline workflow step.
    args.rest_duration = 600
    
    # Configure the runtime device.
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set the random seed.
    args.seed = 2024
    
    return args


def main(args=None):
    """Run the main baseline operation."""
    if args is None:
        # Configure experiment parameters.
        has_cli_args = len(sys.argv) > 1
        
        if has_cli_args:
            # Configure experiment parameters.
            parser = argparse.ArgumentParser(description='Train CIGIN model for LLE prediction')
            parser.add_argument('--data_path', type=str, default=str(TOTAL_CSV),
                               help='Path to dataset file (CSV format)')
            parser.add_argument('--test_data_path', type=str, default=None,
                               help='Path to test dataset file (CSV format, optional)')
            parser.add_argument('--output_dir', type=str,
                               default=str(EXPERIMENT_ROOT / 'runs' / 'cignn' / 'seed_2024'),
                               help='Directory to save checkpoints')
            parser.add_argument('--results_dir', type=str, default=None,
                               help='Directory to save results (default: output_dir/results)')
            parser.add_argument('--batch_size', type=int, default=64,
                               help='Batch size')
            parser.add_argument('--epochs', type=int, default=400,
                               help='Number of epochs')
            parser.add_argument('--lr', type=float, default=0.001,
                               help='Initial learning rate')
            parser.add_argument('--weight_decay', type=float, default=0.0001,
                               help='Weight decay')
            parser.add_argument('--hidden_dim', type=int, default=256,
                               help='Hidden dimension')
            parser.add_argument('--num_mp_layers', type=int, default=3,
                               help='Number of message passing layers')
            parser.add_argument('--use_set2set', action='store_true',
                               help='Use Set2Set instead of sum pooling')
            parser.add_argument('--use_temperature', action='store_true', default=True,
                               help='Use temperature as additional input')
            parser.add_argument('--patience', type=int, default=80,
                               help='Early stopping patience')
            parser.add_argument('--checkpoint_freq', type=int, default=10,
                               help='Checkpoint save frequency (epochs)')
            parser.add_argument('--rest_freq', type=int, default=0,
                               help='Rest frequency (epochs, 0 to disable)')
            parser.add_argument('--rest_duration', type=int, default=600,
                               help='Rest duration (seconds)')
            parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                               help='Device to use')
            parser.add_argument('--seed', type=int, default=2024,
                               help='Random seed')
            
            args = parser.parse_args()
        else:
            # Configure experiment parameters.
            print("=" * 100)
            print("CIGIN model training - use default configuration ( No Command Needed rows parameters )")
            print("=" * 100)
            args = get_default_args()
            
            # Baseline workflow step.
            print("\n training configuration :")
            print(f" Training data : {args.data_path}")
            print(f" Test data : {args.test_data_path}")
            print(f" random seed : {args.seed}")
            print(f" training epochs : {args.epochs}")
            print(f" batch size : {args.batch_size}")
            print(f" learning rate : {args.lr}")
            print(f" hidden-layer dimension : {args.hidden_dim}")
            print(f" Messaging number of layers : {args.num_mp_layers}")
            print(f" use temperature : {args.use_temperature}")
            print(f" use Set2Set: {args.use_set2set}")
            print(f" device : {args.device}")
            print(f" output directory : {args.output_dir}")
            print("=" * 100)
            print()
            
            # Process the experiment data.
            if not os.path.exists(args.data_path):
                print(f" error : Training data file does not exist : {args.data_path}")
                print(" please check file path whether Correct , or use Command rows parameters Designation data path ")
                sys.exit(1)
            
            if args.test_data_path and not os.path.exists(args.test_data_path):
                print(f" warning : Test data file does not exist : {args.test_data_path}")
                print(" will Only use Training data Into rows Training ")
                args.test_data_path = None
    
    # Set the random seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Configure the output artifacts.
    os.makedirs(args.output_dir, exist_ok=True)
    if args.results_dir is None:
        args.results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load the input data.
    print("Loading data from CSV...", flush=True)
    sys.stdout.flush()  # Configure the output artifacts.
    il_smiles, comp2_smiles, comp3_smiles, labels, temperatures = load_csv_data(args.data_path)
    print(f" data load complete , total {len(labels)} samples ", flush=True)
    
    # Process the experiment data.
    # Run the training step.
    # Run the training step.
    split_frame = pd.read_csv(args.data_path)
    canonical_indices = canonical_split_indices(split_frame) if not args.test_data_path else None

    try:
        from sklearn.model_selection import train_test_split
        
        if canonical_indices is not None:
            train_idx = canonical_indices["train"]
            val_idx = canonical_indices["validation"]
            test_idx = canonical_indices["test"]
            test_il_smiles = [il_smiles[i] for i in test_idx]
            test_comp2_smiles = [comp2_smiles[i] for i in test_idx]
            test_comp3_smiles = [comp3_smiles[i] for i in test_idx]
            test_labels = labels[test_idx]
            test_temperatures = temperatures[test_idx] if args.use_temperature else None
            print(
                f"Using canonical split: train={len(train_idx)}, "
                f"validation={len(val_idx)}, test={len(test_idx)}",
                flush=True,
            )
        elif args.test_data_path:
            # Load the input data.
            print(" load Separate test set file ...", flush=True)
            test_il_smiles, test_comp2_smiles, test_comp3_smiles, test_labels, test_temperatures = load_csv_data(args.test_data_path)
            # Run the training step.
            train_idx, val_idx = train_test_split(
                np.arange(len(labels)), 
                test_size=0.2, 
                random_state=args.seed,
                shuffle=True
            )
            print(f" use sklearn Divide : training set {len(train_idx)} bar , validation set {len(val_idx)} bar , test set {len(test_labels)} bar ", flush=True)
        else:
            # Run the training step.
            print(" use sklearn from Total dataset in Divide training set , validation set and test set ...", flush=True)
            # Evaluate the test subset.
            train_val_idx, test_idx = train_test_split(
                np.arange(len(labels)),
                test_size=0.15,
                random_state=args.seed,
                shuffle=True
            )
            # Run the training step.
            # Baseline workflow step.
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=0.235,  # Evaluate the validation subset.
                random_state=args.seed,
                shuffle=True
            )
            
            # Evaluate the test subset.
            test_il_smiles = [il_smiles[i] for i in test_idx]
            test_comp2_smiles = [comp2_smiles[i] for i in test_idx]
            test_comp3_smiles = [comp3_smiles[i] for i in test_idx]
            test_labels = labels[test_idx]
            test_temperatures = temperatures[test_idx] if args.use_temperature else None
            
            print(f" use sklearn split complete : training set {len(train_idx)} bar ({len(train_idx)/len(labels)*100:.1f}%), "
                  f" validation set {len(val_idx)} bar ({len(val_idx)/len(labels)*100:.1f}%), "
                  f" test set {len(test_idx)} bar ({len(test_idx)/len(labels)*100:.1f}%)", flush=True)
    except ImportError:
        # Baseline workflow step.
        print(" hint : sklearn unavailable , use numpy Implementation data Divide ( Features phase same )", flush=True)
        np.random.seed(args.seed)
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        
        if canonical_indices is not None:
            train_idx = canonical_indices["train"]
            val_idx = canonical_indices["validation"]
            test_idx = canonical_indices["test"]
            test_il_smiles = [il_smiles[i] for i in test_idx]
            test_comp2_smiles = [comp2_smiles[i] for i in test_idx]
            test_comp3_smiles = [comp3_smiles[i] for i in test_idx]
            test_labels = labels[test_idx]
            test_temperatures = temperatures[test_idx] if args.use_temperature else None
        elif args.test_data_path:
            # Load the input data.
            test_il_smiles, test_comp2_smiles, test_comp3_smiles, test_labels, test_temperatures = load_csv_data(args.test_data_path)
            # Run the training step.
            split_idx = int(len(indices) * 0.8)
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]
        else:
            # Run the training step.
            test_split_idx = int(len(indices) * 0.15)
            test_idx = indices[:test_split_idx]
            train_val_idx = indices[test_split_idx:]
            # Run the training step.
            val_split_idx = int(len(train_val_idx) * 0.235)  # Baseline workflow step.
            train_idx = train_val_idx[val_split_idx:]
            val_idx = train_val_idx[:val_split_idx]
            
            # Evaluate the test subset.
            test_il_smiles = [il_smiles[i] for i in test_idx]
            test_comp2_smiles = [comp2_smiles[i] for i in test_idx]
            test_comp3_smiles = [comp3_smiles[i] for i in test_idx]
            test_labels = labels[test_idx]
            test_temperatures = temperatures[test_idx] if args.use_temperature else None
            
            print(f" use numpy split complete : training set {len(train_idx)} bar ({len(train_idx)/len(labels)*100:.1f}%), "
                  f" validation set {len(val_idx)} bar ({len(val_idx)/len(labels)*100:.1f}%), "
                  f" test set {len(test_idx)} bar ({len(test_idx)/len(labels)*100:.1f}%)", flush=True)
    
    train_size = len(train_idx)
    val_size = len(val_idx)
    test_size = len(test_labels) if test_labels is not None else 0
    
    # Configure the runtime device.
    device_info = get_device_info(args.device)
    
    # Baseline workflow step.
    print_config_info(args, train_size, val_size, test_size, device_info)
    
    # Process the experiment data.
    train_temps = temperatures[train_idx] if args.use_temperature else None
    val_temps = temperatures[val_idx] if args.use_temperature else None
    
    train_dataset = LLEDataset(
        [il_smiles[i] for i in train_idx],
        [comp2_smiles[i] for i in train_idx],
        [comp3_smiles[i] for i in train_idx],
        labels[train_idx],
        train_temps
    )
    val_dataset = LLEDataset(
        [il_smiles[i] for i in val_idx],
        [comp2_smiles[i] for i in val_idx],
        [comp3_smiles[i] for i in val_idx],
        labels[val_idx],
        val_temps
    )
    
    if test_labels is not None:
        test_temps = test_temperatures if args.use_temperature else None
        test_dataset = LLEDataset(
            test_il_smiles,
            test_comp2_smiles,
            test_comp3_smiles,
            test_labels,
            test_temps
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, collate_fn=collate_fn)
    else:
        test_loader = None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, collate_fn=collate_fn)
    
    # Baseline workflow step.
    # Baseline workflow step.
    node_dim = 33
    # Baseline workflow step.
    edge_dim = 9
    
    # Configure the baseline model.
    model = CIGIN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_mp_layers=args.num_mp_layers,
        use_set2set=args.use_set2set,
        use_temperature=args.use_temperature
    ).to(args.device)
    
    # Compute the training loss.
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                patience=10, min_lr=1e-5)
    
    # Save the generated artifacts.
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the generated artifacts.
    model_dir = args.output_dir  # Save the generated artifacts.
    os.makedirs(model_dir, exist_ok=True)
    
    # Handle model checkpoints.
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    start_epoch = 0
    history = []
    best_val_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(model_dir, 'cigin_best.pt')  # Save the generated artifacts.
    
    if latest_checkpoint:
        restored_state = load_checkpoint(latest_checkpoint, model, optimizer, args.device)
        completed_epoch = restored_state['start_epoch']  # Baseline workflow step.
        start_epoch = completed_epoch  # Run the training step.
        history = restored_state['history']
        best_val_mse = restored_state['best_val_mse']
        best_epoch = restored_state['best_epoch']
        patience_counter = restored_state.get('patience_counter', 0)
        print(f" completed most after M epoch: {completed_epoch}")
        print(f" will load from epoch {start_epoch + 1} continue training to epoch {args.epochs}")
        print()
    else:
        print(" checkpoint file not found , will load from epoch 1 start training ")
        print()
    start_time = time.time()
    
    # Baseline workflow step.
    rest_interval = 2 * 3600  # Baseline workflow step.
    rest_duration = 5 * 60    # Baseline workflow step.
    last_rest_time = time.time()  # Baseline workflow step.
    
    # Run the training step.
    print("\n" + "="*100, flush=True)
    if start_epoch > 0:
        print(f" resume training ( from epoch {start_epoch + 1} Continue )...", flush=True)
    else:
        print(" start training ...", flush=True)
    print(f" rest policy : per run {rest_interval/3600:.1f} hours before cooldown {rest_duration/60:.1f} minutes ", flush=True)
    print("="*100 + "\n", flush=True)
    sys.stdout.flush()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Run the training step.
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        train_loss_val, train_metrics, train_pred, train_true = evaluate(
            model, train_loader, criterion, args.device
        )
        
        # Evaluate the validation subset.
        val_loss_val, val_metrics, val_pred, val_true = evaluate(
            model, val_loader, criterion, args.device
        )
        
        scheduler.step(val_loss_val)
        
        epoch_time = time.time() - epoch_start_time
        
        # Baseline workflow step.
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'epoch_time': epoch_time,
            'train_mse': train_metrics['mse'],
            'train_rmse': train_metrics['rmse'],
            'train_rmse_std': train_metrics['rmse_std'],
            'train_mae': train_metrics['mae'],
            'train_mae_std': train_metrics['mae_std'],
            'train_r2': train_metrics['r2'],
            'train_mae_E': train_metrics['mae_e'],
            'train_mae_E_std': train_metrics['mae_e_std'],
            'train_rmse_E': train_metrics['rmse_e'],
            'train_rmse_E_std': train_metrics['rmse_e_std'],
            'train_r2_E': train_metrics['r2_e'],
            'train_mae_R': train_metrics['mae_r'],
            'train_mae_R_std': train_metrics['mae_r_std'],
            'train_rmse_R': train_metrics['rmse_r'],
            'train_rmse_R_std': train_metrics['rmse_r_std'],
            'train_r2_R': train_metrics['r2_r'],
            'val_mse': val_metrics['mse'],
            'val_rmse': val_metrics['rmse'],
            'val_rmse_std': val_metrics['rmse_std'],
            'val_mae': val_metrics['mae'],
            'val_mae_std': val_metrics['mae_std'],
            'val_r2': val_metrics['r2'],
            'val_mae_E': val_metrics['mae_e'],
            'val_mae_E_std': val_metrics['mae_e_std'],
            'val_rmse_E': val_metrics['rmse_e'],
            'val_rmse_E_std': val_metrics['rmse_e_std'],
            'val_r2_E': val_metrics['r2_e'],
            'val_mae_R': val_metrics['mae_r'],
            'val_mae_R_std': val_metrics['mae_r_std'],
            'val_rmse_R': val_metrics['rmse_r'],
            'val_rmse_R_std': val_metrics['rmse_r_std'],
            'val_r2_R': val_metrics['r2_r']
        })
        
        # Baseline workflow step.
        print("=" * 100)
        best_loss_str = f"{best_val_mse:.6f} (epoch {best_epoch})" if best_val_mse < float('inf') else f"{val_metrics['mse']:.6f} (initial)"
        print(f"Epoch {epoch+1}/{args.epochs} | training time : {epoch_time:.2f} seconds | Train Loss: {train_loss:.6f}")
        print(f"Best Loss: {best_loss_str}")
        print("=" * 100)
        print("[ Training metrics ]")
        print(f" [ mean metrics ] MAE={train_metrics['mae']:.6f}±{train_metrics['mae_std']:.4f} RMSE={train_metrics['rmse']:.6f}±{train_metrics['rmse_std']:.4f} R²={train_metrics['r2']:.6f}")
        print(f" [E phase metrics ] MAE={train_metrics['mae_e']:.6f}±{train_metrics['mae_e_std']:.4f} RMSE={train_metrics['rmse_e']:.6f}±{train_metrics['rmse_e_std']:.4f} R²={train_metrics['r2_e']:.6f}")
        print(f" [R phase metrics ] MAE={train_metrics['mae_r']:.6f}±{train_metrics['mae_r_std']:.4f} RMSE={train_metrics['rmse_r']:.6f}±{train_metrics['rmse_r_std']:.4f} R²={train_metrics['r2_r']:.6f}")
        print()
        print("[ validation metrics ]")
        print(f" [ mean metrics ] MAE={val_metrics['mae']:.6f}±{val_metrics['mae_std']:.4f} RMSE={val_metrics['rmse']:.6f}±{val_metrics['rmse_std']:.4f} R²={val_metrics['r2']:.6f}")
        print(f" [E phase metrics ] MAE={val_metrics['mae_e']:.6f}±{val_metrics['mae_e_std']:.4f} RMSE={val_metrics['rmse_e']:.6f}±{val_metrics['rmse_e_std']:.4f} R²={val_metrics['r2_e']:.6f}")
        print(f" [R phase metrics ] MAE={val_metrics['mae_r']:.6f}±{val_metrics['mae_r_std']:.4f} RMSE={val_metrics['rmse_r']:.6f}±{val_metrics['rmse_r_std']:.4f} R²={val_metrics['r2_r']:.6f}")
        print("=" * 100)
        print()
        
        # Configure the baseline model.
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_epoch = epoch + 1
            patience_counter = 0
            # Save the generated artifacts.
            best_model_path = os.path.join(model_dir, 'cigin_best.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_mse': best_val_mse,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, best_model_path)
        else:
            patience_counter += 1
        
        # Save the generated artifacts.
        if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_mse': val_metrics['mse'],
                'best_val_mse': best_val_mse,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'history': history,
                'args': vars(args)
            }, checkpoint_path)
            print(f" checkpoint saved : {checkpoint_path}")
            print()
        
        # Baseline workflow step.
        current_time = time.time()
        elapsed_since_last_rest = current_time - last_rest_time
        
        if elapsed_since_last_rest >= rest_interval and epoch < args.epochs - 1:
            # Baseline workflow step.
            elapsed_hours = elapsed_since_last_rest / 3600
            print("=" * 100)
            print(f" Already run {elapsed_hours:.2f} hours ({elapsed_since_last_rest:.0f} seconds ), current epoch completed ")
            print(f" Break {rest_duration/60:.1f} minutes ({rest_duration} seconds ) allow CPU/GPU to allow a cooldown period ...")
            print("=" * 100)
            print()
            sys.stdout.flush()
            time.sleep(rest_duration)
            last_rest_time = time.time()  # Baseline workflow step.
            print(" Break end , continue training ...")
            print()
            sys.stdout.flush()
        
        # Apply early stopping.
        if patience_counter >= args.patience:
            print("=" * 100)
            print(f" early stopping triggered ! at epoch {epoch+1} stop training .")
            print(f" best model at epoch {best_epoch}, validation set MSE: {best_val_mse:.6f}")
            print(f" waited {patience_counter}/{args.patience} epoch without improvement ")
            print("=" * 100)
            break
    
    # Run the training step.
    total_time = time.time() - start_time
    avg_time_per_epoch = total_time / len(history)
    
    print("=" * 100)
    print(" training complete !")
    print(f" best model at epoch {best_epoch}, validation set MSE: {best_val_mse:.6f}")
    print(f" total training time : {total_time:.2f} seconds ({total_time/60:.2f} minutes )")
    print(f" mean time per epoch : {avg_time_per_epoch:.2f} seconds ")
    print()
    
    # Load the input data.
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(" warning : not found best model file , use current model Into rows Assessment ")
        print()
    
    # Evaluate the test subset.
    if test_loader is not None:
        _, test_metrics, test_pred, test_true = evaluate(
            model, test_loader, criterion, args.device
        )
        print("[ test metrics ]")
        print(f" [ mean metrics ] MAE={test_metrics['mae']:.6f}±{test_metrics['mae_std']:.4f} RMSE={test_metrics['rmse']:.6f}±{test_metrics['rmse_std']:.4f} R²={test_metrics['r2']:.6f}")
        print(f" [E phase metrics ] MAE={test_metrics['mae_e']:.6f}±{test_metrics['mae_e_std']:.4f} RMSE={test_metrics['rmse_e']:.6f}±{test_metrics['rmse_e_std']:.4f} R²={test_metrics['r2_e']:.6f}")
        print(f" [R phase metrics ] MAE={test_metrics['mae_r']:.6f}±{test_metrics['mae_r_std']:.4f} RMSE={test_metrics['rmse_r']:.6f}±{test_metrics['rmse_r_std']:.4f} R²={test_metrics['r2_r']:.6f}")
        print()
    else:
        test_metrics = None
        test_pred = None
        test_true = None
    
    # Save the generated artifacts.
    save_results(args, history, train_pred, train_true, val_pred, val_true, 
                 test_pred, test_true, best_epoch, best_val_mse, 
                 total_time, avg_time_per_epoch, val_metrics, test_metrics)
    
    print(" result file saved :")
    print(f" - training history CSV: {os.path.join(args.results_dir, 'train_history.csv')}")
    print(f" - Training / validation results CSV: {os.path.join(args.results_dir, 'training_results.csv')}")
    if test_loader is not None:
        print(f" - test-set results CSV: {os.path.join(args.results_dir, 'test_results.csv')}")
    print(f" - best-model metrics TXT: {os.path.join(args.results_dir, 'best_metrics.txt')}")
    print(f" - training metrics TXT: {os.path.join(args.results_dir, 'training_metrics.txt')}")
    print(f" - model checkpoint : {best_model_path}")
    print(f" - checkpoint file clip : {checkpoint_dir}")
    print("=" * 100)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Trained By User in Broken ")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n Training Process in Appearance error : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

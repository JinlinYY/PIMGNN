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
        print(f" warning : Discovery {np.sum(~mask)} non-finite values , will be ignored ")
        y_true = y_true[mask.all(axis=1)]
        y_pred = y_pred[mask.all(axis=1)]
    
    # Process the experiment data.
    if len(y_true) == 0:
        print(" error : valid data is Empty ")
        return {
            'all': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
            'e_phase': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
            'r_phase': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
        }
    
    # Generate model predictions.
    pred_std = np.std(y_pred, axis=0)
    if np.any(pred_std < 0.01):
        print(f"[ warning ] prediction Variance too small , Possible at satiety and Question . prediction standard deviation : {pred_std}")
    
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
            print(f"\n[ Commissioning ] number M batch prediction range : [{pred_np.min():.6f}, {pred_np.max():.6f}]")
            print(f"[ Commissioning ] number M batch target range : [{targets_np.min():.6f}, {targets_np.max():.6f}]")
            print(f"[ Commissioning ] number M batch prediction mean : {pred_np.mean(axis=0)}")
            print(f"[ Commissioning ] number M batch target mean : {targets_np.mean(axis=0)}")
        
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
            print(f"\n[ warning ] prediction range is unusual :")
            print(f" prediction : [{pred_min:.6f}, {pred_max:.6f}], mean : {pred_mean}")
            print(f" target : [{target_min:.6f}, {target_max:.6f}], mean : {target_mean}")
            print(f" recommendation : ensure enabled output approximately beam ")
        
        # Generate model predictions.
        if np.any(pred_std < 0.01):
            print(f"\n[ warning ] prediction May be full and ( standard deviation Too small ):")
            print(f" prediction standard deviation : {pred_std}")
            print(f" target standard deviation : {target_std}")
            print(f" prediction mean : {pred_mean}")
            print(f" target mean : {target_mean}")
            print(f" recommendation : check model Initialization or Decrease learning rate ")
    
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
            print(f"\n[ warning ] prediction range is unusual :")
            print(f" prediction : [{pred_min:.6f}, {pred_max:.6f}], mean : {pred_mean:.6f}")
            print(f" target : [{target_min:.6f}, {target_max:.6f}], mean : {target_mean:.6f}")
            print(f" recommendation : ensure enabled output approximately beam ")
    
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
    print(f"{prefix}[E phase ] MAE: {metrics['e_phase']['mae_mean']:.6f}±{metrics['e_phase']['mae_std']:.6f}, "
          f"RMSE: {metrics['e_phase']['rmse_mean']:.6f}±{metrics['e_phase']['rmse_std']:.6f}, "
          f"R²: {metrics['e_phase']['r2']:.6f}")
    print(f"{prefix}[R phase ] MAE: {metrics['r_phase']['mae_mean']:.6f}±{metrics['r_phase']['mae_std']:.6f}, "
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
        f.write(" best-model metrics \n")
        f.write("=" * 80 + "\n\n")
        
        f.write("[ training information ]\n")
        f.write(f" best epoch: {best_epoch}\n")
        f.write(f" best validation RMSE: {best_val_rmse:.6f}\n")
        f.write(f" total training epochs : {total_epochs}\n")
        f.write(f" total training time : {total_time:.2f} seconds ({total_time/60:.2f} minutes )\n")
        f.write(f" mean time per epoch : {avg_time_per_epoch:.2f} seconds \n\n")
        
        f.write("[ validation metrics ]\n")
        f.write(f"  【Overall】 MAE: {val_metrics['all']['mae_mean']:.6f}±{val_metrics['all']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['all']['rmse_mean']:.6f}±{val_metrics['all']['rmse_std']:.6f}, "
                f"R²: {val_metrics['all']['r2']:.6f}\n")
        f.write(f" [E phase ] MAE: {val_metrics['e_phase']['mae_mean']:.6f}±{val_metrics['e_phase']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['e_phase']['rmse_mean']:.6f}±{val_metrics['e_phase']['rmse_std']:.6f}, "
                f"R²: {val_metrics['e_phase']['r2']:.6f}\n")
        f.write(f" [R phase ] MAE: {val_metrics['r_phase']['mae_mean']:.6f}±{val_metrics['r_phase']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['r_phase']['rmse_mean']:.6f}±{val_metrics['r_phase']['rmse_std']:.6f}, "
                f"R²: {val_metrics['r_phase']['r2']:.6f}\n\n")
        
        if test_metrics is not None:
            f.write("[ test metrics ]\n")
            f.write(f"  【Overall】 MAE: {test_metrics['all']['mae_mean']:.6f}±{test_metrics['all']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['all']['rmse_mean']:.6f}±{test_metrics['all']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['all']['r2']:.6f}\n")
            f.write(f" [E phase ] MAE: {test_metrics['e_phase']['mae_mean']:.6f}±{test_metrics['e_phase']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['e_phase']['rmse_mean']:.6f}±{test_metrics['e_phase']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['e_phase']['r2']:.6f}\n")
            f.write(f" [R phase ] MAE: {test_metrics['r_phase']['mae_mean']:.6f}±{test_metrics['r_phase']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['r_phase']['rmse_mean']:.6f}±{test_metrics['r_phase']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['r_phase']['r2']:.6f}\n")
    
    # Save the generated artifacts.
    with open(os.path.join(results_dir, 'training_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" training-metric summary \n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f" best model at epoch {best_epoch}\n")
        f.write(f" best validation RMSE: {best_val_rmse:.6f}\n\n")
        
        f.write("[ best validation metrics ]\n")
        f.write(f"  【Overall】 MAE: {val_metrics['all']['mae_mean']:.6f}±{val_metrics['all']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['all']['rmse_mean']:.6f}±{val_metrics['all']['rmse_std']:.6f}, "
                f"R²: {val_metrics['all']['r2']:.6f}\n")
        f.write(f" [E phase ] MAE: {val_metrics['e_phase']['mae_mean']:.6f}±{val_metrics['e_phase']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['e_phase']['rmse_mean']:.6f}±{val_metrics['e_phase']['rmse_std']:.6f}, "
                f"R²: {val_metrics['e_phase']['r2']:.6f}\n")
        f.write(f" [R phase ] MAE: {val_metrics['r_phase']['mae_mean']:.6f}±{val_metrics['r_phase']['mae_std']:.6f}, "
                f"RMSE: {val_metrics['r_phase']['rmse_mean']:.6f}±{val_metrics['r_phase']['rmse_std']:.6f}, "
                f"R²: {val_metrics['r_phase']['r2']:.6f}\n\n")
        
        if test_metrics is not None:
            f.write("[ test metrics ]\n")
            f.write(f"  【Overall】 MAE: {test_metrics['all']['mae_mean']:.6f}±{test_metrics['all']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['all']['rmse_mean']:.6f}±{test_metrics['all']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['all']['r2']:.6f}\n")
            f.write(f" [E phase ] MAE: {test_metrics['e_phase']['mae_mean']:.6f}±{test_metrics['e_phase']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['e_phase']['rmse_mean']:.6f}±{test_metrics['e_phase']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['e_phase']['r2']:.6f}\n")
            f.write(f" [R phase ] MAE: {test_metrics['r_phase']['mae_mean']:.6f}±{test_metrics['r_phase']['mae_std']:.6f}, "
                    f"RMSE: {test_metrics['r_phase']['rmse_mean']:.6f}±{test_metrics['r_phase']['rmse_std']:.6f}, "
                    f"R²: {test_metrics['r_phase']['r2']:.6f}\n\n")
        
        f.write(f" total training time : {total_time:.2f} seconds ({total_time/60:.2f} minutes )\n")
        f.write(f" mean time per epoch : {avg_time_per_epoch:.2f} seconds \n")


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
    print("CGIB model training configuration ")
    print("=" * 80)
    print("\n[ dataset information ]")
    
    # Load the input data.
    df = pd.read_csv(args.data_path)
    
    # Process the experiment data.
    if 'IL (Component 1) full name SMILES' in df.columns:
        # Baseline workflow step.
        print(" detected total.csv format , True at convert ...")
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
            raise ValueError(" data file must contain Ex1-Ex3, Rx1-Rx3 column or target column ")
    else:
        raise ValueError(" unsupported data-file format . must contain smiles1/smiles2 column or IL/Component column ")
    
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
    
    print(f" total samples : {total_size}")
    print(f" number of training samples : {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    print(f" number of validation samples : {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    print(f" number of test samples : {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")
    
    print("\n[ Device configuration ]")
    print(f" device type : {args.device}")
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f" GPU name : {torch.cuda.get_device_name(0)}")
        print(f" CUDA version : {torch.version.cuda}")
        print(f" GPU memory Size : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n[ training parameters ]")
    print(f" random seed : {args.seed}")
    print(f" training epochs (epochs): {args.epochs}")
    print(f" batch size (batch_size): {args.batch_size}")
    print(f" learning rate : {args.lr}")
    print(f" weight decay : {args.weight_decay}")
    print(f" early-stopping patience : {args.patience}")
    print(f" minimum early-stopping improvement : {args.min_delta}")
    print(f" checkpoint frequency : per {args.checkpoint_freq} epoch")
    print(f" rest policy : per {args.rest_interval/3600:.1f} hours before cooldown {args.rest_duration/60:.1f} minutes ")
    
    print("\n[ model hyperparameters ]")
    print(f" hidden-layer dimension (hidden_dim): {args.hidden_dim}")
    print(f" number of graph-neural-network layers (num_layers): {args.num_layers}")
    print(f" Set2Set number of steps : {args.set2set_steps}")
    print(f" Dropout rate : 0.0")
    print(f" output dimension : 6 (LLE task : Ex1, Ex2, Ex3, Rx1, Rx2, Rx3)")
    print(f" use Set2Set: yes ")
    print(f" use for Better than learning : {' yes ' if args.use_contrastive else ' no '}")
    
    print("\n[ path information ]")
    print(f" output directory : {output_dir}")
    print(f" result directory : {output_dir}/results")
    
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
    print(f" output approximately beam : {' enabled ' if constrain_output else ' Disabled '}")
    
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
    print(" model checkpoint Initialized ( output Floor : Small weights Initialization , Other Layers :Xavier Initialization ,gain=0.1)")
    
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
        print(f"\n resume training from a checkpoint : epoch {start_epoch}, best RMSE: {best_val_rmse:.6f}")
    
    # Run the training step.
    training_start_time = time.time()
    last_rest_time = time.time()
    
    print("\n" + "=" * 80)
    print(" start training ")
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
        print(f"\nEpoch {epoch+1}/{args.epochs} | training time : {epoch_time:.2f} seconds | Train Loss: {train_results['loss']:.6f}")
        print(f"Best RMSE: {best_val_rmse:.6f} (epoch {best_epoch+1})")
        
        print("\n[ Training metrics ]")
        print_metrics(train_results['metrics'])
        
        print("\n[ validation metrics ]")
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
            print(f"\n checkpoint saved : {checkpoint_path}")
        
        # Apply early stopping.
        if patience_counter >= args.patience:
            print(f"\n early stopping triggered ! at epoch {epoch+1} stop training .")
            print(f" best model at epoch {best_epoch+1}, validation set RMSE: {best_val_rmse:.6f}")
            print(f" waited {patience_counter}/{args.patience} epoch without improvement ")
            break
        
        # Baseline workflow step.
        current_time = time.time()
        elapsed_since_rest = current_time - last_rest_time
        if elapsed_since_rest >= args.rest_interval:
            elapsed_hours = elapsed_since_rest / 3600
            rest_minutes = args.rest_duration / 60
            print(f"\n Already run {elapsed_hours:.2f} hours ({elapsed_since_rest:.0f} seconds ), current epoch completed ")
            print(f" Break {rest_minutes:.1f} minutes ({args.rest_duration:.0f} seconds ) allow CPU/GPU to allow a cooldown period ...")
            time.sleep(args.rest_duration)
            last_rest_time = time.time()
    
    # Run the training step.
    total_time = time.time() - training_start_time
    avg_time_per_epoch = total_time / (epoch + 1 - start_epoch) if epoch + 1 > start_epoch else 0
    
    print("\n" + "=" * 80)
    print(" training complete !")
    print("=" * 80)
    print(f" best model at epoch {best_epoch+1}, validation set RMSE: {best_val_rmse:.6f}")
    print(f" total training time : {total_time:.2f} seconds ({total_time/60:.2f} minutes )")
    print(f" mean time per epoch : {avg_time_per_epoch:.2f} seconds ")
    
    # Load the input data.
    best_model_path = os.path.join(output_dir, f'seed_{args.seed}_best.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Baseline workflow step.
    print("\n[ Final validation metrics ]")
    final_val_results = evaluate(model, val_loader, args.device)
    print_metrics(final_val_results['metrics'])
    
    # Evaluate the test subset.
    print("\n" + "=" * 80)
    print(" starting test-set evaluation ")
    print("=" * 80)
    print(f" number of test samples : {len(test_dataset)}")
    
    test_results = evaluate(model, test_loader, args.device)
    
    print("\n[ test metrics ]")
    print_metrics(test_results['metrics'])
    
    # Save the generated artifacts.
    save_results(test_results['predictions'], test_results['targets'], 'test', 
                 os.path.join(output_dir, 'results'))
    
    # Save the generated artifacts.
    print("\n save results file ...")
    
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
    
    print(" all file saved complete !")


def main():
    parser = argparse.ArgumentParser(description='CGIB Training - supports Single seed or Batch Training all seed')
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
    parser.add_argument('--seed', type=int, default=None, choices=[42, 123, 456, 789, 2024, None], help=' random seed ( if The designation only trains the seed)')
    parser.add_argument('--all_seeds', action='store_true', help=' by Sequential Training all 5 seed(42, 123, 456, 789, 2024)')
    parser.add_argument('--hidden_dim', type=int, default=256, help=' hidden-layer dimension ')
    parser.add_argument('--num_layers', type=int, default=3, help='GNN number of layers ')
    parser.add_argument('--set2set_steps', type=int, default=3, help='Set2Set number of steps ')
    parser.add_argument('--beta', type=float, default=1e-3, help=' information Bottleneck Balance parameters ')
    parser.add_argument('--lr', type=float, default=5e-4, help=' learning rate ( default : 5e-4, Reduced to prevent prediction satiety and )')
    parser.add_argument('--weight_decay', type=float, default=0.0, help=' weight decay ')
    parser.add_argument('--batch_size', type=int, default=64, help=' batch size ')
    parser.add_argument('--epochs', type=int, default=400, help=' training epochs ')
    parser.add_argument('--patience', type=int, default=80, help=' early-stopping patience ')
    parser.add_argument('--min_delta', type=float, default=0.0, help=' minimum early-stopping improvement ')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help=' checkpoint frequency ( per N epoch)')
    parser.add_argument('--rest_interval', type=float, default=7200, help=' rest interval ( seconds ), default 2 hours ')
    parser.add_argument('--rest_duration', type=float, default=300, help=' rest duration ( seconds ), default 5 minutes ')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help=' device ')
    parser.add_argument('--gnn_type', type=str, default='mpnn', choices=['mpnn', 'gin'], help='GNN type ')
    parser.add_argument('--use_contrastive', action='store_true', help=' use for Better than learning ')
    parser.add_argument('--no_constrain_output', dest='constrain_output', action='store_false', default=True, help=' Disabled output approximately beam ( default enabled output approximately beam )')
    parser.add_argument('--resume', type=str, default=None, help=' resume training from a checkpoint ')
    
    args = parser.parse_args()
    
    # Baseline workflow step.
    if not hasattr(args, 'constrain_output') or args.constrain_output is None:
        args.constrain_output = True
    
    # Run the training step.
    if args.all_seeds:
        # Run the training step.
        seeds = [42, 123, 456, 789, 2024]
        print("=" * 80)
        print(" start run all seed Training ")
        print("=" * 80)
        print(f" data path : {args.data_path}")
        print(f"Seeds: {seeds}")
        print("=" * 80 + "\n")
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n{'='*80}")
            print(f" Training Seed {seed} ({i}/{len(seeds)})")
            print(f"{'='*80}\n")
            
            # Baseline workflow step.
            seed_args = argparse.Namespace(**vars(args))
            seed_args.seed = seed
            
            try:
                train_single_seed(seed_args)
                print(f"\nSeed {seed} training complete !\n")
            except Exception as e:
                print(f"\nSeed {seed} training failed , error : {str(e)}\n")
                print(" whether Continue Next seed?(y/n): ", end='')
                try:
                    response = input().strip().lower()
                    if response != 'y':
                        print(" terminate training ")
                        return
                except:
                    print(" terminate training ")
                    return
        
        print("\n" + "=" * 80)
        print(" all seed Training completed !")
        print("=" * 80)
    else:
        # Run the training step.
        if args.seed is None:
            args.seed = 42  # Baseline workflow step.
        train_single_seed(args)


if __name__ == '__main__':
    main()

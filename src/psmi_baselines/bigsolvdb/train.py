# -*- coding: utf-8 -*-
"""Implement the bigsolvdb train baseline module."""
import os
import time
import glob
import argparse
import re
from typing import Dict, List, Tuple
import warnings

# Baseline workflow step.
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Baseline workflow step.
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from psmi_baselines.common.utils import set_seed, Scaler
from .data_loader import load_bigsolvdb_data
from .model import build_solubility_model
from psmi_baselines.paths import BIGSOLVDB_CSV, BIGSOLVDB_EXPERIMENT_ROOT

warnings.filterwarnings('ignore')


class SolubilityDataset(Dataset):
    """Represent the SolubilityDataset baseline component."""
    def __init__(self, df: pd.DataFrame, T_scaler: Scaler, target_scaler: Scaler = None):
        self.df = df.reset_index(drop=True)
        self.T_scaler = T_scaler
        self.target_scaler = target_scaler
        
        # Baseline workflow step.
        self.features = []
        self.targets = []
        
        for idx, row in self.df.iterrows():
            # Baseline workflow step.
            solute_fp = row['solute_fp']
            solvent_fp = row['solvent_fp']
            T_norm = self.T_scaler.transform(np.array([row['T']]))[0]
            
            feature = np.concatenate([solute_fp, solvent_fp, [T_norm]])
            self.features.append(feature.astype(np.float32))
            
            target = row['target']
            if self.target_scaler:
                target = self.target_scaler.transform(np.array([target]))[0]
            self.targets.append(target)
        
        self.features = np.array(self.features)
        self.targets = np.array(self.targets, dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.targets[idx]]).squeeze()


class EarlyStopping:
    """Represent the EarlyStopping baseline component."""
    def __init__(self, patience: int = 50, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Run the compute metrics baseline operation."""
    y_true = y_true.astype(np.float64).reshape(-1)
    y_pred = y_pred.astype(np.float64).reshape(-1)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Run the evaluate baseline operation."""
    model.eval()
    all_preds = []
    all_targets = []
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_metrics(all_targets, all_preds)
    return metrics, all_preds, all_targets


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Run the train epoch baseline operation."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / max(n_samples, 1)


def train_single_seed(
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str,
    model_name: str = "mlp",
    device: str = "cuda",
    batch_size: int = 1024,
    epochs: int = 300,
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    hidden: int = 512,
    dropout: float = 0.15,
    fp_bits: int = 2048,
    patience: int = 50,
    resume_from: str = None,
    save_checkpoint_every: int = 10,
) -> Dict:
    """Run the train single seed baseline operation."""
    
    seed_out_dir = os.path.join(out_dir, f"seed_{seed}")
    os.makedirs(seed_out_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f" Training Seeds : {seed}")
    print(f" output directory : {seed_out_dir}")
    print(f"{'='*80}\n")
    
    # Set the random seed.
    set_seed(seed)
    
    # Baseline workflow step.
    T_scaler = Scaler.fit(train_df["T"].values.astype(np.float32))
    
    # Baseline workflow step.
    target_scaler = None
    
    # Process the experiment data.
    train_dataset = SolubilityDataset(train_df, T_scaler, target_scaler)
    val_dataset = SolubilityDataset(val_df, T_scaler, target_scaler)
    test_dataset = SolubilityDataset(test_df, T_scaler, target_scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Configure the baseline model.
    in_dim = 2 * fp_bits + 1  # Baseline workflow step.
    model = build_solubility_model(
        model_name=model_name,
        in_dim=in_dim,
        fp_bits=fp_bits,
        hidden=hidden,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Run the training step.
    history = {
        'epoch': [],
        'train_loss': [],
        'train_mae': [],
        'train_rmse': [],
        'train_r2': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': [],
        'epoch_time': [],
    }
    
    best_val_mae = float('inf')
    best_model_state = None
    best_epoch = -1
    start_epoch = 1
    
    early_stopping = EarlyStopping(patience=patience)
    
    # Baseline workflow step.
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        print(f" resume from checkpoint : {resume_from}")
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'history' in checkpoint:
            history = checkpoint['history']
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_mae' in checkpoint:
            best_val_mae = checkpoint['best_val_mae']
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']
        if 'best_model_state' in checkpoint:
            best_model_state = checkpoint['best_model_state']
        if 'T_scaler' in checkpoint:
            T_scaler = Scaler.from_state_dict(checkpoint['T_scaler'])
    
    # Run the training step.
    print(f"\n start training ( from epoch {start_epoch}/{epochs})...")
    print(f"{'='*80}")
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        
        # Run the training step.
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Baseline workflow step.
        train_metrics, _, _ = evaluate(model, train_loader, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Baseline workflow step.
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_metrics['mae'])
        history['train_rmse'].append(train_metrics['rmse'])
        history['train_r2'].append(train_metrics['r2'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_r2'].append(val_metrics['r2'])
        history['epoch_time'].append(epoch_time)
        
        # Save the generated artifacts.
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        
        # Baseline workflow step.
        print(f"\nEpoch {epoch}/{epochs} | Time : {epoch_time:.2f} seconds ")
        print(f" training set : MAE={train_metrics['mae']:.6f} RMSE={train_metrics['rmse']:.6f} R²={train_metrics['r2']:.6f}")
        print(f" validation set : MAE={val_metrics['mae']:.6f} RMSE={val_metrics['rmse']:.6f} R²={val_metrics['r2']:.6f}")
        print(f" best validation MAE: {best_val_mae:.6f} (epoch {best_epoch})")
        
        # Save the generated artifacts.
        if epoch % save_checkpoint_every == 0 or epoch == epochs:
            checkpoint_path = os.path.join(seed_out_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'T_scaler': T_scaler.state_dict(),
                'history': history,
                'best_val_mae': best_val_mae,
                'best_epoch': best_epoch,
                'best_model_state': best_model_state,
            }, checkpoint_path)
            print(f" checkpoint saved : {checkpoint_path}")
        
        # Apply early stopping.
        if early_stopping(val_metrics['mae']):
            print(f"\n early stopping triggered ! at epoch {epoch} stop training .")
            print(f" best model at epoch {best_epoch}, validation set MAE: {best_val_mae:.6f}")
            break
    
    # Load the input data.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device)
    
    # Save the generated artifacts.
    pd.DataFrame(history).to_csv(
        os.path.join(seed_out_dir, "train_history.csv"),
        index=False,
        encoding='utf-8-sig'
    )
    
    # Save the generated artifacts.
    test_results = test_df.copy()
    test_results['pred'] = test_preds
    test_results['target'] = test_targets
    test_results.to_csv(
        os.path.join(seed_out_dir, "test_results.csv"),
        index=False,
        encoding='utf-8-sig'
    )
    
    # Save the generated artifacts.
    best_metrics = {
        'best_epoch': best_epoch,
        'best_val_mae': float(best_val_mae),
        'test_mae': float(test_metrics['mae']),
        'test_rmse': float(test_metrics['rmse']),
        'test_r2': float(test_metrics['r2']),
    }
    
    # Save the generated artifacts.
    metrics_txt_path = os.path.join(seed_out_dir, "best_metrics.txt")
    with open(metrics_txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f" Training results metrics (Seed: {seed})\n")
        f.write("="*80 + "\n\n")
        f.write(f" best epoch: {best_epoch}\n")
        f.write(f" best validation MAE: {best_val_mae:.6f}\n\n")
        f.write(" test metrics :\n")
        f.write("-"*80 + "\n")
        f.write(f"MAE:  {test_metrics['mae']:.6f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"R²:   {test_metrics['r2']:.6f}\n")
        f.write("="*80 + "\n")
    
    print(f"\n training complete !")
    print(f" best epoch: {best_epoch}, best validation MAE: {best_val_mae:.6f}")
    print(f" test set : MAE={test_metrics['mae']:.6f} RMSE={test_metrics['rmse']:.6f} R²={test_metrics['r2']:.6f}")
    
    return best_metrics


def collect_all_results(out_dir: str, seeds: List[int]) -> pd.DataFrame:
    """Run the collect all results baseline operation."""
    all_results = []
    
    for seed in seeds:
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        txt_file = os.path.join(seed_dir, "best_metrics.txt")
        history_file = os.path.join(seed_dir, "train_history.csv")
        
        metrics = {'seed': seed}
        
        # Compute evaluation metrics.
        if os.path.exists(txt_file):
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Baseline workflow step.
                epoch_match = re.search(r' best epoch:\s+(\d+)', content)
                if epoch_match:
                    metrics['best_epoch'] = int(epoch_match.group(1))
                
                # Evaluate the validation subset.
                val_mae_match = re.search(r' best validation MAE:\s+([\d.]+)', content)
                if val_mae_match:
                    metrics['best_val_mae'] = float(val_mae_match.group(1))
                
                # Evaluate the test subset.
                mae_match = re.search(r'MAE:\s+([\d.]+)', content)
                rmse_match = re.search(r'RMSE:\s+([\d.]+)', content)
                r2_match = re.search(r'R²:\s+([\d.]+)', content)
                
                if mae_match:
                    metrics['test_mae'] = float(mae_match.group(1))
                if rmse_match:
                    metrics['test_rmse'] = float(rmse_match.group(1))
                if r2_match:
                    metrics['test_r2'] = float(r2_match.group(1))
        
        # Read the input data.
        elif os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            if not history_df.empty:
                best_idx = history_df['val_mae'].idxmin()
                best_row = history_df.iloc[best_idx]
                metrics['best_epoch'] = int(best_row['epoch'])
                metrics['best_val_mae'] = float(best_row['val_mae'])
                # Evaluate the test subset.
        
        if len(metrics) > 1:  # Baseline workflow step.
            all_results.append(metrics)
    
    if not all_results:
        return pd.DataFrame(), {}
    
    df = pd.DataFrame(all_results)
    
    # Compute evaluation metrics.
    stats = {}
    for metric in ['test_mae', 'test_rmse', 'test_r2']:
        if metric in df.columns and len(df) > 0:
            values = df[metric].dropna()
            if len(values) > 0:
                mean_val = float(values.mean())
                # Baseline workflow step.
                if len(values) > 1:
                    std_val = float(values.std(ddof=1))
                else:
                    std_val = 0.0
                stats[f'{metric}_mean'] = round(mean_val, 4)
                stats[f'{metric}_std'] = round(std_val, 4)
                stats[f'{metric}_format'] = f"{mean_val:.4f}±{std_val:.4f}"
    
    # Save the generated artifacts.
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(
        os.path.join(out_dir, "results_summary.csv"),
        index=False,
        encoding='utf-8-sig'
    )
    
    # Save the generated artifacts.
    summary_txt_path = os.path.join(out_dir, "results_summary.txt")
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" training statistics across all random seeds ( mean ± standard deviation )\n")
        f.write("="*80 + "\n\n")
        f.write(f" random seed : {seeds}\n")
        f.write(f" number of seeds : {len(seeds)}\n\n")
        f.write(" test metrics statistics :\n")
        f.write("-"*80 + "\n")
        if 'test_mae_format' in stats:
            f.write(f"MAE:  {stats['test_mae_format']}\n")
            f.write(f" ( mean : {stats['test_mae_mean']:.4f}, standard deviation : {stats['test_mae_std']:.4f})\n")
        if 'test_rmse_format' in stats:
            f.write(f"RMSE: {stats['test_rmse_format']}\n")
            f.write(f" ( mean : {stats['test_rmse_mean']:.4f}, standard deviation : {stats['test_rmse_std']:.4f})\n")
        if 'test_r2_format' in stats:
            f.write(f"R²:   {stats['test_r2_format']}\n")
            f.write(f" ( mean : {stats['test_r2_mean']:.4f}, standard deviation : {stats['test_r2_std']:.4f})\n")
        f.write("\n" + "="*80 + "\n")
        f.write(" per-seed results :\n")
        f.write("-"*80 + "\n")
        for _, row in df.iterrows():
            f.write(f"\nSeed {row['seed']}:\n")
            f.write(f" best epoch: {row['best_epoch']}\n")
            f.write(f" best validation MAE: {row['best_val_mae']:.6f}\n")
            f.write(f" test set MAE: {row['test_mae']:.6f}\n")
            f.write(f" test set RMSE: {row['test_rmse']:.6f}\n")
            f.write(f" test set R²: {row['test_r2']:.6f}\n")
        f.write("="*80 + "\n")
    
    # Save the generated artifacts.
    df.to_csv(
        os.path.join(out_dir, "all_seeds_results.csv"),
        index=False,
        encoding='utf-8-sig'
    )
    
    return stats_df, stats


def main():
    parser = argparse.ArgumentParser(description='BigSolvDB Training Script ')
    parser.add_argument('--data_path', type=str, default=str(BIGSOLVDB_CSV),
                       help='BigSolvDB dataset path.')
    parser.add_argument('--output_dir', type=str,
                       default=str(BIGSOLVDB_EXPERIMENT_ROOT / 'runs'),
                       help='Directory for BigSolvDB run artifacts.')
    parser.add_argument('--model_name', type=str, default='mlp', 
                       choices=['mlp', 'ann', 'lstm', 'transformer', 'tabknet'],
                       help=' model name ')
    parser.add_argument('--seeds', type=int, nargs='+', 
                       default=[42, 123, 456, 789, 2024],
                       help=' random-seed list ')
    parser.add_argument('--batch_size', type=int, default=1024, help=' batch size ')
    parser.add_argument('--epochs', type=int, default=300, help=' training epochs ')
    parser.add_argument('--lr', type=float, default=2e-4, help=' learning rate ')
    parser.add_argument('--patience', type=int, default=50, help=' early-stopping patience ')
    parser.add_argument('--resume', type=str, default=None, help=' resume-checkpoint path ')
    parser.add_argument('--auto_resume', action='store_true', help=' automatically locate the latest checkpoint ')
    
    args = parser.parse_args()
    
    # Configure the output artifacts.
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*80)
    print("BigSolvDB dataset Training ")
    print("="*80)
    print(f" dataset : {args.data_path}")
    print(f" model : {args.model_name}")
    print(f" random seed : {args.seeds}")
    print(f" output directory : {out_dir}")
    print("="*80)
    
    # Set the random seed.
    print("\n load dataset ...")
    train_df, val_df, test_df = load_bigsolvdb_data(
        csv_path=args.data_path,
        target_col="LogS(mol/L)",
        random_state=args.seeds[0]  # Set the random seed.
    )
    
    # Set the random seed.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n use device : {device}")
    print(f"\n total training runs {len(args.seeds)} random seed , will run sequentially ")
    print("="*80)
    
    for idx, seed in enumerate(args.seeds, 1):
        seed_out_dir = os.path.join(out_dir, f"seed_{seed}")
        
        print(f"\n{'='*80}")
        print(f"[ number {idx}/{len(args.seeds)} seeds ] start training seed={seed}")
        print(f"{'='*80}")
        
        # Handle model checkpoints.
        resume_from = args.resume
        if args.auto_resume and not resume_from:
            checkpoint_pattern = os.path.join(seed_out_dir, "checkpoint_epoch_*.pt")
            checkpoint_files = glob.glob(checkpoint_pattern)
            if checkpoint_files:
                def extract_epoch(fpath):
                    try:
                        return int(os.path.basename(fpath).replace("checkpoint_epoch_", "").replace(".pt", ""))
                    except:
                        return -1
                checkpoint_files.sort(key=extract_epoch, reverse=True)
                resume_from = checkpoint_files[0]
                print(f" automatically selected the latest checkpoint : {resume_from}")
        
        # Set the random seed.
        train_single_seed(
            seed=seed,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            out_dir=out_dir,
            model_name=args.model_name,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            resume_from=resume_from,
        )
        
        print(f"\n{'='*80}")
        print(f"[ number {idx}/{len(args.seeds)} seeds ]seed={seed} training complete ")
        print(f"{'='*80}")
        
        # Set the random seed.
        if idx < len(args.seeds):
            remaining = len(args.seeds) - idx
            print(f"\n remaining {remaining} seeds To be trained , starting the next run ...")
            print("-"*80)
    
    # Baseline workflow step.
    print("\n" + "="*80)
    print(" collect results for all seeds ...")
    print("="*80)
    stats_df, stats = collect_all_results(out_dir, args.seeds)
    
    if not stats_df.empty and stats:
        print("\n" + "="*80)
        print(" results statistics ( mean ± standard deviation )")
        print("="*80)
        print("\n test metrics :")
        print("-"*80)
        if 'test_mae_format' in stats:
            print(f"MAE:  {stats['test_mae_format']}")
            print(f" ( mean : {stats['test_mae_mean']:.4f}, standard deviation : {stats['test_mae_std']:.4f})")
        if 'test_rmse_format' in stats:
            print(f"RMSE: {stats['test_rmse_format']}")
            print(f" ( mean : {stats['test_rmse_mean']:.4f}, standard deviation : {stats['test_rmse_std']:.4f})")
        if 'test_r2_format' in stats:
            print(f"R²:   {stats['test_r2_format']}")
            print(f" ( mean : {stats['test_r2_mean']:.4f}, standard deviation : {stats['test_r2_std']:.4f})")
        print("-"*80)
        print(f"\n results saved to :")
        print(f" - CSV format : {os.path.join(out_dir, 'results_summary.csv')}")
        print(f" - TXT format : {os.path.join(out_dir, 'results_summary.txt')}")
    
    print("\n training complete !")


if __name__ == "__main__":
    main()


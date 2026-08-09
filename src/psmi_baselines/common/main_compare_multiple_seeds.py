# -*- coding: utf-8 -*-
"""
Multiple seeds runner for baseline comparison.

Run:
  python main_compare_multiple_seeds.py

It will:
  1) Run main_compare.py with 5 different random seeds
  2) Collect metrics from each run
  3) Compute mean and std for each model
  4) Save aggregated results with 4 decimal places
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import shutil
import warnings
import argparse

# Suppress matplotlib Tkinter warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 5 random seeds (may be overridden by CLI)
SEEDS = [42, 123, 456, 789, 2024]

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]
QUICK_SEEDS = [42, 123]


def parse_args():
    parser = argparse.ArgumentParser(description="Multiple-seed baseline comparison")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only run seeds 42 and 123",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list, e.g. 42,123,456",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Versioned output directory for summaries and per-seed runs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to run; defaults to the configured comparison set",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def run_with_seed(
    seed: int,
    run_number: int,
    total_runs: int,
    base_out_dir: str,
) -> str:
    """Run main_compare with a specific seed and return the results directory"""
    from . import config as C

    print(f"\n{'='*90}")
    print(f"RUN {run_number}/{total_runs}: SEED={seed}")
    print(f"{'='*90}\n")
    
    # Isolate artifacts from each random seed.
    temp_out_dir = os.path.join(base_out_dir, f"seed_{seed}")
    
    os.makedirs(temp_out_dir, exist_ok=True)
    
    # Reuse a complete metric package when requested by the caller.
    try:
        metrics_table = os.path.join(temp_out_dir, "baseline_compare_metrics.csv")
        metric_files = [
            os.path.join(temp_out_dir, f"baseline_{name.lower()}", "best_metrics.json")
            for name in C.MODELS_TO_RUN
        ]
        completed_metrics = []
        for path in metric_files:
            try:
                with open(path, "r", encoding="utf-8") as stream:
                    payload = json.load(stream)
                completed_metrics.append(isinstance(payload.get("test_metrics"), dict))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                completed_metrics.append(False)
        if os.path.isfile(metrics_table) and all(completed_metrics):
            print(f"Found existing results in {temp_out_dir}, skipping re-run.")
            return temp_out_dir
    except Exception:
        pass
    
    try:
        # Import main_compare
        from .main_compare import main as main_compare
        from .utils import set_seed
        # Store original values
        original_seed = C.SEED
        original_out_dir = C.OUT_DIR
        
        # Set new values
        C.SEED = seed
        C.OUT_DIR = temp_out_dir
        
        # Set seed in utils
        set_seed(seed)
        
        # Run the comparison
        main_compare()
        
        # Restore original config
        C.SEED = original_seed
        C.OUT_DIR = original_out_dir
        
    except Exception as e:
        print(f"ERROR in seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise, continue with next seed
    
    return temp_out_dir


def extract_metrics_from_best_metrics_txt(filepath: str) -> dict:
    """Extract metrics from best_metrics.txt file"""
    metrics = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        current_section = None
        for line in content.split('\n'):
            line = line.strip()
            # Support both sklearn and torch formats
            if line in ('val_metrics:', 'best_val_metrics:'):
                current_section = 'val'
                continue
            elif line in ('test_metrics:', 'best_test_metrics:'):
                current_section = 'test'
                continue
            # Ignore other headers like best_epoch / best_val_mse
            if line.startswith('best_epoch:') or line.startswith('best_val_mse:'):
                continue
            elif ':' in line and current_section:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    try:
                        value = float(parts[1].strip())
                        full_key = f"{current_section}_{key}"
                        metrics[full_key] = value
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    
    return metrics


def extract_metrics_from_best_metrics_json(filepath: str) -> dict:
    """Extract metrics from best_metrics.json file"""
    metrics = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # sklearn format
        if 'val_metrics' in data and isinstance(data['val_metrics'], dict):
            for k, v in data['val_metrics'].items():
                metrics[f'val_{k}'] = v
        # torch format
        if 'best_val_metrics' in data and isinstance(data['best_val_metrics'], dict):
            for k, v in data['best_val_metrics'].items():
                metrics[f'val_{k}'] = v
        
        # sklearn format
        if 'test_metrics' in data and isinstance(data['test_metrics'], dict):
            for k, v in data['test_metrics'].items():
                metrics[f'test_{k}'] = v
        # torch format
        if 'best_test_metrics' in data and isinstance(data['best_test_metrics'], dict):
            for k, v in data['best_test_metrics'].items():
                metrics[f'test_{k}'] = v
    except FileNotFoundError:
        pass
    
    return metrics


def collect_all_metrics(run_dirs: dict[int, str]) -> dict:
    """
    Collect all metrics from multiple runs.
    
    Returns:
        {model_name: {seed1: metrics_dict, seed2: metrics_dict, ...}}
    """
    all_metrics = {}
    
    for seed, run_dir in run_dirs.items():
        if not os.path.exists(run_dir):
            print(f"WARNING: Run directory not found: {run_dir}")
            continue
        
        # List all baseline_* directories
        try:
            items = os.listdir(run_dir)
        except Exception as e:
            print(f"ERROR listing {run_dir}: {e}")
            continue
        
        for item in items:
            if item.startswith('baseline_'):
                model_name = item.replace('baseline_', '')
                metrics_json = os.path.join(run_dir, item, 'best_metrics.json')
                metrics_txt = os.path.join(run_dir, item, 'best_metrics.txt')
                
                metrics = {}
                if os.path.exists(metrics_json):
                    metrics = extract_metrics_from_best_metrics_json(metrics_json)
                elif os.path.exists(metrics_txt):
                    metrics = extract_metrics_from_best_metrics_txt(metrics_txt)
                
                if metrics:
                    if model_name not in all_metrics:
                        all_metrics[model_name] = {}
                    all_metrics[model_name][seed] = metrics
    
    return all_metrics


def compute_statistics(all_metrics: dict) -> pd.DataFrame:
    """
    Compute mean and std for each model and metric.
    
    Returns a DataFrame with columns:
    - model
    - metric (e.g., test_mae_E, test_rmse_E, test_r2_E, test_mae_R, ...)
    - mean (4 decimal places)
    - std (4 decimal places)
    """
    
    results = []
    
    for model_name in sorted(all_metrics.keys()):
        seed_metrics = all_metrics[model_name]
        
        if not seed_metrics:
            continue
        
        # Get all metric keys from first seed
        first_seed_data = next(iter(seed_metrics.values()))
        
        for metric_key in sorted(first_seed_data.keys()):
            values = []
            for seed in SEEDS:
                if seed in seed_metrics and metric_key in seed_metrics[seed]:
                    values.append(seed_metrics[seed][metric_key])
            
            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                
                results.append({
                    'model': model_name,
                    'metric': metric_key,
                    'mean': round(mean_val, 4),
                    'std': round(std_val, 4)
                })
    
    df = pd.DataFrame(results)
    return df


def format_summary_table(all_metrics: dict) -> pd.DataFrame:
    
    summary_rows = []
    
    for model_name in sorted(all_metrics.keys()):
        seed_metrics = all_metrics[model_name]
        
        if not seed_metrics:
            continue
        
        # Extract test metrics for E, R, and overall
        metrics_of_interest = [
            'test_mae_E', 'test_rmse_E', 'test_r2_E',
            'test_mae_R', 'test_rmse_R', 'test_r2_R',
            'test_mae', 'test_mse', 'test_rmse', 'test_r2'
        ]
        
        for metric_key in metrics_of_interest:
            values = []
            for seed in SEEDS:
                if seed in seed_metrics and metric_key in seed_metrics[seed]:
                    values.append(seed_metrics[seed][metric_key])
            
            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                
                # Format label
                if 'rmse' in metric_key:
                    metric_label = metric_key.replace('test_rmse', 'RMSE')
                elif 'mae' in metric_key:
                    metric_label = metric_key.replace('test_mae', 'MAE')
                elif 'mse' in metric_key:
                    metric_label = metric_key.replace('test_mse', 'MSE')
                elif 'r2' in metric_key:
                    metric_label = metric_key.replace('test_r2', 'R2')
                else:
                    metric_label = metric_key
                
                summary_rows.append({
                    'Model': model_name,
                    'Metric': metric_label,
                    'Mean': round(mean_val, 4),
                    'Std': round(std_val, 4)
                })
    
    df_summary = pd.DataFrame(summary_rows)
    return df_summary


def main():
    from . import config as C
    global SEEDS
    args = parse_args()
    if args.seeds:
        SEEDS = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    elif args.quick:
        SEEDS = list(QUICK_SEEDS)
        print('QUICK TEST mode enabled')
    if args.out_dir is not None:
        C.OUT_DIR = str(args.out_dir.resolve())
    if args.models is not None:
        C.MODELS_TO_RUN = list(args.models)
    if args.epochs is not None:
        C.EPOCHS = int(args.epochs)
    if args.device is not None:
        C.DEVICE = str(args.device)
    print("Starting Multiple Seeds Baseline Comparison...")
    print(f"Seeds: {SEEDS}")
    print(f"Base Output Dir: {C.OUT_DIR}")
    
    # Create main output directory
    base_out_dir = C.OUT_DIR
    os.makedirs(base_out_dir, exist_ok=True)
    protocol = {
        "protocol_version": "sample_major",
        "seeds": SEEDS,
        "models": list(C.MODELS_TO_RUN),
        "epochs": int(C.EPOCHS),
        "device": str(C.DEVICE),
        "data_path": str(Path(C.EXCEL_PATH).resolve()),
        "data_sha256": hashlib.sha256(Path(C.EXCEL_PATH).read_bytes()).hexdigest(),
        "split_manifest_path": str(Path(C.SPLIT_MANIFEST_PATH).resolve()),
        "split_manifest_sha256": hashlib.sha256(
            Path(C.SPLIT_MANIFEST_PATH).read_bytes()
        ).hexdigest(),
        "selection_policy": "validation_only; test evaluated after model selection",
        "augmentation": "component-2/3 swap on training partition only",
    }
    with open(os.path.join(base_out_dir, "protocol.json"), "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
    
    # Run with each seed
    run_dirs = {}
    failed_seeds = []
    
    for i, seed in enumerate(SEEDS, 1):
        try:
            run_dir = run_with_seed(seed, i, len(SEEDS), base_out_dir)
            if os.path.exists(run_dir):
                run_dirs[seed] = run_dir
            else:
                print(f"WARNING: Run directory {run_dir} does not exist")
                failed_seeds.append(seed)
        except Exception as e:
            print(f"Failed to run with seed {seed}: {e}")
            failed_seeds.append(seed)
    
    if not run_dirs:
        print("\nERROR: No runs completed successfully!")
        sys.exit(1)
    
    if failed_seeds:
        print(f"\nWARNING: {len(failed_seeds)} seeds failed: {failed_seeds}")
    
    # Collect metrics from all runs
    print("\n" + "="*90)
    print("Collecting metrics from all runs...")
    print("="*90)
    
    all_metrics = collect_all_metrics(run_dirs)
    
    if not all_metrics:
        print("\nERROR: No metrics collected from runs!")
        sys.exit(1)
    
    # Compute statistics
    print("\nComputing statistics...")
    df_stats = compute_statistics(all_metrics)
    
    # Create summary table
    df_summary = format_summary_table(all_metrics)
    
    # Save results
    stats_csv = os.path.join(base_out_dir, "multiple_seeds_metrics_detail.csv")
    df_stats.to_csv(stats_csv, index=False, encoding='utf-8-sig')
    print(f"\nDetailed metrics saved to: {stats_csv}")
    
    summary_csv = os.path.join(base_out_dir, "multiple_seeds_summary.csv")
    df_summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"Summary table saved to: {summary_csv}")
    
    # Print summary
    print("\n" + "="*90)
    print(f"SUMMARY TABLE ({len(run_dirs)} Runs - Average +/- Std)")
    print("="*90)
    
    # Group by model and print nicely
    for model_name in sorted(all_metrics.keys()):
        print(f"\n{model_name.upper()}")
        print("-" * 70)
        
        # Filter for this model
        model_data = df_summary[df_summary['Model'] == model_name]
        
        # Group by phase (E, R, Overall)
        e_data = model_data[model_data['Metric'].str.contains('_E')]
        if not e_data.empty:
            print("\nExtract phase (E):")
            for _, row in e_data.iterrows():
                print(f"  {row['Metric']:<15}: {row['Mean']:>8.4f} +/- {row['Std']:>8.4f}")
        
        r_data = model_data[model_data['Metric'].str.contains('_R')]
        if not r_data.empty:
            print("\nRaffinate phase (R):")
            for _, row in r_data.iterrows():
                print(f"  {row['Metric']:<15}: {row['Mean']:>8.4f} +/- {row['Std']:>8.4f}")
        
        overall_data = model_data[~model_data['Metric'].str.contains('_[ER]$', regex=True)]
        if not overall_data.empty:
            print("\nOverall:")
            for _, row in overall_data.iterrows():
                print(f"  {row['Metric']:<15}: {row['Mean']:>8.4f} +/- {row['Std']:>8.4f}")
    
    print("\n" + "="*90)
    print("DONE!")
    print("="*90)
    print(f"\nResults saved to:")
    print(f"  - Summary: {summary_csv}")
    print(f"  - Details: {stats_csv}")
    print(f"\nSuccessfully completed {len(run_dirs)}/{len(SEEDS)} runs")


if __name__ == "__main__":
    main()

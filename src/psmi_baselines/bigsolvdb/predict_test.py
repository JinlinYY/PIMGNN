# -*- coding: utf-8 -*-
"""Implement the bigsolvdb predict_test baseline module."""
import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Baseline workflow step.
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from psmi_baselines.common.utils import set_seed, Scaler
from .data_loader import load_bigsolvdb_data
from .model import build_solubility_model
from .train import SolubilityDataset, compute_metrics, evaluate
from psmi_baselines.paths import BIGSOLVDB_CSV, BIGSOLVDB_EXPERIMENT_ROOT

# Baseline workflow step.
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except:
    pass


def load_best_model(seed_dir: str, model_name: str, device: str, fp_bits: int = 2048, 
                    hidden: int = 512, dropout: float = 0.15):
    """Run the load best model baseline operation."""
    # Baseline workflow step.
    import glob
    checkpoint_pattern = os.path.join(seed_dir, "checkpoint_epoch_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        raise FileNotFoundError(f"未找到检查点文件: {seed_dir}")
    
    # Baseline workflow step.
    def extract_epoch(fpath):
        try:
            return int(os.path.basename(fpath).replace("checkpoint_epoch_", "").replace(".pt", ""))
        except:
            return -1
    checkpoint_files.sort(key=extract_epoch, reverse=True)
    checkpoint_path = checkpoint_files[0]
    
    print(f"  加载检查点: {os.path.basename(checkpoint_path)}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Configure the baseline model.
    in_dim = 2 * fp_bits + 1
    
    # Configure the baseline model.
    model = build_solubility_model(
        model_name=model_name,
        in_dim=in_dim,
        fp_bits=fp_bits,
        hidden=hidden,
        dropout=dropout
    ).to(device)
    
    # Load the input data.
    if 'best_model_state' in checkpoint and checkpoint['best_model_state'] is not None:
        model.load_state_dict(checkpoint['best_model_state'])
        best_epoch = checkpoint.get('best_epoch', -1)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        best_epoch = checkpoint.get('epoch', -1)
    else:
        raise ValueError("检查点中未找到模型权重")
    
    # Load the input data.
    if 'T_scaler' in checkpoint:
        T_scaler = Scaler.from_state_dict(checkpoint['T_scaler'])
    else:
        raise ValueError("检查点中未找到T_scaler")
    
    return model, T_scaler, best_epoch


def predict_test_set(seed: int, results_dir: str, data_path: str, model_name: str = 'mlp',
                     device: str = 'cuda', batch_size: int = 1024, fp_bits: int = 2048,
                     hidden: int = 512, dropout: float = 0.15, all_results_stats: dict = None):
    """Run the predict test set baseline operation."""
    
    seed_dir = os.path.join(results_dir, f"seed_{seed}")
    
    if not os.path.exists(seed_dir):
        print(f"警告: 种子 {seed} 的目录不存在: {seed_dir}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Seed {seed}: 加载最佳模型并预测测试集")
    print(f"{'='*80}")
    
    # Set the random seed.
    print("  加载数据集...")
    train_df, val_df, test_df = load_bigsolvdb_data(
        csv_path=data_path,
        target_col="LogS(mol/L)",
        random_state=42  # Set the random seed.
    )
    
    # Load the input data.
    print("  加载最佳模型...")
    model, T_scaler, best_epoch = load_best_model(
        seed_dir=seed_dir,
        model_name=model_name,
        device=device,
        fp_bits=fp_bits,
        hidden=hidden,
        dropout=dropout
    )
    
    # Evaluate the test subset.
    target_scaler = None
    test_dataset = SolubilityDataset(test_df, T_scaler, target_scaler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Generate model predictions.
    print("  对测试集进行预测...")
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device)
    
    print(f"  测试集指标:")
    print(f"    MAE:  {test_metrics['mae']:.6f}")
    print(f"    RMSE: {test_metrics['rmse']:.6f}")
    print(f"    R²:   {test_metrics['r2']:.6f}")
    
    # Set the random seed.
    summary_txt_path = os.path.join(seed_dir, "results_summary.txt")
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"测试集预测结果 (Seed: {seed})\n")
        f.write("="*80 + "\n\n")
        f.write(f"最佳epoch: {best_epoch}\n")
        f.write(f"使用最佳权重进行测试集预测\n\n")
        f.write("测试集指标:\n")
        f.write("-"*80 + "\n")
        f.write(f"MAE:  {test_metrics['mae']:.6f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"R²:   {test_metrics['r2']:.6f}\n")
        
        # Set the random seed.
        if all_results_stats:
            f.write("\n" + "="*80 + "\n")
            f.write("所有种子的测试集预测结果统计（均值±标准差）\n")
            f.write("="*80 + "\n\n")
            f.write("测试集指标统计:\n")
            f.write("-"*80 + "\n")
            if 'test_mae_format' in all_results_stats:
                f.write(f"MAE:  {all_results_stats['test_mae_format']}\n")
                f.write(f"      (均值: {all_results_stats['test_mae_mean']:.4f}, 标准差: {all_results_stats['test_mae_std']:.4f})\n")
            if 'test_rmse_format' in all_results_stats:
                f.write(f"RMSE: {all_results_stats['test_rmse_format']}\n")
                f.write(f"      (均值: {all_results_stats['test_rmse_mean']:.4f}, 标准差: {all_results_stats['test_rmse_std']:.4f})\n")
            if 'test_r2_format' in all_results_stats:
                f.write(f"R²:   {all_results_stats['test_r2_format']}\n")
                f.write(f"      (均值: {all_results_stats['test_r2_mean']:.4f}, 标准差: {all_results_stats['test_r2_std']:.4f})\n")
        
        f.write("="*80 + "\n")
    
    print(f"  结果已保存到: {summary_txt_path}")
    
    return {
        'seed': seed,
        'best_epoch': best_epoch,
        'test_mae': float(test_metrics['mae']),
        'test_rmse': float(test_metrics['rmse']),
        'test_r2': float(test_metrics['r2']),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用最佳权重对测试集进行预测')
    parser.add_argument('--data_path', type=str, default=str(BIGSOLVDB_CSV),
                       help='BigSolvDB dataset path.')
    parser.add_argument('--results_dir', type=str,
                       default=str(BIGSOLVDB_EXPERIMENT_ROOT / 'runs'),
                       help='Directory containing trained seed runs.')
    parser.add_argument('--model_name', type=str, default='mlp',
                       choices=['mlp', 'ann', 'lstm', 'transformer', 'tabknet'],
                       help='模型名称')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 2024],
                       help='随机种子列表')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*80)
    print("使用最佳权重对测试集进行预测")
    print("="*80)
    print(f"数据集: {args.data_path}")
    print(f"结果目录: {args.results_dir}")
    print(f"模型: {args.model_name}")
    print(f"随机种子: {args.seeds}")
    print(f"设备: {device}")
    print("="*80)
    
    all_results = []
    
    # Set the random seed.
    for seed in args.seeds:
        try:
            result = predict_test_set(
                seed=seed,
                results_dir=args.results_dir,
                data_path=args.data_path,
                model_name=args.model_name,
                device=device,
                batch_size=args.batch_size
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n错误: Seed {seed} 预测失败: {e}")
            import traceback
            traceback.print_exc()
    
    # Baseline workflow step.
    stats = {}
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Baseline workflow step.
        for metric in ['test_mae', 'test_rmse', 'test_r2']:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    mean_val = float(values.mean())
                    if len(values) > 1:
                        std_val = float(values.std(ddof=1))
                    else:
                        std_val = 0.0
                    stats[f'{metric}_mean'] = round(mean_val, 4)
                    stats[f'{metric}_std'] = round(std_val, 4)
                    stats[f'{metric}_format'] = f"{mean_val:.4f}±{std_val:.4f}"
        
        # Set the random seed.
        print("\n更新各种子文件夹下的results_summary.txt，添加统计信息...")
        for seed in args.seeds:
            seed_dir = os.path.join(args.results_dir, f"seed_{seed}")
            summary_txt_path = os.path.join(seed_dir, "results_summary.txt")
            
            if os.path.exists(summary_txt_path):
                # Read the input data.
                with open(summary_txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Baseline workflow step.
                if "所有种子的测试集预测结果统计" not in content:
                    with open(summary_txt_path, 'a', encoding='utf-8') as f:
                        f.write("\n" + "="*80 + "\n")
                        f.write("所有种子的测试集预测结果统计（均值±标准差）\n")
                        f.write("="*80 + "\n\n")
                        f.write("测试集指标统计:\n")
                        f.write("-"*80 + "\n")
                        if 'test_mae_format' in stats:
                            f.write(f"MAE:  {stats['test_mae_format']}\n")
                            f.write(f"      (均值: {stats['test_mae_mean']:.4f}, 标准差: {stats['test_mae_std']:.4f})\n")
                        if 'test_rmse_format' in stats:
                            f.write(f"RMSE: {stats['test_rmse_format']}\n")
                            f.write(f"      (均值: {stats['test_rmse_mean']:.4f}, 标准差: {stats['test_rmse_std']:.4f})\n")
                        if 'test_r2_format' in stats:
                            f.write(f"R²:   {stats['test_r2_format']}\n")
                            f.write(f"      (均值: {stats['test_r2_mean']:.4f}, 标准差: {stats['test_r2_std']:.4f})\n")
                        f.write("="*80 + "\n")
    
    if all_results:
        print("\n" + "="*80)
        print("所有种子的测试集预测结果统计（均值±标准差）")
        print("="*80)
        
        df = pd.DataFrame(all_results)
        
        # Baseline workflow step.
        stats = {}
        for metric in ['test_mae', 'test_rmse', 'test_r2']:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    mean_val = float(values.mean())
                    if len(values) > 1:
                        std_val = float(values.std(ddof=1))
                    else:
                        std_val = 0.0
                    stats[f'{metric}_mean'] = round(mean_val, 4)
                    stats[f'{metric}_std'] = round(std_val, 4)
                    stats[f'{metric}_format'] = f"{mean_val:.4f}±{std_val:.4f}"
        
        print("\n测试集指标:")
        print("-"*80)
        if 'test_mae_format' in stats:
            print(f"MAE:  {stats['test_mae_format']}")
            print(f"      (均值: {stats['test_mae_mean']:.4f}, 标准差: {stats['test_mae_std']:.4f})")
        if 'test_rmse_format' in stats:
            print(f"RMSE: {stats['test_rmse_format']}")
            print(f"      (均值: {stats['test_rmse_mean']:.4f}, 标准差: {stats['test_rmse_std']:.4f})")
        if 'test_r2_format' in stats:
            print(f"R²:   {stats['test_r2_format']}")
            print(f"      (均值: {stats['test_r2_mean']:.4f}, 标准差: {stats['test_r2_std']:.4f})")
        print("-"*80)
        
        print("\n各种子详细结果:")
        print("-"*80)
        for _, row in df.iterrows():
            print(f"\nSeed {row['seed']}:")
            print(f"  最佳epoch: {row['best_epoch']}")
            print(f"  测试集MAE:  {row['test_mae']:.6f}")
            print(f"  测试集RMSE: {row['test_rmse']:.6f}")
            print(f"  测试集R²:   {row['test_r2']:.6f}")
        
        print("\n" + "="*80)
        print("预测完成！")
        print("="*80)
    else:
        print("\n错误: 没有成功完成任何预测")


if __name__ == "__main__":
    main()

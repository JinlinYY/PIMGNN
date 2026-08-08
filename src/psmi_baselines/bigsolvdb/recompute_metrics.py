# -*- coding: utf-8 -*-
"""Implement the bigsolvdb recompute_metrics baseline module."""
import os
import numpy as np
import pandas as pd
from typing import Dict, List

from psmi_baselines.paths import BIGSOLVDB_EXPERIMENT_ROOT


def compute_metrics_from_csv(csv_path: str) -> Dict[str, float]:
    """Run the compute metrics from csv baseline operation."""
    df = pd.read_csv(csv_path)
    
    if 'target' not in df.columns or 'pred' not in df.columns:
        raise ValueError(f"CSV文件必须包含 'target' 和 'pred' 列。当前列: {list(df.columns)}")
    
    y_true = df['target'].values.astype(np.float64)
    y_pred = df['pred'].values.astype(np.float64)
    
    # Baseline workflow step.
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        raise ValueError("没有有效的预测值和真实值")
    
    n_samples = len(y_true)
    
    # Baseline workflow step.
    abs_errors = np.abs(y_true - y_pred)
    # Baseline workflow step.
    mae = float(np.mean(abs_errors))
    mae_std = float(np.std(abs_errors, ddof=1) / np.sqrt(n_samples)) if n_samples > 1 else 0.0
    
    # Baseline workflow step.
    squared_errors = (y_true - y_pred) ** 2
    # Baseline workflow step.
    rmse = float(np.sqrt(np.mean(squared_errors)))
    # Baseline workflow step.
    # Baseline workflow step.
    mean_squared_error = float(np.mean(squared_errors))
    if mean_squared_error > 1e-12 and n_samples > 1:
        rmse_std = float(np.std(squared_errors, ddof=1) / (2 * np.sqrt(mean_squared_error) * np.sqrt(n_samples)))
    else:
        rmse_std = 0.0
    
    # Baseline workflow step.
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)
    
    # Baseline workflow step.
    # Baseline workflow step.
    residuals = y_true - y_pred
    r2_std = float(np.std(residuals, ddof=1) / np.sqrt(n_samples)) if n_samples > 1 else 0.0
    
    return {
        'mae': mae,
        'mae_std': mae_std,
        'rmse': rmse,
        'rmse_std': rmse_std,
        'r2': r2,
        'r2_std': r2_std
    }


def recompute_all_metrics(results_dir: str, seeds: List[int] = [42, 123, 456, 789, 2024]) -> pd.DataFrame:
    """Run the recompute all metrics baseline operation."""
    all_results = []
    
    print("="*80)
    print("从CSV文件重新计算测试集指标")
    print("="*80)
    print(f"结果目录: {results_dir}")
    print(f"随机种子: {seeds}\n")
    
    for seed in seeds:
        seed_dir = os.path.join(results_dir, f"seed_{seed}")
        csv_path = os.path.join(seed_dir, "test_results.csv")
        
        if not os.path.exists(csv_path):
            print(f"警告: Seed {seed} 的test_results.csv不存在: {csv_path}")
            continue
        
        try:
            print(f"处理 Seed {seed}...")
            metrics = compute_metrics_from_csv(csv_path)
            
            result = {
                'seed': seed,
                'test_mae': metrics['mae'],
                'test_mae_std': metrics['mae_std'],
                'test_rmse': metrics['rmse'],
                'test_rmse_std': metrics['rmse_std'],
                'test_r2': metrics['r2'],
                'test_r2_std': metrics['r2_std']
            }
            all_results.append(result)
            
            # Baseline workflow step.
            print(f"  MAE:  {metrics['mae']:.6f} ± {metrics['mae_std']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.6f} ± {metrics['rmse_std']:.4f}")
            print(f"  R²:   {metrics['r2']:.6f} ± {metrics['r2_std']:.4f}")
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\n错误: 没有成功处理任何结果")
        return pd.DataFrame(), {}
    
    df = pd.DataFrame(all_results)
    
    # Set the random seed.
    print("\n" + "="*80)
    print("每个种子的测试集预测结果")
    print("="*80)
    for _, row in df.iterrows():
        print(f"\nSeed {row['seed']}:")
        print(f"  MAE:  {row['test_mae']:.6f}")
        print(f"  RMSE: {row['test_rmse']:.6f}")
        print(f"  R²:   {row['test_r2']:.6f}")
    
    # Compute evaluation metrics.
    stats = {}
    for metric in ['test_mae', 'test_rmse', 'test_r2']:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                mean_val = float(values.mean())
                if len(values) > 1:
                    # Baseline workflow step.
                    std_val = float(values.std(ddof=1))
                    # Baseline workflow step.
                    # Baseline workflow step.
                    if std_val < 0.001:
                        # Baseline workflow step.
                        stats[f'{metric}_mean'] = round(mean_val, 6)
                        stats[f'{metric}_std'] = round(std_val, 6)
                        stats[f'{metric}_format'] = f"{mean_val:.6f}±{std_val:.6f}"
                    elif std_val < 0.01:
                        # Baseline workflow step.
                        stats[f'{metric}_mean'] = round(mean_val, 4)
                        stats[f'{metric}_std'] = round(std_val, 4)
                        stats[f'{metric}_format'] = f"{mean_val:.4f}±{std_val:.4f}"
                    else:
                        # Baseline workflow step.
                        stats[f'{metric}_mean'] = round(mean_val, 3)
                        stats[f'{metric}_std'] = round(std_val, 3)
                        stats[f'{metric}_format'] = f"{mean_val:.3f}±{std_val:.3f}"
                else:
                    std_val = 0.0
                    stats[f'{metric}_mean'] = round(mean_val, 3)
                    stats[f'{metric}_std'] = round(std_val, 3)
                    stats[f'{metric}_format'] = f"{mean_val:.3f}±{std_val:.3f}"
    
    # Set the random seed.
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("所有种子的测试集预测结果统计（均值±标准差）")
        print("="*80)
        print("\n测试集指标统计:")
        print("-"*80)
        if 'test_mae_format' in stats:
            print(f"MAE:  {stats['test_mae_format']}")
            if stats['test_mae_std'] < 0.001:
                print(f"      (均值: {stats['test_mae_mean']:.6f}, 标准差: {stats['test_mae_std']:.6f})")
            elif stats['test_mae_std'] < 0.01:
                print(f"      (均值: {stats['test_mae_mean']:.4f}, 标准差: {stats['test_mae_std']:.4f})")
            else:
                print(f"      (均值: {stats['test_mae_mean']:.3f}, 标准差: {stats['test_mae_std']:.3f})")
        if 'test_rmse_format' in stats:
            print(f"RMSE: {stats['test_rmse_format']}")
            if stats['test_rmse_std'] < 0.001:
                print(f"      (均值: {stats['test_rmse_mean']:.6f}, 标准差: {stats['test_rmse_std']:.6f})")
            elif stats['test_rmse_std'] < 0.01:
                print(f"      (均值: {stats['test_rmse_mean']:.4f}, 标准差: {stats['test_rmse_std']:.4f})")
            else:
                print(f"      (均值: {stats['test_rmse_mean']:.3f}, 标准差: {stats['test_rmse_std']:.3f})")
        if 'test_r2_format' in stats:
            print(f"R²:   {stats['test_r2_format']}")
            if stats['test_r2_std'] < 0.001:
                print(f"      (均值: {stats['test_r2_mean']:.6f}, 标准差: {stats['test_r2_std']:.6f})")
            elif stats['test_r2_std'] < 0.01:
                print(f"      (均值: {stats['test_r2_mean']:.4f}, 标准差: {stats['test_r2_std']:.4f})")
            else:
                print(f"      (均值: {stats['test_r2_mean']:.3f}, 标准差: {stats['test_r2_std']:.3f})")
        print("-"*80)
    else:
        print(f"\n注意: 只处理了 {len(all_results)} 个种子，无法计算标准差")
    
    # Set the random seed.
    print("\n" + "="*80)
    print("更新各种子文件夹下的results_summary.txt（包含标准差）")
    print("="*80)
    
    # Baseline workflow step.
    if not stats and len(all_results) > 1:
        for metric in ['test_mae', 'test_rmse', 'test_r2']:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    mean_val = float(values.mean())
                    if len(values) > 1:
                        std_val = float(values.std(ddof=1))
                        # Baseline workflow step.
                        if std_val < 0.001:
                            stats[f'{metric}_mean'] = round(mean_val, 6)
                            stats[f'{metric}_std'] = round(std_val, 6)
                            stats[f'{metric}_format'] = f"{mean_val:.6f}±{std_val:.6f}"
                        elif std_val < 0.01:
                            stats[f'{metric}_mean'] = round(mean_val, 4)
                            stats[f'{metric}_std'] = round(std_val, 4)
                            stats[f'{metric}_format'] = f"{mean_val:.4f}±{std_val:.4f}"
                        else:
                            stats[f'{metric}_mean'] = round(mean_val, 3)
                            stats[f'{metric}_std'] = round(std_val, 3)
                            stats[f'{metric}_format'] = f"{mean_val:.3f}±{std_val:.3f}"
                    else:
                        std_val = 0.0
                        stats[f'{metric}_mean'] = round(mean_val, 3)
                        stats[f'{metric}_std'] = round(std_val, 3)
                        stats[f'{metric}_format'] = f"{mean_val:.3f}±{std_val:.3f}"
    
    for seed in seeds:
        seed_dir = os.path.join(results_dir, f"seed_{seed}")
        summary_txt_path = os.path.join(seed_dir, "results_summary.txt")
        csv_path = os.path.join(seed_dir, "test_results.csv")
        
        # Baseline workflow step.
        if not os.path.exists(csv_path):
            print(f"  跳过 Seed {seed}: test_results.csv不存在")
            continue
        
        # Set the random seed.
        try:
            seed_metrics = compute_metrics_from_csv(csv_path)
        except Exception as e:
            print(f"  错误: Seed {seed} 计算指标失败: {e}")
            continue
        
        # Read the input data.
        best_epoch = -1
        best_metrics_txt = os.path.join(seed_dir, "best_metrics.txt")
        if os.path.exists(best_metrics_txt):
            with open(best_metrics_txt, 'r', encoding='utf-8') as f:
                content = f.read()
                import re
                epoch_match = re.search(r'最佳epoch:\s+(\d+)', content)
                if epoch_match:
                    best_epoch = int(epoch_match.group(1))
        
        # Set the random seed.
        with open(summary_txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"测试集预测结果 (Seed: {seed})\n")
            f.write("="*80 + "\n\n")
            if best_epoch > 0:
                f.write(f"最佳epoch: {best_epoch}\n")
            f.write(f"使用最佳权重进行测试集预测\n\n")
            f.write("测试集指标:\n")
            f.write("-"*80 + "\n")
            # Set the random seed.
            f.write(f"MAE:  {seed_metrics['mae']:.6f} ± {seed_metrics['mae_std']:.4f}\n")
            f.write(f"RMSE: {seed_metrics['rmse']:.6f} ± {seed_metrics['rmse_std']:.4f}\n")
            f.write(f"R²:   {seed_metrics['r2']:.6f} ± {seed_metrics['r2_std']:.4f}\n")
            
            # Set the random seed.
            if stats:
                f.write("\n" + "="*80 + "\n")
                f.write("所有种子的测试集预测结果统计（均值±标准差）\n")
                f.write("="*80 + "\n\n")
                f.write("测试集指标统计:\n")
                f.write("-"*80 + "\n")
                if 'test_mae_format' in stats:
                    f.write(f"MAE:  {stats['test_mae_format']}\n")
                    if stats['test_mae_std'] < 0.001:
                        f.write(f"      (均值: {stats['test_mae_mean']:.6f}, 标准差: {stats['test_mae_std']:.6f})\n")
                    elif stats['test_mae_std'] < 0.01:
                        f.write(f"      (均值: {stats['test_mae_mean']:.4f}, 标准差: {stats['test_mae_std']:.4f})\n")
                    else:
                        f.write(f"      (均值: {stats['test_mae_mean']:.3f}, 标准差: {stats['test_mae_std']:.3f})\n")
                if 'test_rmse_format' in stats:
                    f.write(f"RMSE: {stats['test_rmse_format']}\n")
                    if stats['test_rmse_std'] < 0.001:
                        f.write(f"      (均值: {stats['test_rmse_mean']:.6f}, 标准差: {stats['test_rmse_std']:.6f})\n")
                    elif stats['test_rmse_std'] < 0.01:
                        f.write(f"      (均值: {stats['test_rmse_mean']:.4f}, 标准差: {stats['test_rmse_std']:.4f})\n")
                    else:
                        f.write(f"      (均值: {stats['test_rmse_mean']:.3f}, 标准差: {stats['test_rmse_std']:.3f})\n")
                if 'test_r2_format' in stats:
                    f.write(f"R²:   {stats['test_r2_format']}\n")
                    if stats['test_r2_std'] < 0.001:
                        f.write(f"      (均值: {stats['test_r2_mean']:.6f}, 标准差: {stats['test_r2_std']:.6f})\n")
                    elif stats['test_r2_std'] < 0.01:
                        f.write(f"      (均值: {stats['test_r2_mean']:.4f}, 标准差: {stats['test_r2_std']:.4f})\n")
                    else:
                        f.write(f"      (均值: {stats['test_r2_mean']:.3f}, 标准差: {stats['test_r2_std']:.3f})\n")
                f.write("="*80 + "\n")
            else:
                f.write("\n注意: 只有一个种子的结果，无法计算标准差\n")
                f.write("="*80 + "\n")
        
        print(f"\nSeed {seed}: {summary_txt_path}")
        print(f"  该种子测试集指标:")
        # Set the random seed.
        print(f"    MAE:  {seed_metrics['mae']:.6f} ± {seed_metrics['mae_std']:.4f}")
        print(f"    RMSE: {seed_metrics['rmse']:.6f} ± {seed_metrics['rmse_std']:.4f}")
        print(f"    R²:   {seed_metrics['r2']:.6f} ± {seed_metrics['r2_std']:.4f}")
    
    # Save the generated artifacts.
    summary_csv_path = os.path.join(results_dir, "recomputed_metrics_summary.csv")
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n汇总结果已保存到: {summary_csv_path}")
    
    # Save the generated artifacts.
    detail_csv_path = os.path.join(results_dir, "recomputed_metrics_detail.csv")
    df.to_csv(detail_csv_path, index=False, encoding='utf-8-sig')
    print(f"详细结果已保存到: {detail_csv_path}")
    
    # Baseline workflow step.
    print("\n" + "="*80)
    print("所有种子的测试集预测结果统计（均值±标准差）")
    print("="*80)
    print("\n测试集指标统计:")
    print("-"*80)
    if 'test_mae_format' in stats:
        print(f"MAE:  {stats['test_mae_format']}")
        print(f"      (均值: {stats['test_mae_mean']:.3f}, 标准差: {stats['test_mae_std']:.3f})")
    if 'test_rmse_format' in stats:
        print(f"RMSE: {stats['test_rmse_format']}")
        print(f"      (均值: {stats['test_rmse_mean']:.3f}, 标准差: {stats['test_rmse_std']:.3f})")
    if 'test_r2_format' in stats:
        print(f"R²:   {stats['test_r2_format']}")
        print(f"      (均值: {stats['test_r2_mean']:.3f}, 标准差: {stats['test_r2_std']:.3f})")
    print("-"*80)
    
    return df, stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从CSV文件重新计算指标')
    parser.add_argument('--results_dir', type=str,
                       default=str(BIGSOLVDB_EXPERIMENT_ROOT / 'runs'),
                       help='Directory containing trained seed runs.')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 2024],
                       help='随机种子列表')
    
    args = parser.parse_args()
    
    recompute_all_metrics(args.results_dir, args.seeds)
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

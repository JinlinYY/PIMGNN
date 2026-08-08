# -*- coding: utf-8 -*-
"""Implement the mmgnn main baseline module."""

import os
import argparse

from psmi_baselines.common import config as C
from psmi_baselines.common.utils import set_seed
from .data_loader import load_split_datasets
from .train import train_mmgnn
from psmi_baselines.paths import DATA_DIR, EXPERIMENT_ROOT


def main():
    parser = argparse.ArgumentParser(description='MMGNN训练脚本')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练（检查点文件路径）')
    parser.add_argument('--checkpoint-every', type=int, default=10,
                        help='每N个epoch保存一次检查点（默认10）')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory containing train/validation/test CSV files.')
    parser.add_argument('--out-dir', type=str,
                        default=str(EXPERIMENT_ROOT / 'runs' / 'mmgnn'),
                        help='Directory for MMGNN run artifacts.')
    parser.add_argument('--seed', type=int, default=C.SEED)
    parser.add_argument('--epochs', type=int, default=C.EPOCHS)
    parser.add_argument('--device', type=str, default=C.DEVICE)
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Configure the output artifacts.
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Run the training step.
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 90)
    print("MMGNN Training for LLE Prediction")
    print("=" * 90)
    
    # Load the input data.
    print("\n1) Loading datasets from specified CSV files...")
    from .data_loader import load_csv_data
    
    # Run the training step.
    train_csv_path = os.path.join(args.data_dir, "train.csv")
    val_csv_path = os.path.join(args.data_dir, "validation.csv")
    
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
    test_csv_path = os.path.join(args.data_dir, "test.csv")
    if os.path.exists(test_csv_path):
        print(f"  加载测试集: {test_csv_path}")
        _, test_df = load_csv_data(
            test_csv_path,
            min_points_per_group=C.MIN_POINTS_PER_GROUP,
            permute_23_aug=C.PERMUTE_23_AUG
        )
    else:
        raise FileNotFoundError(f"Test dataset not found: {test_csv_path}")
    
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_df)} rows | systems: {train_df['system_id'].nunique()}")
    print(f"  验证集: {len(val_df)} rows | systems: {val_df['system_id'].nunique()}")
    print(f"  测试集: {len(test_df)} rows | systems: {test_df['system_id'].nunique()}")
    
    # Run the training step.
    print("\n3) Training MMGNN...")
    if args.resume:
        print(f"  断点续训模式: 从 {args.resume} 恢复训练")
    else:
        print("  从头开始训练")
    
    model, T_scaler, history = train_mmgnn(
        train_df, val_df, test_df,
        out_dir=out_dir,
        device=args.device,
        batch_size=C.BATCH_SIZE,
        epochs=args.epochs,
        lr=C.LR,
        weight_decay=C.WEIGHT_DECAY,
        hidden_dim=256,
        num_layers=3,
        beta=0.2,
        explainer_method='local_mask',
        dropout=C.DROPOUT,
        patience=150,  # Apply early stopping.
        min_delta=0.0,  # Baseline workflow step.
        resume_from=args.resume,  # Baseline workflow step.
        save_checkpoint_every=args.checkpoint_every,  # Save the generated artifacts.
    )
    
    print("\n" + "=" * 90)
    print("Training completed!")
    print(f"Results saved to: {out_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()


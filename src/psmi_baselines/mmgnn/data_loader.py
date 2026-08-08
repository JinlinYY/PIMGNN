# -*- coding: utf-8 -*-
"""Implement the mmgnn data_loader baseline module."""

import pandas as pd
import numpy as np
from typing import Tuple

import os

from psmi_baselines.common.utils import canonicalize_smiles, renorm3, safe_group_apply_t
from psmi_baselines.paths import DATA_DIR, TOTAL_CSV


def load_csv_data(
    csv_path: str,
    min_points_per_group: int = 6,
    permute_23_aug: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the load csv data baseline operation."""
    # Read the input data.
    df = pd.read_csv(csv_path)
    
    # Baseline workflow step.
    df = df.rename(columns={
        "LLE system NO.": "system_id",
        "T/K": "T",
        "IL (Component 1) full name SMILES": "smiles1",
        "Component 2 SMILES": "smiles2",
        "Component 3 SMILES": "smiles3",
    })
    
    # Baseline workflow step.
    needed = ["system_id", "T", "smiles1", "smiles2", "smiles3",
              "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"缺少必要的列: {c}. 可用列: {list(df.columns)}")
    
    # Baseline workflow step.
    for c in ["smiles1", "smiles2", "smiles3"]:
        df[c] = df[c].astype(str).map(canonicalize_smiles)
    df = df[(df["smiles1"] != "") & (df["smiles2"] != "") & (df["smiles3"] != "")].copy()
    
    # Baseline workflow step.
    for c in ["T", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["T", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]).copy()
    
    # Baseline workflow step.
    E = df[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float32)
    R = df[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float32)
    E = np.vstack([renorm3(e) for e in E])
    R = np.vstack([renorm3(r) for r in R])
    df[["Ex1", "Ex2", "Ex3"]] = E
    df[["Rx1", "Rx2", "Rx3"]] = R
    
    # Process the experiment data.
    counts = df.groupby(["system_id", "T"]).size().reset_index(name="n")
    keep = counts[counts["n"] >= min_points_per_group][["system_id", "T"]]
    df = df.merge(keep, on=["system_id", "T"], how="inner")
    
    # Configure experiment parameters.
    df = safe_group_apply_t(df)
    
    # Process the experiment data.
    df_raw = df.copy()
    df_aug = df.copy()
    df_aug["aug_swap23"] = 0
    
    if permute_23_aug:
        df2 = df.copy()
        df2["aug_swap23"] = 1
        df2[["smiles2", "smiles3"]] = df2[["smiles3", "smiles2"]]
        df2[["Ex2", "Ex3"]] = df2[["Ex3", "Ex2"]]
        df2[["Rx2", "Rx3"]] = df2[["Rx3", "Rx2"]]
        df_aug = pd.concat([df_aug, df2], ignore_index=True)
    
    return df_raw, df_aug


def load_split_datasets(
    dataset_dir: str = ".",
    train_file: str = "train.csv",
    val_file: str = "validation.csv",
    test_file: str = "test.csv",
    min_points_per_group: int = 6,
    permute_23_aug: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the load split datasets baseline operation."""
    train_path = os.path.join(dataset_dir, train_file)
    val_path = os.path.join(dataset_dir, val_file)
    test_path = os.path.join(dataset_dir, test_file)
    
    # Load the input data.
    _, train_df_aug = load_csv_data(train_path, min_points_per_group, permute_23_aug)
    val_df_raw, _ = load_csv_data(val_path, min_points_per_group, False)
    test_df_raw, _ = load_csv_data(test_path, min_points_per_group, False)
    
    return train_df_aug, val_df_raw, test_df_raw

# ===== Dataset split utilities (merged from split_dataset.py) =====
SEED = 42


def split_by_system_id(
    df: pd.DataFrame,
    system_col: str = 'LLE system NO.',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the split by system id baseline operation."""
    if system_col not in df.columns:
        raise ValueError(f"列 '{system_col}' 不存在。可用列: {list(df.columns)}")
    
    # Baseline workflow step.
    systems = sorted(df[system_col].unique().tolist())
    print(f"总共有 {len(systems)} 个不同的system ID")
    
    # Set the random seed.
    rng = np.random.RandomState(seed)
    rng.shuffle(systems)
    
    # Baseline workflow step.
    n = len(systems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # Baseline workflow step.
    train_sys = set(systems[:n_train])
    val_sys = set(systems[n_train:n_train + n_val])
    test_sys = set(systems[n_train + n_val:])
    
    print(f"\n划分结果:")
    print(f"  训练集: {len(train_sys)} 个system ({len(train_sys)/n*100:.1f}%)")
    print(f"  验证集: {len(val_sys)} 个system ({len(val_sys)/n*100:.1f}%)")
    print(f"  测试集: {len(test_sys)} 个system ({len(test_sys)/n*100:.1f}%)")
    
    # Process the experiment data.
    train_df = df[df[system_col].isin(train_sys)].copy()
    val_df = df[df[system_col].isin(val_sys)].copy()
    test_df = df[df[system_col].isin(test_sys)].copy()
    
    print(f"\n数据行数:")
    print(f"  训练集: {len(train_df)} 行 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 行 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 行 ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def split_dataset_main():
    """Create an optional MMGNN-specific legacy split without replacing shared data."""
    input_file = str(TOTAL_CSV)
    
    # Configure the output artifacts.
    output_dir = str(DATA_DIR / "legacy_mmgnn_split")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("数据集划分脚本")
    print("=" * 80)
    print(f"\n输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    
    # Read the input data.
    print("\n正在读取数据...")
    df = pd.read_csv(input_file)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # Baseline workflow step.
    required_cols = ['LLE system NO.']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要的列: {col}")
    
    # Process the experiment data.
    print("\n正在划分数据集...")
    train_df, val_df, test_df = split_by_system_id(
        df,
        system_col='LLE system NO.',
        train_ratio=0.8,
        val_ratio=0.1,
        seed=SEED
    )
    
    # Save the generated artifacts.
    print("\n正在保存数据集...")
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    
    print(f"\n[OK] 训练集已保存: {train_path}")
    print(f"[OK] 验证集已保存: {val_path}")
    print(f"[OK] 测试集已保存: {test_path}")
    
    # Save the generated artifacts.
    info_path = os.path.join(output_dir, "split_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("数据集划分信息\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"随机种子: {SEED}\n")
        f.write(f"划分比例: 训练集80% : 验证集10% : 测试集10%\n\n")
        f.write(f"总数据行数: {len(df)}\n")
        f.write(f"总system数: {df['LLE system NO.'].nunique()}\n\n")
        f.write(f"训练集: {len(train_df)} 行, {train_df['LLE system NO.'].nunique()} 个system\n")
        f.write(f"验证集: {len(val_df)} 行, {val_df['LLE system NO.'].nunique()} 个system\n")
        f.write(f"测试集: {len(test_df)} 行, {test_df['LLE system NO.'].nunique()} 个system\n")
    
    print(f"\n[OK] 划分信息已保存: {info_path}")
    print("\n" + "=" * 80)
    print("数据集划分完成！")
    print("=" * 80)


if __name__ == "__main__":
    split_dataset_main()


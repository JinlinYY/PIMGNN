"""
CIGNN dataset utilities: merge splits, analyze total.csv, check CSV format.

Usage:
  python dataset_tools.py merge
  python dataset_tools.py analyze
  python dataset_tools.py check
"""
import argparse
import os

import pandas as pd

from psmi_baselines.paths import DATA_DIR, TOTAL_CSV

try:
    from rdkit import Chem
except ImportError:
    Chem = None


def merge_datasets(dataset_dir=str(DATA_DIR)):
    print("Reading CSV files...")
    train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"), encoding="utf-8")
    test_df = pd.read_csv(os.path.join(dataset_dir, "test.csv"), encoding="utf-8")
    val_df = pd.read_csv(os.path.join(dataset_dir, "validation.csv"), encoding="utf-8")
    print(f"train.csv: {len(train_df)} rows")
    print(f"test.csv: {len(test_df)} rows")
    print(f"val.csv: {len(val_df)} rows")
    total_df = pd.concat([train_df, test_df, val_df], ignore_index=True)
    output_path = os.path.join(dataset_dir, "total.csv")
    total_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Merged {len(total_df)} rows -> {output_path}")


def analyze_total_dataset(csv_path=str(TOTAL_CSV)):
    print("=" * 80)
    print("Dataset summary:", csv_path)
    print("=" * 80)
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"rows={len(df):,}, cols={len(df.columns)}")
    print("columns:", list(df.columns))
    if "T/K" in df.columns:
        print(
            f"T/K range: {df['T/K'].min():.2f} - {df['T/K'].max():.2f}, "
            f"mean={df['T/K'].mean():.2f}, std={df['T/K'].std():.2f}"
        )
    for col in ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]:
        if col in df.columns:
            print(
                f"{col}: [{df[col].min():.4f}, {df[col].max():.4f}], "
                f"mean={df[col].mean():.4f}"
            )
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("missing columns:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count}")
    else:
        print("no missing values")
    smiles_cols = [
        "IL (Component 1) full name SMILES",
        "Component 2 SMILES",
        "Component 3 SMILES",
    ]
    for col in smiles_cols:
        if col in df.columns:
            print(f"unique {col}: {df[col].nunique()}")
    if "LLE system NO." in df.columns:
        print(f"unique systems: {df['LLE system NO.'].nunique()}")
    size = os.path.getsize(csv_path)
    print(f"file size: {size/1024/1024:.2f} MB")
    print("=" * 80)


def check_smiles_validity(smiles):
    if Chem is None:
        return False, "rdkit not installed"
    if pd.isna(smiles) or smiles == "":
        return False, "empty"
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return False, "invalid SMILES"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def check_csv_files(files=None):
    if files is None:
        files = [
            str(DATA_DIR / "train.csv"),
            str(DATA_DIR / "validation.csv"),
            str(DATA_DIR / "test.csv"),
        ]
    print("=" * 80)
    print("CSV format check")
    print("=" * 80)
    for file_path in files:
        print(f"\n[{file_path}]")
        try:
            df = pd.read_csv(file_path)
            print(f"rows={len(df)}, columns={list(df.columns)}")
            smiles_columns = [col for col in df.columns if "SMILES" in col]
            for col in smiles_columns:
                sample_size = min(10, len(df))
                valid_count = sum(
                    1
                    for idx in range(sample_size)
                    if check_smiles_validity(df[col].iloc[idx])[0]
                )
                print(
                    f"  {col}: sample valid {valid_count}/{sample_size}, "
                    f"unique={df[col].nunique()}"
                )
        except Exception as e:
            print(f"  error: {e}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="CIGNN dataset tools")
    parser.add_argument("command", choices=["merge", "analyze", "check"])
    parser.add_argument("--dataset_dir", default=str(DATA_DIR))
    parser.add_argument("--csv", default=str(TOTAL_CSV))
    args = parser.parse_args()
    if args.command == "merge":
        merge_datasets(args.dataset_dir)
    elif args.command == "analyze":
        analyze_total_dataset(args.csv)
    else:
        check_csv_files()


if __name__ == "__main__":
    main()

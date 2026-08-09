# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

warnings.filterwarnings('ignore', category=DeprecationWarning)


def canonicalize_smiles(smi: str) -> str:
    if not isinstance(smi, str) or not smi.strip():
        return ""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return ""


def morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    
    arr = np.zeros((n_bits,), dtype=np.int8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def load_bigsolvdb_data(
    csv_path: str,
    target_col: str = "LogS(mol/L)",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    fp_bits: int = 2048,
    fp_radius: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f" load dataset : {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f" number of raw records : {len(df)}")
    
    # Process the experiment data.
    df = df.copy()
    df['SMILES_Solute'] = df['SMILES_Solute'].astype(str).map(canonicalize_smiles)
    df['SMILES_Solvent'] = df['SMILES_Solvent'].astype(str).map(canonicalize_smiles)
    
    df = df[(df['SMILES_Solute'] != "") & (df['SMILES_Solvent'] != "")].copy()
    
    if target_col not in df.columns:
        raise ValueError(f" target column '{target_col}' does not exist . available columns : {list(df.columns)}")
    
    df = df.dropna(subset=[target_col, 'Temperature_K']).copy()
    
    print(f" number of cleaned records : {len(df)}")
    
    print(" generate molecular fingerprints ...")
    solute_fps = []
    solvent_fps = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        for idx, row in df.iterrows():
            solute_fp = morgan_fp(row['SMILES_Solute'], radius=fp_radius, n_bits=fp_bits)
            solvent_fp = morgan_fp(row['SMILES_Solvent'], radius=fp_radius, n_bits=fp_bits)
            solute_fps.append(solute_fp)
            solvent_fps.append(solvent_fp)
    
    df['solute_fp'] = solute_fps
    df['solvent_fp'] = solvent_fps
    
    # Save the generated artifacts.
    df['T'] = df['Temperature_K'].astype(np.float32)
    df['target'] = df[target_col].astype(np.float32)
    
    print(f" split dataset : test set {test_size:.1%}, validation set {val_size:.1%}")
    
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    print(f" training set : {len(train_df)} sample ")
    print(f" validation set : {len(val_df)} sample ")
    print(f" test set : {len(test_df)} sample ")
    
    return train_df, val_df, test_df


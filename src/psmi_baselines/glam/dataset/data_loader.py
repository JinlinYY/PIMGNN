"""Implement the glam dataset data_loader baseline module."""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from psmi_baselines.protocol import canonical_split_indices
import os


def smiles_to_graph(smiles):
    """Run the smiles to graph baseline operation."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Baseline workflow step.
        num_atoms = mol.GetNumAtoms()
        
        # Baseline workflow step.
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),  # Baseline workflow step.
                atom.GetDegree(),     # Baseline workflow step.
                atom.GetFormalCharge(),  # Baseline workflow step.
                int(atom.GetHybridization()),  # Baseline workflow step.
                int(atom.GetIsAromatic()),  # Baseline workflow step.
                atom.GetTotalNumHs(),  # Baseline workflow step.
                atom.GetNumRadicalElectrons(),  # Baseline workflow step.
                int(atom.GetChiralTag()),  # Baseline workflow step.
            ]
            node_features.append(features)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # Baseline workflow step.
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Baseline workflow step.
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            # Baseline workflow step.
            bond_features = [
                bond.GetBondTypeAsDouble(),  # Baseline workflow step.
                int(bond.GetIsConjugated()),  # Baseline workflow step.
                int(bond.IsInRing()),  # Baseline workflow step.
            ]
            edge_attr.append(bond_features)
            edge_attr.append(bond_features)
        
        if len(edge_index) == 0:
            # Baseline workflow step.
            edge_index = [[0, 0]]
            edge_attr = [[0.0, 0.0, 0.0]]
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


def load_LLE_dataset(csv_path, test_size=0.2, val_size=0.1, random_state=42):
    """Run the load LLE dataset baseline operation."""
    # Read the input data.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file does not exist : {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f" unable to read CSV file {csv_path}: {e}")
    
    if len(df) == 0:
        raise ValueError(f"CSV file is Empty : {csv_path}")
    
    print(f" dataset total samples : {len(df)}")
    print(f" dataset column : {df.columns.tolist()}")
    
    # Baseline workflow step.
    required_columns = [
        'IL (Component 1) full name SMILES',
        'Component 2 SMILES',
        'Component 3 SMILES',
        'Ex1', 'Ex2', 'Ex3',
        'Rx1', 'Rx2', 'Rx3',
        'T/K'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV file missing required columns : {missing_columns}\n Actual column : {df.columns.tolist()}")
    
    # Baseline workflow step.
    data_list = []
    labels = []
    failed_count = 0
    failed_reasons = {'il': 0, 'comp2': 0, 'comp3': 0}
    
    for idx, row in df.iterrows():
        # Baseline workflow step.
        il_smiles = str(row['IL (Component 1) full name SMILES'])
        comp2_smiles = str(row['Component 2 SMILES'])
        comp3_smiles = str(row['Component 3 SMILES'])
        
        # Baseline workflow step.
        if pd.isna(row['IL (Component 1) full name SMILES']) or il_smiles == 'nan' or il_smiles.strip() == '':
            failed_count += 1
            failed_reasons['il'] += 1
            if failed_count <= 5:  # Baseline workflow step.
                print(f" warning : number {idx+1} rows IL SMILES is empty or invalid ")
            continue
        
        if pd.isna(row['Component 2 SMILES']) or comp2_smiles == 'nan' or comp2_smiles.strip() == '':
            failed_count += 1
            failed_reasons['comp2'] += 1
            if failed_count <= 5:
                print(f" warning : number {idx+1} rows Component 2 SMILES is empty or invalid ")
            continue
        
        if pd.isna(row['Component 3 SMILES']) or comp3_smiles == 'nan' or comp3_smiles.strip() == '':
            failed_count += 1
            failed_reasons['comp3'] += 1
            if failed_count <= 5:
                print(f" warning : number {idx+1} rows Component 3 SMILES is empty or invalid ")
            continue
        
        # Configure the output artifacts.
        label = [
            float(row['Ex1']) if pd.notna(row['Ex1']) else 0.0,
            float(row['Ex2']) if pd.notna(row['Ex2']) else 0.0,
            float(row['Ex3']) if pd.notna(row['Ex3']) else 0.0,
            float(row['Rx1']) if pd.notna(row['Rx1']) else 0.0,
            float(row['Rx2']) if pd.notna(row['Rx2']) else 0.0,
            float(row['Rx3']) if pd.notna(row['Rx3']) else 0.0,
        ]
        
        # Baseline workflow step.
        temperature = float(row['T/K']) if pd.notna(row['T/K']) else 298.15
        
        # Baseline workflow step.
        il_graph = smiles_to_graph(il_smiles)
        comp2_graph = smiles_to_graph(comp2_smiles)
        comp3_graph = smiles_to_graph(comp3_smiles)
        
        # Baseline workflow step.
        if il_graph is not None and comp2_graph is not None and comp3_graph is not None:
            # Baseline workflow step.
            system_no = int(row['LLE system NO.']) if pd.notna(row['LLE system NO.']) else idx
            
            data_list.append({
                'il_graph': il_graph,
                'comp2_graph': comp2_graph,
                'comp3_graph': comp3_graph,
                'temperature': temperature,
                'il_smiles': il_smiles,
                'comp2_smiles': comp2_smiles,
                'comp3_smiles': comp3_smiles,
                'system_no': system_no,  # Baseline workflow step.
                'split': str(row['split']).lower() if 'split' in df.columns else None,
            })
            labels.append(label)
        else:
            failed_count += 1
            if il_graph is None:
                failed_reasons['il'] += 1
                if failed_count <= 5:
                    print(f" warning : number {idx+1} rows IL SMILES unable to convert to a graph : {il_smiles[:50]}")
            if comp2_graph is None:
                failed_reasons['comp2'] += 1
                if failed_count <= 5:
                    print(f" warning : number {idx+1} rows Component 2 SMILES unable to convert to a graph : {comp2_smiles[:50]}")
            if comp3_graph is None:
                failed_reasons['comp3'] += 1
                if failed_count <= 5:
                    print(f" warning : number {idx+1} rows Component 3 SMILES unable to convert to a graph : {comp3_smiles[:50]}")
    
    print(f" successful process {len(data_list)} samples ")
    if failed_count > 0:
        print(f" failed {failed_count} samples ")
        print(f" failed reason statistics : IL={failed_reasons['il']}, Component2={failed_reasons['comp2']}, Component3={failed_reasons['comp3']}")
    
    # Process the experiment data.
    if len(data_list) == 0:
        error_msg = (
            " error : None successful process Any sample !\n"
            " May reason :\n"
            "1. CSV file path Incorrect or file does not exist \n"
            "2. CSV file is Empty or format Incorrect \n"
            "3. SMILES String format Problem ,RDKit unable to parse \n"
            "4. column First Name mismatch ( please check column First Name whether Correct )\n"
            f" current CSV path : {csv_path}\n"
            f"CSV file whether Deposit at : {os.path.exists(csv_path)}\n"
            f"CSV Total rows Number : {len(df)}"
        )
        raise ValueError(error_msg)
    
    # Baseline workflow step.
    labels = np.array(labels, dtype=np.float32)

    # Use the canonical system-exclusive split when it is present in total.csv.
    if 'split' in df.columns:
        accepted_frame = pd.DataFrame(
            {
                'split': [item['split'] for item in data_list],
                'system_id': [item['system_no'] for item in data_list],
            }
        )
        split_indices = canonical_split_indices(accepted_frame)

        def _partition(label):
            selected = split_indices[label]
            return {
                'data': [data_list[index] for index in selected],
                'labels': labels[selected],
            }

        return {
            'train': _partition('train'),
            'val': _partition('validation'),
            'test': _partition('test'),
        }
    
    # Process the experiment data.
    # Run the training step.
    indices = np.arange(len(data_list))
    
    # Process the experiment data.
    if len(indices) == 0:
        raise ValueError(" error : dataset is empty , unable to Into rows Divide !")
    
    # Process the experiment data.
    if len(indices) < 10:
        print(f" warning : data Small Amount ({len(indices)} samples ), will use Smaller test set Scale ")
        test_size_adjusted = min(test_size, 0.1)  # Evaluate the test subset.
    else:
        test_size_adjusted = test_size
    
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size_adjusted, random_state=random_state
    )
    
    # Run the training step.
    val_size_adjusted = val_size / (1 - test_size)  # Evaluate the validation subset.
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size_adjusted, random_state=random_state
    )
    
    # Process the experiment data.
    train_data = [data_list[i] for i in train_indices]
    train_labels = labels[train_indices]
    
    val_data = [data_list[i] for i in val_indices]
    val_labels = labels[val_indices]
    
    test_data = [data_list[i] for i in test_indices]
    test_labels = labels[test_indices]
    
    print(f"\n dataset Divide :")
    print(f" training set : {len(train_data)} sample ")
    print(f" validation set : {len(val_data)} sample ")
    print(f" test set : {len(test_data)} sample ")
    
    return {
        'train': {'data': train_data, 'labels': train_labels},
        'val': {'data': val_data, 'labels': val_labels},
        'test': {'data': test_data, 'labels': test_labels}
    }


def collate_fn(batch):
    """Run the collate fn baseline operation."""
    il_graphs = [item['il_graph'] for item in batch]
    comp2_graphs = [item['comp2_graph'] for item in batch]
    comp3_graphs = [item['comp3_graph'] for item in batch]
    temperatures = torch.tensor([item['temperature'] for item in batch], dtype=torch.float32)
    
    # Baseline workflow step.
    il_batch = Batch.from_data_list(il_graphs)
    comp2_batch = Batch.from_data_list(comp2_graphs)
    comp3_batch = Batch.from_data_list(comp3_graphs)
    
    return {
        'il_graph': il_batch,
        'comp2_graph': comp2_batch,
        'comp3_graph': comp3_batch,
        'temperature': temperatures,
    }


if __name__ == "__main__":
    # Load the input data.
    csv_path = "dataset/total.csv"
    datasets = load_LLE_dataset(csv_path)
    
    print("\n example data :")
    sample = datasets['train']['data'][0]
    print(f"IL SMILES: {sample['il_smiles']}")
    print(f"Component 2 SMILES: {sample['comp2_smiles']}")
    print(f"Component 3 SMILES: {sample['comp3_smiles']}")
    print(f"Temperature: {sample['temperature']}")
    print(f"Label: {datasets['train']['labels'][0]}")


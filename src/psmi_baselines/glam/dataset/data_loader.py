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
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        num_atoms = mol.GetNumAtoms()
        
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetChiralTag()),
            ]
            node_features.append(features)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            bond_features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
            ]
            edge_attr.append(bond_features)
            edge_attr.append(bond_features)
        
        if len(edge_index) == 0:
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
    # Read the input data.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file does not exist : {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Unable to read CSV file {csv_path}: {e}")
    
    if len(df) == 0:
        raise ValueError(f"CSV file is Empty : {csv_path}")
    
    print(f" dataset total samples : {len(df)}")
    print(f" dataset column : {df.columns.tolist()}")
    
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
    
    data_list = []
    labels = []
    failed_count = 0
    failed_reasons = {'il': 0, 'comp2': 0, 'comp3': 0}
    
    for idx, row in df.iterrows():
        il_smiles = str(row['IL (Component 1) full name SMILES'])
        comp2_smiles = str(row['Component 2 SMILES'])
        comp3_smiles = str(row['Component 3 SMILES'])
        
        if pd.isna(row['IL (Component 1) full name SMILES']) or il_smiles == 'nan' or il_smiles.strip() == '':
            failed_count += 1
            failed_reasons['il'] += 1
            if failed_count <= 5:
                print(f"Warning: component-1 SMILES is empty or invalid at row {idx + 1}.")
            continue
        
        if pd.isna(row['Component 2 SMILES']) or comp2_smiles == 'nan' or comp2_smiles.strip() == '':
            failed_count += 1
            failed_reasons['comp2'] += 1
            if failed_count <= 5:
                print(f"Warning: component-2 SMILES is empty or invalid at row {idx + 1}.")
            continue
        
        if pd.isna(row['Component 3 SMILES']) or comp3_smiles == 'nan' or comp3_smiles.strip() == '':
            failed_count += 1
            failed_reasons['comp3'] += 1
            if failed_count <= 5:
                print(f"Warning: component-3 SMILES is empty or invalid at row {idx + 1}.")
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
        
        temperature = float(row['T/K']) if pd.notna(row['T/K']) else 298.15
        
        il_graph = smiles_to_graph(il_smiles)
        comp2_graph = smiles_to_graph(comp2_smiles)
        comp3_graph = smiles_to_graph(comp3_smiles)
        
        if il_graph is not None and comp2_graph is not None and comp3_graph is not None:
            system_no = int(row['LLE system NO.']) if pd.notna(row['LLE system NO.']) else idx
            
            data_list.append({
                'il_graph': il_graph,
                'comp2_graph': comp2_graph,
                'comp3_graph': comp3_graph,
                'temperature': temperature,
                'il_smiles': il_smiles,
                'comp2_smiles': comp2_smiles,
                'comp3_smiles': comp3_smiles,
                'system_no': system_no,
                'split': str(row['split']).lower() if 'split' in df.columns else None,
            })
            labels.append(label)
        else:
            failed_count += 1
            if il_graph is None:
                failed_reasons['il'] += 1
                if failed_count <= 5:
                    print(f"Warning: component-1 SMILES could not be graphed at row {idx + 1}: {il_smiles[:50]}")
            if comp2_graph is None:
                failed_reasons['comp2'] += 1
                if failed_count <= 5:
                    print(f"Warning: component-2 SMILES could not be graphed at row {idx + 1}: {comp2_smiles[:50]}")
            if comp3_graph is None:
                failed_reasons['comp3'] += 1
                if failed_count <= 5:
                    print(f"Warning: component-3 SMILES could not be graphed at row {idx + 1}: {comp3_smiles[:50]}")
    
    print(f"Processed {len(data_list)} samples.")
    if failed_count > 0:
        print(f"Rejected {failed_count} samples.")
        print(f"Rejection counts: component1={failed_reasons['il']}, component2={failed_reasons['comp2']}, component3={failed_reasons['comp3']}")
    
    # Process the experiment data.
    if len(data_list) == 0:
        error_msg = (
            "No valid samples could be constructed.\n"
            "Possible causes:\n"
            "1. The CSV path is incorrect or the file is absent.\n"
            "2. The CSV is empty or malformed.\n"
            "3. RDKit cannot parse one or more SMILES strings.\n"
            "4. Required column names are missing.\n"
            f"CSV path: {csv_path}\n"
            f"File exists: {os.path.exists(csv_path)}\n"
            f"Input rows: {len(df)}"
        )
        raise ValueError(error_msg)
    
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
    indices = np.arange(len(data_list))
    
    # Process the experiment data.
    if len(indices) == 0:
        raise ValueError("The dataset is empty and cannot be partitioned.")
    
    # Process the experiment data.
    if len(indices) < 10:
        print(f"Small dataset ({len(indices)} samples); reducing the test fraction.")
        test_size_adjusted = min(test_size, 0.1)  # Evaluate the test subset.
    else:
        test_size_adjusted = test_size
    
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size_adjusted, random_state=random_state
    )
    
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
    il_graphs = [item['il_graph'] for item in batch]
    comp2_graphs = [item['comp2_graph'] for item in batch]
    comp3_graphs = [item['comp3_graph'] for item in batch]
    temperatures = torch.tensor([item['temperature'] for item in batch], dtype=torch.float32)
    
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


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
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"无法读取CSV文件 {csv_path}: {e}")
    
    if len(df) == 0:
        raise ValueError(f"CSV文件为空: {csv_path}")
    
    print(f"数据集总样本数: {len(df)}")
    print(f"数据集列: {df.columns.tolist()}")
    
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
        raise ValueError(f"CSV文件缺少必要的列: {missing_columns}\n实际列: {df.columns.tolist()}")
    
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
                print(f"警告: 第{idx+1}行IL SMILES为空或无效")
            continue
        
        if pd.isna(row['Component 2 SMILES']) or comp2_smiles == 'nan' or comp2_smiles.strip() == '':
            failed_count += 1
            failed_reasons['comp2'] += 1
            if failed_count <= 5:
                print(f"警告: 第{idx+1}行Component 2 SMILES为空或无效")
            continue
        
        if pd.isna(row['Component 3 SMILES']) or comp3_smiles == 'nan' or comp3_smiles.strip() == '':
            failed_count += 1
            failed_reasons['comp3'] += 1
            if failed_count <= 5:
                print(f"警告: 第{idx+1}行Component 3 SMILES为空或无效")
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
                    print(f"警告: 第{idx+1}行IL SMILES无法转换为图: {il_smiles[:50]}")
            if comp2_graph is None:
                failed_reasons['comp2'] += 1
                if failed_count <= 5:
                    print(f"警告: 第{idx+1}行Component 2 SMILES无法转换为图: {comp2_smiles[:50]}")
            if comp3_graph is None:
                failed_reasons['comp3'] += 1
                if failed_count <= 5:
                    print(f"警告: 第{idx+1}行Component 3 SMILES无法转换为图: {comp3_smiles[:50]}")
    
    print(f"成功处理 {len(data_list)} 个样本")
    if failed_count > 0:
        print(f"失败 {failed_count} 个样本")
        print(f"失败原因统计: IL={failed_reasons['il']}, Component2={failed_reasons['comp2']}, Component3={failed_reasons['comp3']}")
    
    # Process the experiment data.
    if len(data_list) == 0:
        error_msg = (
            "错误: 没有成功处理任何样本！\n"
            "可能的原因：\n"
            "1. CSV文件路径不正确或文件不存在\n"
            "2. CSV文件为空或格式不正确\n"
            "3. SMILES字符串格式有问题，RDKit无法解析\n"
            "4. 列名不匹配（请检查列名是否正确）\n"
            f"当前CSV路径: {csv_path}\n"
            f"CSV文件是否存在: {os.path.exists(csv_path)}\n"
            f"CSV总行数: {len(df)}"
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
        raise ValueError("错误: 数据集为空，无法进行划分！")
    
    # Process the experiment data.
    if len(indices) < 10:
        print(f"警告: 数据量很小（{len(indices)}个样本），将使用较小的测试集比例")
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
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    print(f"测试集: {len(test_data)} 样本")
    
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
    
    print("\n示例数据:")
    sample = datasets['train']['data'][0]
    print(f"IL SMILES: {sample['il_smiles']}")
    print(f"Component 2 SMILES: {sample['comp2_smiles']}")
    print(f"Component 3 SMILES: {sample['comp3_smiles']}")
    print(f"Temperature: {sample['temperature']}")
    print(f"Label: {datasets['train']['labels'][0]}")


import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def smiles_to_graph(smiles):
    if '.' in smiles:
        mol_smiles_list = smiles.split('.')
        all_atom_features = []
        all_edge_index = []
        all_edge_attr = []
        node_offset = 0
        
        for mol_smiles in mol_smiles_list:
            mol_smiles = mol_smiles.strip()
            if not mol_smiles:
                continue
                
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                continue
            
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetHybridization()),
                    int(atom.GetIsAromatic()),
                    atom.GetNumRadicalElectrons(),
                    atom.GetTotalNumHs(),
                ]
                atom_features.append(features)
            
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx() + node_offset
                j = bond.GetEndAtomIdx() + node_offset
                
                edge_index.append([i, j])
                edge_index.append([j, i])
                bond_features = [
                    int(bond.GetBondType()),
                    int(bond.GetIsAromatic()),
                    int(bond.IsInRing()),
                ]
                edge_attr.append(bond_features)
                edge_attr.append(bond_features)
            
            all_atom_features.extend(atom_features)
            all_edge_index.extend(edge_index)
            all_edge_attr.extend(edge_attr)
            
            node_offset += len(atom_features)
        
        if len(all_atom_features) == 0:
            return None
        
        if len(all_edge_index) == 0:
            all_edge_index = [[0], [0]]
            all_edge_attr = [[0, 0, 0]]
        
        x = torch.tensor(all_atom_features, dtype=torch.float)
        edge_index = torch.tensor(all_edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(all_edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalNumHs(),
            ]
            atom_features.append(features)
        
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_index.append([i, j])
            edge_index.append([j, i])
            bond_features = [
                int(bond.GetBondType()),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
            ]
            edge_attr.append(bond_features)
            edge_attr.append(bond_features)
        
        if len(edge_index) == 0:
            edge_index = [[0], [0]]
            edge_attr = [[0, 0, 0]]
        
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def create_batch(graphs1, graphs2, targets=None):
    batch1 = Batch.from_data_list(graphs1)
    batch2 = Batch.from_data_list(graphs2)
    
    if targets is not None:
        if isinstance(targets, list):
            targets = np.array(targets)
        targets = torch.tensor(targets, dtype=torch.float)
        return batch1, batch2, targets
    
    return batch1, batch2


class MolecularDataset:
    def __init__(self, smiles1_list, smiles2_list, targets=None):
        self.smiles1_list = smiles1_list
        self.smiles2_list = smiles2_list
        self.targets = targets
        
        self.graphs1 = []
        self.graphs2 = []
        self.valid_indices = []
        
        for i, (s1, s2) in enumerate(zip(smiles1_list, smiles2_list)):
            g1 = smiles_to_graph(s1)
            g2 = smiles_to_graph(s2)
            
            if g1 is not None and g2 is not None:
                self.graphs1.append(g1)
                self.graphs2.append(g2)
                self.valid_indices.append(i)
        
        if targets is not None:
            self.targets = [targets[i] for i in self.valid_indices]
    
    def __len__(self):
        return len(self.graphs1)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.graphs1[idx], self.graphs2[idx], self.targets[idx]
        return self.graphs1[idx], self.graphs2[idx]


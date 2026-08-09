# -*- coding: utf-8 -*-

from typing import Dict, Tuple, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

ATOM_FEAT_DIM = 44
BOND_FEAT_DIM = 10
def get_atom_features(atom) -> np.ndarray:
    features = []
    
    atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'Sn']
    atom_type_onehot = [1 if atom.GetSymbol() == t else 0 for t in atom_types]
    features.extend(atom_type_onehot)
    
    degree = atom.GetDegree()
    degree_onehot = [1 if degree == i else 0 for i in range(6)]
    features.extend(degree_onehot)
    
    formal_charge = atom.GetFormalCharge()
    features.append(formal_charge)
    
    hybridization = atom.GetHybridization()
    hybrid_onehot = [1 if hybridization == h else 0 for h in [Chem.HybridizationType.SP, 
                                                              Chem.HybridizationType.SP2,
                                                              Chem.HybridizationType.SP3,
                                                              Chem.HybridizationType.SP3D,
                                                              Chem.HybridizationType.SP3D2]]
    features.extend(hybrid_onehot)
    
    features.append(1 if atom.GetIsAromatic() else 0)
    
    num_h = atom.GetTotalNumHs()
    features.append(num_h)
    
    features.append(1 if atom.IsInRing() else 0)
    
    mass = atom.GetMass() / 200.0
    features.append(mass)
    
    try:
        radius = Descriptors.Crippen.MR(atom) / 10.0
    except:
        radius = 0.0
    features.append(radius)
    
    while len(features) < ATOM_FEAT_DIM:
        features.append(0.0)
    
    return np.array(features[:ATOM_FEAT_DIM], dtype=np.float32)


def get_bond_features(bond) -> np.ndarray:
    features = []
    
    bond_type = bond.GetBondType()
    bond_type_onehot = [
        1 if bond_type == Chem.BondType.SINGLE else 0,
        1 if bond_type == Chem.BondType.DOUBLE else 0,
        1 if bond_type == Chem.BondType.TRIPLE else 0,
        1 if bond_type == Chem.BondType.AROMATIC else 0,
    ]
    features.extend(bond_type_onehot)
    
    features.append(1 if bond.GetIsConjugated() else 0)
    
    features.append(1 if bond.IsInRing() else 0)
    
    stereo = bond.GetStereo()
    stereo_onehot = [1 if stereo == s else 0 for s in [Chem.BondStereo.STEREONONE,
                                                        Chem.BondStereo.STEREOANY,
                                                        Chem.BondStereo.STEREOZ,
                                                        Chem.BondStereo.STEREOE]]
    features.extend(stereo_onehot)
    
    while len(features) < BOND_FEAT_DIM:
        features.append(0.0)
    
    return np.array(features[:BOND_FEAT_DIM], dtype=np.float32)


def get_molecular_global_features(mol) -> np.ndarray:
    features = []
    
    try:
        mw = Descriptors.MolWt(mol) / 1000.0
    except:
        mw = 0.0
    features.append(mw)
    
    # LogP
    try:
        logp = Descriptors.MolLogP(mol)
    except:
        logp = 0.0
    features.append(logp)
    
    try:
        tpsa = Descriptors.TPSA(mol) / 200.0
    except:
        tpsa = 0.0
    features.append(tpsa)
    
    num_atoms = mol.GetNumAtoms()
    features.append(num_atoms / 100.0)
    num_bonds = mol.GetNumBonds()
    features.append(num_bonds / 100.0)
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    features.append(num_rings / 10.0)
    return np.array(features, dtype=np.float32)


class MoleculeGraphBuilder:
    
    def __init__(self):
        self.atom_feat_dim = ATOM_FEAT_DIM
        self.bond_feat_dim = BOND_FEAT_DIM
    
    def smiles_to_graph(self, smiles: str) -> Dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'node_features': np.zeros((1, self.atom_feat_dim), dtype=np.float32),
                'edge_index': np.zeros((2, 0), dtype=np.int64),
                'edge_features': np.zeros((0, self.bond_feat_dim), dtype=np.float32),
                'global_features': np.zeros(6, dtype=np.float32),
                'num_nodes': 1,
                'num_edges': 0
            }
        
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(get_atom_features(atom))
        node_features = np.array(node_features, dtype=np.float32)
        
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
            bond_feat = get_bond_features(bond)
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)
        
        edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_features = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, self.bond_feat_dim), dtype=np.float32)
        
        global_features = get_molecular_global_features(mol)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'global_features': global_features,
            'num_nodes': len(node_features),
            'num_edges': len(edge_features)
        }
    
    def merge_graphs(self, graphs: list, add_intermolecular_edges: bool = True) -> Dict:
        if len(graphs) == 0:
            raise ValueError("graphs list cannot be empty")
        
        node_offsets = [0]
        for g in graphs:
            node_offsets.append(node_offsets[-1] + g['num_nodes'])
        
        all_node_features = []
        for g in graphs:
            all_node_features.append(g['node_features'])
        all_node_features = np.concatenate(all_node_features, axis=0)
        
        all_edge_index = []
        all_edge_features = []
        for idx, g in enumerate(graphs):
            offset = node_offsets[idx]
            edge_idx = g['edge_index'] + offset
            all_edge_index.append(edge_idx)
            all_edge_features.append(g['edge_features'])
        
        if add_intermolecular_edges and len(graphs) > 1:
            intermolecular_edges = []
            for i in range(len(graphs)):
                for j in range(i + 1, len(graphs)):
                    nodes_i = list(range(node_offsets[i], node_offsets[i+1]))
                    nodes_j = list(range(node_offsets[j], node_offsets[j+1]))
                    for ni in nodes_i:
                        for nj in nodes_j:
                            intermolecular_edges.append([ni, nj])
                            intermolecular_edges.append([nj, ni])
            if intermolecular_edges:
                intermolecular_edge_index = np.array(intermolecular_edges, dtype=np.int64).T
                intermolecular_edge_features = np.zeros((len(intermolecular_edges), self.bond_feat_dim), dtype=np.float32)
                for k, (ni, nj) in enumerate(intermolecular_edges):
                    node_i_feat = all_node_features[ni]
                    node_j_feat = all_node_features[nj]
                    similarity = 1.0 / (1.0 + np.linalg.norm(node_i_feat - node_j_feat))
                    intermolecular_edge_features[k, 0] = similarity
                all_edge_index.append(intermolecular_edge_index)
                all_edge_features.append(intermolecular_edge_features)
        
        if all_edge_index:
            all_edge_index = np.concatenate(all_edge_index, axis=1)
            all_edge_features = np.concatenate(all_edge_features, axis=0)
        else:
            all_edge_index = np.zeros((2, 0), dtype=np.int64)
            all_edge_features = np.zeros((0, self.bond_feat_dim), dtype=np.float32)
        
        all_global_features = np.concatenate([g['global_features'] for g in graphs], axis=0)
        
        molecule_ranges = [(node_offsets[i], node_offsets[i+1]) for i in range(len(graphs))]
        
        return {
            'node_features': all_node_features,
            'edge_index': all_edge_index,
            'edge_features': all_edge_features,
            'global_features': all_global_features,
            'num_nodes': len(all_node_features),
            'num_edges': len(all_edge_features),
            'molecule_ranges': molecule_ranges,
            'num_molecules': len(graphs)
        }


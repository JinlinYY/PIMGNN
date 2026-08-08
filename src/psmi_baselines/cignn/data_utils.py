"""Implement the cignn data_utils baseline module."""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch_geometric.data import Data, Batch


def get_atom_features(atom):
    """Run the get atom features baseline operation."""
    features = []
    
    # Baseline workflow step.
    atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'unknown']
    atom_type = atom.GetSymbol()
    if atom_type not in atom_types:
        atom_type = 'unknown'
    atom_type_onehot = [1 if atom_type == t else 0 for t in atom_types]
    features.extend(atom_type_onehot)
    
    # 2. Implicit Valence (Binary)
    # Baseline workflow step.
    # Baseline workflow step.
    try:
        # Baseline workflow step.
        implicit_valence = atom.GetValence(getExplicit=False)
    except (TypeError, AttributeError):
        # Baseline workflow step.
        # Baseline workflow step.
        try:
            max_valence = atom.GetMaxValence()
            current_valence = atom.GetValence()
            implicit_valence = max(0, max_valence - current_valence)
        except:
            implicit_valence = 0
    features.append(1 if implicit_valence > 0 else 0)
    
    # 3. Radical Electrons (Binary)
    features.append(1 if atom.GetNumRadicalElectrons() > 0 else 0)
    
    # 4. Chirality (one-hot: R, S, None)
    try:
        chiral_tag = atom.GetChiralTag()
        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            features.extend([1, 0, 0])  # R
        elif chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            features.extend([0, 1, 0])  # S
        else:
            features.extend([0, 0, 1])  # None
    except:
        features.extend([0, 0, 1])
    
    # 5. Number of Hydrogens (one-hot: 0, 1, 2, 3, 4+)
    num_h = atom.GetTotalNumHs()
    num_h_onehot = [1 if num_h == i else 0 for i in range(5)]
    features.extend(num_h_onehot)
    
    # 6. Hybridization (one-hot: sp, sp2, sp3, sp3d)
    hyb = atom.GetHybridization()
    hyb_map = {
        Chem.HybridizationType.SP: [1, 0, 0, 0],
        Chem.HybridizationType.SP2: [0, 1, 0, 0],
        Chem.HybridizationType.SP3: [0, 0, 1, 0],
        Chem.HybridizationType.SP3D: [0, 0, 0, 1]
    }
    features.extend(hyb_map.get(hyb, [0, 0, 0, 1]))
    
    # 7. Acidic (Binary)
    features.append(1 if atom.GetFormalCharge() < 0 else 0)
    
    # 8. Basic (Binary)
    features.append(1 if atom.GetFormalCharge() > 0 else 0)
    
    # 9. Aromatic (Binary)
    features.append(1 if atom.GetIsAromatic() else 0)
    
    # Baseline workflow step.
    features.append(1 if atom.GetNumRadicalElectrons() > 0 or atom.GetFormalCharge() < 0 else 0)
    
    # Baseline workflow step.
    features.append(1 if atom.GetFormalCharge() > 0 else 0)
    
    return np.array(features, dtype=np.float32)


def get_bond_features(bond):
    """Run the get bond features baseline operation."""
    features = []
    
    # 1. Bond Type (one-hot: Single, Double, Triple, Aromatic)
    bt = bond.GetBondType()
    bond_type_map = {
        Chem.BondType.SINGLE: [1, 0, 0, 0],
        Chem.BondType.DOUBLE: [0, 1, 0, 0],
        Chem.BondType.TRIPLE: [0, 0, 1, 0],
        Chem.BondType.AROMATIC: [0, 0, 0, 1]
    }
    features.extend(bond_type_map.get(bt, [1, 0, 0, 0]))
    
    # 2. Bond is in Conjugation (Binary)
    features.append(1 if bond.GetIsConjugated() else 0)
    
    # 3. Bond is in Ring (Binary)
    features.append(1 if bond.IsInRing() else 0)
    
    # 4. Bond Chirality (one-hot: E, Z, None)
    try:
        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREOE:
            features.extend([1, 0, 0])  # E
        elif stereo == Chem.BondStereo.STEREOZ:
            features.extend([0, 1, 0])  # Z
        else:
            features.extend([0, 0, 1])  # None
    except:
        features.extend([0, 0, 1])
    
    return np.array(features, dtype=np.float32)


def smiles_to_graph(smiles):
    """Run the smiles to graph baseline operation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Baseline workflow step.
    mol = Chem.RemoveHs(mol)
    
    # Baseline workflow step.
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    
    node_features = np.array(node_features)
    
    # Baseline workflow step.
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Baseline workflow step.
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        bond_features = get_bond_features(bond)
        edge_features.append(bond_features)
        edge_features.append(bond_features)  # Baseline workflow step.
    
    if len(edge_indices) == 0:
        # Baseline workflow step.
        # Baseline workflow step.
        edge_indices = [[0, 0]]
        edge_features = [np.zeros(9, dtype=np.float32)]
    
    edge_indices = np.array(edge_indices).T
    edge_features = np.array(edge_features)
    
    # Baseline workflow step.
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_indices, dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float32)
    )
    
    return data


def batch_graphs(graphs):
    """Run the batch graphs baseline operation."""
    return Batch.from_data_list(graphs)


"""Implement the cgib example baseline module."""
import torch
import numpy as np
from .models import CGIB
from .utils import MolecularDataset, create_batch

# Baseline workflow step.
smiles1_list = [
    'CCO',  # Baseline workflow step.
    'CC(=O)O',  # Baseline workflow step.
    'CCCC',  # Baseline workflow step.
]

smiles2_list = [
    'O',  # Baseline workflow step.
    'O',  # Baseline workflow step.
    'O',  # Baseline workflow step.
]

# Baseline workflow step.
targets = [0.5, 0.8, 0.3]

# Process the experiment data.
dataset = MolecularDataset(smiles1_list, smiles2_list, targets)

print(f" dataset size : {len(dataset)}")
print(f" number of valid samples : {len(dataset.valid_indices)}")

if len(dataset) > 0:
    # Baseline workflow step.
    sample_graph = dataset.graphs1[0]
    input_dim = sample_graph.x.size(1)
    print(f" input-feature dimension : {input_dim}")
    
    # Configure the baseline model.
    model = CGIB(
        input_dim=input_dim,
        hidden_dim=32,
        output_dim=1,
        num_layers=2,
        beta=1e-3,
        gnn_type='mpnn'
    )
    
    print(f" number of model parameters : {sum(p.numel() for p in model.parameters())}")
    
    # Baseline workflow step.
    batch1, batch2, batch_targets = create_batch(
        dataset.graphs1,
        dataset.graphs2,
        dataset.targets
    )
    
    # Baseline workflow step.
    model.eval()
    with torch.no_grad():
        pred, loss_components = model(batch1, batch2, return_loss_components=True)
        print(f"\n prediction shape : {pred.shape}")
        print(f" prediction : {pred.squeeze().numpy()}")
        print(f"\n loss Components :")
        print(f" MI1 loss : {loss_components['mi1'].item():.4f}")
        print(f" MI2 loss : {loss_components['mi2'].item():.4f}")
        print(f" Lambda mean : {loss_components['lambda'].mean().item():.4f}")
        print(f" P mean : {loss_components['p'].mean().item():.4f}")
    
    print("\n model test passed !")
else:
    print(" dataset is empty , please check SMILES String format ")


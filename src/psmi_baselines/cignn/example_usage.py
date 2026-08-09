import torch
from .model import CIGIN
from .data_utils import smiles_to_graph, batch_graphs

# Configure the runtime device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure the baseline model.
node_dim = 33
edge_dim = 10
model = CIGIN(
    node_dim=node_dim,
    edge_dim=edge_dim,
    hidden_dim=64,
    num_mp_layers=3,
    use_set2set=False
).to(device)

# Generate model predictions.
solute_smiles = "CCO"
solvent_smiles = "O"
solute_graph = smiles_to_graph(solute_smiles)
solvent_graph = smiles_to_graph(solvent_smiles)

if solute_graph is not None and solvent_graph is not None:
    solute_batch = batch_graphs([solute_graph])
    solvent_batch = batch_graphs([solvent_graph])
    
    # Configure the runtime device.
    solute_batch = solute_batch.to(device)
    solvent_batch = solvent_batch.to(device)
    
    # Generate model predictions.
    model.eval()
    with torch.no_grad():
        prediction, interaction_map = model(solute_batch, solvent_batch)
        print(f" solute : {solute_smiles}")
        print(f" solvent : {solvent_smiles}")
        print(f" prediction solvent Transforming Free Energy : {prediction.item():.4f} kcal/mol")
        print(f" Interaction Mapping shape : {interaction_map.shape}")
else:
    print("Unable to parse the SMILES string.")


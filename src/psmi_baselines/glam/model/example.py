"""Implement the glam model example baseline module."""
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from .glam import GLAM, GLAMEnsemble, ConfigurationSpace


def example_single_graph():
    """Run the example single graph baseline operation."""
    print("=" * 50)
    print(" Single graph Architecture example - molecule Nature prediction ")
    print("=" * 50)
    
    # Configure the baseline model.
    node_dim = 9  # Baseline workflow step.
    out_dim = 1   # Configure the output artifacts.
    
    # Configure the baseline model.
    model = GLAM(
        node_dim=node_dim,
        out_dim=out_dim,
        task_type='property'
    )
    
    print(f" model configuration : {model.config}")
    
    # Process the experiment data.
    num_nodes = 10
    num_edges = 20
    x = torch.randn(num_nodes, node_dim)  # Baseline workflow step.
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Baseline workflow step.
    
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])
    
    # Baseline workflow step.
    model.eval()
    with torch.no_grad():
        output = model(
            batch.x, 
            batch.edge_index, 
            batch=batch.batch
        )
    
    print(f" input node count : {num_nodes}")
    print(f" input edge count : {num_edges}")
    print(f" output shape : {output.shape}")
    print(f" prediction : {output.item():.4f}")
    print()


def example_pair_graph():
    """Run the example pair graph baseline operation."""
    print("=" * 50)
    print(" Double graph Architecture example - molecule phase Interaction prediction ")
    print("=" * 50)
    
    # Configure the baseline model.
    node_dim = 9
    out_dim = 1  # Baseline workflow step.
    
    # Configure the baseline model.
    model = GLAM(
        node_dim=node_dim,
        out_dim=out_dim,
        task_type='interaction'
    )
    
    print(f" model configuration : {model.config}")
    
    # Process the experiment data.
    num_nodes1 = 8
    num_nodes2 = 12
    num_edges1 = 15
    num_edges2 = 20
    
    x1 = torch.randn(num_nodes1, node_dim)
    edge_index1 = torch.randint(0, num_nodes1, (2, num_edges1))
    
    x2 = torch.randn(num_nodes2, node_dim)
    edge_index2 = torch.randint(0, num_nodes2, (2, num_edges2))
    
    # Baseline workflow step.
    model.eval()
    with torch.no_grad():
        output = model(
            x1, edge_index1,
            x2, edge_index2
        )
    
    print(f" graph 1 node count : {num_nodes1}, edge count : {num_edges1}")
    print(f" graph 2 node count : {num_nodes2}, edge count : {num_edges2}")
    print(f" output shape : {output.shape}")
    print(f" prediction : {output.item():.4f}")
    print()


def example_ensemble():
    """Run the example ensemble baseline operation."""
    print("=" * 50)
    print(" Integration model example ")
    print("=" * 50)
    
    node_dim = 9
    out_dim = 1
    ensemble_size = 3
    
    # Configure the baseline model.
    ensemble_model = GLAMEnsemble(
        node_dim=node_dim,
        out_dim=out_dim,
        task_type='property',
        ensemble_size=ensemble_size
    )
    
    print(f" Integration model count : {ensemble_size}")
    for i, model in enumerate(ensemble_model.models):
        print(f" model {i+1} configuration : {model.config}")
    
    # Process the experiment data.
    num_nodes = 10
    x = torch.randn(num_nodes, node_dim)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])
    
    # Baseline workflow step.
    ensemble_model.eval()
    with torch.no_grad():
        output = ensemble_model(
            batch.x,
            batch.edge_index,
            batch=batch.batch
        )
    
    print(f" Integration prediction output shape : {output.shape}")
    print(f" Integration prediction : {output.item():.4f}")
    print()


def example_config_space():
    """Run the example config space baseline operation."""
    print("=" * 50)
    print(" configuration Space example ")
    print("=" * 50)
    
    config_space = ConfigurationSpace()
    
    # Baseline workflow step.
    print(" sampling 5 molecule Nature prediction configuration :")
    configs_property = config_space.sample_configs(5, task_type='property')
    for i, config in enumerate(configs_property):
        print(f" configuration {i+1}:")
        print(f" MP type : {config['mp_type']}")
        print(f" hidden dimension : {config['hidden_dim']}")
        print(f" MP number of layers : {config['num_mp_layers']}")
        print(f" learning rate : {config['learning_rate']}")
        print()
    
    print(" sampling 3 molecule phase Interaction prediction configuration :")
    configs_interaction = config_space.sample_configs(3, task_type='interaction')
    for i, config in enumerate(configs_interaction):
        print(f" configuration {i+1}:")
        print(f" MP type : {config['mp_type']}")
        print(f" Fusion type : {config['fusion_type']}")
        print(f" hidden dimension : {config['hidden_dim']}")
        print()


def example_training():
    """Run the example training baseline operation."""
    print("=" * 50)
    print(" Training example ")
    print("=" * 50)
    
    node_dim = 9
    out_dim = 1
    
    # Configure the baseline model.
    model = GLAM(node_dim=node_dim, out_dim=out_dim, task_type='property')
    
    # Process the experiment data.
    num_samples = 5
    data_list = []
    labels = []
    
    for _ in range(num_samples):
        num_nodes = torch.randint(5, 15, (1,)).item()
        num_edges = torch.randint(10, 30, (1,)).item()
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
        labels.append(torch.randn(1))
    
    batch = Batch.from_data_list(data_list)
    labels = torch.stack(labels)
    
    # Compute the training loss.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run the training step.
    model.train()
    optimizer.zero_grad()
    
    output = model(batch.x, batch.edge_index, batch=batch.batch)
    loss = criterion(output, labels)
    
    loss.backward()
    optimizer.step()
    
    print(f" training loss : {loss.item():.4f}")
    print(f" prediction output shape : {output.shape}")
    print(f" Real label shape : {labels.shape}")
    print()


if __name__ == "__main__":
    # Baseline workflow step.
    example_single_graph()
    example_pair_graph()
    example_ensemble()
    example_config_space()
    example_training()
    
    print("=" * 50)
    print(" all example run complete !")
    print("=" * 50)


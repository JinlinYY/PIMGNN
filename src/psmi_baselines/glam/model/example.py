"""Implement the glam model example baseline module."""
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from .glam import GLAM, GLAMEnsemble, ConfigurationSpace


def example_single_graph():
    """Run the example single graph baseline operation."""
    print("=" * 50)
    print("单图架构示例 - 分子性质预测")
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
    
    print(f"模型配置: {model.config}")
    
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
    
    print(f"输入节点数: {num_nodes}")
    print(f"输入边数: {num_edges}")
    print(f"输出形状: {output.shape}")
    print(f"预测值: {output.item():.4f}")
    print()


def example_pair_graph():
    """Run the example pair graph baseline operation."""
    print("=" * 50)
    print("双图架构示例 - 分子相互作用预测")
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
    
    print(f"模型配置: {model.config}")
    
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
    
    print(f"图1节点数: {num_nodes1}, 边数: {num_edges1}")
    print(f"图2节点数: {num_nodes2}, 边数: {num_edges2}")
    print(f"输出形状: {output.shape}")
    print(f"预测值: {output.item():.4f}")
    print()


def example_ensemble():
    """Run the example ensemble baseline operation."""
    print("=" * 50)
    print("集成模型示例")
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
    
    print(f"集成模型数量: {ensemble_size}")
    for i, model in enumerate(ensemble_model.models):
        print(f"模型 {i+1} 配置: {model.config}")
    
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
    
    print(f"集成预测输出形状: {output.shape}")
    print(f"集成预测值: {output.item():.4f}")
    print()


def example_config_space():
    """Run the example config space baseline operation."""
    print("=" * 50)
    print("配置空间示例")
    print("=" * 50)
    
    config_space = ConfigurationSpace()
    
    # Baseline workflow step.
    print("采样5个分子性质预测配置:")
    configs_property = config_space.sample_configs(5, task_type='property')
    for i, config in enumerate(configs_property):
        print(f"配置 {i+1}:")
        print(f"  MP类型: {config['mp_type']}")
        print(f"  隐藏维度: {config['hidden_dim']}")
        print(f"  MP层数: {config['num_mp_layers']}")
        print(f"  学习率: {config['learning_rate']}")
        print()
    
    print("采样3个分子相互作用预测配置:")
    configs_interaction = config_space.sample_configs(3, task_type='interaction')
    for i, config in enumerate(configs_interaction):
        print(f"配置 {i+1}:")
        print(f"  MP类型: {config['mp_type']}")
        print(f"  融合类型: {config['fusion_type']}")
        print(f"  隐藏维度: {config['hidden_dim']}")
        print()


def example_training():
    """Run the example training baseline operation."""
    print("=" * 50)
    print("训练示例")
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
    
    print(f"训练损失: {loss.item():.4f}")
    print(f"预测输出形状: {output.shape}")
    print(f"真实标签形状: {labels.shape}")
    print()


if __name__ == "__main__":
    # Baseline workflow step.
    example_single_graph()
    example_pair_graph()
    example_ensemble()
    example_config_space()
    example_training()
    
    print("=" * 50)
    print("所有示例运行完成！")
    print("=" * 50)


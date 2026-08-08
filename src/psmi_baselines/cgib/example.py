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

print(f"数据集大小: {len(dataset)}")
print(f"有效样本数: {len(dataset.valid_indices)}")

if len(dataset) > 0:
    # Baseline workflow step.
    sample_graph = dataset.graphs1[0]
    input_dim = sample_graph.x.size(1)
    print(f"输入特征维度: {input_dim}")
    
    # Configure the baseline model.
    model = CGIB(
        input_dim=input_dim,
        hidden_dim=32,
        output_dim=1,
        num_layers=2,
        beta=1e-3,
        gnn_type='mpnn'
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
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
        print(f"\n预测形状: {pred.shape}")
        print(f"预测值: {pred.squeeze().numpy()}")
        print(f"\n损失组件:")
        print(f"  MI1损失: {loss_components['mi1'].item():.4f}")
        print(f"  MI2损失: {loss_components['mi2'].item():.4f}")
        print(f"  Lambda均值: {loss_components['lambda'].mean().item():.4f}")
        print(f"  P均值: {loss_components['p'].mean().item():.4f}")
    
    print("\n模型测试成功！")
else:
    print("数据集为空，请检查SMILES字符串格式")


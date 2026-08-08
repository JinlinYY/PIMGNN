"""Implement the cignn evaluate baseline module."""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import CIGIN
from .data_utils import smiles_to_graph, batch_graphs


class LLEDataset(Dataset):
    """Represent the LLEDataset baseline component."""
    def __init__(self, il_smiles_list, comp2_smiles_list, comp3_smiles_list, 
                 labels, temperatures=None):
        self.il_smiles_list = il_smiles_list
        self.comp2_smiles_list = comp2_smiles_list
        self.comp3_smiles_list = comp3_smiles_list
        self.labels = labels
        self.temperatures = temperatures if temperatures is not None else [None] * len(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'il_smiles': self.il_smiles_list[idx],
            'comp2_smiles': self.comp2_smiles_list[idx],
            'comp3_smiles': self.comp3_smiles_list[idx],
            'label': self.labels[idx],
            'temperature': self.temperatures[idx]
        }


def collate_fn(batch):
    """Run the collate fn baseline operation."""
    il_graphs = []
    comp2_graphs = []
    comp3_graphs = []
    labels = []
    temperatures = []
    
    for item in batch:
        il_graph = smiles_to_graph(item['il_smiles'])
        comp2_graph = smiles_to_graph(item['comp2_smiles'])
        comp3_graph = smiles_to_graph(item['comp3_smiles'])
        
        if il_graph is not None and comp2_graph is not None and comp3_graph is not None:
            il_graphs.append(il_graph)
            comp2_graphs.append(comp2_graph)
            comp3_graphs.append(comp3_graph)
            labels.append(item['label'])
            temperatures.append(item['temperature'])
    
    if len(il_graphs) == 0:
        return None, None, None, None, None
    
    il_batch = batch_graphs(il_graphs)
    comp2_batch = batch_graphs(comp2_graphs)
    comp3_batch = batch_graphs(comp3_graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    if temperatures[0] is not None:
        temperatures = torch.tensor(temperatures, dtype=torch.float32)
    else:
        temperatures = None
    
    return il_batch, comp2_batch, comp3_batch, labels, temperatures


def evaluate_model(model, dataloader, device, save_plot=False, plot_path=None):
    """Run the evaluate model baseline operation."""
    model.eval()
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch[0] is None:
                continue
            
            il_batch, comp2_batch, comp3_batch, labels, temperatures = batch
            il_batch = il_batch.to(device)
            comp2_batch = comp2_batch.to(device)
            comp3_batch = comp3_batch.to(device)
            labels = labels.to(device)
            
            if temperatures is not None:
                temperatures = temperatures.to(device)
            
            predictions, _ = model(il_batch, comp2_batch, comp3_batch, temperatures)
            # predictions shape: [batch_size, 6]
            
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    predictions_array = np.array(predictions_list)  # [N, 6]
    labels_array = np.array(labels_list)  # [N, 6]
    
    # Configure the output artifacts.
    rmse = np.sqrt(np.mean((predictions_array - labels_array) ** 2))
    mae = np.mean(np.abs(predictions_array - labels_array))
    r2 = 1 - np.sum((labels_array - predictions_array) ** 2) / np.sum((labels_array - np.mean(labels_array, axis=0)) ** 2)
    
    # Configure the output artifacts.
    rmse_per_output = np.sqrt(np.mean((predictions_array - labels_array) ** 2, axis=0))
    mae_per_output = np.mean(np.abs(predictions_array - labels_array), axis=0)
    
    print(f"\n评估结果:")
    print(f"总体 RMSE: {rmse:.4f}")
    print(f"总体 MAE: {mae:.4f}")
    print(f"总体 R²: {r2:.4f}")
    print(f"\n各输出指标:")
    output_names = ['Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3']
    for i, name in enumerate(output_names):
        print(f"  {name}: RMSE={rmse_per_output[i]:.4f}, MAE={mae_per_output[i]:.4f}")
    
    # Configure the output artifacts.
    if save_plot and plot_path:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        output_names = ['Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3']
        
        for i, (ax, name) in enumerate(zip(axes, output_names)):
            ax.scatter(labels_array[:, i], predictions_array[:, i], alpha=0.5)
            min_val = min(labels_array[:, i].min(), predictions_array[:, i].min())
            max_val = max(labels_array[:, i].max(), predictions_array[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax.set_xlabel(f'Experimental {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name}\nRMSE: {rmse_per_output[i]:.4f}')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        print(f"\n预测图已保存到: {plot_path}")
    
    return {
        'predictions': predictions_array,
        'labels': labels_array,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'rmse_per_output': rmse_per_output,
        'mae_per_output': mae_per_output
    }


def load_csv_data(csv_path):
    """Run the load csv data baseline operation."""
    df = pd.read_csv(csv_path)
    
    il_smiles = df['IL (Component 1) full name SMILES'].tolist()
    comp2_smiles = df['Component 2 SMILES'].tolist()
    comp3_smiles = df['Component 3 SMILES'].tolist()
    
    labels = df[['Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3']].values.astype(np.float32)
    
    temperatures = df['T/K'].values.astype(np.float32)
    temperatures = (temperatures - 250.0) / 150.0  # Baseline workflow step.
    
    return il_smiles, comp2_smiles, comp3_smiles, labels, temperatures


def main():
    parser = argparse.ArgumentParser(description='Evaluate CIGIN model for LLE prediction')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test dataset file (CSV format)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num_mp_layers', type=int, default=3,
                       help='Number of message passing layers')
    parser.add_argument('--use_set2set', action='store_true',
                       help='Use Set2Set instead of sum pooling')
    parser.add_argument('--use_temperature', action='store_true', default=True,
                       help='Use temperature as additional input')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--save_plot', action='store_true',
                       help='Save prediction plot')
    parser.add_argument('--plot_path', type=str, default='prediction_plot.png',
                       help='Path to save prediction plot')
    
    args = parser.parse_args()
    
    # Load the input data.
    print("Loading data from CSV...")
    il_smiles, comp2_smiles, comp3_smiles, labels, temperatures = load_csv_data(args.data_path)
    
    test_temps = temperatures if args.use_temperature else None
    test_dataset = LLEDataset(il_smiles, comp2_smiles, comp3_smiles, labels, test_temps)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    # Configure the baseline model.
    node_dim = 33
    edge_dim = 10
    model = CIGIN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_mp_layers=args.num_mp_layers,
        use_set2set=args.use_set2set,
        use_temperature=args.use_temperature
    ).to(args.device)
    
    # Load the input data.
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Baseline workflow step.
    results = evaluate_model(model, test_loader, args.device, 
                           args.save_plot, args.plot_path)
    
    # Save the generated artifacts.
    results_path = args.model_path.replace('.pth', '_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'rmse': float(results['rmse']),
            'mae': float(results['mae']),
            'r2': float(results['r2']),
            'rmse_per_output': results.get('rmse_per_output', []).tolist() if 'rmse_per_output' in results else None,
            'mae_per_output': results.get('mae_per_output', []).tolist() if 'mae_per_output' in results else None
        }, f, indent=2)
    print(f"\n结果已保存到: {results_path}")


if __name__ == '__main__':
    main()


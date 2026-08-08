"""Implement the glam test baseline module."""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

from .dataset.data_loader import load_LLE_dataset, collate_fn
from .model.glam import GLAM_LLE
from .config import Config, default_config
from .train import LLEDataset, evaluate
from torch.utils.data import DataLoader
from psmi_baselines.paths import TOTAL_CSV


def load_model(model_path, device):
    """Run the load model baseline operation."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Baseline workflow step.
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = Config()
        # Baseline workflow step.
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
    else:
        config = default_config
    
    # Process the experiment data.
    # Process the experiment data.
    node_dim = config.model.node_dim if config.model.node_dim > 0 else 8
    out_dim = config.model.out_dim
    
    # Configure the baseline model.
    model_config = {
        'norm_type': config.model.norm_type,
        'activation': config.model.activation,
        'mp_type': config.model.mp_type,
        'pool_type': config.model.pool_type,
        'dropout': config.model.dropout,
        'num_mp_layers': config.model.num_mp_layers,
        'hidden_dim': config.model.hidden_dim,
        'fusion_type': config.model.fusion_type
    }
    
    # Configure the baseline model.
    model = GLAM_LLE(
        node_dim=node_dim,
        out_dim=out_dim,
        config=model_config
    ).to(device)
    
    # Load the input data.
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, checkpoint


def plot_predictions(predictions, labels, save_path=None):
    """Run the plot predictions baseline operation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    component_names = ['Component 1', 'Component 2', 'Component 3']
    
    for i in range(3):
        ax = axes[i]
        pred = predictions[:, i]
        true = labels[:, i]
        
        # Baseline workflow step.
        ax.scatter(true, pred, alpha=0.6, s=20)
        
        # Baseline workflow step.
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Baseline workflow step.
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        
        ax.set_xlabel('True Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{component_names[i]}\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" prediction for Ratio graph saved to : {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, save_path=None):
    """Run the plot training history baseline operation."""
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    val_r2s = [h['val_r2'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute the training loss.
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    ax1.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Baseline workflow step.
    ax2 = axes[1]
    ax2.plot(epochs, val_r2s, label='Val R²', marker='^', markersize=3, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('Validation R²', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" training history graph saved to : {save_path}")
    else:
        plt.show()
    
    plt.close()


def main(model_path=None, test_only=False):
    """Run the main baseline operation."""
    # Configure the runtime device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("GLAM model Test - LLE prediction ")
    print("=" * 80)
    print(f" use device : {device}")
    
    # Configure the baseline model.
    if model_path is None:
        model_path = os.path.join(default_config.model_save_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f" error : model file does not exist : {model_path}")
        print(" First run train.py Training model ")
        return
    
    # Load the input data.
    print(f"\n load model : {model_path}")
    model, config, checkpoint = load_model(model_path, device)
    
    print(f" model training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f" best validation loss : {checkpoint.get('val_loss', 'N/A'):.6f}")
    
    # Load the input data.
    print("\n load dataset ...")
    datasets = load_LLE_dataset(
        config.data.csv_path,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_state=config.data.random_state
    )
    
    # Load the input data.
    test_dataset = LLEDataset(datasets['test'])
    
    def custom_collate_fn(batch):
        """Run the custom collate fn baseline operation."""
        data_items = [item[0] for item in batch]
        labels = np.array([item[1] for item in batch])
        graph_batch = collate_fn(data_items)
        return graph_batch, labels
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.data.num_workers
    )
    
    # Configure the baseline model.
    print("\n Assessment model ...")
    criterion = nn.MSELoss()
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # Baseline workflow step.
    print("\n" + "=" * 80)
    print(" test-set results :")
    print("=" * 80)
    print(f" test loss : {test_metrics['loss']:.6f}")
    print(f" Test MSE: {test_metrics['mse']:.6f}")
    print(f" Test MAE: {test_metrics['mae']:.6f}")
    print(f" Test RMSE: {test_metrics['rmse']:.6f}")
    print(f" Test R²: {test_metrics['r2']:.6f}")
    
    print("\n Each component Detail metrics :")
    for comp_name, comp_metrics in test_metrics['component_metrics'].items():
        print(f"  {comp_name}:")
        print(f"    MSE: {comp_metrics['mse']:.6f}")
        print(f"    MAE: {comp_metrics['mae']:.6f}")
        print(f"    RMSE: {comp_metrics['rmse']:.6f}")
        print(f"    R²: {comp_metrics['r2']:.6f}")
    
    # Generate model predictions.
    print("\n generate prediction for Ratio graph ...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(config.result_dir, f'predictions_{timestamp}.png')
    plot_predictions(test_metrics['predictions'], test_metrics['labels'], plot_path)
    
    # Run the training step.
    if not test_only:
        # Load the input data.
        if 'train_history' in checkpoint:
            print("\n generate training-history plot ...")
            history_path = os.path.join(config.result_dir, f'training_history_{timestamp}.png')
            plot_training_history(checkpoint['train_history'], history_path)
        else:
            # Load the input data.
            if os.path.exists(config.result_dir):
                result_files = [f for f in os.listdir(config.result_dir) if f.startswith('results_') and f.endswith('.json')]
                if result_files:
                    latest_result = max(result_files, key=lambda x: os.path.getmtime(os.path.join(config.result_dir, x)))
                    result_file_path = os.path.join(config.result_dir, latest_result)
                    try:
                        with open(result_file_path, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                            if 'train_history' in result_data.get('results', {}):
                                print("\n generate training-history plot ...")
                                history_path = os.path.join(config.result_dir, f'training_history_{timestamp}.png')
                                plot_training_history(result_data['results']['train_history'], history_path)
                    except Exception as e:
                        print(f" unable to load training history : {e}")
    
    # Save the generated artifacts.
    test_results = {
        'model_path': model_path,
        'test_metrics': {
            'loss': test_metrics['loss'],
            'mse': test_metrics['mse'],
            'mae': test_metrics['mae'],
            'rmse': test_metrics['rmse'],
            'r2': test_metrics['r2'],
            'component_metrics': test_metrics['component_metrics']
        },
        'predictions': test_metrics['predictions'].tolist(),
        'labels': test_metrics['labels'].tolist()
    }
    
    result_path = os.path.join(config.result_dir, f'test_results_{timestamp}.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n Test results saved to : {result_path}")
    print("=" * 80)

def test_data_loading(csv_path=str(TOTAL_CSV)):
    """Smoke-test dataset loading / collation."""
    print("=" * 60)
    print("Testing data loading")
    print("=" * 60)
    print(f"csv: {csv_path}")
    datasets = load_LLE_dataset(csv_path, test_size=0.2, val_size=0.1, random_state=42)
    sample = datasets["train"]["data"][0]
    print(f"IL nodes: {sample['il_graph'].x.shape[0]}")
    print(f"node dim: {sample['il_graph'].x.shape[1]}")
    n = min(3, len(datasets["train"]["data"]))
    batch = collate_fn([datasets["train"]["data"][i] for i in range(n)])
    print(f"batch IL nodes: {batch['il_graph'].x.shape[0]}")
    print("data loading OK")
    return True


def run_model_test():
        import argparse
    
        parser = argparse.ArgumentParser(description=' Test GLAM model ')
        parser.add_argument('--model', type=str, default=None, help=' model file path ')
        parser.add_argument('--test-only', action='store_true', help=' Test Only , No plot training history ')
    
        args = parser.parse_args()
    
        main(model_path=args.model, test_only=args.test_only)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["model", "data"], default="model")
    parser.add_argument("--csv", default=str(TOTAL_CSV))
    _args = parser.parse_args()
    if _args.mode == "data":
        test_data_loading(_args.csv)
    else:
        run_model_test()

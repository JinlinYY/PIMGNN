"""Implement the cgib test_model baseline module."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
import os
import pandas as pd

from .models import CGIB
from .utils.data_loader import MolecularDataset, create_batch
from .train import set_seed, evaluate, compute_metrics, print_metrics, save_results
from psmi_baselines.paths import TOTAL_CSV


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Run the load model from checkpoint baseline operation."""
    print(f" loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist : {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Baseline workflow step.
    if 'args' in checkpoint:
        args_dict = checkpoint['args']
        # Baseline workflow step.
        args = argparse.Namespace(**args_dict)
    else:
        raise ValueError("Checkpoint in not found args parameters , unable to reconstruct the model ")
    
    print(f" model parameters :")
    print(f"    seed: {args.seed}")
    print(f"    hidden_dim: {args.hidden_dim}")
    print(f"    num_layers: {args.num_layers}")
    print(f"    set2set_steps: {args.set2set_steps}")
    print(f"    beta: {args.beta}")
    print(f"    gnn_type: {args.gnn_type}")
    print(f"    use_contrastive: {args.use_contrastive}")
    print(f"    constrain_output: {getattr(args, 'constrain_output', True)}")
    
    return checkpoint, args


def create_model_from_args(args, input_dim, output_dim=6, device='cuda'):
    """Run the create model from args baseline operation."""
    constrain_output = getattr(args, 'constrain_output', True)
    
    model = CGIB(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        beta=args.beta,
        gnn_type=args.gnn_type,
        use_contrastive=args.use_contrastive,
        set2set_steps=args.set2set_steps,
        constrain_output=constrain_output
    ).to(device)
    
    return model


def main():
    parser = argparse.ArgumentParser(description=' evaluate the test set with the trained model ')
    parser.add_argument('--checkpoint', type=str, default='seed_2024/seed_2024_best.pt', 
                       help=' model checkpoint path ( default : seed_2024/seed_2024_best.pt)')
    parser.add_argument('--data_path', type=str, default=str(TOTAL_CSV),
                       help='Input comparison CSV.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help=' device ')
    parser.add_argument('--batch_size', type=int, default=64, help=' batch size ')
    
    args_cmd = parser.parse_args()
    
    print("=" * 80)
    print(" model evaluation script ")
    print("=" * 80)
    
    # Load the input data.
    checkpoint, args_train = load_model_from_checkpoint(args_cmd.checkpoint, args_cmd.device)
    
    # Set the random seed.
    seed = args_train.seed
    set_seed(seed)
    print(f"\n random seed set to : {seed}")
    
    # Load the input data.
    print(f"\n loading data : {args_cmd.data_path}")
    df = pd.read_csv(args_cmd.data_path)
    
    # Process the experiment data.
    if 'IL (Component 1) full name SMILES' in df.columns:
        print(" detected total.csv format , True at convert ...")
        smiles1_list = []
        smiles2_list = []
        
        for idx, row in df.iterrows():
            il_smiles = str(row['IL (Component 1) full name SMILES']).strip()
            comp2_smiles = str(row['Component 2 SMILES']).strip()
            comp3_smiles = str(row['Component 3 SMILES']).strip()
            
            if pd.notna(comp3_smiles) and comp3_smiles != '' and comp3_smiles != 'nan':
                combined_smiles = f"{comp2_smiles}.{comp3_smiles}"
            else:
                combined_smiles = comp2_smiles
            
            smiles1_list.append(il_smiles)
            smiles2_list.append(combined_smiles)
        
        targets = df[['Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3']].values
    elif 'smiles1' in df.columns and 'smiles2' in df.columns:
        smiles1_list = df['smiles1'].tolist()
        smiles2_list = df['smiles2'].tolist()
        
        if 'Ex1' in df.columns:
            targets = df[['Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3']].values
        elif 'target' in df.columns:
            targets = np.array([eval(x) if isinstance(x, str) else x for x in df['target']])
            if targets.ndim == 1:
                targets = targets.reshape(-1, 1)
        else:
            raise ValueError(" data file must contain Ex1-Ex3, Rx1-Rx3 column or target column ")
    else:
        raise ValueError(" unsupported data-file format . must contain smiles1/smiles2 column or IL/Component column ")
    
    # Process the experiment data.
    dataset = MolecularDataset(smiles1_list, smiles2_list, targets)
    
    # Run the training step.
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f" total samples : {total_size}")
    print(f" number of training samples : {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    print(f" number of validation samples : {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    print(f" number of test samples : {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")
    
    # Baseline workflow step.
    sample_graph = dataset.graphs1[0]
    input_dim = sample_graph.x.size(1)
    output_dim = 6
    
    print(f"\n input dimension : {input_dim}")
    print(f" output dimension : {output_dim}")
    
    # Configure the baseline model.
    print("\n True at create model ...")
    model = create_model_from_args(args_train, input_dim, output_dim, args_cmd.device)
    
    # Load the input data.
    print(" True at load model checkpoint ...")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(" model checkpoint loaded !")
    
    # Load the input data.
    def collate_fn(batch):
        graphs1 = [item[0] for item in batch]
        graphs2 = [item[1] for item in batch]
        targets = [item[2] for item in batch]
        return create_batch(graphs1, graphs2, targets)
    
    test_loader = DataLoader(test_dataset, batch_size=args_cmd.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Evaluate the test subset.
    print("\n" + "=" * 80)
    print(" starting test-set evaluation ")
    print("=" * 80)
    
    test_results = evaluate(model, test_loader, args_cmd.device)
    
    print("\n[ test metrics ]")
    print_metrics(test_results['metrics'])
    
    # Save the generated artifacts.
    output_dir = os.path.dirname(args_cmd.checkpoint) if os.path.dirname(args_cmd.checkpoint) else '.'
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n saving Test results to : {results_dir}")
    save_results(test_results['predictions'], test_results['targets'], 'test', results_dir)
    
    # Save the generated artifacts.
    metrics_file = os.path.join(results_dir, 'test_metrics.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" test-set evaluation metrics \n")
        f.write("=" * 80 + "\n\n")
        f.write(f" model : {args_cmd.checkpoint}\n")
        f.write(f" dataset : {args_cmd.data_path}\n")
        f.write(f" number of test samples : {len(test_dataset)}\n")
        f.write(f" random seed : {seed}\n\n")
        
        f.write("【Overall】\n")
        f.write(f"  MAE: {test_results['metrics']['all']['mae_mean']:.6f}±{test_results['metrics']['all']['mae_std']:.6f}\n")
        f.write(f"  RMSE: {test_results['metrics']['all']['rmse_mean']:.6f}±{test_results['metrics']['all']['rmse_std']:.6f}\n")
        f.write(f"  R²: {test_results['metrics']['all']['r2']:.6f}\n\n")
        
        f.write("【E-phase】\n")
        f.write(f"  MAE: {test_results['metrics']['e_phase']['mae_mean']:.6f}±{test_results['metrics']['e_phase']['mae_std']:.6f}\n")
        f.write(f"  RMSE: {test_results['metrics']['e_phase']['rmse_mean']:.6f}±{test_results['metrics']['e_phase']['rmse_std']:.6f}\n")
        f.write(f"  R²: {test_results['metrics']['e_phase']['r2']:.6f}\n\n")
        
        f.write("【R-phase】\n")
        f.write(f"  MAE: {test_results['metrics']['r_phase']['mae_mean']:.6f}±{test_results['metrics']['r_phase']['mae_std']:.6f}\n")
        f.write(f"  RMSE: {test_results['metrics']['r_phase']['rmse_mean']:.6f}±{test_results['metrics']['r_phase']['rmse_std']:.6f}\n")
        f.write(f"  R²: {test_results['metrics']['r_phase']['r2']:.6f}\n")
    
    print(f" test metrics saved to : {metrics_file}")
    print("\n Test complete !")


if __name__ == '__main__':
    main()


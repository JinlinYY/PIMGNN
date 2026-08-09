import os
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

try:
    from ._bootstrap import add_src_to_path
except ImportError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from psmi_checkpoint_compat import config as C
from psmi_checkpoint_compat.data import GraphCache, collate_graph_batch
from psmi_checkpoint_compat.model import LLEGraphNet
from psmi_checkpoint_compat.paths import (
    BASE_TERNARY_CHECKPOINT,
    BIGSOLVDB_DATA,
    BIGSOLVDB_EXPERIMENT_ROOT,
)

class BigSolDataset(Dataset):
    def __init__(self, df, g_cache, target_col='y_true'):
        self.df = df.dropna(subset=['smiles1', target_col]).reset_index(drop=True)
        self.y_log = np.log10(self.df[target_col].values + 1e-12)
        self.g_cache = g_cache
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        def fetch_g(s):
            s_str = str(s)
            for attr in ['graphs', 'cache', '_graphs']:
                if hasattr(self.g_cache, attr):
                    d = getattr(self.g_cache, attr)
                    if s in d: return d[s]
                    if s_str in d: return d[s_str]
                    
                    if pd.isna(s) or s_str.lower() in ['-', 'nan', '', 'smiles']:
                        if '' in d: return d['']
                        if len(d) > 0: return list(d.values())[0]
            
            for attr in ['graphs', 'cache', '_graphs']:
                if hasattr(self.g_cache, attr):
                    d = getattr(self.g_cache, attr)
                    if len(d) > 0: return list(d.values())[0]
            return None

        g1 = fetch_g(row['smiles1'])
        g2 = fetch_g(row['smiles2'])
        
        g3 = g2  
        
        scalars = torch.tensor([row['T'] / 298.15 if 'T' in row else 1.0, 0.5], dtype=torch.float32) 
        
        return {'g1': g1, 'g2': g2, 'g3': g3, 'scalars': scalars, 
                'system_id': torch.tensor(idx, dtype=torch.long),
                'aug_swap12': torch.tensor(0), 'aug_swap13': torch.tensor(0), 'aug_swap23': torch.tensor(0)},\
               torch.tensor(self.y_log[idx], dtype=torch.float32)


def compatible_backbone_weights(checkpoint, model_state):
    """Select shape-compatible non-head tensors from a published checkpoint."""
    pretrained = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    return {
        key: value
        for key, value in pretrained.items()
        if isinstance(value, torch.Tensor)
        and key in model_state
        and value.shape == model_state[key].shape
        and "head" not in key
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune the checkpoint-compatible binary model on BigSolDB.")
    parser.add_argument("--data", type=Path, default=BIGSOLVDB_DATA)
    parser.add_argument("--pretrained", type=Path, default=BASE_TERNARY_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=BIGSOLVDB_EXPERIMENT_ROOT / "degenerate_predictions.csv",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    csv_path = str(args.data)
    epochs = args.epochs
    batch_size = args.batch_size
    
    print("Starting the PSMI degenerate-mixture transfer experiment...")
    df = pd.read_csv(csv_path)
    
    g_cache = GraphCache(add_hs=C.GRAPH_ADD_HS, use_gasteiger=C.GRAPH_USE_GASTEIGER, max_atoms=C.GRAPH_MAX_ATOMS)
    smiles_clean = [s for s in pd.concat([df["smiles1"], df["smiles2"]]).unique() if pd.notna(s) and str(s) != '-']
    g_cache.build_from_smiles(smiles_clean)
    
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    train_loader = DataLoader(BigSolDataset(train_df, g_cache), batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
    test_loader = DataLoader(BigSolDataset(test_df, g_cache), batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
    
    model = LLEGraphNet(gnn_hidden=C.GNN_HIDDEN, gnn_layers=C.GNN_LAYERS, mlp_hidden=C.GNN_HEAD_HIDDEN, is_binary=True).to(C.DEVICE)
    
    model_path = str(args.pretrained)
    if os.path.exists(model_path):
        print(f"Loading ternary pretrained checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=C.DEVICE, weights_only=True)
        model_dict = model.state_dict()
        transfer_dict = compatible_backbone_weights(checkpoint, model_dict)
        model_dict.update(transfer_dict)
        model.load_state_dict(model_dict)
        print(f"Transferred {len(transfer_dict)} compatible ternary-interaction layers.")
    else:
        print(f"Pretrained checkpoint not found: {model_path}")
        return
        
    head_params, base_params = [], []
    for name, param in model.named_parameters():
        if 'head' in name: head_params.append(param)
        else: base_params.append(param)
            
    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-5}, 
        {'params': head_params, 'lr': 1e-3}
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()
    
    print("\nDegenerate-composition fine-tuning started...")
    for epoch in range(epochs):
        model.train() 
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = {k: (v.to(C.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch_x.items()}
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y.to(C.DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{epochs} | Log-MSE Loss: {total_loss / train_size:.4f}")

    print("\n generating final predictions ...")
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = {k: (v.to(C.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch_x.items()}
            all_preds.extend(10**model(batch_x).cpu().numpy()) 
            all_trues.extend(10**batch_y.numpy())
            
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'y_true': all_trues, 'y_pred': all_preds}).to_csv(args.output, index=False)
    print(f"Fine-tuning complete. Predictions written to {args.output}")

if __name__ == "__main__":
    main()

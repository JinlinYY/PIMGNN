import os
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
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
    BIGSOLVDB_DATA,
    BIGSOLVDB_EXPERIMENT_ROOT,
    BIGSOLVDB_PRETRAINED_CHECKPOINT,
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
                    
                    if pd.isna(s) or s_str == '-' or s_str == 'nan':
                        if '' in d: return d['']
                        return list(d.values())[0]
            
            for attr in ['graphs', 'cache', '_graphs']:
                if hasattr(self.g_cache, attr):
                    return list(getattr(self.g_cache, attr).values())[0]
            return None

        g1 = fetch_g(row['smiles1']); g2 = fetch_g(row['smiles2']); g3 = fetch_g(row['smiles3'])
        scalars = torch.tensor([row['T'] / 298.15 if 'T' in row else 1.0, 0.5], dtype=torch.float32) 
        return {'g1': g1, 'g2': g2, 'g3': g3, 'scalars': scalars, 
                'system_id': torch.tensor(idx, dtype=torch.long),
                'aug_swap12': torch.tensor(0), 'aug_swap13': torch.tensor(0), 'aug_swap23': torch.tensor(0)},\
               torch.tensor(self.y_log[idx], dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Predict BigSolDB with the reference binary checkpoint.")
    parser.add_argument("--data", type=Path, default=BIGSOLVDB_DATA)
    parser.add_argument("--checkpoint", type=Path, default=BIGSOLVDB_PRETRAINED_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=BIGSOLVDB_EXPERIMENT_ROOT / "predictions.csv",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    csv_path = str(args.data)
    batch_size = args.batch_size
    
    print("Starting checkpoint-only BigSolDB prediction...")
    df = pd.read_csv(csv_path)
    g_cache = GraphCache(add_hs=C.GRAPH_ADD_HS, use_gasteiger=C.GRAPH_USE_GASTEIGER, max_atoms=C.GRAPH_MAX_ATOMS)
    
    smiles_all = pd.concat([df["smiles1"], df["smiles2"], df["smiles3"]]).unique()
    smiles_clean = [s for s in smiles_all if pd.notna(s) and str(s) != '-']
    print(" True at Rebuild molecule graph Cache ...")
    g_cache.build_from_smiles(smiles_clean)
    
    full_loader = DataLoader(BigSolDataset(df, g_cache), batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
    
    model = LLEGraphNet(gnn_hidden=C.GNN_HIDDEN, gnn_layers=C.GNN_LAYERS, mlp_hidden=C.GNN_HEAD_HIDDEN, is_binary=True).to(C.DEVICE)
    
    model_path = str(args.checkpoint)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=C.DEVICE, weights_only=True))
        print(f"Loaded checkpoint: {model_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        return
        
    print(" True at Sweep Tail generate results file , Never False ...")
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in full_loader:
            batch_x = {k: (v.to(C.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch_x.items()}
            all_preds.extend(10**model(batch_x).cpu().numpy()) 
            all_trues.extend(10**batch_y.numpy())
            
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'y_true': all_trues, 'y_pred': all_preds}).to_csv(args.output, index=False)
    print(f"Prediction complete. Results written to {args.output}")

if __name__ == "__main__":
    main()

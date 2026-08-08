import os
import argparse
from pathlib import Path
import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

try:
    from ._bootstrap import add_src_to_path
except ImportError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from psmi_legacy_public import config as C
from psmi_legacy_public.data import GraphCache, collate_graph_batch
from psmi_legacy_public.model import LLEGraphNet
from psmi_legacy_public.paths import COMPSOL_DATA, COMPSOL_EXPERIMENT_ROOT

# ==========================================
# Legacy public-release workflow step.
# ==========================================
class CompSolDataset(Dataset):
    def __init__(self, df, g_cache, target_col='value'):
        # Legacy public-release workflow step.
        self.df = df.dropna(subset=['smiles1', target_col]).reset_index(drop=True)
        self.g_cache = g_cache
        self.target_col = target_col
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        def fetch_g(s):
            s_str = str(s)
            # Legacy public-release workflow step.
            for attr in ['graphs', 'cache', '_graphs']:
                if hasattr(self.g_cache, attr):
                    d = getattr(self.g_cache, attr)
                    if s in d: return d[s]
                    if s_str in d: return d[s_str]
                    if pd.isna(s) or s_str in ['-', 'nan']:
                        if '' in d: return d['']
                        return list(d.values())[0]
            # Legacy public-release workflow step.
            for attr in ['graphs', 'cache', '_graphs']:
                if hasattr(self.g_cache, attr):
                    return list(getattr(self.g_cache, attr).values())[0]
            return None

        g1 = fetch_g(row['smiles1']); g2 = fetch_g(row['smiles2']); g3 = fetch_g(row['smiles3'])
        scalars = torch.tensor([row['T'] / 298.15 if 'T' in row else 1.0, 0.5], dtype=torch.float32) 
        
        y_true = torch.tensor(row[self.target_col], dtype=torch.float32)
        
        return {'g1': g1, 'g2': g2, 'g3': g3, 'scalars': scalars, 
                'system_id': torch.tensor(idx, dtype=torch.long),
                'aug_swap12': torch.tensor(0), 'aug_swap13': torch.tensor(0), 'aug_swap23': torch.tensor(0)},\
               y_true

# ==========================================
# Legacy public-release workflow step.
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train the archived CompSol binary model.")
    parser.add_argument("--data", type=Path, default=COMPSOL_DATA)
    parser.add_argument("--output-dir", type=Path, default=COMPSOL_EXPERIMENT_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    file_path = str(args.data)
    
    # Legacy public-release workflow step.
    epochs = 40           # Legacy public-release workflow step.
    batch_size = 128      # Legacy public-release workflow step.
    base_lr = 1e-3        # Legacy public-release workflow step.
    
    print(f"🚀 启动 CompSol 深度精调任务 (目标 {epochs} 轮)...")
    epochs = args.epochs
    batch_size = args.batch_size
    base_lr = args.learning_rate
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件 {file_path}")
        return

    print("正在加载 Excel 数据...")
    df = pd.read_excel(file_path)
    
    # Legacy public-release workflow step.
    g_cache = GraphCache(add_hs=C.GRAPH_ADD_HS, use_gasteiger=C.GRAPH_USE_GASTEIGER, max_atoms=C.GRAPH_MAX_ATOMS)
    smiles_clean = [s for s in pd.concat([df["smiles1"], df["smiles2"], df["smiles3"]]).unique() if pd.notna(s) and str(s) != '-']
    print(f"正在构建 {len(smiles_clean)} 个分子的缓存，请稍候...")
    g_cache.build_from_smiles(smiles_clean)
    
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    train_loader = DataLoader(CompSolDataset(train_df, g_cache), batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
    test_loader = DataLoader(CompSolDataset(test_df, g_cache), batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
    
    model = LLEGraphNet(gnn_hidden=C.GNN_HIDDEN, gnn_layers=C.GNN_LAYERS, mlp_hidden=C.GNN_HEAD_HIDDEN, is_binary=True).to(C.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    
    # Legacy public-release workflow step.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()
    
    print("\n🔥 训练正式开始...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = {k: (v.to(C.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch_x.items()}
            
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y.to(C.DEVICE))
            loss.backward()
            
            # Legacy public-release workflow step.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        avg_loss = total_loss / train_size
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} | MSE Loss: {avg_loss:.4f} | LR: {current_lr:.1e}")
        
        # Legacy public-release workflow step.
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), args.output_dir / f"checkpoint_epoch_{epoch+1}.pt")
            print(f"   [保存] 已生成检查点 CompSol_checkpoint_ep{epoch+1}.pt")

    print("\n正在生成最终预测结果...")
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = {k: (v.to(C.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch_x.items()}
            all_preds.extend(model(batch_x).cpu().numpy()) 
            all_trues.extend(batch_y.numpy())
            
    pd.DataFrame({'y_true': all_trues, 'y_pred': all_preds}).to_csv(
        args.output_dir / "predictions.csv", index=False
    )
    print("✅ 深度训练圆满完成！结果已更新至 CompSol_results.csv")

if __name__ == "__main__":
    main()

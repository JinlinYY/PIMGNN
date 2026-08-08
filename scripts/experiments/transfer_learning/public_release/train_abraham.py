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
from psmi_legacy_public.paths import ABRAHAM_DATA, ABRAHAM_EXPERIMENT_ROOT

class AbrahamDataset(Dataset):
    def __init__(self, df, g_cache, target_col):
        # Legacy public-release workflow step.
        self.df = df.dropna(subset=[target_col]).reset_index(drop=True)
        # Build molecular graph features.
        self.df = self.df[~self.df['smiles1'].astype(str).str.lower().isin(['smiles', '-', 'nan', ''])]
        self.df = self.df.reset_index(drop=True)
        
        self.g_cache = g_cache
        self.target_col = target_col
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        def fetch_g(s):
            s_str = str(s).strip()
            for attr in ['graphs', 'cache', '_graphs']:
                if hasattr(self.g_cache, attr):
                    d = getattr(self.g_cache, attr)
                    if s in d: return d[s]
                    if s_str in d: return d[s_str]
                    if pd.isna(s) or s_str.lower() in ['-', 'nan', 'smiles']:
                        if '' in d: return d['']
                        return list(d.values())[0]
            for attr in ['graphs', 'cache', '_graphs']:
                if hasattr(self.g_cache, attr):
                    return list(getattr(self.g_cache, attr).values())[0]
            return None

        g1 = fetch_g(row['smiles1'])
        g2 = fetch_g(row.get('smiles2', '-'))
        g3 = fetch_g(row.get('smiles3', '-'))
        
        scalars = torch.tensor([row['T'] / 298.15 if 'T' in row else 1.0, 0.5], dtype=torch.float32) 
        y_true = torch.tensor(row[self.target_col], dtype=torch.float32)
        
        return {'g1': g1, 'g2': g2, 'g3': g3, 'scalars': scalars, 
                'system_id': torch.tensor(idx, dtype=torch.long),
                'aug_swap12': torch.tensor(0), 'aug_swap13': torch.tensor(0), 'aug_swap23': torch.tensor(0)},\
               y_true

def main():
    parser = argparse.ArgumentParser(description="Train the archived Abraham binary head.")
    parser.add_argument("--data", type=Path, default=ABRAHAM_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=ABRAHAM_EXPERIMENT_ROOT / "predictions.csv",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    file_path = str(args.data)
    epochs = args.epochs
    batch_size = args.batch_size
    base_lr = args.learning_rate
    
    print(f"🚀 启动 Abraham 专项精调任务 (自动容错模式)...")
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件 {file_path}")
        return

    df = pd.read_excel(file_path)
    print(f"📊 当前 Excel 包含的列名有: {df.columns.tolist()}")
    
    # Legacy public-release workflow step.
    target_col = None
    possible_targets = ['value', 'L', 'S', 'A', 'B', 'V', 'E', 'y', 'target', 'exp']
    for pt in possible_targets:
        if pt in df.columns:
            target_col = pt
            break
            
    # Legacy public-release workflow step.
    if target_col is None:
        print("⚠️ 无法自动识别要预测的列！请将下方代码中的 `target_col = None` 修改为您想预测的列名 (例如 'L' 或 'S')。")
        target_col = df.columns[-1] # Legacy public-release workflow step.
        print(f"👉 暂时自动使用最后一列 [{target_col}] 作为预测目标。")
    else:
        print(f"🎯 成功锁定预测目标列: [{target_col}]")

    # Legacy public-release workflow step.
    g_cache = GraphCache(add_hs=C.GRAPH_ADD_HS, use_gasteiger=C.GRAPH_USE_GASTEIGER, max_atoms=C.GRAPH_MAX_ATOMS)
    smiles_all = pd.concat([df["smiles1"], df.get("smiles2", pd.Series()), df.get("smiles3", pd.Series())]).unique()
    smiles_clean = [s for s in smiles_all if pd.notna(s) and str(s).strip().lower() not in ['-', 'smiles', 'nan', '']]
    
    print(f"正在构建 {len(smiles_clean)} 个分子的图缓存...")
    g_cache.build_from_smiles(smiles_clean)
    
    dataset = AbrahamDataset(df, g_cache, target_col)
    train_size = int(0.8 * len(dataset))
    
    train_df = dataset.df.iloc[:train_size].copy()
    test_df = dataset.df.iloc[train_size:].copy()
    train_loader = DataLoader(AbrahamDataset(train_df, g_cache, target_col), batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
    test_loader = DataLoader(AbrahamDataset(test_df, g_cache, target_col), batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
    
    model = LLEGraphNet(gnn_hidden=C.GNN_HIDDEN, gnn_layers=C.GNN_LAYERS, mlp_hidden=C.GNN_HEAD_HIDDEN, is_binary=True).to(C.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print("\n🔥 训练正式开始...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        valid_samples = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = {k: (v.to(C.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch_x.items()}
            batch_y = batch_y.to(C.DEVICE)
            
            optimizer.zero_grad()
            preds = model(batch_x).squeeze()
            
            # Legacy public-release workflow step.
            valid_mask = batch_y > -100.0
            if valid_mask.sum() > 0:
                loss = nn.functional.mse_loss(preds[valid_mask], batch_y[valid_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * valid_mask.sum().item()
                valid_samples += valid_mask.sum().item()
            
        scheduler.step()
        avg_loss = total_loss / max(1, valid_samples)
        print(f"Epoch {epoch+1:02d}/{epochs} | Masked MSE: {avg_loss:.4f} | 有效样本: {valid_samples}")

    print("\n正在生成最终预测结果...")
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = {k: (v.to(C.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch_x.items()}
            preds = model(batch_x).cpu().numpy().flatten()
            trues = batch_y.numpy().flatten()
            
            all_preds.extend(preds)
            all_trues.extend(trues)
            
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'y_true': all_trues, 'y_pred': all_preds}).to_csv(args.output, index=False)
    print("✅ Abraham 深度训练圆满完成！结果已更新至 Abraham_results_new.csv")

if __name__ == "__main__":
    main()

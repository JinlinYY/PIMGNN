# -*- coding: utf-8 -*-
"""
main.py - 项目主入口

功能流程：
1. 加载并准备 Excel 数据
2. 按体系 ID 分割训练/验证/测试集
3. 训练或加载预训练模型
4. 在验证和测试集上评估模型
5. 对原始测试集进行点态预测
6. 生成奇偶图和三角相图

运行命令：
  python main.py

输出文件：
  - train/val/loss/metrics 曲线图
  - test_df_raw_pointwise_predictions.csv（预测结果）
  - parity_E.png / parity_R.png（奇偶图）
  - test_ternary_png/（三角相图）+ combined_ternary.pdf（合并PDF）
"""
import os
import json
# 设置 matplotlib 后端为 Agg（不依赖 X11）
os.environ.setdefault("MPLBACKEND", "Agg")

# 解决 Intel MKL 重复加载问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from torch.utils.data import DataLoader

import config as C
from utils import set_seed
from data import (
    load_and_prepare_excel, split_by_system, stratified_split_by_system,
    FingerprintCache, LLEDataset,
    FunctionalGroupCache,
    GraphCache, GraphLLEDataset, collate_graph_batch
)
from train import train_or_load
from metrics import evaluate_loader, print_metrics
from predict import predict_pointwise_df_raw
from viz import parity_plots, visualize_all_test_groups


def main():
    """
    主函数：完整的训练-评估-预测流程
    """
    # 设定全局随机种子，保证可复现
    set_seed(C.SEED)
    # 创建输出目录
    os.makedirs(C.OUT_DIR, exist_ok=True)

    # ============ 步骤 1：加载并准备数据 ============
    print("1) Load & prepare Excel ...")
    # 加载原始数据和增强后数据（包含2<->3 分量交换增强）
    df_raw, df_aug = load_and_prepare_excel(C.EXCEL_PATH, C.MIN_POINTS_PER_GROUP, C.PERMUTE_23_AUG)
    print("df_raw:", len(df_raw), "rows | systems:", df_raw["system_id"].nunique())
    print("df_aug:", len(df_aug), "rows | systems:", df_aug["system_id"].nunique())

    # ============ 步骤 2：数据集分割 ============
    print("2) Split by system_id (on df_aug) ...")
    # 使用分层分割以保证不同系统在各集合均匀分布
    # train_df, val_df, test_df = split_by_system(df_aug, train_ratio=0.8, val_ratio=0.1, seed=C.SEED)
    train_df, val_df, test_df = stratified_split_by_system(
        df_aug, train_ratio=0.8, val_ratio=0.1, seed=C.SEED, n_bins=8, min_bin_size=3
    )

    train_system_ids = set(train_df["system_id"].unique().tolist())
    val_system_ids = set(val_df["system_id"].unique().tolist())
    test_system_ids = set(test_df["system_id"].unique().tolist())
    print(
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)} | "
        f"train systems={len(train_system_ids)} val systems={len(val_system_ids)} test systems={len(test_system_ids)}"
    )

    # ============ 步骤 3：训练或加载模型 ============
    print("3) Train / load model ...")
    model, T_scaler, _history = train_or_load(train_df, val_df, test_df)

    # 构建功能基团（FG）缓存用于多尺度特征
    fg_cache = None
    if getattr(C, 'USE_FG', False):
        # 先尝试从模型获取 FG 词汇表
        corpus = getattr(model, 'fg_corpus', None)
        if (corpus is None) or (isinstance(corpus, list) and len(corpus) == 0):
            # 降级方案：从输出目录加载 FG 词汇表
            fg_path = os.path.join(C.OUT_DIR, 'fg_corpus.json')
            if os.path.exists(fg_path) and os.path.getsize(fg_path) > 0:
                try:
                    with open(fg_path, 'r', encoding='utf-8') as f:
                        corpus = json.load(f)
                except Exception:
                    corpus = None
        if corpus is not None:
            fg_cache = FunctionalGroupCache(corpus=corpus)

    # 根据模式选择构建相应的数据加载器
    if getattr(C, "USE_GRAPH", False):
        # 图模式：使用 RDKit 构建分子图 + GNN 编码
        print("  Using GRAPH mode")
        g_cache = GraphCache(
            add_hs=getattr(C, "GRAPH_ADD_HS", False),
            add_3d=getattr(C, "GRAPH_ADD_3D", False),
            use_gasteiger=getattr(C, "GRAPH_USE_GASTEIGER", True),
            max_atoms=getattr(C, "GRAPH_MAX_ATOMS", 256),
        )
        # 构建所有 SMILES 的分子图缓存
        smiles_all = pd.concat([val_df[["smiles1","smiles2","smiles3"]],
                                test_df[["smiles1","smiles2","smiles3"]]], axis=0).values.reshape(-1).tolist()
        g_cache.build_from_smiles(smiles_all)

        # 验证集加载器
        val_loader = DataLoader(
            GraphLLEDataset(val_df, T_scaler, g_cache, mix_cache=None, fg_cache=fg_cache, use_fg=getattr(C,'USE_FG',False), precompute_scalars=True),
            batch_size=getattr(C, "BATCH_SIZE_GRAPH", 64),
            shuffle=False,
            num_workers=0,
            pin_memory=C.DEVICE.startswith("cuda"),
            collate_fn=collate_graph_batch,
        )
        # 测试集加载器
        test_loader = DataLoader(
            GraphLLEDataset(test_df, T_scaler, g_cache, mix_cache=None, fg_cache=fg_cache, use_fg=getattr(C,'USE_FG',False), precompute_scalars=True),
            batch_size=getattr(C, "BATCH_SIZE_GRAPH", 64),
            shuffle=False,
            num_workers=0,
            pin_memory=C.DEVICE.startswith("cuda"),
            collate_fn=collate_graph_batch,
        )
    else:
        # 指纹模式：使用 Morgan 指纹 + MLP
        print("  Using Fingerprint mode")
        fp_cache = FingerprintCache()
        val_loader = DataLoader(
            LLEDataset(val_df, T_scaler, fp_cache, fg_cache=fg_cache, use_fg=getattr(C,'USE_FG',False)),
            batch_size=C.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            LLEDataset(test_df, T_scaler, fp_cache, fg_cache=fg_cache, use_fg=getattr(C,'USE_FG',False)),
            batch_size=C.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

    # ============ 步骤 4：评估最佳模型 ============
    val_m = evaluate_loader(model, val_loader, C.DEVICE)
    test_m = evaluate_loader(model, test_loader, C.DEVICE)
    print("\nFinal metrics (best-by-val model):")
    print_metrics("  Val :", val_m)
    print_metrics("  Test:", test_m)

    # ============ 步骤 5：点态预测 ============
    print("\n4) Test pointwise predictions on df_raw (no augmentation) ...")
    # 从原始数据中提取测试集样本进行预测（不使用增强数据）
    df_raw_test = df_raw[df_raw["system_id"].isin(test_system_ids)].copy()
    df_pred = predict_pointwise_df_raw(model, T_scaler, df_raw_test)

    # 保存预测结果
    pred_csv = os.path.join(C.OUT_DIR, "test_df_raw_pointwise_predictions.csv")
    df_pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print("Saved test predictions CSV:", pred_csv)

    # ============ 步骤 6：生成奇偶图 ============
    print("5) Parity plots ...")
    parity_plots(df_pred, C.OUT_DIR)
    print("Saved parity plots: parity_E.png / parity_R.png")

    # ============ 步骤 7：生成三角相图 ============
    print("6) Visualize ALL test groups ternary + PDF ...")
    visualize_all_test_groups(model, T_scaler, df_raw, test_system_ids, df_pred, C.OUT_DIR)

    print("\nDONE. Everything is in:", C.OUT_DIR)


if __name__ == "__main__":
    # 程序入口
    main()

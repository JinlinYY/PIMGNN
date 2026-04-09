# -*- coding: utf-8 -*-

import os
import torch

# -------------------------
# 路径
# -------------------------
# EXCEL_PATH = r"D:\GGNN\YXFL-github\data_update\update-LLE-all-with-smiles_min3.xlsx"
# OUT_DIR = r"./lle_run_aichej_fineturned"

EXCEL_PATH = r"D:\GGNN\YXFL-github\data_update\LLE-literature-data-boosted.xlsx"
OUT_DIR = r"./lle_run_literature_test"

# ======================================
# 可复现性配置
# ======================================

SEED = 42

# ======================================
# 数据过滤和增强配置
# ======================================

MIN_POINTS_PER_GROUP = 6
PERMUTE_23_AUG = True

# ======================================
# 指纹基线配置（向后兼容）
# ======================================

FP_RADIUS = 2
FP_BITS = 2048

# ======================================
# 主模式开关
# ======================================

USE_GRAPH = True

# ======================================
# 混合物图编码开关
# ======================================

USE_MIX_GRAPH = True

# 混合物图构建（基于3D几何）
MIX_USE_3D = True
MIX_ADD_HS_3D = True
MIX_NUM_ORIENT = 4
MIX_CONTACT_CUTOFF = 4.5
MIX_ELEC_CUTOFF = 8.0
MIX_HB_CUTOFF = 3.4
MIX_XB_CUTOFF = 3.6
MIX_PI_CUTOFF = 5.5
MIX_EDGE_MIN_CONTACTS = 1
MIX_ELEC_KEEP_THRESH = 0.05
MIX_PACKING_BUFFER = 0.8

# 混合物图编码参数
MIX_LAYERS = 3
MIX_HIDDEN = 256
MIX_DROPOUT = 0.10

# 图构建参数
GRAPH_ADD_HS = False
GRAPH_ADD_3D = False
GRAPH_USE_GASTEIGER = True
GRAPH_MAX_ATOMS = 256

# -------------------------
# 模型
# -------------------------

# 指纹 MLP
HIDDEN = 1024
DROPOUT = 0.15

# GNN
GNN_HIDDEN = 256
GNN_LAYERS = 3
GNN_POOL = "mean"
GNN_INTERACTION = True
GNN_HEAD_HIDDEN = 512

# -------------------------
# 训练
# -------------------------

BATCH_SIZE = 1024
BATCH_SIZE_GRAPH = 256
EPOCHS = 200
LR = 5e-5
WEIGHT_DECAY = 1e-5
USE_AMP = True
GRAD_CLIP = 1.0

# 损失开关：False -> 先用纯MSE预训练；True -> 启用 MechanisticNRTLLoss（物理约束微调）
USE_MECH_LOSS = False

# 第二阶段微调时是否冻结主干网络（只训练输出头）
FREEZE_BACKBONE = False

# NRTL 参数文件路径（物理约束损失中使用）
NRTL_PARAMS_PATH = r"/root/autodl-tmp/YXFL/nrtl_param/nrtl_params_all.json"

# MechanisticNRTLLoss 超参数
LAMBDA_PHY = 1e-3  # 物理损失权重系数（机理约束的强度；TPD约束需要更大权重；改为1e-2增强优化信号）
WARMUP_EPOCHS = 0  # 预热阶段epoch数（物理损失权重从0开始）
RAMP_EPOCHS = 5  # 斜坡阶段epoch数（物理损失权重从0线性增加到LAMBDA_PHY）
ROBUST_DELTA = 5.0  # Huber loss delta参数（鲁棒性阈值）
TAU_CLIP = 10.0  # NRTL模型tau参数的裁剪上界
LN_GAMMA_CLIP = 20.0  # NRTL模型ln(gamma)的裁剪上界
MECH_USE_KELVIN = None  # 温度单位控制（None自动判断，True摄氏转K，False不转换）
MECH_W_EQ = 1.0  # 化学势平衡残差权重（主物理约束项）
MECH_W_GD = 0.0  # Gibbs-Duhem约束权重（热力学一致性）
MECH_W_STAB = 1.0  # TPD稳定性约束权重（防止不稳定相）
MECH_GD_N_DIR = 2  # GD约束的随机方向采样数量（越大越严格）
MECH_GD_EPS = 1e-4  # GD约束有限差分步长
MECH_STAB_N_TRIAL = 64  # TPD稳定性随机采样次数（越大越严格，减少波动；改为64减少蒙特卡洛方差）
MECH_STAB_SIGMA = 0.05  # TPD扰动幅度（组成噪声标准差）
MECH_STAB_MARGIN = 0.05  # TPD安全边际（0严格，正值保守；改为0.05加强梯度信号并保持保守）

EVAL_EVERY = 1  # 每隔多少epoch在验证测试集上评估一次（每个epoch评估以捕捉TPD变化）
PLOT_EVERY = 5  # 每隔多少epoch更新并保存训练曲线图
SAVE_EVERY = 10  # 周期性保存间隔（当前训练逻辑已停用）

# ======================================
# 早停（Early Stopping）配置
# ======================================

USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 50
EARLY_STOP_METRIC = "r2"
EARLY_STOP_MIN_DELTA = 1e-5

# ---- 物理微调阶段配置 ----
USE_PHYSICS_FINETUNE = False
FINETUNE_EARLY_STOP_METRIC = "mu_res_mae"  # 监控化学势残差MAE（化学势损失项，越小越好）
FINETUNE_PATIENCE = 30

# 物理微调阶段默认从预训练权重启动（建议指向第一阶段 pure-MSE 的 best_model.pt）
FINETUNE_PRETRAIN_CKPT_PATH = r"./lle_run_aichej/best_model.pt"

NUM_WORKERS = min(16, os.cpu_count() or 8)
NUM_WORKERS_GRAPH = 0
PREFETCH_FACTOR = 4

# ---- 混合物图缓存大小 ----
MIX_TRIPLE_CACHE_SIZE = 4096

# 训练入口统一读取 LOAD_CKPT_PATH；开启物理微调时自动使用预训练模型路径
LOAD_CKPT_PATH = FINETUNE_PRETRAIN_CKPT_PATH if USE_PHYSICS_FINETUNE else ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 预测 / 可视化
# -------------------------

PRED_BATCH_SIZE = 2048
PRED_BATCH_SIZE_GRAPH = 128

# ======================================
# 可视化参数
# ======================================

N_SWEEP = 80
DRAW_TIELINES_MAX = 14

# ======================================
# 功能基团（FG）多尺度特征 - 消融实验配置
# ======================================

USE_FG = True

FG_PRESET = "F3"

# FG 特征词表配置
FG_TOPK = 512
FG_MIN_FREQ = 3

# FG 编码器配置
FG_MLP_HIDDEN = 256
FG_DROPOUT = 0.10

PRECOMPUTE_FG = True

# FG Token 模式配置
FG_TOKEN_MODE = True
FG_MAX_TOKENS = 32

# FG 跨分子注意力配置
FG_CROSS_ATTN = True
FG_ATTN_HEADS = 8

# ======================================
# 三分子特征融合策略
# ======================================

S3_EQUIVARIANT = True

FUSION_MODE = "tf"

# Transformer 超参数（图模式）
TF_DIM = GNN_HIDDEN
TF_LAYERS = 2
TF_HEADS = 8
TF_FF = 1024
TF_DROPOUT = 0.15
TF_POOL = "cls"
TF_MAX_LEN = 32
TF_TYPE_VOCAB = 16

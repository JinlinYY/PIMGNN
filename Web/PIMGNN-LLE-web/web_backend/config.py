# web_backend/config.py
import os

# 设备配置
DEVICE = "cpu"  # 或 "cuda" 如果有GPU

# 模型路径（指向原项目的模型权重）
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "default")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

# 其他配置
USE_GRAPH = True  # 使用图模式
FP_BITS = 2048
GNN_HIDDEN = 256
GNN_LAYERS = 4
GNN_HEAD_HIDDEN = 512
DROPOUT = 0.15
GNN_POOL = "mean"
GNN_INTERACTION = True
USE_MIX_GRAPH = False
MIX_LAYERS = 2
MIX_HIDDEN = 256
MIX_DROPOUT = 0.10
USE_FG = False
FG_TOPK = 0
FG_MLP_HIDDEN = 256
FG_DROPOUT = 0.10
FG_TOKEN_MODE = False
FG_MAX_TOKENS = 32
FG_CROSS_ATTN = False
FG_ATTN_HEADS = 8
S3_EQUIVARIANT = False
FUSION_MODE = "concat"
TF_DIM = 256
TF_LAYERS = 2
TF_HEADS = 8
TF_FF = 1024
TF_DROPOUT = 0.10
TF_POOL = "cls"
TF_MAX_LEN = 32
TF_TYPE_VOCAB = 16

# Explainability CSV directory (precomputed saliency outputs)
EXPLAIN_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "explainability",
    "explain_test_saliency_20260119-123603",
)
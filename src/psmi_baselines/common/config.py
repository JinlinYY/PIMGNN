# -*- coding: utf-8 -*-
"""
Project config (single source of truth).
Edit EXCEL_PATH / OUT_DIR here (or override via CLI in main.py).
"""

from pathlib import Path

import torch

from psmi_baselines.paths import EXPERIMENT_ROOT, PROJECT_ROOT

# -------------------------
# Paths
# -------------------------
EXCEL_PATH = str(
    PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx"
)
SPLIT_MANIFEST_PATH = str(
    PROJECT_ROOT / "datasets" / "splits" / "main_benchmark_corrected_v2.json"
)
OUT_DIR = str(EXPERIMENT_ROOT / "runs" / "classical")

# -------------------------
# Reproducibility
# -------------------------
SEED = 42

# -------------------------
# Data rules
# -------------------------
MIN_POINTS_PER_GROUP = 6   # each (system_id, T) must have >= this many tie-lines
PERMUTE_23_AUG = True      # training augmentation by swapping component-2/3 (test visualization uses df_raw)

# -------------------------
# Checkpoint
# -------------------------
LOAD_CKPT_PATH = ""        # if provided and exists, skip training and load this ckpt

# -------------------------
# Model & training
# -------------------------
FP_BITS = 2048
FP_RADIUS = 2

HIDDEN = 1024
DROPOUT = 0.15

BATCH_SIZE = 1024
EPOCHS = 300
LR = 2e-4
WEIGHT_DECAY = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Visualization
# -------------------------
N_SWEEP = 80
DRAW_TIELINES_MAX = 14     # max tie-lines drawn per ternary plot (avoid clutter)
# -------------------------
# Baseline comparison
# -------------------------
# Torch model name used by main.py / train.py (default keeps your original behavior)
MODEL_NAME = "ann"

# If you want to run multiple models for comparison, use main_compare.py
# Available:
#   torch: "mlp", "ann", "lstm", "transformer", "tabknet", "smiles_rnn", "gnn"
#   sklearn: "xgboost", "random_forest", "tabnet" (requires pytorch-tabnet)
MODELS_TO_RUN = [
    "mlp",
    "ann",
    "lstm",
    "transformer",
    "tabknet",
    "smiles_rnn",
    "gnn",
    "random_forest",
    "tabnet",
    "xgboost",
]

# Auto-load OUT_DIR/<model_name>.pt if exists (useful for repeated runs)
AUTO_LOAD_IF_EXISTS = True

# Visualization control (viz.visualize_all_test_groups)
SAVE_TERNARY_PDF = False          # set True if you also want the multi-page PDF
COMPARE_DRAW_TERNARY = False      # set True to generate ternary plots for *each* model (slow)

# -------------------------
# SMILES RNN (no fingerprints)
# -------------------------
SMILES_MAX_LEN = 256      # max total tokens after concatenating 3 SMILES with separators
SMILES_EMB_DIM = 256      # embedding dim for SMILES chars
SMILES_HIDDEN = 384       # RNN hidden size
SMILES_LAYERS = 2         # RNN layers
SMILES_DROPOUT = 0.15     # dropout for embeddings/RNN/MLP head
SMILES_USE_T = True       # include normalized temperature token
SMILES_USE_TIE_T = True   # include tie-line parameter t token

# Per-component GCN baseline
GNN_NODE_DIM = 11
GNN_HIDDEN = 256
GNN_LAYERS = 3
GNN_DROPOUT = 0.10
GNN_MLP = 256
GNN_SCALAR_DIM = 2

# -*- coding: utf-8 -*-
"""Define reproducible paths and runtime settings for PSMI experiments."""


import os
from pathlib import Path
try:
    import torch
except ModuleNotFoundError:  # Paths and metadata remain inspectable before setup.
    torch = None

# Repository paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

EXCEL_PATH = str(DATASETS_DIR / "processed" / "LLE-literature-data-boosted.xlsx")
OUT_DIR = str(EXPERIMENTS_DIR / "07_external_validation" / "runs" / "literature_reproduction")

# Reproducibility
SEED = 42

# Dataset filtering and augmentation
MIN_POINTS_PER_GROUP = 6
PERMUTE_23_AUG = True
SPLIT_STRATEGY = "random"
SPLIT_MANIFEST_PATH = ""
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
STRATIFIED_N_BINS = 3
STRATIFIED_MIN_BIN_SIZE = 5

# Temperature feature basis. ``linear_quadratic`` preserves the original
# implementation; ``inverse`` uses a Gibbs--Helmholtz-motivated reciprocal
# basis while retaining the same two mixture-edge feature dimensions.
TEMPERATURE_ENCODING = "linear_quadratic"
TEMPERATURE_REFERENCE_K = 500.0

PRECOMPUTE_SCALARS = True
PRECOMPUTE_FEATURES = True
# Benchmark checkpoints use [temperature, phase path]; expanded-data checkpoints
# may opt into pressure as a third scalar through an explicit profile.
SCALAR_DIM = 3

# Fingerprint baseline
FP_RADIUS = 2
FP_BITS = 2048

# Molecular and mixture graph representation
USE_GRAPH = True
USE_MIX_GRAPH = True
MIX_HB_CUTOFF = 3.4
MIX_XB_CUTOFF = 3.6
MIX_PI_CUTOFF = 5.5
MIX_EDGE_MIN_CONTACTS = 1
MIX_ELEC_KEEP_THRESH = 0.05
MIX_PACKING_BUFFER = 0.8

# Mixture-graph encoder
MIX_LAYERS = 2
MIX_HIDDEN = 256
MIX_DROPOUT = 0.30
# Mixture graph batches store nodes sample by sample. Historical checkpoints
# were trained with component-major embeddings; use ``legacy_component_major``
# only when reproducing those archived results.
MIXTURE_NODE_LAYOUT = "sample_major"

# Molecular graph construction
GRAPH_ADD_HS = False
GRAPH_ADD_3D = False
GRAPH_USE_GASTEIGER = True
GRAPH_MAX_ATOMS = 256

# Model dimensions
HIDDEN = 1024
DROPOUT = 0.25

# GNN
GNN_HIDDEN = 256
GNN_LAYERS = 4
GNN_POOL = "mean"
GNN_INTERACTION = True
GNN_HEAD_HIDDEN = 512

# Training hyperparameters
NUM_WORKERS_GRAPH = 4  
PREFETCH_FACTOR = 4    

BATCH_SIZE = 1024
BATCH_SIZE_GRAPH = 256 
EPOCHS = 200
LR = 5e-6
WEIGHT_DECAY = 1e-3
USE_AMP = True
GRAD_CLIP = 1.0

# Enable the mechanistic loss only for documented physics-informed runs.
USE_MECH_LOSS = False

# Freeze non-output layers during a head-only fine-tuning stage.
FREEZE_BACKBONE = False

# Thermodynamic parameter sources. The training store is deliberately distinct
# from the all-system store used only after checkpoint selection.
NRTL_TRAIN_PARAMS_PATH = str(DATASETS_DIR / "parameters" / "nrtl_params_train.json")
NRTL_EVAL_PARAMS_PATH = str(DATASETS_DIR / "parameters" / "nrtl_params_all.json")
# Supervised transfer runs may disable unrelated post-hoc NRTL diagnostics.
COMPUTE_FINAL_PHYSICS_METRICS = True
GE_MODEL = "nrtl"

# Mechanistic loss weights and numerical safeguards
LAMBDA_PHY = 1e-3  
WARMUP_EPOCHS = 0  
RAMP_EPOCHS = 5  
ROBUST_DELTA = 5.0  
TAU_CLIP = 10.0  
LN_GAMMA_CLIP = 20.0  
MECH_USE_KELVIN = None  
MECH_W_EQ = 1.0  
MECH_W_GD = 0.0  
MECH_W_STAB = 0.0  
MECH_GD_N_DIR = 2  
MECH_GD_EPS = 1e-4  
MECH_STAB_N_TRIAL = 64  
MECH_STAB_SIGMA = 0.05  
MECH_STAB_MARGIN = 0.05  

EVAL_EVERY = 1  
PLOT_EVERY = 5  
# Keep expensive per-system phase-diagram rendering separate from training runs.
GENERATE_PHASE_DIAGRAMS = True
EVALUATE_TEST_DURING_TRAINING = False

# Validation-based early stopping
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 15
EARLY_STOP_METRIC = "rmse"
EARLY_STOP_MIN_DELTA = 1e-4

# Physics-stage early stopping
USE_PHYSICS_FINETUNE = False
FINETUNE_EARLY_STOP_METRIC = "rmse"
FINETUNE_PATIENCE = 30

NUM_WORKERS = min(16, os.cpu_count() or 8)
NUM_WORKERS_GRAPH = 0
PREFETCH_FACTOR = 4

# Mixture-graph cache
MIX_TRIPLE_CACHE_SIZE = 4096

# Optional checkpoint used to resume training or run evaluation.
LOAD_CKPT_PATH = str(
    MODELS_DIR
    / "06_transfer_learning"
    / "public_release"
    / "base_ternary"
    / "best_model.pt"
)
# LOAD_CKPT_PATH = ""
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

# Prediction and visualization
PRED_BATCH_SIZE_GRAPH = 128


N_SWEEP = 80
DRAW_TIELINES_MAX = 14

# Functional-group representation
USE_FG = True

# Functional-group vocabulary
FG_TOPK = 512
FG_MIN_FREQ = 3

# Functional-group encoder
FG_MLP_HIDDEN = 256
FG_DROPOUT = 0.20

PRECOMPUTE_FG = True

# Functional-group tokenization
FG_TOKEN_MODE = True
FG_MAX_TOKENS = 32

# Cross-molecule functional-group attention
FG_CROSS_ATTN = True
FG_ATTN_HEADS = 8

# Permutation-aware component fusion
# Current name: enables only the S3-aware component embedding, not full-model
# output equivariance. The legacy alias remains for historical profiles.
USE_S3_COMPONENT_EMBEDDING = None
S3_EQUIVARIANT = True

# Saved main-model checkpoints used multiscale concatenation. Historical run
# names used the ambiguous label ``tf`` even though their state dictionaries
# contain no Transformer module.
FUSION_MODE = "concat"

# Transformer fusion settings
TF_DIM = GNN_HIDDEN
TF_LAYERS = 2
TF_HEADS = 8
TF_FF = 1024
TF_DROPOUT = 0.35
TF_POOL = "cls"
TF_MAX_LEN = 32
TF_TYPE_VOCAB = 16

# Expanded-dataset fine-tuning
USE_FINE_TUNE = False
FINE_TUNE_EXCEL_PATH = str(
    DATASETS_DIR / "processed" / "LLE-literature-data-case-mul-去掉IL.xlsx"
)
PRETRAINED_MODEL_PATH = LOAD_CKPT_PATH
FINE_TUNE_LR = 2e-5
FINE_TUNE_EPOCHS = 200

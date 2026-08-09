"""Centralize repository paths for all comparison-model implementations."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "baseline_comparison"
TOTAL_CSV = DATA_DIR / "total.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "baselines"
MODEL_ROOT = EXPERIMENT_ROOT / "models"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"

BIGSOLVDB_CSV = PROJECT_ROOT / "datasets" / "external" / "BigSolDBv2.0.csv"
BIGSOLVDB_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "outputs" / "binary_solubility" / "bigsolvdb"
)

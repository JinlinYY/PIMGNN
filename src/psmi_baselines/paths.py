"""Centralize repository paths for all comparison-model implementations."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "baseline_comparison"
TOTAL_CSV = DATA_DIR / "total.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "01_baselines" / "comparative_models"
MODEL_ROOT = PROJECT_ROOT / "models" / "01_baselines" / "comparative_models"
FIGURE_ROOT = PROJECT_ROOT / "figures" / "01_baselines" / "comparative_models"

BIGSOLVDB_CSV = PROJECT_ROOT / "datasets" / "external" / "BigSolDBv2.0.csv"
BIGSOLVDB_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "experiments" / "07_external_validation" / "bigsolvdb"
)

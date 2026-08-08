"""Repository paths for the historical public code-and-weights release."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_ROOT = PROJECT_ROOT / "datasets" / "external"

ABRAHAM_DATA = EXTERNAL_DATA_ROOT / "abraham" / "Abraham_Ready.xlsx"
COMPSOL_DATA = EXTERNAL_DATA_ROOT / "compsol" / "CompSol_Ready.xlsx"
FREESOLV_DATA = EXTERNAL_DATA_ROOT / "freesolv" / "FreeSolv_Ready.xlsx"
BIGSOLVDB_DATA = EXTERNAL_DATA_ROOT / "bigsolvdb" / "BigSolDB_ready.csv"
BIGSOLVDB_TRAIN_DATA = EXTERNAL_DATA_ROOT / "bigsolvdb" / "train_BigSolDB_pseudo_ternary.csv"
BIGSOLVDB_TEST_DATA = EXTERNAL_DATA_ROOT / "bigsolvdb" / "test_BigSolDB_pseudo_ternary.csv"

BASE_TERNARY_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "06_transfer_learning"
    / "public_release"
    / "base_ternary"
    / "best_model.pt"
)
COMPSOL_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "06_transfer_learning"
    / "public_release"
    / "compsol"
    / "checkpoint_epoch_40.pt"
)
BIGSOLVDB_TEMP_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "07_external_validation"
    / "bigsolvdb"
    / "public_release"
    / "bigsoldb_temp_model.pt"
)
BIGSOLVDB_FINETUNED_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "07_external_validation"
    / "bigsolvdb"
    / "public_release"
    / "binary_finetuned_model.pt"
)

ABRAHAM_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "experiments" / "06_transfer_learning" / "public_release" / "abraham"
)
COMPSOL_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "experiments" / "06_transfer_learning" / "public_release" / "compsol"
)
BIGSOLVDB_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "07_external_validation"
    / "bigsolvdb"
    / "public_release"
)

NRTL_PARAMETER_FILE = PROJECT_ROOT / "datasets" / "parameters" / "nrtl_params_all.json"

"""Repository paths for the binary-solubility validation package."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_ROOT = PROJECT_ROOT / "datasets" / "external"

ABRAHAM_DATA = EXTERNAL_DATA_ROOT / "abraham" / "Abraham_Ready.xlsx"
COMPSOL_DATA = EXTERNAL_DATA_ROOT / "compsol" / "CompSol_Ready.xlsx"
FREESOLV_DATA = EXTERNAL_DATA_ROOT / "freesolv" / "FreeSolv_Ready.xlsx"
BIGSOLVDB_DATA = EXTERNAL_DATA_ROOT / "bigsolvdb" / "BigSolDB_ready.csv"
BIGSOLVDB_TRAIN_DATA = EXTERNAL_DATA_ROOT / "bigsolvdb" / "train_BigSolDB_pseudo_ternary.csv"
BIGSOLVDB_TEST_DATA = EXTERNAL_DATA_ROOT / "bigsolvdb" / "test_BigSolDB_pseudo_ternary.csv"

BINARY_VALIDATION_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "section_3_results"
    / "3_3_binary_solubility_validation"
)

BASE_TERNARY_CHECKPOINT = (
    BINARY_VALIDATION_ROOT
    / "models"
    / "base_ternary"
    / "best_model.pt"
)
COMPSOL_CHECKPOINT = (
    BINARY_VALIDATION_ROOT
    / "models"
    / "compsol"
    / "checkpoint_epoch_40.pt"
)
BIGSOLVDB_PRETRAINED_CHECKPOINT = (
    BINARY_VALIDATION_ROOT
    / "models"
    / "bigsolvdb"
    / "pretrained_bigsolvdb_model.pt"
)
BIGSOLVDB_FINETUNED_CHECKPOINT = (
    BINARY_VALIDATION_ROOT
    / "models"
    / "bigsolvdb"
    / "binary_finetuned_model.pt"
)

ABRAHAM_EXPERIMENT_ROOT = BINARY_VALIDATION_ROOT / "results" / "abraham"
COMPSOL_EXPERIMENT_ROOT = BINARY_VALIDATION_ROOT / "results" / "compsol"
BIGSOLVDB_EXPERIMENT_ROOT = BINARY_VALIDATION_ROOT / "results" / "bigsolvdb"

NRTL_PARAMETER_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "parameters"
    / "main_benchmark"
    / "nrtl_params_all.json"
)

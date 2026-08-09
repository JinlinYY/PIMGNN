"""Regression tests for the organized comparison-model package and dataset."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from psmi_baselines import AVAILABLE_BASELINES
from psmi_baselines.cgib.utils.data_loader import smiles_to_graph as cgib_graph
from psmi_baselines.cignn.data_utils import smiles_to_graph as cignn_graph
from psmi_baselines.glam.dataset.data_loader import smiles_to_graph as glam_graph
from psmi_baselines.paths import DATA_DIR
from psmi_baselines.protocol import canonical_split_indices
from psmi_baselines.common.main_compare_multiple_seeds import (
    extract_metrics_from_best_metrics_json,
)
from psmi_baselines.common.utils import canonicalize_smiles


class BaselineComparisonTest(unittest.TestCase):
    """Check package registration and the shared comparison data contract."""

    def test_expected_baselines_are_registered(self) -> None:
        self.assertEqual(
            set(AVAILABLE_BASELINES),
            {"classical", "cgib", "cignn", "glam", "mmgnn", "solvbert", "bigsolvdb"},
        )

    def test_exported_splits_are_system_exclusive(self) -> None:
        frames = {
            split: pd.read_csv(DATA_DIR / f"{split}.csv")
            for split in ("train", "validation", "test")
        }
        systems = {split: set(frame["system_id"]) for split, frame in frames.items()}

        self.assertFalse(systems["train"] & systems["validation"])
        self.assertFalse(systems["train"] & systems["test"])
        self.assertFalse(systems["validation"] & systems["test"])
        self.assertEqual(
            sum(len(frame) for frame in frames.values()),
            len(pd.read_csv(DATA_DIR / "total.csv")),
        )

    def test_exported_splits_match_the_canonical_benchmark(self) -> None:
        manifest_path = PROJECT_ROOT / "datasets" / "splits" / "main_benchmark_system_split.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "train": (6092, 612, "train"),
            "validation": (788, 75, "validation"),
            "test": (803, 78, "test"),
        }
        for split, (rows, systems, partition_key) in expected.items():
            frame = pd.read_csv(DATA_DIR / f"{split}.csv")
            observed_systems = {int(value) for value in frame["system_id"].unique()}
            self.assertEqual(len(frame), rows)
            self.assertEqual(len(observed_systems), systems)
            self.assertEqual(observed_systems, set(manifest["partitions"][partition_key]))

    def test_total_table_exposes_validated_canonical_indices(self) -> None:
        total = pd.read_csv(DATA_DIR / "total.csv")
        indices = canonical_split_indices(total)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices["train"]), 6092)
        self.assertEqual(len(indices["validation"]), 788)
        self.assertEqual(len(indices["test"]), 803)

    def test_graph_loaders_accept_the_shared_smiles_columns(self) -> None:
        sample = pd.read_csv(DATA_DIR / "test.csv", nrows=1).iloc[0]

        self.assertIsNotNone(cgib_graph(sample["smiles1"]))
        self.assertIsNotNone(cignn_graph(sample["smiles2"]))
        self.assertIsNotNone(glam_graph(sample["smiles3"]))

    def test_missing_spreadsheet_smiles_are_rejected_before_rdkit(self) -> None:
        self.assertEqual(canonicalize_smiles(float("nan")), "")

    def test_multiseed_reader_accepts_post_selection_test_metrics(self) -> None:
        import tempfile

        payload = {
            "best_val_metrics": {"mae": 0.2},
            "test_metrics": {"mae": 0.3, "rmse": 0.4},
        }
        with tempfile.TemporaryDirectory(prefix="psmi-baseline-metrics-") as directory:
            path = Path(directory) / "best_metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            metrics = extract_metrics_from_best_metrics_json(str(path))
        self.assertEqual(metrics["val_mae"], 0.2)
        self.assertEqual(metrics["test_mae"], 0.3)
        self.assertEqual(metrics["test_rmse"], 0.4)


if __name__ == "__main__":
    unittest.main()

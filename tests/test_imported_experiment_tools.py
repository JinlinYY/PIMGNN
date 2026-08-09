"""Regression tests for the system-level split utilities."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.experiments.data_splitting.run_kfold_cv import (
    build_stratified_system_folds,
    make_811_fold_split,
)
from scripts.experiments.data_splitting.run_split_strategy_benchmark import (
    split_system_random,
)
class DataSplitUtilityTest(unittest.TestCase):
    @staticmethod
    def _example_frame() -> pd.DataFrame:
        rows = []
        for system_id in range(30):
            for offset in range(2):
                rows.append(
                    {
                        "system_id": system_id,
                        "T": 290.0 + system_id + offset,
                        "smiles1": "CC",
                        "smiles2": "O",
                        "smiles3": "CCC",
                    }
                )
        return pd.DataFrame(rows)

    def test_kfold_split_is_system_exclusive(self) -> None:
        frame = self._example_frame()
        folds = build_stratified_system_folds(frame, folds=10, seed=42)
        train, validation, test = make_811_fold_split(frame, folds, fold_idx=0)

        train_ids = set(train["system_id"])
        validation_ids = set(validation["system_id"])
        test_ids = set(test["system_id"])
        self.assertFalse(train_ids & validation_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(validation_ids & test_ids)
        self.assertEqual(train_ids | validation_ids | test_ids, set(range(30)))

    def test_system_random_split_is_system_exclusive(self) -> None:
        frame = self._example_frame()
        train, validation, test, _ = split_system_random(frame, seed=42)

        subsets = [set(part["system_id"]) for part in (train, validation, test)]
        self.assertFalse(subsets[0] & subsets[1])
        self.assertFalse(subsets[0] & subsets[2])
        self.assertFalse(subsets[1] & subsets[2])


if __name__ == "__main__":
    unittest.main()

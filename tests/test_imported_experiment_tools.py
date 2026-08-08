"""Regression tests for the imported visualization and split utilities."""

from __future__ import annotations

import unittest

import pandas as pd
from rdkit import Chem

from scripts.experiments.data_splitting.run_kfold_cv import (
    build_stratified_system_folds,
    make_811_fold_split,
)
from scripts.experiments.data_splitting.run_split_strategy_benchmark import (
    split_system_random,
)
from scripts.visualization.explainability._common import (
    functional_group_weights,
    normalize_weights,
    parse_bond_importance,
    parse_node_importance,
)


class ExplainabilityUtilityTest(unittest.TestCase):
    def test_node_and_bond_labels_map_to_expected_atoms(self) -> None:
        node_rows = pd.DataFrame(
            {
                "node_label": ["[g1] C:0", "[g2] O:3"],
                "importance": [-0.25, 0.75],
            }
        )
        bond_rows = pd.DataFrame(
            {
                "bond_label": ["[g1] C:0-C:1", "[g1] C:1-O:2"],
                "importance": [0.4, 0.6],
            }
        )

        self.assertEqual(parse_node_importance(node_rows)[1][0], -0.25)
        self.assertEqual(parse_node_importance(node_rows)[2][3], 0.75)
        bond_weights = parse_bond_importance(bond_rows)[1]
        self.assertAlmostEqual(bond_weights[0], 0.4)
        self.assertAlmostEqual(bond_weights[1], 1.0)
        self.assertAlmostEqual(bond_weights[2], 0.6)

    def test_functional_group_weights_are_normalized(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        weights = functional_group_weights(mol, [("CO", 2.0), ("C", 1.0)])
        normalized = normalize_weights(weights)

        self.assertEqual(len(normalized), mol.GetNumAtoms())
        self.assertAlmostEqual(max(normalized), 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in normalized))


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

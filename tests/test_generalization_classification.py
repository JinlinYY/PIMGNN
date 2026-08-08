"""Test held-out phase-diagram generalization classification."""

import unittest

import numpy as np
import pandas as pd

from scripts.analysis.classify_phase_diagram_generalization import (
    analyze_predictions,
    calculate_system_metrics,
    classify_metrics,
    ternary_to_cartesian,
)


class GeneralizationClassificationTest(unittest.TestCase):
    def test_ternary_vertices_map_to_equilateral_triangle(self) -> None:
        points = ternary_to_cartesian(np.eye(3))
        distances = [
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[1] - points[2]),
            np.linalg.norm(points[2] - points[0]),
        ]
        np.testing.assert_allclose(distances, [1.0, 1.0, 1.0])

    def test_exact_predictions_are_quantitative(self) -> None:
        frame = pd.DataFrame(
            {
                "Ex1": [0.8, 0.7, 0.6],
                "Ex2": [0.1, 0.2, 0.3],
                "Ex3": [0.1, 0.1, 0.1],
                "Rx1": [0.1, 0.1, 0.1],
                "Rx2": [0.2, 0.3, 0.4],
                "Rx3": [0.7, 0.6, 0.5],
            }
        )
        for true_column, pred_column in zip(
            ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"],
            ["pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3"],
        ):
            frame[pred_column] = frame[true_column]
        metrics = calculate_system_metrics(frame)
        category, _ = classify_metrics(metrics, composition_tolerance=0.02)
        self.assertEqual(category, "Quantitative within tolerance")

    def test_missing_required_columns_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required prediction columns"):
            analyze_predictions(pd.DataFrame({"system_id": [1], "T": [298.15]}), 0.02, 0.05)


if __name__ == "__main__":
    unittest.main()

"""Numerical tests for the NRTL ternary flash solver."""
from __future__ import annotations

import unittest

import numpy as np
import torch


class NRTLFlashTest(unittest.TestCase):
    def test_numpy_activity_coefficients_match_training_implementation(self) -> None:
        from psmi.loss import nrtl_ln_gamma
        from psmi.nrtl_flash import nrtl_ln_gamma_numpy

        composition = np.asarray([0.22, 0.33, 0.45], dtype=float)
        interaction = np.asarray(
            [[0.0, 1200.0, -800.0], [2200.0, 0.0, 900.0], [400.0, -500.0, 0.0]],
            dtype=float,
        )
        expected = nrtl_ln_gamma(
            torch.tensor(composition[None, :], dtype=torch.float64),
            torch.tensor([318.15], dtype=torch.float64),
            torch.tensor(interaction[None, :, :], dtype=torch.float64),
            ln_gamma_clip=None,
        ).numpy()[0]
        actual = nrtl_ln_gamma_numpy(composition, 318.15, interaction)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_ideal_mixture_does_not_create_a_false_phase_split(self) -> None:
        from psmi.nrtl_flash import flash_nrtl

        result = flash_nrtl(
            overall=np.asarray([0.35, 0.40, 0.25]),
            temperature=298.15,
            interaction=np.zeros((3, 3), dtype=float),
            initial_extract=np.asarray([0.75, 0.15, 0.10]),
            initial_raffinate=np.asarray([0.10, 0.75, 0.15]),
            random_starts=2,
        )
        self.assertFalse(result.success)
        self.assertIn(result.status, {"single_phase", "no_stable_split"})

    def test_binary_like_immiscibility_returns_balanced_stable_phases(self) -> None:
        from psmi.nrtl_flash import flash_nrtl

        interaction = np.asarray(
            [[0.0, 9000.0, 0.0], [9000.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=float,
        )
        overall = np.asarray([0.475, 0.475, 0.05], dtype=float)
        result = flash_nrtl(
            overall=overall,
            temperature=298.15,
            interaction=interaction,
            initial_extract=np.asarray([0.88, 0.07, 0.05]),
            initial_raffinate=np.asarray([0.07, 0.88, 0.05]),
            random_starts=6,
        )
        self.assertTrue(result.success, msg=result)
        reconstructed = (
            result.phase_fraction * result.extract
            + (1.0 - result.phase_fraction) * result.raffinate
        )
        np.testing.assert_allclose(reconstructed, overall, atol=2e-6)
        self.assertGreater(result.phase_separation, 0.05)
        self.assertLess(result.mu_residual_max, 2e-3)
        self.assertGreaterEqual(result.minimum_tpd, -2e-4)


if __name__ == "__main__":
    unittest.main()

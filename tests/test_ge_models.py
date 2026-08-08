"""Test excess-Gibbs-energy models and physics-informed losses."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import json

import pandas as pd
import torch


class GEActivityModelTest(unittest.TestCase):
    def test_pairwise_margules_matches_binary_closed_form(self) -> None:
        from psmi.ge_models import ge_ln_gamma

        dtype = torch.float64
        temperature = torch.tensor([300.0], dtype=dtype)
        gas_constant = 8.314462618
        composition = torch.tensor([[0.3, 0.7, 0.0]], dtype=dtype)

        # Binary three-suffix Margules parameters A12=2 and A21=1.
        energies = torch.zeros((1, 3, 3), dtype=dtype)
        energies[0, 0, 1] = 2.0 * gas_constant * temperature[0]
        energies[0, 1, 0] = 1.0 * gas_constant * temperature[0]

        actual = ge_ln_gamma("margules", composition, temperature, energies)
        expected_1 = 0.7**2 * (2.0 + 2.0 * (1.0 - 2.0) * 0.3)
        expected_2 = 0.3**2 * (1.0 + 2.0 * (2.0 - 1.0) * 0.7)

        torch.testing.assert_close(
            actual[0, :2],
            torch.tensor([expected_1, expected_2], dtype=dtype),
            rtol=1e-8,
            atol=1e-8,
        )

    def test_pairwise_van_laar_matches_binary_closed_form(self) -> None:
        from psmi.ge_models import ge_ln_gamma

        dtype = torch.float64
        temperature = torch.tensor([300.0], dtype=dtype)
        gas_constant = 8.314462618
        composition = torch.tensor([[0.3, 0.7, 0.0]], dtype=dtype)

        # Binary van Laar parameters A12=2 and A21=1.
        energies = torch.zeros((1, 3, 3), dtype=dtype)
        energies[0, 0, 1] = 2.0 * gas_constant * temperature[0]
        energies[0, 1, 0] = 1.0 * gas_constant * temperature[0]

        actual = ge_ln_gamma("van_laar", composition, temperature, energies)
        denominator = 2.0 * 0.3 + 1.0 * 0.7
        expected_1 = 2.0 * (1.0 * 0.7 / denominator) ** 2
        expected_2 = 1.0 * (2.0 * 0.3 / denominator) ** 2

        torch.testing.assert_close(
            actual[0, :2],
            torch.tensor([expected_1, expected_2], dtype=dtype),
            rtol=1e-8,
            atol=1e-8,
        )

    def test_activity_models_are_component_permutation_equivariant(self) -> None:
        from psmi.ge_models import ge_ln_gamma

        dtype = torch.float64
        composition = torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype)
        temperature = torch.tensor([315.0], dtype=dtype)
        energies = torch.tensor(
            [[[0.0, 1200.0, 800.0], [900.0, 0.0, 1500.0], [700.0, 1100.0, 0.0]]],
            dtype=dtype,
        )
        permutation = torch.tensor([0, 2, 1])

        for model in ("nrtl", "margules", "van_laar"):
            with self.subTest(model=model):
                original = ge_ln_gamma(model, composition, temperature, energies)
                permuted = ge_ln_gamma(
                    model,
                    composition[:, permutation],
                    temperature,
                    energies[:, permutation][:, :, permutation],
                )
                torch.testing.assert_close(
                    permuted,
                    original[:, permutation],
                    rtol=1e-8,
                    atol=1e-8,
                )

    def test_activity_models_satisfy_gibbs_duhem_numerically(self) -> None:
        from psmi.ge_models import ge_ln_gamma

        dtype = torch.float64
        composition = torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype)
        direction = torch.tensor([[1.0, -0.4, -0.6]], dtype=dtype)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        temperature = torch.tensor([315.0], dtype=dtype)
        energies = torch.tensor(
            [[[0.0, 1200.0, 800.0], [900.0, 0.0, 1500.0], [700.0, 1100.0, 0.0]]],
            dtype=dtype,
        )
        step = 1e-5

        for model in ("nrtl", "margules", "van_laar"):
            with self.subTest(model=model):
                ln_plus = ge_ln_gamma(
                    model, composition + step * direction, temperature, energies
                )
                ln_minus = ge_ln_gamma(
                    model, composition - step * direction, temperature, energies
                )
                derivative = (ln_plus - ln_minus) / (2.0 * step)
                residual = (composition * derivative).sum()
                self.assertLess(abs(float(residual)), 1e-7)

    def test_activity_models_support_parameter_optimization(self) -> None:
        from psmi.ge_models import ge_ln_gamma

        composition = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
        temperature = torch.tensor([315.0], dtype=torch.float64)
        for model in ("margules", "van_laar"):
            with self.subTest(model=model):
                energies = torch.full(
                    (1, 3, 3), 1000.0, dtype=torch.float64, requires_grad=True
                )
                objective = ge_ln_gamma(
                    model, composition, temperature, energies
                ).square().mean()
                objective.backward()
                self.assertIsNotNone(energies.grad)
                self.assertTrue(torch.isfinite(energies.grad).all())

    def test_equal_phase_compositions_have_zero_equilibrium_residual(self) -> None:
        from psmi.ge_models import ge_mu_residual

        composition = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
        temperature = torch.tensor([315.0], dtype=torch.float64)
        energies = torch.tensor(
            [[[0.0, 1200.0, 800.0], [900.0, 0.0, 1500.0], [700.0, 1100.0, 0.0]]],
            dtype=torch.float64,
        )
        for model in ("nrtl", "margules", "van_laar"):
            with self.subTest(model=model):
                residual = ge_mu_residual(
                    model, composition, composition, temperature, energies
                )
                torch.testing.assert_close(
                    residual, torch.zeros_like(residual), rtol=0.0, atol=1e-12
                )


class GEParameterDataIsolationTest(unittest.TestCase):
    def test_parameter_store_masks_unseen_systems(self) -> None:
        from psmi.ge_parameters import GEParameterStore

        payload = {
            "meta": {"model": "margules", "R": 8.314462618},
            "params": {"7": [[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]]},
        }
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "params.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            store = GEParameterStore(path)
            parameters, mask = store.get_batch(
                torch.tensor([7, 99]), swap23=torch.tensor([0, 1])
            )

        self.assertEqual(store.model, "margules")
        self.assertEqual(mask.tolist(), [True, False])
        torch.testing.assert_close(
            parameters[0],
            torch.tensor(payload["params"]["7"], dtype=parameters.dtype),
        )
        torch.testing.assert_close(parameters[1], torch.zeros_like(parameters[1]))

    def test_parameter_fitting_is_limited_to_declared_training_systems(self) -> None:
        from psmi.ge_fit import fit_parameter_store

        rows = []
        for system_id in (1, 2):
            for offset in (0.0, 0.02, 0.04):
                rows.append(
                    {
                        "system_id": system_id,
                        "T": 298.15,
                        "Ex1": 0.10 + offset,
                        "Ex2": 0.20,
                        "Ex3": 0.70 - offset,
                        "Rx1": 0.70 - offset,
                        "Rx2": 0.20,
                        "Rx3": 0.10 + offset,
                    }
                )
        data = pd.DataFrame(rows)
        fitted = fit_parameter_store(
            data,
            model="margules",
            training_system_ids={1},
            steps=2,
            learning_rate=0.01,
        )

        self.assertEqual(set(fitted["params"]), {"1"})
        self.assertEqual(fitted["meta"]["fit_scope"], "training_systems_only")

    def test_vectorized_parameter_fitting_preserves_training_scope(self) -> None:
        from psmi.ge_fit import fit_parameter_store

        data = pd.DataFrame(
            [
                {
                    "system_id": system_id,
                    "T": 298.15,
                    "Ex1": 0.1 + offset,
                    "Ex2": 0.2,
                    "Ex3": 0.7 - offset,
                    "Rx1": 0.7 - offset,
                    "Rx2": 0.2,
                    "Rx3": 0.1 + offset,
                }
                for system_id in (1, 2)
                for offset in (0.0, 0.02, 0.04)
            ]
        )
        fitted = fit_parameter_store(
            data,
            model="van_laar",
            training_system_ids={2},
            steps=2,
            learning_rate=0.01,
            vectorized=True,
        )

        self.assertEqual(set(fitted["params"]), {"2"})
        self.assertEqual(fitted["meta"]["optimizer_layout"], "vectorized")


class MechanisticGELossTest(unittest.TestCase):
    def test_alternative_ge_model_rejects_nrtl_only_auxiliary_terms(self) -> None:
        from psmi.loss import MechanisticNRTLLoss

        payload = {"meta": {"model": "margules"}, "params": {}}
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "params.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NRTL-only"):
                MechanisticNRTLLoss(
                    T_mean=300.0,
                    T_std=10.0,
                    nrtl_params_path=str(path),
                    ge_model="margules",
                    w_gd=0.1,
                    w_stab=0.0,
                )

    def test_mechanistic_loss_accepts_alternative_ge_model(self) -> None:
        from psmi.loss import MechanisticNRTLLoss

        payload = {
            "meta": {"model": "margules", "R": 8.314462618},
            "params": {
                "7": [[0.0, 2000.0, 1800.0], [1500.0, 0.0, 1700.0], [1400.0, 1600.0, 0.0]]
            },
        }
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "params.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loss = MechanisticNRTLLoss(
                T_mean=300.0,
                T_std=10.0,
                nrtl_params_path=str(path),
                ge_model="margules",
                lambda_phy=1e-2,
                warmup_epochs=0,
                ramp_epochs=1,
                w_gd=0.0,
                w_stab=0.0,
            )
            loss.set_epoch(1)
            prediction = torch.tensor(
                [[0.1, 0.2, 0.7, 0.7, 0.2, 0.1]],
                dtype=torch.float32,
                requires_grad=True,
            )
            target = prediction.detach().clone()
            batch = {
                "scalars": torch.tensor([[0.0, 0.5]], dtype=torch.float32),
                "system_id": torch.tensor([7]),
                "aug_swap23": torch.tensor([0]),
            }
            result = loss(prediction, target, batch)
            result["loss"].backward()

        self.assertGreater(float(result["phy"]), 0.0)
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()

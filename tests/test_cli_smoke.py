"""Regression tests for repository entry-point behavior on Windows."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "section_3_results"
    / "3_4_industrial_extraction_design"
    / "application_workflow"
    / "results"
    / "application_case_predictions.csv"
)
APPLICATION_EXCEL = PROJECT_ROOT / "datasets" / "raw" / "application_case_1.xlsx"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "section_3_results"
    / "3_3_binary_solubility_validation"
    / "models"
    / "base_ternary"
    / "best_model.pt"
)


class ApplicationCliSmokeTest(unittest.TestCase):
    def test_application_full_workflow_respects_cpu_device(self) -> None:
        with tempfile.TemporaryDirectory(prefix="psmi-full-case-") as output_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_application_case.py",
                    "--excel",
                    str(APPLICATION_EXCEL),
                    "--ckpt",
                    str(DEFAULT_CHECKPOINT),
                    "--out_dir",
                    output_dir,
                    "--device",
                    "cpu",
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            output = result.stdout.decode(errors="replace")
            self.assertEqual(result.returncode, 0, output)
            self.assertTrue(
                (Path(output_dir) / "application_case_predictions.csv").is_file()
            )

    def test_default_checkpoint_loads_with_pressure_compatibility(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        self.addCleanup(lambda: sys.path.remove(str(PROJECT_ROOT)))
        application = importlib.import_module("scripts.run_application_case")
        from psmi import config

        model, scaler, checkpoint_config = application.load_model_and_scaler(
            config.LOAD_CKPT_PATH, "cpu"
        )
        self.assertEqual(model.backbone[0].in_features, 3331)
        self.assertEqual(float(model.backbone[0].weight[:, -1].abs().max()), 0.0)
        self.assertGreater(scaler.std, 0)
        self.assertIsInstance(checkpoint_config, dict)

    def test_application_analysis_is_console_safe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="psmi-cli-") as output_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_application_case.py",
                    "--csv",
                    str(APPLICATION_CSV),
                    "--out_dir",
                    output_dir,
                    "--analyze_only",
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            output = result.stdout.decode(errors="replace")
            self.assertEqual(result.returncode, 0, output)
            self.assertTrue((Path(output_dir) / "detailed_analysis.txt").is_file())

    def test_application_script_is_importable_as_a_module(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        self.addCleanup(lambda: sys.path.remove(str(PROJECT_ROOT)))
        module = importlib.import_module("scripts.run_application_case")
        self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()

"""Behavioral tests for split-bound NRTL parameter generation and routing."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class NRTLParameterIsolationTest(unittest.TestCase):
    def _write_parameter_file(self, path: Path, system_ids: list[int], role: str) -> None:
        payload = {
            "meta": {
                "schema_version": 2,
                "role": role,
                "fitted_independently_by_system": True,
            },
            "params": {
                str(system_id): [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                for system_id in system_ids
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_training_parameter_validation_rejects_validation_or_test_systems(self) -> None:
        from psmi.nrtl_isolation import validate_training_parameter_file

        train_df = pd.DataFrame({"system_id": [1, 1, 2, 2]})
        val_df = pd.DataFrame({"system_id": [3, 3]})
        test_df = pd.DataFrame({"system_id": [4, 4]})

        with tempfile.TemporaryDirectory(prefix="psmi-nrtl-isolation-") as temp_dir:
            safe_path = Path(temp_dir) / "nrtl_params_train.json"
            self._write_parameter_file(safe_path, [1, 2], "training_loss")
            audit = validate_training_parameter_file(
                safe_path, train_df=train_df, val_df=val_df, test_df=test_df
            )
            self.assertEqual(audit["unexpected_parameter_system_ids"], [])
            self.assertEqual(audit["test_parameter_overlap"], [])

            incomplete_path = Path(temp_dir) / "nrtl_params_train_incomplete.json"
            self._write_parameter_file(incomplete_path, [1], "training_loss")
            with self.assertRaisesRegex(ValueError, "does not cover"):
                validate_training_parameter_file(
                    incomplete_path, train_df=train_df, val_df=val_df, test_df=test_df
                )

            unsafe_path = Path(temp_dir) / "nrtl_params_all.json"
            self._write_parameter_file(unsafe_path, [1, 2, 3, 4], "posthoc_evaluation")
            with self.assertRaisesRegex(ValueError, "validation/test systems"):
                validate_training_parameter_file(
                    unsafe_path, train_df=train_df, val_df=val_df, test_df=test_df
                )

    def test_posthoc_parameter_validation_requires_heldout_coverage(self) -> None:
        from psmi.nrtl_isolation import validate_evaluation_parameter_file

        val_df = pd.DataFrame({"system_id": [3, 3]})
        test_df = pd.DataFrame({"system_id": [4, 4]})

        with tempfile.TemporaryDirectory(prefix="psmi-nrtl-posthoc-") as temp_dir:
            complete_path = Path(temp_dir) / "nrtl_params_all.json"
            self._write_parameter_file(complete_path, [1, 2, 3, 4], "posthoc_evaluation")
            audit = validate_evaluation_parameter_file(
                complete_path, val_df=val_df, test_df=test_df
            )
            self.assertEqual(audit["missing_validation_parameter_system_ids"], [])
            self.assertEqual(audit["missing_test_parameter_system_ids"], [])

            incomplete_path = Path(temp_dir) / "nrtl_params_incomplete.json"
            self._write_parameter_file(incomplete_path, [1, 2, 3], "posthoc_evaluation")
            with self.assertRaisesRegex(ValueError, "does not cover"):
                validate_evaluation_parameter_file(
                    incomplete_path, val_df=val_df, test_df=test_df
                )

    def test_config_declares_distinct_training_and_posthoc_paths(self) -> None:
        from psmi import config

        training_path = Path(config.NRTL_TRAIN_PARAMS_PATH)
        evaluation_path = Path(config.NRTL_EVAL_PARAMS_PATH)
        self.assertEqual(training_path.name, "nrtl_params_train.json")
        self.assertEqual(evaluation_path.name, "nrtl_params_all.json")
        self.assertNotEqual(training_path.resolve(), evaluation_path.resolve())

    def test_parameter_dataset_hash_mismatch_is_rejected(self) -> None:
        from psmi.nrtl_isolation import validate_training_parameter_file

        train_df = pd.DataFrame({"system_id": [1, 1]})
        empty_df = pd.DataFrame({"system_id": pd.Series(dtype=int)})
        with tempfile.TemporaryDirectory(prefix="psmi-nrtl-provenance-") as temp_dir:
            temp_root = Path(temp_dir)
            dataset_path = temp_root / "dataset.xlsx"
            dataset_path.write_bytes(b"current dataset")
            parameter_path = temp_root / "nrtl_params_train.json"
            self._write_parameter_file(parameter_path, [1], "training_loss")
            payload = json.loads(parameter_path.read_text(encoding="utf-8"))
            payload["meta"]["dataset_sha256"] = "0" * 64
            parameter_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "different dataset"):
                validate_training_parameter_file(
                    parameter_path,
                    train_df=train_df,
                    val_df=empty_df,
                    test_df=empty_df,
                    dataset_path=dataset_path,
                )

    def test_fit_cli_generates_split_bound_train_and_all_files(self) -> None:
        rows = []
        for system_id in range(1, 11):
            for point in range(3):
                fraction = 0.1 + 0.05 * point
                rows.append(
                    {
                        "system_id": system_id,
                        "T": 298.15,
                        "smiles1": "O",
                        "smiles2": "CC",
                        "smiles3": "CCC",
                        "Ex1": 0.7 - fraction,
                        "Ex2": fraction,
                        "Ex3": 0.3,
                        "Rx1": 0.1,
                        "Rx2": 0.2 + fraction,
                        "Rx3": 0.7 - fraction,
                    }
                )

        with tempfile.TemporaryDirectory(prefix="psmi-nrtl-fit-") as temp_dir:
            temp_root = Path(temp_dir)
            excel_path = temp_root / "tiny_lle.xlsx"
            output_dir = temp_root / "parameters"
            pd.DataFrame(rows).to_excel(excel_path, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/fit_nrtl.py",
                    "--excel_path",
                    str(excel_path),
                    "--out_dir",
                    str(output_dir),
                    "--scope",
                    "both",
                    "--split-strategy",
                    "random",
                    "--seed",
                    "42",
                    "--min-points",
                    "3",
                    "--steps",
                    "1",
                    "--workers",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            output = result.stdout.decode(errors="replace")
            self.assertEqual(result.returncode, 0, output)

            manifest = json.loads(
                (output_dir / "nrtl_split_manifest.json").read_text(encoding="utf-8")
            )
            train_payload = json.loads(
                (output_dir / "nrtl_params_train.json").read_text(encoding="utf-8")
            )
            all_payload = json.loads(
                (output_dir / "nrtl_params_all.json").read_text(encoding="utf-8")
            )

            train_ids = set(manifest["partitions"]["train"])
            held_out_ids = set(manifest["partitions"]["validation"]) | set(
                manifest["partitions"]["test"]
            )
            self.assertEqual(set(train_payload["params"]), train_ids)
            self.assertTrue(set(train_payload["params"]).isdisjoint(held_out_ids))
            self.assertEqual(set(all_payload["params"]), {str(i) for i in range(1, 11)})
            self.assertEqual(train_payload["meta"]["role"], "training_loss")
            self.assertEqual(all_payload["meta"]["role"], "posthoc_evaluation")
            self.assertTrue(train_payload["meta"]["fitted_independently_by_system"])


if __name__ == "__main__":
    unittest.main()

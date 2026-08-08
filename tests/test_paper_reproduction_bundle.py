"""Tests for the inference-only paper reproduction package."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_registered_reproduction_inputs_exist() -> None:
    """Every registry entry must resolve to an existing config and checkpoint."""
    registry_dir = PROJECT_ROOT / "configs" / "reproduction"
    for registry_path in sorted(registry_dir.glob("*_registry.json")):
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        assert registry["runs"], registry_path
        for run in registry["runs"]:
            for field in ("config", "checkpoint"):
                assert (PROJECT_ROOT / run[field]).is_file(), (registry_path, run["id"], field)
            for field in ("fg_corpus", "reference_predictions"):
                if run.get(field):
                    assert (PROJECT_ROOT / run[field]).is_file(), (registry_path, run["id"], field)


def test_reproduction_entry_points_are_inference_only() -> None:
    """The public reproduction commands must not invoke the training entry point."""
    for relative in ("scripts/evaluate_checkpoint.py", "scripts/reproduce_current_weights.py"):
        content = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "scripts/train.py" not in content
        assert "optimizer.step" not in content


def test_organized_bundle_contains_required_evidence() -> None:
    """The generated bundle must contain paper tables, predictions, figures, and audit data."""
    bundle = PROJECT_ROOT / "results" / "paper_reproduction"
    required = [
        "README.md",
        "tables/paper_table_1_reported.csv",
        "tables/paper_table_2_reported.csv",
        "tables/paper_table_3_reported.csv",
        "tables/current_weight_metrics.csv",
        "tables/historical_weight_metrics.csv",
        "data/predictions/historical/figure2a_psmi.csv",
        "figures/historical/figure2a_psmi/parity_E.png",
        "figures/historical/figure2a_psmi/parity_R.png",
        "audit/protocol_alignment.csv",
        "audit/artifact_manifest.csv",
    ]
    for relative in required:
        path = bundle / relative
        assert path.is_file() and path.stat().st_size > 0, path

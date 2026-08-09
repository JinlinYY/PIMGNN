"""Tests for the canonical Figure 2a public result."""

import csv
import hashlib
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


def test_reproduction_entry_points_are_checkpoint_evaluators() -> None:
    """The public reproduction commands must not invoke the training entry point."""
    for relative in ("scripts/evaluate_checkpoint.py", "scripts/evaluate_checkpoint_registry.py"):
        content = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "scripts/train.py" not in content
        assert "optimizer.step" not in content


def _sha256(path: Path) -> str:
    """Return the uppercase SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_results_directory_directly_contains_canonical_figure_2a() -> None:
    """The public results directory must directly expose Figure 2a and named run packages."""
    results = PROJECT_ROOT / "results"
    assert {path.name for path in results.iterdir()} == {
        "README.md",
        "figure_2a.png",
        "data_driven",
        "chemical_potential_regularized",
    }
    figure = results / "figure_2a.png"
    source = (
        PROJECT_ROOT
        / "experiments"
        / "section_3_results"
        / "3_1_lle_prediction"
        / "main_benchmark"
        / "figures"
        / "figure_2a_parity.png"
    )
    assert figure.is_file() and figure.stat().st_size > 0
    assert _sha256(figure) == "2E60A8072F0F47A10CFEE6B468A6FB6DBE416C665B0C2DD849EB12A7C9B630F7"
    assert _sha256(figure) == _sha256(source)


def test_model_variant_result_packages_use_english_researcher_facing_paths() -> None:
    """Every reference run must use the common English result structure."""
    results = PROJECT_ROOT / "results"
    runs = (
        "data_driven",
        "chemical_potential_regularized",
    )
    required = (
        "README.md",
        "artifact_manifest.csv",
        "checkpoints/best_model.pt",
        "metrics/best_metrics.json",
        "metrics/best_metrics.txt",
        "metrics/training_metrics_log.csv",
        "predictions/test_pointwise_predictions.csv",
        "artifacts/functional_group_corpus.json",
        "figures/parity/extract_phase_parity.png",
        "figures/parity/raffinate_phase_parity.png",
        "figures/ternary_phase_diagrams/all_test_systems.pdf",
    )
    for run_name in runs:
        run = results / run_name
        for relative in required:
            path = run / relative
            assert path.is_file() and path.stat().st_size > 0, path
        for path in run.rglob("*"):
            path.relative_to(results).as_posix().encode("ascii")


def test_model_variant_artifact_manifests_match_distributed_files() -> None:
    """Every result-package manifest must match the distributed file bytes."""
    for manifest in sorted((PROJECT_ROOT / "results").glob("*/artifact_manifest.csv")):
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, manifest
        assert all(row["relative_path"] != "checkpoints/last_model.pt" for row in rows)
        for row in rows:
            artifact = manifest.parent / row["relative_path"]
            assert artifact.is_file(), artifact
            assert artifact.stat().st_size == int(row["bytes"]), artifact
            assert _sha256(artifact) == row["sha256"], artifact


def test_figure_2a_sources_remain_in_the_section_experiment() -> None:
    """Numerical data and the associated checkpoint remain in the experiment directory."""
    experiment = (
        PROJECT_ROOT
        / "experiments"
        / "section_3_results"
        / "3_1_lle_prediction"
        / "main_benchmark"
    )
    predictions = experiment / "data" / "figure_2a_predictions.csv"
    checkpoint = experiment / "models" / "figure_2a_psmi" / "best_model.pt"
    assert _sha256(predictions) == "18FE725C7D9D60EA294C2CE54B3A979A3D6B03185F59B937D74977D4082E1B86"
    assert _sha256(checkpoint) == "72C432BC7FD48CB44B52402AD01393BA5A8B47737FA1E286658928670C10F380"

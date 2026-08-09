"""Release-quality checks for the public PSMI repository."""

from __future__ import annotations

import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".vue",
    ".yaml",
    ".yml",
}


def _public_text_files() -> list[Path]:
    return [
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and "tmp" not in path.relative_to(PROJECT_ROOT).parts
        and path.name != Path(__file__).name
    ]


def test_open_source_policy_files_are_present() -> None:
    """A public research repository must declare reuse and contribution terms."""
    for name in (
        "LICENSE",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        assert (PROJECT_ROOT / name).is_file(), name


def test_public_paths_use_scientific_names() -> None:
    """Public paths describe scientific roles instead of development history."""
    forbidden = re.compile(r"historical|legacy|corrected_v2|archive|20260119", re.I)
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in PROJECT_ROOT.rglob("*")
        if forbidden.search(str(path.relative_to(PROJECT_ROOT)))
    ]
    assert offenders == []


def test_public_text_omits_internal_release_language() -> None:
    """Documentation and code must not read like an internal migration log."""
    forbidden = re.compile(
        r"PIMGNN|PI\s*[+_-]?\s*MGNN|"
        r"(?:legacy|baseline) public-release workflow step|"
        r"SI-Ready Text|Discrepancy Check Against Manuscript Counts|"
        r"code_only_result_not_found|source archive|"
        r"update-LLE-all-with-smiles_(?:min3|no-missing-smiles)|"
        r"(?:00_dataset_construction|01_baselines|07_external_validation|"
        r"08_interpretability|08_temperature_robustness|09_application_cases|"
        r"09_data_splitting|11_tieline_sensitivity|11_ge_model_sensitivity|"
        r"12_temperature_encoding|12_thermodynamic_audit)",
        re.I,
    )
    offenders: list[str] = []
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if forbidden.search(text):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def test_dataset_manifests_use_one_explicit_counting_contract() -> None:
    """The documented benchmark counts match the frozen system split manifests."""
    expected = {
        "main_benchmark_system_split.json": (7683, 765, 612, 75, 78),
        "expanded_lle_system_split.json": (6709, 719, 575, 72, 72),
    }
    for filename, counts in expected.items():
        manifest = json.loads(
            (PROJECT_ROOT / "datasets" / "splits" / filename).read_text(
                encoding="utf-8"
            )
        )
        actual = manifest["counts"]
        assert (
            actual["records_without_augmentation"],
            actual["systems"],
            actual["train_systems"],
            actual["validation_systems"],
            actual["test_systems"],
        ) == counts
        assert set(manifest["partitions"]) == {"train", "validation", "test"}
        assert sum(len(values) for values in manifest["partitions"].values()) == counts[1]


def test_experiment_catalog_references_existing_code() -> None:
    """Every public experiment entry point must resolve inside the repository."""
    catalog = json.loads(
        (PROJECT_ROOT / "experiments" / "experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for experiment in catalog["experiments"]:
        assert experiment["code"], experiment["title"]
        for relative in experiment["code"]:
            assert (PROJECT_ROOT / relative).exists(), (experiment["title"], relative)


def test_dataset_analysis_references_distributed_workbooks() -> None:
    """The data-audit entry point must run against files shipped in the release."""
    script = (
        PROJECT_ROOT / "scripts" / "analysis" / "analyze_dataset_distribution.py"
    ).read_text(encoding="utf-8")
    assert "update-LLE-all-with-smiles_min3.xlsx" not in script
    assert "update-LLE-all-with-smiles.xlsx" in script
    assert "dataset_report.md" in script


def test_results_exclude_nonselected_training_snapshots() -> None:
    """Reference packages retain selected checkpoints, not last-epoch snapshots."""
    assert list((PROJECT_ROOT / "results").rglob("last_model.pt")) == []


def test_readme_is_researcher_facing() -> None:
    """The landing page covers the scientific and reproducibility interfaces."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for heading in (
        "## Scientific scope",
        "## Model architecture",
        "## Installation",
        "## Quick start",
        "## Data contract",
        "## Experiments and reference results",
        "## Web application",
        "## Citation and license",
    ):
        assert heading in readme

"""Regression tests for the deployable public Web application package."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGED_WEB = PROJECT_ROOT / "public_release" / "PSMI-public" / "Web" / "PSMI-LLE-web"
PUBLIC_WEB = STAGED_WEB if STAGED_WEB.is_dir() else PROJECT_ROOT / "Web" / "PSMI-LLE-web"


def test_public_web_package_contains_runtime_and_frontend_files() -> None:
    """The public release must contain everything referenced by the launch guide."""
    required = (
        "requirements.txt",
        "scripts/run_backend.ps1",
        "scripts/run_frontend.ps1",
        "frontend/src/App.vue",
        "frontend/src/components/SmilesInput.vue",
        "frontend/src/components/PhaseDiagram.vue",
        "frontend/src/services/api.js",
        "checkpoints/default/best_model.pt",
        "checkpoints/default/fg_corpus.json",
    )
    missing = [relative for relative in required if not (PUBLIC_WEB / relative).is_file()]
    assert missing == []


def test_public_web_readme_documents_complete_local_deployment() -> None:
    """The Web README must cover installation, startup, validation, and settings."""
    text = (PUBLIC_WEB / "README.md").read_text(encoding="utf-8")
    required_fragments = (
        "## Local deployment",
        "conda env create -f environment.yml",
        "python -m pip install -r Web/PSMI-LLE-web/requirements.txt",
        "npm ci",
        "Web/PSMI-LLE-web/scripts/run_backend.ps1",
        "Web/PSMI-LLE-web/scripts/run_frontend.ps1",
        "http://localhost:8000/health",
        "Invoke-RestMethod",
        "PSMI_WEB_DEVICE",
        "## Troubleshooting",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    assert missing == []


def test_frontend_development_server_is_local_only_by_default() -> None:
    """The Vite proxy must not expose local model inference unintentionally."""
    config = (PUBLIC_WEB / "frontend" / "vite.config.js").read_text(encoding="utf-8")
    assert "host: '127.0.0.1'" in config

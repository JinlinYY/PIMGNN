"""Runtime paths and environment settings for the PSMI web backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SRC = PROJECT_ROOT / "src"

# Make the shared PSMI package importable when the backend is run as a module.
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

DEVICE = os.getenv("PSMI_WEB_DEVICE", "cpu")
MODEL_DIR = Path(
    os.getenv("PSMI_WEB_MODEL_DIR", str(WEB_ROOT / "checkpoints" / "default"))
).resolve()
MODEL_PATH = Path(os.getenv("PSMI_WEB_MODEL_PATH", str(MODEL_DIR / "best_model.pt"))).resolve()
EXPLAIN_DIR = Path(
    os.getenv(
        "PSMI_WEB_EXPLAIN_DIR",
        str(WEB_ROOT / "assets" / "explainability" / "default_saliency"),
    )
).resolve()

DEFAULT_PRESSURE_KPA = 101.325
DEFAULT_TIE_LINES = 14
API_HOST = os.getenv("PSMI_WEB_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PSMI_WEB_PORT", "8000"))


"""Make the repository source packages importable from nested experiment scripts."""

from pathlib import Path
import sys


def add_src_to_path() -> Path:
    """Add the local ``src`` directory and return the repository root."""
    project_root = Path(__file__).resolve().parents[4]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root

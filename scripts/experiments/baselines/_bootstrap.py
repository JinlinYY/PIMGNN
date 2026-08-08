"""Bootstrap the repository package path for nested baseline scripts."""

from pathlib import Path
import sys


def add_baseline_package_to_path() -> Path:
    """Add the repository root and local src directory to ``sys.path``."""
    project_root = Path(__file__).resolve().parents[3]
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from scripts._bootstrap import add_src_to_path

    add_src_to_path()
    return project_root

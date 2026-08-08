"""Make the local ``src`` tree importable for repository entry scripts."""

from pathlib import Path
import sys


def _make_console_output_safe() -> None:
    """Prevent Windows legacy code pages from crashing on report symbols."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def add_src_to_path() -> Path:
    _make_console_output_safe()
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    src_text = str(src_dir)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return project_root

"""Render functional-group importance as molecular heatmaps."""

try:
    from ._common import run_cli
except ImportError:
    from _common import run_cli


if __name__ == "__main__":
    run_cli("functional_group", style="heatmap", colorbar=False)

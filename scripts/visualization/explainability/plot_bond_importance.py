"""Render bond-importance heatmaps with a shared color scale."""

try:
    from ._common import run_cli
except ImportError:
    from _common import run_cli


if __name__ == "__main__":
    run_cli("bond", style="heatmap", colorbar=True)

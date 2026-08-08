"""Render functional-group importance with solid highlights and a colorbar."""

try:
    from ._common import run_cli
except ImportError:
    from _common import run_cli


if __name__ == "__main__":
    run_cli("functional_group", style="highlight", colorbar=True)

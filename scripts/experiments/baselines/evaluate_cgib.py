"""Evaluate a trained CGIB comparison model."""

try:
    from ._bootstrap import add_baseline_package_to_path
except ImportError:
    from _bootstrap import add_baseline_package_to_path

add_baseline_package_to_path()

from psmi_baselines.cgib.test_model import main


if __name__ == "__main__":
    main()

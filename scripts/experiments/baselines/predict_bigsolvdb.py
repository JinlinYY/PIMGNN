"""Predict the BigSolvDB test subset with trained seed models."""

try:
    from ._bootstrap import add_baseline_package_to_path
except ImportError:
    from _bootstrap import add_baseline_package_to_path

add_baseline_package_to_path()

from psmi_baselines.bigsolvdb.predict_test import main


if __name__ == "__main__":
    main()

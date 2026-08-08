"""Run the shared classical comparison over multiple random seeds."""

try:
    from ._bootstrap import add_baseline_package_to_path
except ImportError:
    from _bootstrap import add_baseline_package_to_path

add_baseline_package_to_path()

from psmi_baselines.common.main_compare_multiple_seeds import main


if __name__ == "__main__":
    main()

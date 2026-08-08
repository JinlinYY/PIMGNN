"""Regenerate extended prediction diagnostics from a CSV result file."""

try:
    from _bootstrap import add_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_src_to_path

add_src_to_path()
from psmi.plot_test_viz_from_csv_extra import main

if __name__ == "__main__":
    main()

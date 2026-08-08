"""Fit NRTL parameters used by PSMI physics constraints."""

try:
    from _bootstrap import add_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_src_to_path

add_src_to_path()
from psmi.fit_nrtl_params import main

if __name__ == "__main__":
    main()

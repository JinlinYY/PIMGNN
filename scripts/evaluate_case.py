"""Evaluate one configured LLE application case."""

try:
    from _bootstrap import add_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_src_to_path

add_src_to_path()
from psmi.testcase import main

if __name__ == "__main__":
    main()

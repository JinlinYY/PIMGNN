"""Run PSMI interpretability analysis."""

try:
    from _bootstrap import add_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_src_to_path

add_src_to_path()
from psmi.eval_explain import main

if __name__ == "__main__":
    main()

"""Run PSMI training, evaluation, and visualization from layered YAML profiles."""

try:
    from _bootstrap import add_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_src_to_path

add_src_to_path()
import argparse

from psmi.configuration import apply_config_files, apply_config_overrides
from psmi.main import main


def parse_args() -> argparse.Namespace:
    """Parse reproducible profile paths applied from left to right."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML profile to apply; may be supplied multiple times",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="validated runtime override; YAML scalar/list syntax is supported",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {}
    if args.config:
        overrides.update(apply_config_files(args.config))
    if args.set:
        overrides.update(apply_config_overrides(args.set))
    if overrides:
        print("Applied configuration profiles:")
        for key in sorted(overrides):
            print(f"  {key}={overrides[key]!r}")
    main()

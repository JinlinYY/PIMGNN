"""Evaluate one saved PSMI checkpoint without training or weight updates."""

try:
    from _bootstrap import add_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_src_to_path

add_src_to_path()

import argparse
import json

from psmi.configuration import apply_config_files, apply_config_overrides
from psmi.reproduction import evaluate_saved_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse a checkpoint evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None, help="cpu, cuda, or a CUDA device such as cuda:0")
    parser.add_argument("--reference-predictions", default=None)
    parser.add_argument("--fg-corpus", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--allow-input-hash-mismatch", action="store_true")
    parser.add_argument("--allow-derived-scalers", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    apply_config_files(args.config)
    apply_config_overrides(args.set)
    report = evaluate_saved_checkpoint(
        args.checkpoint,
        args.output_dir,
        device=args.device,
        reference_predictions=args.reference_predictions,
        functional_group_corpus=args.fg_corpus,
        require_hash_match=not args.allow_input_hash_mismatch,
        allow_derived_scalers=args.allow_derived_scalers,
        make_plots=not args.no_plots,
    )
    print(json.dumps(report["metrics"], indent=2))

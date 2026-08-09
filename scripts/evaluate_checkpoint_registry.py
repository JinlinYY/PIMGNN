"""Evaluate registered PSMI checkpoints in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

try:
    from _bootstrap import add_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from psmi.reproduction import summarize_reports


def parse_args() -> argparse.Namespace:
    """Parse registry selection and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/reproduction/multiseed_checkpoint_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/checkpoint_evaluation/registry_runs",
    )
    parser.add_argument("--only", action="append", default=[], help="Run id or group; repeatable")
    parser.add_argument("--device", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    """Resolve a registry path relative to the repository root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def main() -> int:
    """Evaluate selected checkpoints and write one combined metric table."""
    args = parse_args()
    registry_path = _resolve(args.registry)
    with registry_path.open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    runs = registry.get("runs", [])
    if args.list:
        for run in runs:
            print(f"{run['id']}: group={run['group']} checkpoint={run['checkpoint']}")
        return 0

    selectors = set(args.only)
    selected = [
        run
        for run in runs
        if not selectors or run["id"] in selectors or run["group"] in selectors
    ]
    if not selected:
        raise ValueError(f"No registry entries match: {sorted(selectors)}")

    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_paths = []
    for run in selected:
        run_output = output_root / run["id"]
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_checkpoint.py"),
            "--config",
            str(_resolve(run["config"])),
            "--checkpoint",
            str(_resolve(run["checkpoint"])),
            "--output-dir",
            str(run_output),
            "--set",
            f"SEED={int(run['seed'])}",
        ]
        for override in run.get("set", []):
            command.extend(["--set", str(override)])
        optional_paths = {
            "reference_predictions": "--reference-predictions",
            "fg_corpus": "--fg-corpus",
        }
        for field, flag in optional_paths.items():
            if run.get(field):
                command.extend([flag, str(_resolve(run[field]))])
        if run.get("allow_input_hash_mismatch", False):
            command.append("--allow-input-hash-mismatch")
        if run.get("allow_derived_scalers", False):
            command.append("--allow-derived-scalers")
        if args.device:
            command.extend(["--device", args.device])
        if args.no_plots:
            command.append("--no-plots")
        print(f"[checkpoint-evaluation] {run['id']}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        report_paths.append(run_output / "reproduction_report.json")

    summary = summarize_reports(report_paths)
    summary_path = output_root / "reproduced_metrics.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved combined metrics: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

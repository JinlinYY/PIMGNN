"""Quantify temperature coverage and repeated-temperature support in PSMI data.

This analysis distinguishes a broad global temperature range from genuine
within-system temperature variation.  The distinction is essential when making
claims about temperature extrapolation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from psmi.data import load_and_prepare_excel  # noqa: E402


def parse_dataset_spec(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Dataset must be specified as NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Dataset name cannot be empty")
    path = Path(raw_path.strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return name, path.resolve()


def summarize_dataset(name: str, path: Path, min_tie_lines: int) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    df, _ = load_and_prepare_excel(
        str(path),
        min_points_per_group=int(min_tie_lines),
        permute_23_aug=False,
    )
    per_system = (
        df.groupby("system_id", sort=True)["T"]
        .agg(n_temperatures="nunique", T_min_K="min", T_max_K="max", n_tie_lines="size")
        .reset_index()
    )
    per_system["temperature_span_K"] = per_system["T_max_K"] - per_system["T_min_K"]

    frequency = (
        df.groupby("T", sort=True)
        .agg(n_tie_lines=("T", "size"), n_systems=("system_id", "nunique"))
        .reset_index()
        .rename(columns={"T": "temperature_K"})
    )

    temperature = df["T"].to_numpy(dtype=float)
    multi = per_system[per_system["n_temperatures"] >= 2]
    quantiles = np.quantile(temperature, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    summary: Dict[str, object] = {
        "dataset": name,
        "path": str(path),
        "min_tie_lines_per_system_temperature": int(min_tie_lines),
        "n_tie_lines": int(len(df)),
        "n_systems": int(df["system_id"].nunique()),
        "n_system_temperature_groups": int(df.groupby(["system_id", "T"]).ngroups),
        "temperature_min_K": float(temperature.min()),
        "temperature_max_K": float(temperature.max()),
        "temperature_mean_K": float(temperature.mean()),
        "temperature_std_K": float(temperature.std(ddof=1)),
        "temperature_quantiles_K": {
            key: float(value)
            for key, value in zip(["q00", "q05", "q25", "q50", "q75", "q95", "q100"], quantiles)
        },
        "n_unique_temperatures": int(df["T"].nunique()),
        "n_single_temperature_systems": int((per_system["n_temperatures"] == 1).sum()),
        "n_multi_temperature_systems": int(len(multi)),
        "multi_temperature_fraction": float(len(multi) / len(per_system)),
        "multi_temperature_span_median_K": float(multi["temperature_span_K"].median()) if len(multi) else None,
        "multi_temperature_span_max_K": float(multi["temperature_span_K"].max()) if len(multi) else None,
    }
    per_system.insert(0, "dataset", name)
    frequency.insert(0, "dataset", name)
    return summary, per_system, frequency


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_temperature_coverage(
    summaries: List[Dict[str, object]],
    frequencies: pd.DataFrame,
    per_system: pd.DataFrame,
    out_dir: Path,
) -> None:
    apply_style()
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))

    for idx, summary in enumerate(summaries):
        name = str(summary["dataset"])
        subset = frequencies[frequencies["dataset"] == name]
        axes[0].plot(
            subset["temperature_K"],
            subset["n_tie_lines"],
            color=colors[idx % len(colors)],
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            label=name,
        )
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Tie-line records")
    axes[0].set_title("a  Temperature coverage", loc="left", fontweight="bold")
    axes[0].legend()

    names = [str(item["dataset"]) for item in summaries]
    single = [int(item["n_single_temperature_systems"]) for item in summaries]
    multi = [int(item["n_multi_temperature_systems"]) for item in summaries]
    x = np.arange(len(names))
    axes[1].bar(x, single, color="#9AA6B2", width=0.62, label="One temperature")
    axes[1].bar(x, multi, bottom=single, color="#E69F00", width=0.62, label="Two or more")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylabel("Chemical systems")
    axes[1].set_title("b  Within-system support", loc="left", fontweight="bold")
    axes[1].legend()

    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_dir / "temperature_coverage.pdf")
    fig.savefig(out_dir / "temperature_coverage.png", dpi=300)
    plt.close(fig)


def write_report(summaries: Iterable[Dict[str, object]], out_path: Path) -> None:
    lines = ["# Temperature variability audit", ""]
    for item in summaries:
        q = item["temperature_quantiles_K"]
        lines.extend(
            [
                f"## {item['dataset']}",
                "",
                f"- Tie-line records: {item['n_tie_lines']}",
                f"- Chemical systems: {item['n_systems']}",
                f"- System-temperature groups: {item['n_system_temperature_groups']}",
                f"- Range: {item['temperature_min_K']:.2f}-{item['temperature_max_K']:.2f} K",
                f"- Mean +/- SD: {item['temperature_mean_K']:.2f} +/- {item['temperature_std_K']:.2f} K",
                f"- Median (IQR): {q['q50']:.2f} K ({q['q25']:.2f}-{q['q75']:.2f} K)",
                f"- Unique reported temperatures: {item['n_unique_temperatures']}",
                f"- Single-temperature systems: {item['n_single_temperature_systems']}",
                f"- Multi-temperature systems: {item['n_multi_temperature_systems']} ({100.0 * item['multi_temperature_fraction']:.2f}%)",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "A broad global temperature range does not by itself demonstrate temperature extrapolation. ",
            "Strict within-system extrapolation requires at least three temperatures for a system so that an extreme temperature can be held out while at least one interior temperature remains available for training.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_spec,
        required=True,
        help="Repeatable NAME=PATH dataset specification",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--min_tie_lines", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_tie_lines < 1:
        raise ValueError("min_tie_lines must be positive")
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    systems: List[pd.DataFrame] = []
    frequencies: List[pd.DataFrame] = []
    for name, path in args.dataset:
        summary, per_system, frequency = summarize_dataset(name, path, args.min_tie_lines)
        summaries.append(summary)
        systems.append(per_system)
        frequencies.append(frequency)

    system_table = pd.concat(systems, ignore_index=True)
    frequency_table = pd.concat(frequencies, ignore_index=True)
    (out_dir / "temperature_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    system_table.to_csv(out_dir / "system_temperature_summary.csv", index=False, encoding="utf-8-sig")
    frequency_table.to_csv(out_dir / "temperature_frequency.csv", index=False, encoding="utf-8-sig")
    write_report(summaries, out_dir / "temperature_variability.md")
    plot_temperature_coverage(summaries, frequency_table, system_table, out_dir)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved temperature audit to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

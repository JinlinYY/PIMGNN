"""Generate the publication figure for same-chemistry temperature extrapolation."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "supporting_information"
    / "s3_additional_evaluation_and_validation"
    / "s3_11_conditional_same_system_temperature_extrapolation"
    / "results"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_DIR.parent / "figures"

PSMI_COLOR = "#D55E00"
BASELINE_COLOR = "#9AA6B2"
DIRECTION_MARKERS = {"cold_extrapolation": "v", "hot_extrapolation": "^"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def make_figure(summary: pd.DataFrame, groups: pd.DataFrame, output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.8))

    scopes = ["cold_extrapolation", "hot_extrapolation", "overall"]
    scope_labels = ["Cold", "Hot", "Combined"]
    methods = ["PSMI", "Nearest observed temperature"]
    colors = {"PSMI": PSMI_COLOR, "Nearest observed temperature": BASELINE_COLOR}
    x = np.arange(len(scopes))
    width = 0.34
    maximum_bar_value = 0.0
    for method_index, method in enumerate(methods):
        values = [
            float(
                summary.loc[
                    (summary["scope"] == scope) & (summary["method"] == method),
                    "rmse",
                ].iloc[0]
            )
            for scope in scopes
        ]
        offset = (method_index - 0.5) * width
        maximum_bar_value = max(maximum_bar_value, max(values))
        bars = axes[0].bar(
            x + offset,
            values,
            width=width * 0.92,
            color=colors[method],
            edgecolor="white",
            linewidth=0.5,
            label=method,
        )
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444444",
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scope_labels)
    axes[0].set_ylabel("Composition RMSE")
    axes[0].set_title("a  Held-out extreme temperatures", loc="left")
    axes[0].set_ylim(0.0, maximum_bar_value * 1.30)
    axes[0].legend(loc="upper center", ncol=2, columnspacing=0.9, handletextpad=0.4)

    for method in methods:
        method_data = groups[groups["method"] == method]
        for direction, subset in method_data.groupby("temperature_direction"):
            axes[1].scatter(
                subset["temperature_gap_K"],
                subset["rmse"],
                s=25 if method == "PSMI" else 18,
                marker=DIRECTION_MARKERS[direction],
                facecolor=colors[method],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.78 if method == "PSMI" else 0.55,
            )
    axes[1].set_xlabel("Distance to nearest training temperature (K)")
    axes[1].set_ylabel("System-temperature RMSE")
    axes[1].set_title("b  Error versus extrapolation distance", loc="left")
    axes[1].set_ylim(0.0, float(groups["rmse"].max()) * 1.28)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=PSMI_COLOR, label="PSMI"),
        Line2D([0], [0], marker="o", linestyle="", color=BASELINE_COLOR, label="Nearest temperature"),
        Line2D([0], [0], marker="v", linestyle="", color="#555555", label="Cold"),
        Line2D([0], [0], marker="^", linestyle="", color="#555555", label="Hot"),
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    fig.tight_layout(w_pad=2.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure_s7.pdf")
    fig.savefig(output_dir / "figure_s7.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir", "--result_dir", dest="result_dir", type=Path, default=DEFAULT_RESULT_DIR
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    args = parser.parse_args()
    result_dir = args.result_dir if args.result_dir.is_absolute() else PROJECT_ROOT / args.result_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    summary = pd.read_csv(result_dir / "summary.csv")
    groups = pd.read_csv(result_dir / "by_system_temperature.csv")
    make_figure(summary, groups, output_dir)
    print(f"Saved publication figure to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

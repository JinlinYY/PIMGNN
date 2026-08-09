"""Plot Figure 3c-d for the industrial extraction case studies.

All experimental and model endpoints are read from the chapter-aligned
application workbook. Experimental observations are drawn above the model
predictions, and tie lines remain visually subordinate to the phase endpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["Experiment", "PSMI", "COSMO-RS", "NRTL", "UNIFAC"]
PREDICTION_DRAW_ORDER = ["UNIFAC", "NRTL", "COSMO-RS", "PSMI"]
MODEL_COLORS = {
    "Experiment": "#000000",
    "PSMI": "#E64B35",
    "COSMO-RS": "#4DBBD5",
    "NRTL": "#00A087",
    "UNIFAC": "#3C5488",
}
PHASE_MARKERS = {"Extract": "o", "Raffinate": "^"}
# Matplotlib interprets ``s`` as marker area. A 1.5x linear scale therefore
# requires 2.25x the base area: 36 * 1.5^2 = 81 pt^2.
ORIGINAL_MARKER_AREA = 36.0
MARKER_LINEAR_SCALE = 1.5
MARKER_AREA = ORIGINAL_MARKER_AREA * MARKER_LINEAR_SCALE**2


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    public_experiment_root = (
        project_root
        / "experiments"
        / "section_3_results"
        / "3_4_industrial_extraction_design"
    )
    public_data = (
        public_experiment_root
        / "3_4_1_sulfolane_aromatic_extraction"
        / "data"
        / "industrial_extraction_lle_data.xlsx"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=public_data,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "experiments"
            / "section_3_results"
            / "3_4_industrial_extraction_design"
            / "figures"
        ),
    )
    parser.add_argument(
        "--experiment-figure-dir",
        type=Path,
        default=public_experiment_root / "figures",
    )
    return parser.parse_args()


def ternary_xy(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map component fractions to an equilateral ternary triangle."""
    total = x1 + x2 + x3
    if np.any(~np.isfinite(total)) or np.any(total <= 0):
        raise ValueError("All ternary compositions must have a positive finite sum.")
    x1, x2, x3 = x1 / total, x2 / total, x3 / total
    return x2 + 0.5 * x3, (np.sqrt(3.0) / 2.0) * x3


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_excel(path)
    required = {
        "LLE system NO.",
        "Model",
        "Component 1",
        "Component 2",
        "Component 3",
        "T/K",
        "Ex1",
        "Ex2",
        "Ex3",
        "Rx1",
        "Rx2",
        "Rx3",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.copy()
    data["Model"] = data["Model"].replace({"COSMO-rs": "COSMO-RS"})
    unknown = sorted(set(data["Model"].dropna()) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unexpected model labels: {unknown}")

    extract_error = (data[["Ex1", "Ex2", "Ex3"]].sum(axis=1) - 1.0).abs()
    raffinate_error = (data[["Rx1", "Rx2", "Rx3"]].sum(axis=1) - 1.0).abs()
    if max(extract_error.max(), raffinate_error.max()) > 2e-3:
        raise ValueError("At least one phase composition differs from unity by more than 0.002.")
    return data


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 22.0,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_triangle(ax: plt.Axes, component_names: tuple[str, str, str]) -> None:
    height = np.sqrt(3.0) / 2.0
    ax.plot([0.0, 1.0], [0.0, 0.0], color="#222222", lw=0.9, zorder=20)
    ax.plot([0.0, 0.5], [0.0, height], color="#222222", lw=0.9, zorder=20)
    ax.plot([0.5, 1.0], [height, 0.0], color="#222222", lw=0.9, zorder=20)

    c1, c2, c3 = component_names
    ax.text(-0.045, -0.035, c1, ha="right", va="top", fontsize=20.4)
    ax.text(1.045, -0.035, c2, ha="left", va="top", fontsize=20.4)
    ax.text(0.5, height + 0.035, c3, ha="center", va="bottom", fontsize=20.4)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.16, height + 0.30)
    ax.set_aspect("equal")
    ax.axis("off")


def phase_coordinates(model_data: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ex = ternary_xy(
        model_data["Ex1"].to_numpy(float),
        model_data["Ex2"].to_numpy(float),
        model_data["Ex3"].to_numpy(float),
    )
    rx = ternary_xy(
        model_data["Rx1"].to_numpy(float),
        model_data["Rx2"].to_numpy(float),
        model_data["Rx3"].to_numpy(float),
    )
    return {"Extract": ex, "Raffinate": rx}


def draw_model(ax: plt.Axes, model_data: pd.DataFrame, model: str) -> None:
    coordinates = phase_coordinates(model_data)
    ex_x, ex_y = coordinates["Extract"]
    rx_x, rx_y = coordinates["Raffinate"]
    color = MODEL_COLORS[model]

    # Tie lines are intentionally subordinate to markers.  The experimental
    # lines are slightly more visible but remain light gray.
    if model == "Experiment":
        tie_color = "#7A7A7A"
        tie_alpha = 0.24
        tie_width = 0.55
        tie_zorder = 2
    else:
        tie_color = color
        tie_alpha = 0.11
        tie_width = 0.45
        tie_zorder = 1

    for x1, y1, x2, y2 in zip(ex_x, ex_y, rx_x, rx_y):
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=tie_color,
            alpha=tie_alpha,
            lw=tie_width,
            solid_capstyle="round",
            zorder=tie_zorder,
        )

    is_experiment = model == "Experiment"
    marker_zorder = 12 if is_experiment else 5 + PREDICTION_DRAW_ORDER.index(model)
    edge_color = "white" if is_experiment else "none"
    edge_width = 0.8 if is_experiment else 0.0
    for phase, (x, y) in coordinates.items():
        ax.scatter(
            x,
            y,
            s=MARKER_AREA,
            marker=PHASE_MARKERS[phase],
            facecolor=color,
            edgecolor=edge_color,
            linewidth=edge_width,
            alpha=0.98,
            zorder=marker_zorder,
        )


def legend_handles(system_data: pd.DataFrame) -> list[mlines.Line2D]:
    handles: list[mlines.Line2D] = []
    present = set(system_data["Model"])
    # sqrt(81) = 9 pt, matching the data marker's nominal linear size.
    for model in MODEL_ORDER:
        if model not in present:
            continue
        for phase in ("Extract", "Raffinate"):
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker=PHASE_MARKERS[phase],
                    markersize=np.sqrt(MARKER_AREA),
                    markerfacecolor=MODEL_COLORS[model],
                    markeredgecolor="white" if model == "Experiment" else "none",
                    markeredgewidth=0.6 if model == "Experiment" else 0.0,
                    label=f"{model} {phase}",
                )
            )
    return handles


def draw_panel(
    ax: plt.Axes,
    legend_ax: plt.Axes,
    system_data: pd.DataFrame,
    panel_label: str,
) -> None:
    system_data = system_data.copy()
    c1 = str(system_data["Component 1"].iloc[0])
    c2 = str(system_data["Component 2"].iloc[0])
    c3 = str(system_data["Component 3"].iloc[0])
    temperature = float(system_data["T/K"].iloc[0])

    draw_triangle(ax, (c1, c2, c3))
    for model in PREDICTION_DRAW_ORDER:
        model_data = system_data[system_data["Model"] == model]
        if not model_data.empty:
            draw_model(ax, model_data, model)
    experiment = system_data[system_data["Model"] == "Experiment"]
    if not experiment.empty:
        draw_model(ax, experiment, "Experiment")

    ax.set_title(f"{c1} + {c2} + {c3}", fontsize=22.4, pad=20)
    ax.text(
        0.5,
        np.sqrt(3.0) / 2.0 + 0.145,
        f"{temperature:.2f} K",
        ha="center",
        va="bottom",
        fontsize=20.4,
    )
    ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes, fontsize=36, fontweight="bold")

    legend_ax.axis("off")
    legend_ax.legend(
        handles=legend_handles(system_data),
        loc="upper left",
        frameon=False,
        fontsize=20.0,
        handletextpad=0.5,
        labelspacing=0.48,
        borderaxespad=0.0,
    )


def save_figure(fig: plt.Figure, output_dir: Path, experiment_figure_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_figure_dir.mkdir(parents=True, exist_ok=True)
    for directory in (output_dir, experiment_figure_dir):
        png = directory / "figure3cd_industrial_extraction_validation.png"
        pdf = directory / "figure3cd_industrial_extraction_validation.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved: {png}")
        print(f"Saved: {pdf}")


def main() -> None:
    args = parse_args()
    setup_style()
    data = load_data(args.data)

    # Match the submitted Figure 3 order: c = system 2, d = system 1.
    panel_systems = [(2, "c"), (1, "d")]
    fig = plt.figure(figsize=(19.5, 6.2), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1.18, 0.92, 1.18, 0.92),
        left=0.035,
        right=0.995,
        bottom=0.08,
        top=0.91,
        wspace=0.02,
    )

    for panel_index, (system_id, label) in enumerate(panel_systems):
        ax = fig.add_subplot(grid[0, panel_index * 2])
        legend_ax = fig.add_subplot(grid[0, panel_index * 2 + 1])
        system_data = data[data["LLE system NO."] == system_id]
        if system_data.empty:
            raise ValueError(f"No rows found for LLE system {system_id}.")
        draw_panel(ax, legend_ax, system_data, label)

    save_figure(fig, args.output_dir, args.experiment_figure_dir)
    plt.close(fig)

    counts = data.groupby(["LLE system NO.", "Model"]).size().unstack(fill_value=0)
    print("Rows used by system/model:")
    print(counts.to_string())
    print(f"Marker area: {MARKER_AREA:.0f} pt^2 (linear scale: {MARKER_LINEAR_SCALE:.1f}x)")


if __name__ == "__main__":
    main()

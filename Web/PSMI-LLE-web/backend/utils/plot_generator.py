"""Generate ternary plots from PSMI curve predictions."""

from __future__ import annotations

import base64
import math
from io import BytesIO
from typing import Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def apply_publication_style() -> None:
    """Apply a compact, publication-friendly plotting style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def ternary_to_xy(composition: Sequence[float]) -> Tuple[float, float]:
    """Map a normalized ternary composition to Cartesian coordinates."""
    values = np.asarray(composition, dtype=float)
    total = float(values.sum())
    if total <= 1e-12:
        values = np.full(3, 1.0 / 3.0)
    else:
        values = values / total
    return float(values[1] + 0.5 * values[2]), float(math.sqrt(3) * 0.5 * values[2])


def _draw_axes(ax, labels: Sequence[str]) -> None:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2.0]])
    closed = np.vstack([vertices, vertices[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, math.sqrt(3) / 2.0 + 0.08)
    ax.axis("off")
    ax.text(-0.02, -0.035, labels[0], ha="right", va="top")
    ax.text(1.02, -0.035, labels[1], ha="left", va="top")
    ax.text(0.5, math.sqrt(3) / 2.0 + 0.04, labels[2], ha="center", va="bottom")

    for tick in (0.2, 0.4, 0.6, 0.8):
        tick_style = {"fontsize": 8, "color": "#555555"}
        ax.text(tick, -0.03, f"{tick:.1f}", ha="center", va="top", **tick_style)
        ax.text(0.5 * tick - 0.02, math.sqrt(3) * 0.5 * tick, f"{tick:.1f}", ha="right", **tick_style)
        ax.text(1.0 - 0.5 * tick + 0.02, math.sqrt(3) * 0.5 * tick, f"{tick:.1f}", ha="left", **tick_style)


def generate_ternary_plot(
    t_grid: np.ndarray,
    extract: np.ndarray,
    raffinate: np.ndarray,
    temperature: float,
    pressure: float | None,
    labels: Sequence[str],
    tie_lines_count: int,
) -> str:
    """Return a base64 PNG for one predicted ternary LLE curve."""
    apply_publication_style()
    extract_xy = np.asarray([ternary_to_xy(row) for row in extract])
    raffinate_xy = np.asarray([ternary_to_xy(row) for row in raffinate])
    count = max(1, min(int(tie_lines_count), len(t_grid)))
    indices = np.linspace(0, len(t_grid) - 1, count, dtype=int)

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    _draw_axes(ax, labels)
    ax.plot(extract_xy[:, 0], extract_xy[:, 1], linewidth=2.0, label="Extract curve")
    ax.plot(raffinate_xy[:, 0], raffinate_xy[:, 1], linewidth=2.0, label="Raffinate curve")
    ax.scatter(extract_xy[indices, 0], extract_xy[indices, 1], s=16, marker="^")
    ax.scatter(raffinate_xy[indices, 0], raffinate_xy[indices, 1], s=16, marker="v")
    for position, index in enumerate(indices):
        ax.plot(
            [extract_xy[index, 0], raffinate_xy[index, 0]],
            [extract_xy[index, 1], raffinate_xy[index, 1]],
            linewidth=1.0,
            linestyle="--",
            label="Predicted tie-lines" if position == 0 else None,
        )
    title = f"PSMI prediction | T={temperature:.2f} K"
    if pressure is not None:
        title += f" | P={pressure:.3f} kPa"
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.9, bottom=0.1)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=260)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

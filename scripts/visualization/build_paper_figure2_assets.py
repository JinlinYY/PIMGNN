"""Rebuild manuscript Figure 2 from section-aligned experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import add_src_to_path

add_src_to_path()

from psmi.eval_explain import save_barh
from psmi.plot_test_viz_from_csv_extra import (
    apply_style,
    normalize_columns,
    parity_plot_combined,
    plot_group_ternary_from_csv,
)
from psmi.viz_advanced import plot_combined_rank_heatmaps


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _asset_root(project_root: Path) -> Path:
    public_root = project_root / "public_release" / "PSMI-public"
    return public_root if public_root.is_dir() else project_root


def _read_importance(path: Path) -> tuple[list[str], np.ndarray]:
    table = pd.read_csv(path)
    required = {"name", "importance"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return table["name"].astype(str).tolist(), table["importance"].to_numpy(float)


def _plot_system(prediction_csv: Path, system_id: int, output_path: Path) -> None:
    table = normalize_columns(pd.read_csv(prediction_csv))
    selected = table.loc[table["system_id"] == system_id].copy()
    if selected.empty:
        raise ValueError(f"System {system_id} is absent from {prediction_csv}")
    temperatures = sorted(selected["T"].unique())
    if len(temperatures) != 1:
        raise ValueError(
            f"Expected one temperature for system {system_id}; found {temperatures}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_group_ternary_from_csv(selected, str(output_path), draw_tielines_max=18)


def build(asset_root: Path, output_root: Path | None = None) -> None:
    main_experiment = (
        asset_root
        / "experiments"
        / "section_3_results"
        / "3_1_lle_prediction"
        / "main_benchmark"
    )
    mechanism_experiment = (
        asset_root
        / "experiments"
        / "section_3_results"
        / "3_2_molecular_interaction_mechanisms"
    )
    figure2a_csv = main_experiment / "data" / "figure_2a_predictions.csv"
    saliency_root = mechanism_experiment / "data" / "global_saliency"

    if output_root is None:
        main_figure_dir = main_experiment / "figures"
        mechanism_figure_dir = mechanism_experiment / "figures"
    else:
        main_figure_dir = output_root / "section_3_1"
        mechanism_figure_dir = output_root / "section_3_2"

    apply_style(1.7)

    panel_a = main_figure_dir / "figure_2a_parity.png"
    panel_a.parent.mkdir(parents=True, exist_ok=True)
    parity_plot_combined(normalize_columns(pd.read_csv(figure2a_csv)), str(panel_a))

    mix_names, mix_values = _read_importance(
        saliency_root / "mix_edge_feature_importance_grad.csv"
    )
    panel_b = mechanism_figure_dir / "figure_2b_mixture_edge_importance.png"
    panel_b.parent.mkdir(parents=True, exist_ok=True)
    save_barh(
        mix_names,
        mix_values,
        str(panel_b),
        "Mixture edge-feature importance",
        topk=10,
        font_scale=1.0,
    )

    importance: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}
    for component in ("g1", "g2", "g3"):
        for feature_type, filename_type in (
            ("node", "atom"),
            ("edge", "bond"),
            ("glob", "global"),
        ):
            names, values = _read_importance(
                saliency_root
                / f"{component}_{filename_type}_feature_importance.csv"
            )
            key = f"{component}_{feature_type}_feat"
            feature_names[key] = names
            importance[key] = values

    panel_c = mechanism_figure_dir / "figure_2c_feature_rank_heatmap.png"
    panel_c.parent.mkdir(parents=True, exist_ok=True)
    plot_combined_rank_heatmaps(
        importance,
        feature_names,
        str(panel_c),
        top_k=12,
        color_scheme="nature_green",
        font_scale=1.0,
    )

    _plot_system(
        main_experiment / "data" / "figure_2d_system_22_source_predictions.csv",
        22,
        main_figure_dir / "figure_2d_system_22.png",
    )
    _plot_system(
        main_experiment / "data" / "figure_2d_system_826_source_predictions.csv",
        826,
        main_figure_dir / "figure_2d_system_826.png",
    )

    print(f"[OK] Figure 2a and 2d were written to {main_figure_dir}")
    print(f"[OK] Figure 2b and 2c were written to {mechanism_figure_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Figure 2 assets from manuscript-section experiments."
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Repository containing experiments/ and results/; detected by default.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional staging directory; by default figures are written to their experiments.",
    )
    args = parser.parse_args()

    project_root = _project_root()
    asset_root = args.asset_root.resolve() if args.asset_root else _asset_root(project_root)
    output_root = args.output_root.resolve() if args.output_root else None
    build(asset_root, output_root)


if __name__ == "__main__":
    main()

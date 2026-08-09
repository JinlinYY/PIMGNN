"""Build Supplementary Figures S1-S3 from saved pointwise predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import add_src_to_path

add_src_to_path()

from psmi.plot_test_viz_from_csv_extra import (
    add_error_columns_for_group_plots,
    apply_style,
    normalize_columns,
    plot_bland_altman_combined,
    plot_cdf_abs_error,
    plot_error_hist_kde_combined,
    plot_residual_vs_true_combined,
    plot_sum_to_one_combined,
    plot_violin_combined_categories,
)


def build(prediction_csv: Path, output_dir: Path) -> None:
    table = normalize_columns(pd.read_csv(prediction_csv))
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)

    apply_style(1.7)
    plot_violin_combined_categories(
        add_error_columns_for_group_plots(table),
        str(output_dir / "figure_s1_category_error_distributions.png"),
        top_n=12,
        kind="violin",
    )
    plot_bland_altman_combined(
        table,
        str(output_dir / "figure_s2_bland_altman.png"),
        max_points=8000,
        seed=0,
    )
    plot_error_hist_kde_combined(
        table,
        str(output_dir / "figure_s3_residual_distributions.png"),
    )
    plot_cdf_abs_error(table, str(diagnostics / "cdf_abs_error.png"))
    plot_residual_vs_true_combined(
        table,
        str(diagnostics / "residual_vs_true_combined.png"),
        max_points=8000,
        seed=0,
    )
    plot_sum_to_one_combined(table, str(diagnostics / "sum_to_one_combined.png"))
    print(f"[OK] Supplementary error figures were written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Supplementary Figures S1-S3 from saved predictions."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    build(args.predictions.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Format aggregated common-baseline metrics for reports and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAME_MAP = {
    "mlp": "MLP (3 layers)",
    "ann": "MLP (2 layers)",
    "lstm": "LSTM",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "transformer": "Transformer",
    "tabnet": "TabNet",
    "tabknet": "TabKNet",
    "smiles_rnn": "SMILES-RNN",
    "gnn": "GNN",
}

METRICS_E = ["MAE_E", "RMSE_E", "R2_E"]
METRICS_R = ["MAE_R", "RMSE_R", "R2_R"]
METRICS_OVERALL = ["MAE", "RMSE", "R2"]

METRIC_SYNONYMS = {
    "test_rmse_E": "RMSE_E",
    "test_rmse_R": "RMSE_R",
    "test_rmse": "RMSE",
    "rmse_e": "RMSE_E",
    "rmse_r": "RMSE_R",
    "rmse": "RMSE",
}


def fmt(mean: float | None, std: float | None) -> str:
    """Return a compact mean-and-standard-deviation cell."""
    if mean is None or std is None:
        return ""
    try:
        if np.isnan(mean) or np.isnan(std):
            return ""
    except TypeError:
        return ""
    return f"{float(mean):.4f}({float(std):.4f})"


def build_formatted_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a long metric table into one publication-friendly row per model."""
    data = df.copy()
    data.columns = [str(column).strip() for column in data.columns]
    required_columns = {"Model", "Metric", "Mean", "Std"}
    missing = required_columns.difference(data.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")

    data["Metric"] = data["Metric"].astype(str).str.strip()
    data["Metric_norm"] = data["Metric"].map(METRIC_SYNONYMS).fillna(data["Metric"])
    data = data[data["Metric_norm"].isin(METRICS_E + METRICS_R + METRICS_OVERALL)]

    normalized_names = data["Model"].astype(str).str.strip()
    data["Model_display"] = normalized_names.map(MODEL_NAME_MAP).fillna(normalized_names)
    models = list(dict.fromkeys(data["Model_display"].tolist()))
    values = {
        (row["Model_display"], row["Metric_norm"]): (row["Mean"], row["Std"])
        for _, row in data.iterrows()
    }

    rows = []
    for model in models:
        row = {"Model": model}
        for phase, metrics in (("E_phase", METRICS_E), ("R_phase", METRICS_R)):
            for metric in metrics:
                name = metric.split("_")[0]
                row[f"{phase}_{name}"] = fmt(*values.get((model, metric), (None, None)))
        for metric in METRICS_OVERALL:
            row[f"Overall_{metric}"] = fmt(*values.get((model, metric), (None, None)))
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Read an aggregated result file and write its formatted counterpart."""
    parser = argparse.ArgumentParser(
        description="Format multiple_seeds_summary.csv for reports and figures."
    )
    parser.add_argument("--input", type=Path, help="Input summary CSV path.")
    parser.add_argument("--output", type=Path, help="Output formatted CSV path.")
    args = parser.parse_args()

    try:
        from . import config as baseline_config

        default_directory = Path(baseline_config.OUT_DIR)
    except ImportError:
        default_directory = Path("baseline_multi_seed")

    input_path = args.input or default_directory / "multiple_seeds_summary.csv"
    output_path = args.output or default_directory / "multiple_seeds_summary_formatted.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output = build_formatted_table(pd.read_csv(input_path, encoding="utf-8-sig"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved formatted summary to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze the datasets distributed with PSMI.

Run from the repository root:

    python scripts/analysis/analyze_dataset_distribution.py

The script intentionally separates:
1. raw workbook rows;
2. validated records immediately before the tie-line-density filter;
3. filtered analysis records after psmi.data.load_and_prepare_excel(...);
4. training augmentation, which is reported separately from experimental data.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

from psmi.data import load_and_prepare_excel  # noqa: E402
from psmi.utils import canonicalize_smiles  # noqa: E402


DATASETS = [
    {
        "dataset_id": "Curated IL-LLE",
        "dataset_name": "Curated ionic-liquid LLE workbook",
        "path": ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles.xlsx",
    },
    {
        "dataset_id": "Expanded literature LLE",
        "dataset_name": "Expanded literature-collected LLE workbook",
        "path": ROOT / "datasets" / "processed" / "LLE-literature-data-boosted.xlsx",
    },
]

SYSTEM_CANDIDATES = [
    "system_id",
    "system id",
    "System ID",
    "System_ID",
    "LLE system NO.",
    "LLE system NO",
    "LLE system No.",
    "LLE system No",
    "LLE system number",
    "LLE system#",
    "LLE system #",
    "System NO.",
    "System No.",
    "System NO",
    "System No",
    "System number",
]
TEMP_CANDIDATES = [
    "T/K",
    "T / K",
    "T (K)",
    "T",
    "Temp",
    "Temperature",
    "Temperature/K",
    "Temperature (K)",
]
SMILES_CANDIDATES = {
    "component1": [
        "IL (Component 1) full name SMILES",
        "IL (Component 1) SMILES",
        "Component 1 SMILES",
        "Comp 1 SMILES",
        "Component1-SMILES",
        "Component1 SMILES",
        "smiles1",
        "SMILES1",
        "SMILES 1",
    ],
    "component2": [
        "Component 2 SMILES",
        "Comp 2 SMILES",
        "Component2-SMILES",
        "Component2 SMILES",
        "smiles2",
        "SMILES2",
        "SMILES 2",
    ],
    "component3": [
        "Component 3 SMILES",
        "Comp 3 SMILES",
        "Component3-SMILES",
        "Component3 SMILES",
        "smiles3",
        "SMILES3",
        "SMILES 3",
    ],
}
NAME_CANDIDATES = {
    "component1": [
        "IL (Component 1) full name",
        "Component1",
        "Component 1",
        "IL full name",
    ],
    "component2": ["Component 2", "Component2"],
    "component3": ["Component 3", "Component3"],
}
FAMILY_CANDIDATES = {
    "component2": ["Family of component 2", "Component 2 family", "Family2"],
    "component3": ["Family of component 3", "Component 3 family", "Family3"],
}


@dataclass
class PreparedDataset:
    dataset_id: str
    dataset_name: str
    source_file: Path
    raw: pd.DataFrame
    validated: pd.DataFrame
    filtered: pd.DataFrame
    raw_cols: dict[str, str | None]


def norm_col(value: object) -> str:
    return " ".join(str(value).strip().replace("\n", " ").replace("\r", " ").split())


def compact(value: str) -> str:
    return "".join(ch for ch in norm_col(value).lower() if ch.isalnum())


def find_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns)
    exact = {norm_col(c).lower(): c for c in cols}
    compact_map = {compact(c): c for c in cols}
    for cand in candidates:
        key = norm_col(cand).lower()
        if key in exact:
            return exact[key]
        ckey = compact(cand)
        if ckey in compact_map:
            return compact_map[ckey]
    return None


def clean_string_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.mask(out.str.lower().isin(["", "nan", "none", "<na>"]))


def nunique_clean(df: pd.DataFrame, col: str | None) -> int:
    if col is None or col not in df.columns:
        return 0
    return int(clean_string_series(df[col]).dropna().nunique())


def read_raw_excel(path: Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = pd.read_excel(path)
    df.columns = [norm_col(c) for c in df.columns]
    cols: dict[str, str | None] = {
        "system": find_col(df.columns, SYSTEM_CANDIDATES),
        "T": find_col(df.columns, TEMP_CANDIDATES),
    }
    for role, candidates in SMILES_CANDIDATES.items():
        cols[f"{role}_smiles"] = find_col(df.columns, candidates)
    for role, candidates in NAME_CANDIDATES.items():
        cols[f"{role}_name"] = find_col(df.columns, candidates)
    for role, candidates in FAMILY_CANDIDATES.items():
        cols[f"{role}_family"] = find_col(df.columns, candidates)
    return df, cols


def normalize_filtered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["system_id"] = out["system_id"].astype(str)
    out["T"] = pd.to_numeric(out["T"], errors="coerce")
    return out


def validate_before_density_filter(
    raw: pd.DataFrame,
    raw_cols: dict[str, str | None],
) -> pd.DataFrame:
    """Reproduce loader validation immediately before density filtering."""
    required = {
        "system_id": raw_cols["system"],
        "T": raw_cols["T"],
        "smiles1": raw_cols["component1_smiles"],
        "smiles2": raw_cols["component2_smiles"],
        "smiles3": raw_cols["component3_smiles"],
    }
    for name in ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]:
        candidates = [
            name,
            name.upper(),
            name.lower(),
            name.replace("x", "X"),
            name.replace("X", "x"),
            name.replace("x", "").replace("X", ""),
        ]
        required[name] = find_col(raw.columns, candidates)

    missing = [name for name, source in required.items() if source is None]
    if missing:
        raise KeyError(f"Cannot validate pre-density records; missing columns: {missing}")

    work = raw.rename(columns={source: name for name, source in required.items()}).copy()
    pressure_col = find_col(
        raw.columns,
        [
            "P/kPa",
            "P / kPa",
            "P(kPa)",
            "P",
            "Pressure",
            "Pressure/kPa",
            "Pressure (kPa)",
            "P/bar",
            "P(bar)",
            "P / bar",
        ],
    )
    if pressure_col is not None:
        if pressure_col != "P":
            work = work.rename(columns={pressure_col: "P"})
        work["P"] = pd.to_numeric(work["P"], errors="coerce")
        if work["P"].isnull().all():
            work["P"] = 101.325
        else:
            work["P"] = work["P"].fillna(101.325)
    else:
        work["P"] = 101.325

    for column in ["smiles1", "smiles2", "smiles3"]:
        work[column] = work[column].astype(str).map(canonicalize_smiles)
    work = work[
        (work["smiles1"] != "")
        & (work["smiles2"] != "")
        & (work["smiles3"] != "")
    ].copy()

    numeric_columns = ["T", "P", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work.dropna(subset=numeric_columns).copy()


def prepare_dataset(spec: dict[str, object], min_points_per_group: int) -> PreparedDataset:
    path = Path(spec["path"])
    raw, cols = read_raw_excel(path)
    validated = validate_before_density_filter(raw, cols)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered, _ = load_and_prepare_excel(
            str(path),
            min_points_per_group=min_points_per_group,
            permute_23_aug=False,
        )

    return PreparedDataset(
        dataset_id=str(spec["dataset_id"]),
        dataset_name=str(spec["dataset_name"]),
        source_file=path,
        raw=raw,
        validated=validated,
        filtered=normalize_filtered(filtered),
        raw_cols=cols,
    )


def points_summary(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return {
            "n_systems": 0,
            "mean_points_per_system": math.nan,
            "std_points_per_system": math.nan,
            "min_points_per_system": math.nan,
            "q1_points_per_system": math.nan,
            "median_points_per_system": math.nan,
            "q3_points_per_system": math.nan,
            "max_points_per_system": math.nan,
            "iqr_points_per_system": math.nan,
        }
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return {
        "n_systems": int(values.shape[0]),
        "mean_points_per_system": float(values.mean()),
        "std_points_per_system": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min_points_per_system": int(values.min()),
        "q1_points_per_system": q1,
        "median_points_per_system": float(values.median()),
        "q3_points_per_system": q3,
        "max_points_per_system": int(values.max()),
        "iqr_points_per_system": float(q3 - q1),
    }


def raw_system_counts(prepared: PreparedDataset) -> pd.Series:
    system_col = prepared.raw_cols["system"]
    if system_col is None:
        return pd.Series(dtype=int)
    return prepared.raw.groupby(system_col, dropna=True).size()


def filtered_system_counts(prepared: PreparedDataset) -> pd.Series:
    return prepared.filtered.groupby("system_id", dropna=True).size()


def overview_rows(prepared: PreparedDataset, min_points_per_group: int) -> list[dict[str, object]]:
    rows = []
    for stage, df in [
        ("raw_workbook", prepared.raw),
        ("validated_pre_density", prepared.validated),
        ("filtered_min6_no_aug", prepared.filtered),
    ]:
        if stage == "raw_workbook":
            system_col = prepared.raw_cols["system"]
            t_col = prepared.raw_cols["T"]
        else:
            system_col = "system_id"
            t_col = "T"
        row: dict[str, object] = {
            "dataset_id": prepared.dataset_id,
            "dataset_name": prepared.dataset_name,
            "stage": stage,
            "source_file": str(prepared.source_file.relative_to(ROOT)),
            "min_points_per_system_temperature_group": (
                min_points_per_group if stage == "filtered_min6_no_aug" else ""
            ),
            "experimental_or_analysis_rows": int(len(df)),
            "training_rows_if_component23_swap_augmented": int(len(df) * 2) if stage == "filtered_min6_no_aug" else "",
        }
        if system_col and system_col in df.columns:
            row["unique_system_id"] = int(df[system_col].dropna().nunique())
        else:
            row["unique_system_id"] = ""
        if system_col and t_col and system_col in df.columns and t_col in df.columns:
            row["unique_system_temperature_groups"] = int(df[[system_col, t_col]].dropna().drop_duplicates().shape[0])
        else:
            row["unique_system_temperature_groups"] = ""
        if t_col and t_col in df.columns:
            t_values = pd.to_numeric(df[t_col], errors="coerce").dropna()
            row["temperature_min_K"] = float(t_values.min()) if len(t_values) else ""
            row["temperature_max_K"] = float(t_values.max()) if len(t_values) else ""
            row["unique_temperatures"] = int(t_values.nunique()) if len(t_values) else 0
        else:
            row["temperature_min_K"] = ""
            row["temperature_max_K"] = ""
            row["unique_temperatures"] = ""
        rows.append(row)
    return rows


def manuscript_table_s15(overview: pd.DataFrame) -> pd.DataFrame:
    """Map repository counting stages to the final manuscript Table S15."""
    selections = [
        ("Curated IL-LLE", "Before filtering", "validated_pre_density"),
        ("Curated IL-LLE", "After filtering", "filtered_min6_no_aug"),
        ("Expanded literature LLE", "Before filtering", "raw_workbook"),
        ("Expanded literature LLE", "After filtering", "filtered_min6_no_aug"),
    ]
    rows: list[dict[str, object]] = []
    for dataset_id, paper_stage, repository_stage in selections:
        match = overview[
            (overview["dataset_id"] == dataset_id)
            & (overview["stage"] == repository_stage)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one overview row for {dataset_id}/{repository_stage}, got {len(match)}"
            )
        source = match.iloc[0]
        rows.append(
            {
                "paper_item": "Table S15",
                "dataset_id": dataset_id,
                "paper_stage": paper_stage,
                "repository_stage": repository_stage,
                "records": int(source["experimental_or_analysis_rows"]),
                "systems": int(source["unique_system_id"]),
                "system_temperature_groups": int(source["unique_system_temperature_groups"]),
            }
        )
    return pd.DataFrame(rows)


def points_summary_rows(prepared: PreparedDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage, counts in [
        ("raw_workbook", raw_system_counts(prepared)),
        ("filtered_min6_no_aug", filtered_system_counts(prepared)),
    ]:
        row: dict[str, object] = {
            "dataset_id": prepared.dataset_id,
            "dataset_name": prepared.dataset_name,
            "stage": stage,
        }
        row.update(points_summary(counts))
        rows.append(row)
    return rows


def points_count_rows(prepared: PreparedDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage, counts in [
        ("raw_workbook", raw_system_counts(prepared)),
        ("filtered_min6_no_aug", filtered_system_counts(prepared)),
    ]:
        for system_id, n_points in counts.items():
            rows.append(
                {
                    "dataset_id": prepared.dataset_id,
                    "stage": stage,
                    "system_id": system_id,
                    "n_points": int(n_points),
                }
            )
    return rows


def component_summary_rows(prepared: PreparedDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage, df in [("raw_workbook", prepared.raw), ("filtered_min6_no_aug", prepared.filtered)]:
        if stage == "raw_workbook":
            smiles_cols = {role: prepared.raw_cols[f"{role}_smiles"] for role in SMILES_CANDIDATES}
            name_cols = {role: prepared.raw_cols[f"{role}_name"] for role in NAME_CANDIDATES}
        else:
            smiles_cols = {"component1": "smiles1", "component2": "smiles2", "component3": "smiles3"}
            name_cols = {
                role: prepared.raw_cols.get(f"{role}_name")
                for role in NAME_CANDIDATES
            }
        all_smiles = []
        all_names = []
        role_rows = []
        for role in ["component1", "component2", "component3"]:
            smiles_col = smiles_cols.get(role)
            name_col = name_cols.get(role)
            unique_smiles = nunique_clean(df, smiles_col)
            unique_names = nunique_clean(df, name_col)
            if smiles_col in df.columns:
                all_smiles.append(clean_string_series(df[smiles_col]).dropna())
            if name_col in df.columns:
                all_names.append(clean_string_series(df[name_col]).dropna())
            role_rows.append(
                {
                    "dataset_id": prepared.dataset_id,
                    "dataset_name": prepared.dataset_name,
                    "stage": stage,
                    "component_role": role,
                    "unique_smiles": unique_smiles,
                    "unique_component_names": unique_names,
                    "smiles_column": smiles_col or "",
                    "name_column": name_col or "",
                }
            )
        if all_smiles:
            union_smiles = int(pd.concat(all_smiles, ignore_index=True).nunique())
        else:
            union_smiles = 0
        if all_names:
            union_names = int(pd.concat(all_names, ignore_index=True).nunique())
        else:
            union_names = 0
        for row in role_rows:
            row["union_unique_smiles_all_roles"] = union_smiles
            row["union_unique_names_all_roles"] = union_names
        rows.extend(role_rows)
    return rows


def family_distribution_rows(prepared: PreparedDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage, df in [("raw_workbook", prepared.raw), ("filtered_min6_no_aug", prepared.filtered)]:
        for role in ["component2", "component3"]:
            family_col = prepared.raw_cols.get(f"{role}_family")
            if family_col is None or family_col not in df.columns:
                continue
            values = clean_string_series(df[family_col]).dropna()
            if values.empty:
                continue
            counts = values.value_counts()
            total = float(counts.sum())
            for family, n in counts.items():
                rows.append(
                    {
                        "dataset_id": prepared.dataset_id,
                        "dataset_name": prepared.dataset_name,
                        "stage": stage,
                        "component_role": role,
                        "family": family,
                        "count": int(n),
                        "fraction": float(n / total) if total else 0.0,
                        "family_column": family_col,
                    }
                )
    return rows


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}".rstrip("0").rstrip(".")
            )
    headers = [str(c) for c in work.columns]
    rows = [[str(v) for v in row] for row in work.fillna("").to_numpy()]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(row[i])) for i in range(len(headers))]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + row_lines)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_points_plots(points_counts: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    filtered = points_counts[points_counts["stage"].isin(["raw_workbook", "filtered_min6_no_aug"])]
    datasets = filtered["dataset_id"].drop_duplicates().tolist()
    colors = {"raw_workbook": "#4E79A7", "filtered_min6_no_aug": "#E15759"}
    labels = {"raw_workbook": "Raw workbook", "filtered_min6_no_aug": "Filtered, no augmentation"}

    fig, axes = plt.subplots(len(datasets), 1, figsize=(7.2, 3.4 * len(datasets)), squeeze=False)
    for ax, dataset_id in zip(axes[:, 0], datasets):
        sub = filtered[filtered["dataset_id"] == dataset_id]
        max_points = max(1, int(sub["n_points"].max()))
        bins = np.arange(0.5, max_points + 1.5, 1.0)
        for stage in ["raw_workbook", "filtered_min6_no_aug"]:
            vals = sub[sub["stage"] == stage]["n_points"].to_numpy()
            if len(vals) == 0:
                continue
            ax.hist(
                vals,
                bins=bins,
                alpha=0.58,
                color=colors[stage],
                label=labels[stage],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_title(dataset_id)
        ax.set_xlabel("Tie-line records per system")
        ax.set_ylabel("Number of systems")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "points_per_system_histograms.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    data = []
    tick_labels = []
    box_colors = []
    for dataset_id in datasets:
        for stage in ["raw_workbook", "filtered_min6_no_aug"]:
            vals = filtered[(filtered["dataset_id"] == dataset_id) & (filtered["stage"] == stage)]["n_points"].to_numpy()
            if len(vals):
                data.append(vals)
                tick_labels.append(f"{dataset_id}\n{labels[stage]}")
                box_colors.append(colors[stage])
    box = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("#333333")
    for key in ["whiskers", "caps", "medians"]:
        for item in box[key]:
            item.set_color("#333333")
            item.set_linewidth(0.8)
    ax.set_ylabel("Tie-line records per system")
    ax.set_xticklabels(tick_labels)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    fig.tight_layout()
    path = out_dir / "points_per_system_boxplots.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def save_temperature_plot(prepared: list[PreparedDataset], out_dir: Path) -> Path:
    colors = {"raw_workbook": "#4E79A7", "filtered_min6_no_aug": "#E15759"}
    labels = {"raw_workbook": "Raw workbook", "filtered_min6_no_aug": "Filtered, no augmentation"}
    fig, axes = plt.subplots(len(prepared), 1, figsize=(7.2, 3.4 * len(prepared)), squeeze=False)
    for ax, item in zip(axes[:, 0], prepared):
        raw_t_col = item.raw_cols["T"]
        stages = []
        if raw_t_col is not None:
            stages.append(("raw_workbook", pd.to_numeric(item.raw[raw_t_col], errors="coerce").dropna()))
        stages.append(("filtered_min6_no_aug", pd.to_numeric(item.filtered["T"], errors="coerce").dropna()))
        all_t = pd.concat([values for _, values in stages], ignore_index=True)
        bins = np.linspace(float(all_t.min()), float(all_t.max()), 24) if len(all_t) else 10
        for stage, values in stages:
            ax.hist(
                values.to_numpy(),
                bins=bins,
                alpha=0.58,
                color=colors[stage],
                label=labels[stage],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_title(item.dataset_id)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Number of records")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "temperature_distributions.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_component_plot(component_summary: pd.DataFrame, out_dir: Path) -> Path:
    sub = component_summary[component_summary["stage"] == "filtered_min6_no_aug"].copy()
    dataset_ids = sub["dataset_id"].drop_duplicates().tolist()
    roles = ["component1", "component2", "component3"]
    colors = {"component1": "#59A14F", "component2": "#F28E2B", "component3": "#B07AA1"}
    x = np.arange(len(dataset_ids))
    width = 0.23

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, role in enumerate(roles):
        vals = []
        for dataset_id in dataset_ids:
            match = sub[(sub["dataset_id"] == dataset_id) & (sub["component_role"] == role)]
            vals.append(int(match["unique_smiles"].iloc[0]) if not match.empty else 0)
        ax.bar(x + (i - 1) * width, vals, width=width, label=role.replace("component", "Component "), color=colors[role])
    union_vals = []
    for dataset_id in dataset_ids:
        match = sub[sub["dataset_id"] == dataset_id]
        union_vals.append(int(match["union_unique_smiles_all_roles"].iloc[0]) if not match.empty else 0)
    ax.plot(x, union_vals, color="#111111", marker="o", linewidth=1.2, label="Union across roles")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_ids)
    ax.set_ylabel("Unique canonical SMILES")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = out_dir / "component_unique_smiles_filtered.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_combined_dataset_summary_plot(
    points_counts: pd.DataFrame,
    prepared: list[PreparedDataset],
    component_summary: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Create a polished 2x2 panel figure for SI dataset-distribution reporting."""
    display = {
        "Curated IL-LLE": "Curated",
        "Expanded literature LLE": "Literature",
    }
    stage_display = {
        "raw_workbook": "Raw",
        "filtered_min6_no_aug": "Filtered",
    }
    dataset_ids = points_counts["dataset_id"].drop_duplicates().tolist()
    colors = {
        "Curated IL-LLE": "#2F6DB3",
        "Expanded literature LLE": "#D84A4A",
        "raw_workbook": "#9C755F",
        "filtered_min6_no_aug": "#59A14F",
        "component1": "#2F6DB3",
        "component2": "#D84A4A",
        "component3": "#7A6BB7",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.4))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) Filtered tie-line records per system.
    filtered_counts = points_counts[points_counts["stage"] == "filtered_min6_no_aug"].copy()
    max_points = max(1, int(filtered_counts["n_points"].max()))
    bins = np.arange(0.5, max_points + 1.5, 1.0)
    for dataset_id in dataset_ids:
        vals = filtered_counts[filtered_counts["dataset_id"] == dataset_id]["n_points"].to_numpy()
        if len(vals) == 0:
            continue
        ax_a.hist(
            vals,
            bins=bins,
            alpha=0.62,
            color=colors.get(dataset_id, "#4E79A7"),
            edgecolor="white",
            linewidth=0.5,
            label=display.get(dataset_id, dataset_id),
        )
    ax_a.set_xlabel("Tie-line records per system")
    ax_a.set_ylabel("Number of systems")
    ax_a.set_title("Filtered system coverage")
    ax_a.legend(frameon=False)
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.6)

    # (b) Raw vs filtered points-per-system boxplots.
    data = []
    labels = []
    box_colors = []
    for dataset_id in dataset_ids:
        for stage in ["raw_workbook", "filtered_min6_no_aug"]:
            vals = points_counts[
                (points_counts["dataset_id"] == dataset_id) & (points_counts["stage"] == stage)
            ]["n_points"].to_numpy()
            if len(vals):
                data.append(vals)
                labels.append(f"{display.get(dataset_id, dataset_id)}\n{stage_display[stage]}")
                box_colors.append(colors[stage])
    box = ax_b.boxplot(data, patch_artist=True, widths=0.58, showfliers=False)
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#333333")
    for key in ["whiskers", "caps", "medians"]:
        for item in box[key]:
            item.set_color("#333333")
            item.set_linewidth(0.9)
    ax_b.set_ylabel("Tie-line records per system")
    ax_b.set_title("Effect of preprocessing")
    ax_b.set_xticklabels(labels)
    ax_b.grid(axis="y", color="#D9D9D9", linewidth=0.6)

    # (c) Temperature distributions after filtering.
    for item in prepared:
        values = pd.to_numeric(item.filtered["T"], errors="coerce").dropna()
        if values.empty:
            continue
        ax_c.hist(
            values.to_numpy(),
            bins=22,
            alpha=0.58,
            color=colors.get(item.dataset_id, "#4E79A7"),
            edgecolor="white",
            linewidth=0.5,
            label=display.get(item.dataset_id, item.dataset_id),
        )
    ax_c.set_xlabel("Temperature (K)")
    ax_c.set_ylabel("Number of records")
    ax_c.set_title("Filtered temperature coverage")
    ax_c.legend(frameon=False)
    ax_c.grid(axis="y", color="#D9D9D9", linewidth=0.6)

    # (d) Unique molecular species by component role.
    sub = component_summary[component_summary["stage"] == "filtered_min6_no_aug"].copy()
    roles = ["component1", "component2", "component3"]
    x = np.arange(len(dataset_ids))
    width = 0.22
    for i, role in enumerate(roles):
        vals = []
        for dataset_id in dataset_ids:
            match = sub[(sub["dataset_id"] == dataset_id) & (sub["component_role"] == role)]
            vals.append(int(match["unique_smiles"].iloc[0]) if not match.empty else 0)
        ax_d.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=colors[role],
            label=role.replace("component", "Component "),
        )
    union_vals = []
    for dataset_id in dataset_ids:
        match = sub[sub["dataset_id"] == dataset_id]
        union_vals.append(int(match["union_unique_smiles_all_roles"].iloc[0]) if not match.empty else 0)
    ax_d.plot(x, union_vals, color="#111111", marker="o", linewidth=1.4, label="Union")
    ymax = max(union_vals + [0])
    for role in roles:
        for dataset_id in dataset_ids:
            match = sub[(sub["dataset_id"] == dataset_id) & (sub["component_role"] == role)]
            if not match.empty:
                ymax = max(ymax, int(match["unique_smiles"].iloc[0]))
    ax_d.set_ylim(0, ymax * 1.42)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([display.get(d, d) for d in dataset_ids])
    ax_d.set_ylabel("Unique canonical SMILES")
    ax_d.set_title("Molecular species")
    leg = ax_d.legend(
        frameon=True,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.0,
        columnspacing=0.9,
        handlelength=1.2,
        handletextpad=0.4,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.86)
    leg.get_frame().set_edgecolor("#CCCCCC")
    leg.get_frame().set_linewidth(0.6)
    ax_d.grid(axis="y", color="#D9D9D9", linewidth=0.6)

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(
            -0.12,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    path = out_dir / "dataset_distribution_combined.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_family_plot(family_distribution: pd.DataFrame, out_dir: Path) -> Path | None:
    if family_distribution.empty:
        return None
    plot_df = family_distribution[family_distribution["stage"] == "filtered_min6_no_aug"].copy()
    if plot_df.empty:
        plot_df = family_distribution.copy()
    groups = plot_df[["dataset_id", "stage", "component_role"]].drop_duplicates().to_records(index=False).tolist()
    if not groups:
        return None
    nrows = len(groups)
    fig, axes = plt.subplots(nrows, 1, figsize=(7.2, max(3.2, 3.0 * nrows)), squeeze=False)
    for ax, (dataset_id, stage, role) in zip(axes[:, 0], groups):
        sub = plot_df[
            (plot_df["dataset_id"] == dataset_id)
            & (plot_df["stage"] == stage)
            & (plot_df["component_role"] == role)
        ].sort_values("count", ascending=False).head(12)
        sub = sub.sort_values("count", ascending=True)
        ax.barh(sub["family"], sub["count"], color="#76B7B2")
        ax.set_title(f"{dataset_id} - {role.replace('component', 'Component ')} family ({stage})")
        ax.set_xlabel("Number of records")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    fig.tight_layout()
    path = out_dir / "component_family_distributions.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_markdown_report(
    out_dir: Path,
    overview: pd.DataFrame,
    table_s15: pd.DataFrame,
    points_summary_df: pd.DataFrame,
    component_summary: pd.DataFrame,
    family_distribution: pd.DataFrame,
    figure_paths: list[Path],
    min_points_per_group: int,
) -> Path:
    def rel(path: Path) -> str:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")

    curated_overview = overview[overview["dataset_id"] == "Curated IL-LLE"]
    expanded_overview = overview[overview["dataset_id"] == "Expanded literature LLE"]
    curated_raw = curated_overview[curated_overview["stage"] == "raw_workbook"].iloc[0]
    curated_validated = curated_overview[curated_overview["stage"] == "validated_pre_density"].iloc[0]
    curated_filtered = curated_overview[curated_overview["stage"] == "filtered_min6_no_aug"].iloc[0]
    expanded_raw = expanded_overview[expanded_overview["stage"] == "raw_workbook"].iloc[0]
    expanded_validated = expanded_overview[expanded_overview["stage"] == "validated_pre_density"].iloc[0]
    expanded_filtered = expanded_overview[expanded_overview["stage"] == "filtered_min6_no_aug"].iloc[0]

    comp_filtered = component_summary[component_summary["stage"] == "filtered_min6_no_aug"]
    comp_wide_rows = []
    for dataset_id, sub in comp_filtered.groupby("dataset_id", sort=False):
        comp_wide_rows.append(
            {
                "dataset_id": dataset_id,
                "component1_unique_smiles": int(sub[sub["component_role"] == "component1"]["unique_smiles"].iloc[0]),
                "component2_unique_smiles": int(sub[sub["component_role"] == "component2"]["unique_smiles"].iloc[0]),
                "component3_unique_smiles": int(sub[sub["component_role"] == "component3"]["unique_smiles"].iloc[0]),
                "union_unique_smiles": int(sub["union_unique_smiles_all_roles"].iloc[0]),
            }
        )
    comp_wide = pd.DataFrame(comp_wide_rows)

    family_note = (
        "Component-family annotations were available for at least one workbook and are summarized in "
        "`family_distribution.csv`."
        if not family_distribution.empty
        else "No non-empty component-family annotation was detected in the analyzed workbooks."
    )

    lines = [
        "# Dataset Coverage Report",
        "",
        "## Reproducibility",
        "",
        "Run from the project root:",
        "",
        "```bash",
        "python scripts/analysis/analyze_dataset_distribution.py",
        "```",
        "",
        f"The preprocessing summary reuses `psmi.data.load_and_prepare_excel(..., min_points_per_group={min_points_per_group}, permute_23_aug=False)`. The no-augmentation setting is deliberate: swapping Components 2 and 3 is a training-time symmetry augmentation and is not counted as additional experimental LLE measurements.",
        "",
        "## Dataset Overview",
        "",
        markdown_table(
            overview[
                [
                    "dataset_id",
                    "stage",
                    "experimental_or_analysis_rows",
                    "unique_system_id",
                    "unique_system_temperature_groups",
                    "temperature_min_K",
                    "temperature_max_K",
                    "unique_temperatures",
                    "training_rows_if_component23_swap_augmented",
                ]
            ]
        ),
        "",
        "## Manuscript Table S15",
        "",
        markdown_table(
            table_s15[
                [
                    "dataset_id",
                    "paper_stage",
                    "repository_stage",
                    "records",
                    "systems",
                    "system_temperature_groups",
                ]
            ]
        ),
        "",
        "The final manuscript's `Before filtering` row maps to `validated_pre_density` for the curated benchmark and to `raw_workbook` for the expanded literature dataset. This explicit mapping preserves the reported Table S15 values without conflating workbook ingestion with molecular-record validation.",
        "",
        "## Points Per System",
        "",
        markdown_table(
            points_summary_df[
                [
                    "dataset_id",
                    "stage",
                    "n_systems",
                    "mean_points_per_system",
                    "std_points_per_system",
                    "min_points_per_system",
                    "median_points_per_system",
                    "max_points_per_system",
                    "iqr_points_per_system",
                ]
            ]
        ),
        "",
        "## Component Coverage",
        "",
        markdown_table(comp_wide),
        "",
        "## Figures",
        "",
    ]
    for path in figure_paths:
        lines.append(f"- `{rel(path)}`")
    lines.extend(
        [
            "",
            "## Counting Contract",
            "",
            "One workbook row is one candidate experimental tie-line record. A `system_id` identifies one ternary chemical system, while `(system_id, T)` identifies that system at a specific temperature. The paired extract-phase (`Ex1-Ex3`) and raffinate-phase (`Rx1-Rx3`) compositions define the measured equilibrium point. Molecular-record validation canonicalizes all three SMILES strings and removes rows with an invalid component representation or missing required numerical field. Preprocessing then assigns a continuous phase-path coordinate `t` within each `(system_id, T)` group and retains only groups meeting the configured minimum tie-line density.",
            "",
            f"The distributed main workbook contains {int(curated_raw['experimental_or_analysis_rows'])} rows, {int(curated_raw['unique_system_id'])} unique `system_id` values, and {int(curated_raw['unique_system_temperature_groups'])} unique `(system_id, T)` groups. Molecular-record validation retains {int(curated_validated['experimental_or_analysis_rows'])} records, {int(curated_validated['unique_system_id'])} systems, and {int(curated_validated['unique_system_temperature_groups'])} groups. Requiring at least {min_points_per_group} records per group then retains {int(curated_filtered['experimental_or_analysis_rows'])} records, {int(curated_filtered['unique_system_id'])} systems, and {int(curated_filtered['unique_system_temperature_groups'])} groups. The expanded workbook contains {int(expanded_raw['experimental_or_analysis_rows'])} rows, {int(expanded_raw['unique_system_id'])} systems, and {int(expanded_raw['unique_system_temperature_groups'])} groups; validation retains {int(expanded_validated['experimental_or_analysis_rows'])} records, {int(expanded_validated['unique_system_id'])} systems, and {int(expanded_validated['unique_system_temperature_groups'])} groups; the density filter retains {int(expanded_filtered['experimental_or_analysis_rows'])} records, {int(expanded_filtered['unique_system_id'])} systems, and {int(expanded_filtered['unique_system_temperature_groups'])} groups.",
            "",
            "The component counts are computed from canonical SMILES. Component-2/component-3 permutation is a training-time symmetry augmentation: it can double training examples, but it does not create experimental tie-line records and is excluded from dataset-size reporting.",
            "",
            f"{family_note}",
            "",
            "## Generated Tables",
            "",
            "- `dataset_overview.csv`",
            "- `table_s15_counts.csv`",
            "- `points_per_system_summary.csv`",
            "- `points_per_system_counts.csv`",
            "- `component_summary.csv`",
            "- `family_distribution.csv`",
        ]
    )

    path = out_dir / "dataset_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        "--out-dir",
        "--out_dir",
        dest="results_dir",
        default=str(
            ROOT
            / "experiments"
            / "supporting_information"
            / "s5_dataset_construction_and_distribution"
            / "results"
        ),
        help="Output directory for CSV tables and the dataset report.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(
            ROOT
            / "experiments"
            / "supporting_information"
            / "s5_dataset_construction_and_distribution"
            / "figures"
        ),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--min-points-per-group",
        "--min_points_per_group",
        dest="min_points_per_group",
        type=int,
        default=6,
        help="Minimum records retained per (system_id, T) group, matching src/config.py by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    prepared = [prepare_dataset(spec, args.min_points_per_group) for spec in DATASETS]

    overview = pd.DataFrame([row for item in prepared for row in overview_rows(item, args.min_points_per_group)])
    table_s15 = manuscript_table_s15(overview)
    points_summary_df = pd.DataFrame([row for item in prepared for row in points_summary_rows(item)])
    points_counts = pd.DataFrame([row for item in prepared for row in points_count_rows(item)])
    component_summary = pd.DataFrame([row for item in prepared for row in component_summary_rows(item)])
    family_distribution = pd.DataFrame([row for item in prepared for row in family_distribution_rows(item)])

    overview.to_csv(results_dir / "dataset_overview.csv", index=False, encoding="utf-8-sig")
    table_s15.to_csv(results_dir / "table_s15_counts.csv", index=False, encoding="utf-8-sig")
    points_summary_df.to_csv(results_dir / "points_per_system_summary.csv", index=False, encoding="utf-8-sig")
    points_counts.to_csv(results_dir / "points_per_system_counts.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(results_dir / "component_summary.csv", index=False, encoding="utf-8-sig")
    family_distribution.to_csv(results_dir / "family_distribution.csv", index=False, encoding="utf-8-sig")

    figure_paths: list[Path] = []
    figure_paths.append(save_combined_dataset_summary_plot(points_counts, prepared, component_summary, figures_dir))
    figure_paths.extend(save_points_plots(points_counts, figures_dir))
    figure_paths.append(save_temperature_plot(prepared, figures_dir))
    figure_paths.append(save_component_plot(component_summary, figures_dir))
    family_path = save_family_plot(family_distribution, figures_dir)
    if family_path is not None:
        figure_paths.append(family_path)

    report_path = write_markdown_report(
        out_dir=results_dir,
        overview=overview,
        table_s15=table_s15,
        points_summary_df=points_summary_df,
        component_summary=component_summary,
        family_distribution=family_distribution,
        figure_paths=figure_paths,
        min_points_per_group=args.min_points_per_group,
    )

    print("Generated dataset summary outputs:")
    for path in [
        results_dir / "dataset_overview.csv",
        results_dir / "table_s15_counts.csv",
        results_dir / "points_per_system_summary.csv",
        results_dir / "points_per_system_counts.csv",
        results_dir / "component_summary.csv",
        results_dir / "family_distribution.csv",
        report_path,
        *figure_paths,
    ]:
        print(f"  {path.resolve().relative_to(ROOT.resolve())}")


if __name__ == "__main__":
    main()

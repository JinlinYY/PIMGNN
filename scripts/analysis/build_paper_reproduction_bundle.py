"""Organize paper-reported values and checkpoint outputs into one result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "paper_reproduction"


TABLE1_ROWS = [
    ("MLP", .0603, .0064, .0987, .0131, .9024, .0280, .0620, .0070, .1181, .0186, .8881, .0366, .0612, .0062, .1090, .0146, .8948, .0290),
    ("LSTM", .0535, .0066, .0943, .0146, .9101, .0304, .0549, .0040, .1143, .0116, .8971, .0202, .0542, .0049, .1050, .0109, .9031, .0205),
    ("XGBoost", .0433, .0078, .0790, .0171, .9354, .0312, .0441, .0052, .0964, .0106, .9264, .0178, .0437, .0061, .0883, .0128, .9306, .0225),
    ("Transformer", .0495, .0054, .0850, .0135, .9270, .0247, .0493, .0038, .0997, .0124, .9209, .0221, .0494, .0043, .0928, .0116, .9238, .0214),
    ("TabNet", .0873, .0112, .1304, .0098, .8319, .0292, .0893, .0074, .1616, .0061, .7956, .0202, .0883, .0086, .1470, .0047, .8121, .0148),
    ("KAN", .0510, .0061, .0895, .0135, .9192, .0269, .0496, .0061, .1007, .0165, .9182, .0296, .0503, .0057, .0955, .0137, .9189, .0263),
    ("SMILES-RNN", .1604, .0027, .2025, .0032, .5978, .0213, .1956, .0065, .2707, .0067, .4292, .0129, .1780, .0028, .2391, .0032, .5039, .0128),
    ("MMGNN", .1694, .0023, .2095, .0007, .5971, .0026, .1978, .0012, .2766, .0010, .4005, .0069, .1831, .0014, .2457, .0010, .4988, .0048),
    ("CIGIN", .0938, .0023, .1395, .0033, .5754, .0216, .1236, .0015, .1945, .0015, .3140, .0106, .1087, .0018, .1693, .0018, .4447, .0161),
    ("SolvBERT", .0950, .0018, .1287, .0013, .6259, .0172, .1192, .0050, .1561, .0004, .2228, .0180, .1068, .0010, .1659, .0008, .4244, .0176),
    ("CGIB", .1368, .0232, .1776, .0216, .7224, .0361, .1427, .0204, .2006, .0046, .6649, .0609, .1240, .0023, .1936, .0188, .6936, .0485),
    ("GLAM", .1061, .0012, .1415, .0009, .5465, .0056, .1324, .0003, .1678, .0002, .1649, .0011, .1173, .0042, .1577, .0071, .3557, .0034),
    ("ANN", .0575, .0069, .0950, .0142, .9093, .0284, .0597, .0095, .1179, .0241, .8864, .0462, .0586, .0077, .1072, .0185, .8969, .0349),
    ("RF", .0458, .0076, .0820, .0151, .9314, .0287, .0439, .0043, .0925, .0088, .9325, .0147, .0449, .0056, .0875, .0114, .9321, .0201),
    ("UALF-GNN", .0522, .0064, .0886, .0150, .9204, .0294, .0576, .0076, .1113, .0176, .9006, .0315, .0549, .0063, .1008, .0152, .9096, .0279),
    ("PSMI", .0355, .0006, .0577, .0010, .9654, .0014, .0356, .0025, .0801, .0146, .9516, .0153, .0356, .0011, .0700, .0084, .9585, .0083),
]


TABLE2_ROWS = [
    (1, "none", "none", "permutation", "concat", .0493, .0927, .9118, .0433, .0873, .9445, .0463, .0900, .9309),
    (2, "single", "none", "permutation", "concat", .0402, .0702, .9493, .0409, .0740, .9601, .0405, .0722, .9556),
    (3, "mixture", "none", "permutation", "concat", .0368, .0594, .9638, .0362, .0716, .9627, .0365, .0658, .9631),
    (4, "mixture", "top-k", "permutation", "concat", .0359, .0596, .9635, .0340, .0621, .9719, .0350, .0608, .9685),
    (5, "mixture", "top-k", "permutation", "FiLM", .0373, .0635, .9586, .0375, .0710, .9633, .0374, .0674, .9613),
    (6, "mixture", "top-k", "permutation", "Transformer", .0376, .0582, .9651, .0338, .0659, .9684, .0357, .0622, .9670),
    (7, "mixture", "cross", "permutation", "concat", .0338, .0545, .9695, .0320, .0577, .9758, .0329, .0561, .9732),
    (8, "mixture", "cross", "permutation", "Hadamard", .0333, .0542, .9698, .0300, .0561, .9770, .0317, .0552, .9740),
    (9, "mixture", "cross", "permutation", "bilinear", .0330, .0525, .9717, .0309, .0577, .9758, .0319, .0552, .9741),
    (10, "mixture", "cross", "permutation", "Transformer", .0349, .0564, .9673, .0321, .0564, .9768, .0335, .0564, .9729),
    (11, "mixture", "cross", "S3", "S3-Set", .0345, .0572, .9664, .0325, .0599, .9738, .0335, .0586, .9707),
    (12, "mixture", "cross", "S3", "Transformer", .0351, .0555, .9684, .0309, .0546, .9783, .0330, .0550, .9742),
]


TABLE3_ROWS = [
    ("data_driven", .0351, .0555, .9684, .0309, .0546, .9783, .0330, .0550, .9742, 1.771093, 2.612110),
    ("physics_informed", .0351, .0530, .9711, .0346, .0557, .9774, .0349, .0544, .9748, .541074, .788265),
]


TABLE_S5_ROWS = [
    ("stratified_10_fold", .0479, .0038, .0911, .0094, .9277, .0145),
    ("system_level_random", .0538, .0028, .1061, .0112, .8995, .0223),
    ("point_level_random", .0331, .0006, .0581, .0010, .9708, .0010),
    ("structure_family", .0737, .0127, .1259, .0194, .8593, .0401),
]


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Write one UTF-8 CSV with stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _copy(source: Path, destination: Path) -> bool:
    """Copy one existing result file and return whether it was available."""
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _table_headers() -> tuple[list[str], list[str], list[str]]:
    """Return explicit manuscript table schemas."""
    table1 = ["model"]
    for phase in ("extract", "raffinate", "overall"):
        for metric in ("mae", "rmse", "r2"):
            table1.extend([f"{phase}_{metric}_mean", f"{phase}_{metric}_std"])
    table2 = [
        "row", "graph_representation", "functional_group_interaction", "component_encoding", "fusion",
        "extract_mae", "extract_rmse", "extract_r2", "raffinate_mae", "raffinate_rmse", "raffinate_r2",
        "overall_mae", "overall_rmse", "overall_r2",
    ]
    table3 = [
        "variant", "extract_mae", "extract_rmse", "extract_r2", "raffinate_mae", "raffinate_rmse", "raffinate_r2",
        "overall_mae", "overall_rmse", "overall_r2", "chemical_potential_mae", "chemical_potential_rmse",
    ]
    return table1, table2, table3


def _aggregate_current_metrics(source: Path, destination: Path) -> None:
    """Summarize the three saved corrected-v2 seeds without training."""
    frame = pd.read_csv(source)
    frame["group"] = frame["run_id"].str.replace(r"_seed\d+$", "", regex=True)
    metric_columns = [column for column in frame if column.startswith(("mae_", "rmse_", "r2_"))]
    rows = []
    for group, group_frame in frame.groupby("group", sort=True):
        row: dict[str, object] = {"group": group, "seed_count": len(group_frame)}
        for metric in metric_columns:
            row[f"{metric}_mean"] = group_frame[metric].mean()
            row[f"{metric}_sample_std"] = group_frame[metric].std(ddof=1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(destination, index=False, encoding="utf-8-sig")


def _copy_inference_outputs(output: Path) -> None:
    """Collect predictions and figures while retaining the raw audit reports."""
    sources = {
        "current": output / "current_weight_inference",
        "historical": output / "historical_weight_inference",
    }
    for protocol, root in sources.items():
        if not root.is_dir():
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            _copy(run_dir / "test_predictions.csv", output / "data" / "predictions" / protocol / f"{run_dir.name}.csv")
            _copy(run_dir / "parity_E.png", output / "figures" / protocol / run_dir.name / "parity_E.png")
            _copy(run_dir / "parity_R.png", output / "figures" / protocol / run_dir.name / "parity_R.png")


def _write_paper_comparison(output: Path) -> None:
    """Compare rounded paper values with direct historical-weight inference."""
    runs = {
        "figure2a": ("figure2a_psmi", {
            "mae_E": .0371, "rmse_E": .0566, "r2_E": .9671,
            "mae_R": .0318, "rmse_R": .0545, "r2_R": .9784,
        }),
        "table3_data_driven": ("table3_data_driven", {
            "mae_E": .0351, "rmse_E": .0555, "r2_E": .9684,
            "mae_R": .0309, "rmse_R": .0546, "r2_R": .9783,
            "mae": .0330, "rmse": .0550, "r2": .9742,
        }),
        "table3_physics_informed": ("table3_physics_informed", {
            "mae_E": .0351, "rmse_E": .0530, "r2_E": .9711,
            "mae_R": .0346, "rmse_R": .0557, "r2_R": .9774,
            "mae": .0349, "rmse": .0544, "r2": .9748,
        }),
    }
    rows = []
    report_root = output / "historical_weight_inference"
    for paper_item, (run_id, paper_metrics) in runs.items():
        report_path = report_root / run_id / "reproduction_report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for metric, reported in paper_metrics.items():
            reproduced = float(report["metrics"][metric])
            rows.append((paper_item, run_id, metric, reported, reproduced, abs(reported - reproduced)))
    _write_csv(
        output / "audit" / "paper_vs_weight_inference.csv",
        ["paper_item", "run_id", "metric", "paper_reported", "weight_inference", "absolute_difference"],
        rows,
    )


def _write_saved_table3_metrics(output: Path) -> None:
    """Flatten the historical training-time best-test records behind Table 3."""
    records = {
        "data_driven": PROJECT_ROOT / "experiments/paper_historical/table3_data_driven/metrics/best_metrics.json",
        "physics_informed": PROJECT_ROOT / "experiments/paper_historical/table3_physics_informed/metrics/best_metrics.json",
    }
    public_records = {
        "data_driven": PROJECT_ROOT / "experiments/paper_historical/table3_data_driven/metrics/best_metrics.json",
        "physics_informed": PROJECT_ROOT / "experiments/paper_historical/table3_physics_informed/metrics/best_metrics.json",
    }
    if (PROJECT_ROOT / "experiments/paper_historical").is_dir():
        records = public_records
    destination = output / "tables" / "table3_saved_best_metrics.csv"
    if not all(path.is_file() for path in records.values()):
        if destination.is_file():
            return
        missing = [str(path) for path in records.values() if not path.is_file()]
        raise FileNotFoundError(f"Table 3 metric records are unavailable: {missing}")
    metrics = ("mae_E", "rmse_E", "r2_E", "mae_R", "rmse_R", "r2_R", "mae", "rmse", "r2", "mu_res_mae", "mu_res_rmse")
    rows = []
    for variant, path in records.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload["best_test"]
        row = [variant, payload["best_epoch"]]
        row.extend(test.get(metric) for metric in metrics)
        rows.append(row)
    _write_csv(destination, ["variant", "best_epoch", *metrics], rows)


def _copy_supplementary_sources(output: Path) -> list[tuple[str, bool]]:
    """Copy existing SI table data into neutrally named table files."""
    mappings = {
        "table_s3_s4_temperature_encoding.csv": "experiments/12_temperature_encoding/encoding_metrics_with_ci.csv",
        "table_s6_tieline_threshold.csv": "experiments/11_tieline_sensitivity/threshold_metrics_with_ci.csv",
        "table_s7_tieline_location.csv": "experiments/11_tieline_sensitivity/location_metrics_with_ci.csv",
        "table_s8_ge_model_sensitivity.csv": "experiments/03_physics_constraints/comment10_ge_model_sensitivity/summary.csv",
        "table_s9_s10_thermodynamic_thresholds.csv": "experiments/12_thermodynamic_audit/results/threshold_sensitivity.csv",
        "table_s11_s14_system_categories.csv": "experiments/13_system_generalization/results/category_summary.csv",
        "table_s15_dataset_overview.csv": "experiments/00_dataset_construction/results/dataset_overview.csv",
        "table_s16_component_summary.csv": "experiments/00_dataset_construction/results/component_summary.csv",
        "table_s17_system_classification.csv": "experiments/13_system_generalization/results/system_classification.csv",
        "efficiency_summary.csv": "experiments/10_efficiency/runs/psmi_rtx3090_ti/latency_aggregate.csv",
    }
    public_mappings = {
        "table_s3_s4_temperature_encoding.csv": "experiments/08_temperature_robustness/02_encoding_and_tail/results/seed42_and_multiseed/encoding_metrics_with_ci.csv",
        "table_s6_tieline_threshold.csv": "experiments/10_tieline_sensitivity/results/threshold_metrics_with_ci.csv",
        "table_s7_tieline_location.csv": "experiments/10_tieline_sensitivity/results/location_metrics_with_ci.csv",
        "table_s8_ge_model_sensitivity.csv": "experiments/11_ge_model_sensitivity/results/summary.csv",
        "table_s9_s10_thermodynamic_thresholds.csv": "experiments/12_thermodynamic_audit/results/threshold_sensitivity.csv",
        "table_s11_s14_system_categories.csv": "experiments/13_system_generalization/results/category_summary.csv",
        "table_s15_dataset_overview.csv": "experiments/00_dataset_construction/results/dataset_overview.csv",
        "table_s16_component_summary.csv": "experiments/00_dataset_construction/results/component_summary.csv",
        "table_s17_system_classification.csv": "experiments/13_system_generalization/results/system_classification.csv",
        "efficiency_summary.csv": "experiments/15_efficiency/results/psmi_rtx3090_ti/latency_aggregate.csv",
    }
    if (PROJECT_ROOT / "experiments/08_temperature_robustness").is_dir():
        mappings = public_mappings
    status = []
    for destination_name, relative_source in mappings.items():
        available = _copy(PROJECT_ROOT / relative_source, output / "tables" / destination_name)
        status.append((relative_source, available))
    return status


def _sha256(path: Path) -> str:
    """Hash one result artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output: Path) -> None:
    """Write a machine-readable inventory for the organized result package."""
    rows = []
    for category in ("tables", "data", "figures"):
        root = output / category
        if not root.is_dir():
            continue
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            rows.append((path.relative_to(output).as_posix(), path.stat().st_size, _sha256(path)))
    _write_csv(output / "audit" / "artifact_manifest.csv", ["relative_path", "size_bytes", "sha256"], rows)


def _write_weight_registry(output: Path) -> None:
    """Inventory all checkpoints used by the two evaluation registries."""
    registry_paths = {
        "current_corrected_v2": PROJECT_ROOT / "configs/reproduction/current_weight_registry.json",
        "historical_paper": PROJECT_ROOT / "configs/reproduction/historical_paper_weight_registry.json",
    }
    rows = []
    for protocol, registry_path in registry_paths.items():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for run in registry["runs"]:
            relative = run["checkpoint"].replace("\\", "/")
            public_relative = relative
            if relative.startswith("results/main_benchmark/corrected_v2/"):
                public_relative = "experiments/01_main_benchmark/results/corrected_v2/" + relative.removeprefix("results/main_benchmark/corrected_v2/")
            elif relative.startswith("results/transfer_evaluation/expanded_lle/corrected_v2/"):
                public_relative = "experiments/07_external_transfer/results/expanded_lle_corrected_v2/" + relative.removeprefix("results/transfer_evaluation/expanded_lle/corrected_v2/")
            checkpoint = PROJECT_ROOT / relative
            rows.append((
                protocol,
                run["id"],
                run["group"],
                run["seed"],
                run["config"],
                relative,
                public_relative,
                checkpoint.stat().st_size,
                _sha256(checkpoint),
            ))
    _write_csv(
        output / "audit" / "weight_registry.csv",
        [
            "protocol", "run_id", "group", "seed", "config", "workspace_checkpoint",
            "public_checkpoint", "size_bytes", "sha256",
        ],
        rows,
    )


def _write_readme(output: Path) -> None:
    """Document the result package and protocol boundaries."""
    text = """# Paper Result Reproduction Package

This directory separates published reference values, checkpoint-based evaluation outputs, pointwise predictions, parity plots, and artifact provenance.

## Contents

- `tables/`: main-text tables, Supporting Information tables, and multi-seed summaries.
- `data/predictions/`: pointwise predictions for the corrected and historical protocols.
- `figures/`: parity plots generated from published checkpoints.
- `current_weight_inference/`: hash-verified reports for the `corrected_v2` protocol.
- `historical_weight_inference/`: reports for the Figure 2a and Table 3 checkpoint protocols.
- `audit/`: protocol alignment, numerical comparisons, checkpoint digests, and artifact hashes.

## Protocol boundaries

The historical protocol uses the component-major mixture-node layout required by the paper checkpoints. The `corrected_v2` protocol uses the sample-major layout and must be evaluated as a separate protocol. Table 3 stores both the training-time best metrics and metrics recomputed from checkpoint evaluation because exporter and floating-point differences can introduce small numerical deviations.

The expanded-LLE checkpoint contract includes normalized pressure. Dataset digests, split-manifest digests, checkpoint metadata migrations, and public artifact hashes are recorded under `audit/`.

## Commands

```powershell
python scripts/reproduce_current_weights.py --device cuda
python scripts/reproduce_current_weights.py `
  --registry configs/reproduction/historical_paper_weight_registry.json `
  --output-root results/paper_reproduction/historical_weight_inference `
  --device cuda
python scripts/analysis/build_paper_reproduction_bundle.py
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")

def build(output: Path) -> None:
    """Build all tables, copies, summaries, and audit files."""
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("tables", "data", "figures", "audit"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    table1_header, table2_header, table3_header = _table_headers()
    _write_csv(output / "tables" / "paper_table_1_reported.csv", table1_header, TABLE1_ROWS)
    _write_csv(output / "tables" / "paper_table_2_reported.csv", table2_header, TABLE2_ROWS)
    _write_csv(output / "tables" / "paper_table_3_reported.csv", table3_header, TABLE3_ROWS)
    _write_csv(
        output / "tables" / "paper_table_s5_reported.csv",
        ["split_strategy", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std"],
        TABLE_S5_ROWS,
    )
    _write_csv(
        output / "tables" / "paper_temperature_sensitivity_reported.csv",
        ["quantity", "mean", "maximum", "unit_or_note"],
        [
            ("local_temperature_sensitivity", 5.82e-4, 3.09e-3, "K^-1"),
            ("phase_path_length", .2870, 1.1909, "composition distance"),
            ("composition_sum_deviation", None, 1.19e-7, "absolute"),
            ("zero_perturbation_mean_absolute_difference", .0060, None, "composition fraction"),
        ],
    )
    _write_saved_table3_metrics(output)

    current_metrics = output / "current_weight_inference" / "reproduced_metrics.csv"
    if current_metrics.is_file():
        _copy(current_metrics, output / "tables" / "current_weight_metrics.csv")
        _aggregate_current_metrics(current_metrics, output / "tables" / "current_weight_multiseed_summary.csv")

    historical_metrics = output / "historical_weight_inference" / "reproduced_metrics.csv"
    if historical_metrics.is_file():
        _copy(historical_metrics, output / "tables" / "historical_weight_metrics.csv")

    source_status = _copy_supplementary_sources(output)
    _copy_inference_outputs(output)
    _write_paper_comparison(output)
    protocol_rows = [
        ("Figure 2a", "historical", "weight inference reproduced", "legacy component-major layout; prediction max absolute difference about 3.8e-4"),
        ("Table 1 baselines", "historical", "saved five-seed summaries available", "historical split protocol; do not mix with corrected_v2"),
        ("Table 1 PSMI", "historical", "paper aggregate available", "exact seed-to-run mapping is not uniquely recoverable"),
        ("Table 2", "historical", "paper table and saved best metrics available", "individual historical checkpoints are archived"),
        ("Table 3", "historical", "weights and training-time best metrics available", "inference is close; table values remain the stored best metrics"),
        ("corrected_v2 main", "current", "strict hash-verified inference", "scientifically corrected protocol; not a replacement for historical paper values"),
        ("corrected_v2 expanded", "current", "strict hash-verified inference", "current exporter includes pressure scaling; legacy CSV did not"),
    ]
    _write_csv(output / "audit" / "protocol_alignment.csv", ["paper_item", "protocol", "status", "boundary"], protocol_rows)
    _write_csv(output / "audit" / "supplementary_source_status.csv", ["source", "available"], source_status)
    _write_readme(output)
    _write_manifest(output)
    _write_weight_registry(output)


def parse_args() -> argparse.Namespace:
    """Parse the optional bundle destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    build(destination.resolve())
    print(f"Saved paper reproduction bundle: {destination.resolve()}")

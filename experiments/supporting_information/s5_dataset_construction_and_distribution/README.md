# Dataset Construction and Distribution

## Paper mapping

- Main text Section 2.1
- SI Section S5
- Tables S15-S16
- Figure S8

## Scientific scope

Processed data, fixed split manifests, component summaries, and distribution
figures are available. Dataset counts are separated into three auditable
stages: workbook ingestion, molecular-record validation before the density
filter, and the final minimum-density dataset.

## Manuscript-aligned results

- `results/table_s15_counts.csv` reproduces the four rows of Table S15 and
  records which repository stage supplies each manuscript row.
- `results/dataset_overview.csv` retains all three counting stages.
- `results/component_summary.csv` supports Table S16.
- `figures/dataset_distribution_combined.png` is the Figure S8 asset.

The curated Table S15 `Before filtering` value is the validated pre-density
stage (7,953 records, 830 systems, and 842 system-temperature groups), not the
8,343-row workbook-ingestion count. The expanded `Before filtering` value uses
the 7,134-row workbook count. This mapping is intentionally explicit.

## Code entry points

- `scripts/data_preparation/build_canonical_split.py`
- `scripts/data_preparation/build_expanded_split.py`
- `scripts/analysis/analyze_dataset_distribution.py`
- `scripts/analysis/export_temperature_range_audit.py`

## Representative commands

```bash
python scripts/analysis/analyze_dataset_distribution.py --help
python scripts/data_preparation/build_canonical_split.py --help
```

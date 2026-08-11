# Dataset Construction and Distribution

## Paper mapping

- Main text Section 2.1
- SI Section S5
- Tables S18-S19
- Figure S9

## Scientific scope

Processed data, fixed split manifests, component summaries, and distribution
figures are available. Dataset counts are separated into three auditable
stages: workbook ingestion, molecular-record validation before the density
filter, and the final minimum-density dataset.

## Manuscript-aligned results

- `results/table_s15_counts.csv` reproduces the four rows of Table S18 and
  records which repository stage supplies each manuscript row. The filename is
  retained to avoid breaking existing links created before final SI renumbering.
- `results/dataset_overview.csv` retains all three counting stages.
- `results/component_summary.csv` supports Table S19.
- `figures/dataset_distribution_combined.png` is the Figure S9 asset.

The curated Table S18 `Before filtering` value is the validated pre-density
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

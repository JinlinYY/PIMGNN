# Dataset Construction and Distribution

## Paper mapping

- Main text Section 2.1
- SI Section S5
- Tables S15-S16
- Figure S8

## Scientific scope

Processed data, fixed split manifests, component summaries, and distribution figures are available.

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

# Dataset Construction, Filtering, and Distribution Analysis

## Paper mapping

- Main text Section 2.1
- SI Section S5
- Tables S15-S16
- Figure S8

## Available resources

Code, processed data, split manifests, summary tables, and distribution figures are available.

## Evidence status

Executable code and reference artifacts available.

## Code entry points

- `scripts/data_preparation/build_canonical_split.py`
- `scripts/data_preparation/build_expanded_split.py`
- `scripts/data_preparation/filter_sparse_systems.py`
- `scripts/analysis/analyze_dataset_distribution.py`
- `scripts/analysis/export_temperature_range_audit.py`

## Commands

```powershell
python scripts/analysis/analyze_dataset_distribution.py --help
python scripts/data_preparation/build_canonical_split.py --help
```

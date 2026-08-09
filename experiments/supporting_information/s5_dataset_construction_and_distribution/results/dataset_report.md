# Dataset Coverage Report

## Reproducibility

Run from the project root:

```bash
python scripts/analysis/analyze_dataset_distribution.py
```

The preprocessing summary reuses `psmi.data.load_and_prepare_excel(..., min_points_per_group=6, permute_23_aug=False)`. The no-augmentation setting is deliberate: swapping Components 2 and 3 is a training-time symmetry augmentation and is not counted as additional experimental LLE measurements.

## Dataset Overview

| dataset_id              | stage                | experimental_or_analysis_rows | unique_system_id | unique_system_temperature_groups | temperature_min_K | temperature_max_K | unique_temperatures | training_rows_if_component23_swap_augmented |
| ----------------------- | -------------------- | ----------------------------- | ---------------- | -------------------------------- | ----------------- | ----------------- | ------------------- | ------------------------------------------- |
| Curated IL-LLE          | raw_workbook         | 8343                          | 860              | 872                              | 278.15            | 353.2             | 35                  |                                             |
| Curated IL-LLE          | filtered_min6_no_aug | 7683                          | 765              | 766                              | 278.15            | 353.2             | 33                  | 15366                                       |
| Expanded literature LLE | raw_workbook         | 7134                          | 830              | 883                              | 278.15            | 373.15            | 85                  |                                             |
| Expanded literature LLE | filtered_min6_no_aug | 6709                          | 719              | 764                              | 278.15            | 373.15            | 66                  | 13418                                       |

## Points Per System

| dataset_id              | stage                | n_systems | mean_points_per_system | std_points_per_system | min_points_per_system | median_points_per_system | max_points_per_system | iqr_points_per_system |
| ----------------------- | -------------------- | --------- | ---------------------- | --------------------- | --------------------- | ------------------------ | --------------------- | --------------------- |
| Curated IL-LLE          | raw_workbook         | 860       | 9.7                    | 3.624                 | 1                     | 9                        | 30                    | 3                     |
| Curated IL-LLE          | filtered_min6_no_aug | 765       | 10.043                 | 3.189                 | 6                     | 10                       | 30                    | 3                     |
| Expanded literature LLE | raw_workbook         | 830       | 8.595                  | 4.35                  | 1                     | 8                        | 52                    | 3                     |
| Expanded literature LLE | filtered_min6_no_aug | 719       | 9.331                  | 4.134                 | 6                     | 8                        | 52                    | 2                     |

## Component Coverage

| dataset_id              | component1_unique_smiles | component2_unique_smiles | component3_unique_smiles | union_unique_smiles |
| ----------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------- |
| Curated IL-LLE          | 142                      | 33                       | 31                       | 194                 |
| Expanded literature LLE | 61                       | 83                       | 101                      | 186                 |

## Figures

- `tmp/open_source_audit/figures/dataset_distribution_combined.png`
- `tmp/open_source_audit/figures/points_per_system_histograms.png`
- `tmp/open_source_audit/figures/points_per_system_boxplots.png`
- `tmp/open_source_audit/figures/temperature_distributions.png`
- `tmp/open_source_audit/figures/component_unique_smiles_filtered.png`
- `tmp/open_source_audit/figures/component_family_distributions.png`

## Counting Contract

One workbook row is one experimental tie-line record. A `system_id` identifies one ternary chemical system, while `(system_id, T)` identifies that system at a specific temperature. The paired extract-phase (`Ex1-Ex3`) and raffinate-phase (`Rx1-Rx3`) compositions define the measured equilibrium point. Preprocessing assigns a continuous phase-path coordinate `t` within each `(system_id, T)` group and retains only groups meeting the configured minimum tie-line density.

The distributed main workbook contains 8343 raw tie-line records, 860 unique `system_id` values, and 872 unique `(system_id, T)` groups over 278.15-353.20 K. Requiring at least 6 records per `(system_id, T)` group retains 7683 records, 765 systems, and 766 system-temperature groups. The distributed expanded workbook contains 7134 raw records, 830 systems, and 883 system-temperature groups over 278.15-373.15 K; the same filter retains 6709 records, 719 systems, and 764 system-temperature groups.

The component counts are computed from canonical SMILES. Component-2/component-3 permutation is a training-time symmetry augmentation: it can double training examples, but it does not create experimental tie-line records and is excluded from dataset-size reporting.

Component-family annotations were available for at least one workbook and are summarized in `family_distribution.csv`.

## Generated Tables

- `dataset_overview.csv`
- `points_per_system_summary.csv`
- `points_per_system_counts.csv`
- `component_summary.csv`
- `family_distribution.csv`

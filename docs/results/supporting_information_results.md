# Supporting Information Result Mapping

This index follows the final Supporting Information numbering. Method-only
sections are omitted unless they provide provenance for a reported result.

## SI Section S3: additional evaluation and validation

| Paper item | Public evidence | Status |
| --- | --- | --- |
| Figures S1-S3: prediction-error analysis | [`s3_1_prediction_error_analysis/figures`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_1_prediction_error_analysis/figures) and [`test_pointwise_predictions.csv`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_1_prediction_error_analysis/results/test_pointwise_predictions.csv) | Complete |
| Section S3.2: baseline protocol | Main-text [`3_1_1_baseline_comparison`](../../experiments/section_3_results/3_1_lle_prediction/3_1_1_baseline_comparison) | Partial; same Table 1 boundary described in [main-text mapping](main_text_results.md#table-1-comparison-with-baseline-methods) |
| Figure S4: local temperature perturbation and phase-path continuity | [`run_sensitivity_analysis.py`](../../scripts/analysis/run_sensitivity_analysis.py) and experiment [`README`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_3_temperature_robustness/01_local_perturbation/README.md) | Partial; exact archived Figure S4 output is absent |
| Tables S3-S4 and Figure S5: temperature representation and tail robustness | [`02_encoding_and_tail`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_3_temperature_robustness/02_encoding_and_tail) | Complete |
| Table S5: data-splitting strategies | [`s3_4_data_splitting`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_4_data_splitting) | Partial; code and split manifests exist, but the final aggregate metric table is absent |
| Tables S6-S7 and Figure S6: tie-line density and phase-path location | [`s3_5_tieline_density_and_phase_path`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_5_tieline_density_and_phase_path) | Complete |
| Table S8: excess-Gibbs-energy model sensitivity | [`results/summary.csv`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_6_excess_gibbs_energy_model_sensitivity/results/summary.csv) and per-seed records | Complete |
| Tables S9-S10: thermodynamic-consistency audit | [`summary.json`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_7_thermodynamic_consistency_audit/results/summary.json), [`threshold_sensitivity.csv`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_7_thermodynamic_consistency_audit/results/threshold_sensitivity.csv), and per-prediction residuals | Complete |
| Tables S11-S14 and S17: system-level reconstruction | [`s3_8_system_generalization/results`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_8_system_generalization/results) | Complete |
| Section S3.9: inference efficiency | [`psmi_rtx3090_ti`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_9_inference_efficiency/results/psmi_rtx3090_ti) | Complete |

### Tables S3-S4

The manuscript uses random seeds 7, 42, and 2024. The multi-seed source tables
are:

- [`multi_seed_encoding_metrics.csv`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_3_temperature_robustness/02_encoding_and_tail/results/seed42_and_multiseed/multi_seed_encoding_metrics.csv)
- [`encoding_metrics_with_ci.csv`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_3_temperature_robustness/02_encoding_and_tail/results/seed42_and_multiseed/encoding_metrics_with_ci.csv)
- [`distance_metrics_with_ci.csv`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_3_temperature_robustness/02_encoding_and_tail/results/seed42_and_multiseed/distance_metrics_with_ci.csv)

These seeds are specific to the temperature-encoding experiment and are not
the Table 1 five-seed protocol.

### Tables S6-S7

The primary manuscript tables are backed by:

- [`threshold_metrics_with_ci.csv`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_5_tieline_density_and_phase_path/results/threshold_metrics_with_ci.csv)
- [`location_metrics_with_ci.csv`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_5_tieline_density_and_phase_path/results/location_metrics_with_ci.csv)
- [`dataset_threshold_counts.csv`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_5_tieline_density_and_phase_path/results/dataset_threshold_counts.csv)
- [`tieline_threshold_sensitivity.png`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_5_tieline_density_and_phase_path/figures/tieline_threshold_sensitivity.png)

The threshold-6 phase-path MAEs reported in Table S7 range from 0.05262 in the
central interval `0.4-0.6` to 0.12006 in the terminal interval `0.8-1.0`.

### Table S8

The five split seeds are 42-46. The reported test MAEs are 0.04087 +/- 0.00342
for NRTL, 0.04099 +/- 0.00361 for pairwise three-suffix Margules, and 0.04072
+/- 0.00345 for pairwise van Laar. These values are stored in the public
summary with the per-seed metrics, split manifests, and fitted parameter files.

### Tables S9-S10

The audit covers 803 predictions from 78 unseen systems. The manuscript reports
component residual MAE/RMSE values of 0.5400/0.7852 for physics-informed PSMI
and 1.7358/2.5876 for the data-driven baseline. The threshold table records the
exceedance fraction at every stated epsilon and must be interpreted as a
post-hoc diagnostic, not as a hard equilibrium guarantee.

### Tables S11-S14 and S17

The primary classification at composition RMSE tolerance 0.02 contains 4
quantitatively consistent, 72 qualitatively consistent, and 2 failed systems.
The public result directory contains the per-system table, category summary,
and tolerance-sensitivity analysis used for these supplementary tables.

## SI Section S4: Web application

### Figure S7

Status: **Partial**

The FastAPI backend, Vue frontend, default checkpoint contract, and deployment
scripts are available under [`Web/PSMI-LLE-web`](../../Web/PSMI-LLE-web).
The exact manuscript prototype screenshot used as Figure S7 is not distributed
as a standalone image.

## SI Section S5: dataset construction and distribution

| Paper item | Public evidence | Status |
| --- | --- | --- |
| Table S15: records and systems before/after density filtering | [`table_s15_counts.csv`](../../experiments/supporting_information/s5_dataset_construction_and_distribution/results/table_s15_counts.csv) and the three-stage [`dataset_overview.csv`](../../experiments/supporting_information/s5_dataset_construction_and_distribution/results/dataset_overview.csv) | Complete; the manuscript-stage mapping is explicit |
| Table S16: molecular-species coverage | [`component_summary.csv`](../../experiments/supporting_information/s5_dataset_construction_and_distribution/results/component_summary.csv) | Complete |
| Figure S8: dataset distribution and molecular coverage | [`dataset_distribution_combined.png`](../../experiments/supporting_information/s5_dataset_construction_and_distribution/figures/dataset_distribution_combined.png) and component plots | Complete |

The exact Table S15 mapping is:

| Dataset | Manuscript stage | Repository counting stage | Records | Systems | System-temperature groups |
| --- | --- | --- | ---: | ---: | ---: |
| Curated IL-LLE | Before filtering | `validated_pre_density` | 7,953 | 830 | 842 |
| Curated IL-LLE | After filtering | `filtered_min6_no_aug` | 7,683 | 765 | 766 |
| Expanded literature LLE | Before filtering | `raw_workbook` | 7,134 | 830 | 883 |
| Expanded literature LLE | After filtering | `filtered_min6_no_aug` | 6,709 | 719 | 764 |

The curated workbook contains 8,343 candidate rows before molecular-record
validation. Canonical-SMILES and required-field validation removes 390 rows,
which explains why the manuscript's curated `Before filtering` value is 7,953
rather than 8,343. For auditability, `dataset_overview.csv` keeps all three
stages. The manuscript used the expanded workbook ingestion count for its
expanded `Before filtering` row; this selection is recorded explicitly instead
of silently applying a different label. All counts refer to unaugmented
experimental records.

Table S16 is reproduced from the filtered rows using canonical SMILES:

| Dataset | Component 1 | Component 2 | Component 3 | Union |
| --- | ---: | ---: | ---: | ---: |
| Curated IL-LLE | 142 | 33 | 31 | 194 |
| Expanded literature LLE | 61 | 83 | 101 | 186 |

The `unique_component_names` columns in `component_summary.csv` are auxiliary
label-quality diagnostics; Table S16 uses `unique_smiles` and
`union_unique_smiles_all_roles` only.

## SI Section S6: phase-diagram system classification

Table S17 reuses the authoritative output from SI Section S3.8 rather than
duplicating it. Use
[`system_classification.csv`](../../experiments/supporting_information/s3_additional_evaluation_and_validation/s3_8_system_generalization/results/system_classification.csv)
for the complete 78-system record.

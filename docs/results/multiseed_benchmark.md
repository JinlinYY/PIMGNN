# PSMI Multi-Seed Benchmarks

## Scope

This document summarizes the maintained sample-major multi-seed protocol for
seeds 42, 43, and 44. It is distinct from the component-major manuscript
checkpoint registry used to reproduce Figure 2a and Table 3.

All seeds use the same fixed system-level partition. Component permutation is
applied only to the training partition. Checkpoints are selected by validation
performance and the test partition is evaluated after selection.

## Main benchmark dataset

The filtered main dataset contains 7,683 unaugmented tie-line records from 765
systems. The fixed partition is:

| Partition | Systems | Records |
| --- | ---: | ---: |
| Train | 612 | 6,092 |
| Validation | 75 | 788 |
| Test | 78 | 803 |

## Main benchmark results

| Stage | Test MAE | Test RMSE | Test R2 | Chemical-potential residual MAE | TPD violation rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supervised | 0.05145 +/- 0.00089 | 0.09270 +/- 0.00136 | 0.92674 +/- 0.00216 | 1.45072 +/- 0.03246 | 20.96% +/- 0.73% |
| NRTL regularized | 0.05176 +/- 0.00127 | 0.09264 +/- 0.00129 | 0.92682 +/- 0.00204 | 1.36934 +/- 0.12601 | 20.63% +/- 2.21% |

Values are mean plus or minus sample standard deviation across the three seeds.
Predictive and thermodynamic changes should be read separately: a smaller
chemical-potential residual does not necessarily imply a smaller composition
MAE, and the stage-2 public objective does not directly optimize the reported
TPD diagnostic.

The per-seed archived metric records for the main stages are located at:

```text
experiments/section_3_results/3_1_lle_prediction/
  3_1_2_ablation_analysis/physics_regularization/
  results/stage_comparison/seed*/
```

The cleaned public release retains these metrics as evidence. It does not
present missing main-stage sample-major checkpoints as downloadable artifacts.
For executable manuscript checkpoints, use the published component-major
registry described in the [checkpoint reference](../reference/checkpoints_and_artifacts.md).

## Expanded-LLE dataset

The expanded dataset contains 6,709 filtered unaugmented records from 719
systems:

| Partition | Systems | Records |
| --- | ---: | ---: |
| Train | 575 | 5,370 |
| Validation | 72 | 707 |
| Test | 72 | 632 |

## Expanded-LLE results

| Test MAE | Test RMSE | Test R2 |
| ---: | ---: | ---: |
| 0.10787 +/- 0.00474 | 0.17931 +/- 0.00576 | 0.74036 +/- 0.01662 |

Per-seed checkpoints, metrics, predictions, parity plots, and functional-group
corpora are distributed under:

```text
experiments/section_3_results/3_4_industrial_extraction_design/
  expanded_lle_adaptation/results/multiseed_reference/
```

List these checkpoints with:

```bash
python scripts/evaluate_checkpoint_registry.py \
  --registry configs/reproduction/multiseed_checkpoint_registry.json \
  --list
```

## Protocol differences that prevent direct pooling

The main and expanded rows should not be pooled into one performance estimate:

- they use different source workbooks and system populations;
- the expanded input adds pressure as a third scalar;
- the expanded profile performs supervised full-network fine-tuning;
- output errors reflect different test-system distributions.

Likewise, sample-major multi-seed results and component-major manuscript
checkpoint results belong to separate compatibility tracks.

## Recomputing a summary

The summarization entry point is:

```bash
python scripts/analysis/summarize_multiseed_benchmark.py --help
```

Run it on a new output tree rather than overwriting the archived summary. A
valid summary should identify included seeds, checkpoint-selection metric,
missing runs, aggregation convention, and source metric files.

## Reporting guidance

When citing these results, include:

- sample-major protocol name;
- seeds 42, 43, and 44;
- fixed system-level split;
- unaugmented test counts;
- mean and sample standard deviation;
- whether a value is predictive or thermodynamic;
- availability status of the associated checkpoint.

# Auxiliary Sample-Major Multi-Seed Benchmarks

## Scope

This document summarizes maintained `sample_major` runs for seeds 42, 43, and
44. These values are auxiliary repository benchmarks. They are not the
numerical source for main-text Tables 1-3.

The manuscript uses several experiment-specific replication protocols:

| Paper item | Replication protocol |
| --- | --- |
| Table 1 | Five independent runs; reported as mean and sample standard deviation |
| Table 2 | Validation-selected historical architecture-ablation records |
| Table 3 | Historical data-driven and physics-informed checkpoint records |
| Tables S3-S4 | Seeds 7, 42, and 2024 |
| Table S8 | Five system splits with seeds 42-46 |
| Section S3.9 efficiency | Seeds 42, 43, and 44 |

Never combine uncertainty estimates from these protocols.

## Main sample-major benchmark

The filtered main dataset contains 7,683 unaugmented records from 765 systems.
The fixed partition is:

| Partition | Systems | Records |
| --- | ---: | ---: |
| Train | 612 | 6,092 |
| Validation | 75 | 788 |
| Test | 78 | 803 |

All three auxiliary seeds use this fixed system-level partition. Component
permutation is restricted to the training partition, checkpoints are selected
by validation performance, and the unaugmented test partition is evaluated
after selection.

| Stage | Test MAE | Test RMSE | Test R2 | Chemical-potential residual MAE | TPD violation rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supervised | 0.05145 +/- 0.00089 | 0.09270 +/- 0.00136 | 0.92674 +/- 0.00216 | 1.45072 +/- 0.03246 | 20.96% +/- 0.73% |
| NRTL regularized | 0.05176 +/- 0.00127 | 0.09264 +/- 0.00129 | 0.92682 +/- 0.00204 | 1.36934 +/- 0.12601 | 20.63% +/- 2.21% |

Values are arithmetic mean plus or minus sample standard deviation across
seeds 42, 43, and 44. Predictive and thermodynamic changes answer different
questions: a smaller chemical-potential residual does not imply a smaller
composition MAE, and the auxiliary stage-2 objective does not directly optimize
the reported TPD diagnostic.

Per-seed metric records are under:

```text
experiments/section_3_results/3_1_lle_prediction/
  3_1_2_ablation_analysis/physics_regularization/
  results/stage_comparison/seed*/
```

These metric files document maintained training behavior. For executable
manuscript checkpoints, use the component-major registry described in
[Checkpoint and Artifact Reference](../reference/checkpoints_and_artifacts.md).

## Expanded-LLE auxiliary benchmark

The expanded dataset contains 6,709 filtered unaugmented records from 719
systems:

| Partition | Systems | Records |
| --- | ---: | ---: |
| Train | 575 | 5,370 |
| Validation | 72 | 707 |
| Test | 72 | 632 |

The auxiliary three-seed summary is:

| Test MAE | Test RMSE | Test R2 |
| ---: | ---: | ---: |
| 0.10787 +/- 0.00474 | 0.17931 +/- 0.00576 | 0.74036 +/- 0.01662 |

Per-seed checkpoints, metrics, predictions, parity plots, and functional-group
corpora are distributed under:

```text
experiments/section_3_results/3_4_industrial_extraction_design/
  expanded_lle_adaptation/results/multiseed_reference/
```

List the associated checkpoints with:

```bash
python scripts/evaluate_checkpoint_registry.py \
  --registry configs/reproduction/multiseed_checkpoint_registry.json \
  --list
```

The expanded input adds pressure as a third scalar and uses full-parameter
supervised adaptation. Its metrics must not be pooled with the main benchmark.

## Recomputing an auxiliary summary

Inspect the summarization interface with:

```bash
python scripts/analysis/summarize_multiseed_benchmark.py --help
```

Write new summaries to a new output directory. A valid report identifies:

- included and missing seeds;
- exact checkpoint-selection metric;
- node layout and scalar dimension;
- fixed split manifest;
- augmented training and unaugmented evaluation counts;
- arithmetic mean and sample standard deviation;
- source metric files and software environment.

See [Main-Text Result Mapping](main_text_results.md) before comparing an
auxiliary value with a manuscript table.

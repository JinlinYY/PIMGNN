# PSMI Sample-Major Multi-Seed Results

## Protocol

Seeds 42, 43, and 44 use the same fixed system-level split and sample-major mixture-node layout. Component permutation is applied only to the training partition. Checkpoints are selected by validation performance and evaluated on the test partition after selection.

## Main benchmark

The dataset contains 7,683 tie lines from 765 systems. The train/validation/test split contains 612/75/78 systems and 6,092/788/803 unaugmented records.

| Stage | Test MAE | Test RMSE | Test R2 | Chemical-potential residual MAE | TPD violation rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supervised | 0.05145 +/- 0.00089 | 0.09270 +/- 0.00136 | 0.92674 +/- 0.00216 | 1.45072 +/- 0.03246 | 20.96% +/- 0.73% |
| NRTL regularized | 0.05176 +/- 0.00127 | 0.09264 +/- 0.00129 | 0.92682 +/- 0.00204 | 1.36934 +/- 0.12601 | 20.63% +/- 2.21% |

## Expanded LLE adaptation

The expanded dataset contains 6,709 tie lines from 719 systems. The fixed split contains 575/72/72 systems and 5,370/707/632 unaugmented records.

| Test MAE | Test RMSE | Test R2 |
| ---: | ---: | ---: |
| 0.10787 +/- 0.00474 | 0.17931 +/- 0.00576 | 0.74036 +/- 0.01662 |

Machine-readable per-seed metrics, predictions, figures, and checkpoints are stored under `experiments/section_3_results/3_1_lle_prediction/main_benchmark/results/multiseed_reference/`. The canonical manuscript parity plot is stored separately at `results/figure_2a.png`.

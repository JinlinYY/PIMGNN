# Evaluation Metrics

Metric definitions do not establish result identity. Before quoting a value,
first locate its figure or table in the [Paper-Aligned Results Index](README.md)
and confirm the dataset, split, checkpoint, seed aggregation, and stored versus
recomputed status.

## Prediction vector

For `N` records, the true and predicted arrays have shape `(N, 6)` with order:

```text
[Ex1, Ex2, Ex3, Rx1, Rx2, Rx3]
```

Unless a phase-specific suffix is present, predictive metrics flatten all six
components across all records.

## Mean absolute error

```text
MAE = mean(|y_true - y_pred|)
```

MAE has mole-fraction units and weights every component equally. `mae_E` and
`mae_R` apply the same definition to the three extract or raffinate components.

## Root mean squared error

```text
RMSE = sqrt(mean((y_true - y_pred)^2))
```

RMSE emphasizes larger component errors more strongly than MAE. `rmse_E` and
`rmse_R` are phase-specific.

## Coefficient of determination

```text
R2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
```

`r2_E` and `r2_R` use phase-specific flattened arrays. R2 can be negative for a
poor model and becomes undefined when the true array has negligible variance.
Do not compare an overall flattened R2 with a system-averaged or
component-averaged value from another implementation without aligning the
aggregation convention.

## Composition closure

The model applies softmax within each phase. The evaluator nevertheless reports:

| Metric | Definition |
| --- | --- |
| `sum_err_E` | Mean `|sum(Ex) - 1|` |
| `sum_err_R` | Mean `|sum(Rx) - 1|` |
| `sum_err_95` | 95th percentile across both phase errors |
| `neg_frac` | Fraction of six-component predictions below zero |

These values detect checkpoint incompatibility, unexpected postprocessing, or
numerical problems. They are structural diagnostics, not independent measures
of phase-equilibrium accuracy.

## Thermodynamic parameter coverage

`param_cov` is the fraction of evaluated samples for which the selected
thermodynamic parameter store provides a matching system entry. Chemical-
potential, Gibbs-Duhem, and TPD diagnostics should be interpreted together with
coverage. Comparing physics metrics at different coverage can be misleading.

## Chemical-potential residual

For component `i`:

```text
r_i = ln(x_i^E * gamma_i^E) - ln(x_i^R * gamma_i^R)
```

The evaluator reports:

- `mu_res_mae`: mean absolute residual across available samples and components;
- `mu_res_rmse`: root mean square residual;
- `mu_res_max`: maximum absolute residual when exported by the run.

The residual is dimensionless. It depends on the predicted compositions,
temperature, selected excess-Gibbs-energy model, and fitted interaction
parameters. It is not directly comparable across different parameter stores or
coverage without qualification.

## Gibbs-Duhem diagnostic

The evaluator estimates the directional residual of:

```text
sum_i x_i * d(ln gamma_i) = 0
```

using finite differences along simplex directions. Reported fields include:

- `gd_penalty_mean`;
- `gd_penalty_p95`;
- `gd_res_mae` in extended exports.

The public stage-2 training profile sets `MECH_W_GD: 0`. Therefore this value is
a post-hoc consistency diagnostic, not evidence that a separate Gibbs-Duhem
loss was minimized.

## Tangent-plane-distance diagnostic

Local trial compositions are sampled around each phase, and the implemented TPD
penalty records instability relative to the configured margin:

- `tpd_viol_rate`: fraction of evaluated samples with positive penalty;
- `tpd_viol_mean`: mean of positive violation penalties.

This is a stochastic local diagnostic controlled by trial count, perturbation
scale, margin, parameter coverage, and random state. The public stage-2 profile
sets `MECH_W_STAB: 0`, so TPD is not an optimized loss in that profile.

## Top-level package metrics

| Run | Best epoch | Test MAE | Test RMSE | Test R2 |
| --- | ---: | ---: | ---: | ---: |
| Data-driven | 101 | 0.033925 | 0.055169 | 0.974053 |
| Chemical-potential regularized | 29 | 0.034851 | 0.054381 | 0.974788 |

These values come from the `best_metrics.json` files in the two top-level result
packages. They describe specific archived checkpoints and must not be cited as
the complete main-text Table 3 without checking provenance. The
physics-informed values round to the manuscript Table 3 row, but the
data-driven package's composition metrics differ from the data-driven
composition metrics printed in that table. The exact boundary is documented in
[Main-Text Result Mapping](main_text_results.md#table-3-chemical-potential-regularization)
and [Artifact Status and Result Discrepancies](artifact_status_and_discrepancies.md#table-3-data-driven-row).

## Pointwise versus system-level reporting

Overall MAE/RMSE weight records and components, so systems with more retained
tie lines contribute more terms. System-level generalization analyses first
aggregate or classify by system/category and answer a different question.

When comparing to another work, document whether the metric is:

- pointwise over all six components;
- phase-specific;
- component-specific;
- averaged by system;
- averaged across seeds;
- computed on the exact same test systems.

## Multi-seed uncertainty

The repository contains several experiment-specific seed sets. Table 1 uses
five independent runs, Tables S3-S4 use seeds 7/42/2024, Table S8 uses split
seeds 42-46, and the auxiliary sample-major and efficiency summaries use seeds
42/43/44. In every case, the reported spread is a sample standard deviation
under that fixed protocol; it is not a confidence interval over all possible
datasets or chemical systems.

## Thermodynamic-threshold sensitivity

No universal cutoff is assumed for the dimensionless chemical-potential
residual. The consistency audit reports violation rates over a threshold grid:

```bash
python scripts/analysis/evaluate_thermodynamic_consistency.py --help
```

Report the threshold, parameter store, coverage, and aggregation rule whenever
quoting a violation percentage.

## Minimum comparison checklist

Before comparing two metric rows, align:

1. dataset and minimum-density filter;
2. system-level split;
3. unaugmented evaluation set;
4. checkpoint-selection rule;
5. model layout and scalar definition;
6. aggregation convention;
7. thermodynamic model and parameter coverage;
8. seed aggregation.

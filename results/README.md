# Published Results

This directory contains the canonical main-text Figure 2a and two reference result packages used to compare data-driven and chemical-potential-regularized PSMI models.

## Canonical figure

- `figure_2a.png`: complete extract- and raffinate-phase parity plot used as the public Figure 2a asset.

## Reference result packages

| Directory | Training objective | Best epoch | Test MAE | Test RMSE | Test R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `data_driven/` | Composition prediction loss | 101 | 0.033925 | 0.055169 | 0.974053 |
| `chemical_potential_regularized/` | Chemical-potential regularization | 29 | 0.034851 | 0.054381 | 0.974788 |

Each package uses the same researcher-facing structure:

```text
<run>/
|- checkpoints/                    Validation-selected checkpoint
|- metrics/                        Best metrics and epoch-wise metric log
|- predictions/                    Pointwise test-set predictions
|- artifacts/                      Functional-group corpus
|- figures/parity/                 Extract- and raffinate-phase parity plots
|- figures/training_curves/        Prediction and physics diagnostic curves
|- figures/ternary_phase_diagrams/ Aggregate PDF and individual system plots
|- artifact_manifest.csv           File sizes and SHA-256 identities
`- README.md                       Run scope and directory guide
```

The data-driven package provides the source checkpoint and predictions associated with Figure 2a. The chemical-potential-regularized package follows the same evaluation structure and additionally reports thermodynamic consistency diagnostics.

The section-aligned experiment code and canonical source assets remain under `experiments/section_3_results/3_1_lle_prediction/`.

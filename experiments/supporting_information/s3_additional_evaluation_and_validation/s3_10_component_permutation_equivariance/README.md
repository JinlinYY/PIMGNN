# Component-2/3 Permutation-Equivariance Audit

This experiment tests whether the released PSMI checkpoint responds consistently
when Components 2 and 3 are exchanged. It is a paired, post-hoc evaluation of
the model and molecular-graph pipeline used for Figure 2a.

## Scientific question

For an original test record,

```text
X = (M1, M2, M3, T, t)
Y = (Ex1, Ex2, Ex3, Rx1, Rx2, Rx3)
```

the exchange operator is

```text
P23 X = (M1, M3, M2, T, t)
P23 Y = (Ex1, Ex3, Ex2, Rx1, Rx3, Rx2).
```

The molecular inputs and the corresponding extract- and raffinate-phase mole
fractions are exchanged together. Temperature and the phase-path coordinate
remain unchanged. The Figure 2a checkpoint has two scalar inputs and does not
consume pressure; the pressure column and system identifier are preserved as
tabular metadata. Because the output is component indexed, the relevant
property is equivariance rather than invariance:

```text
f(P23 X) approximately equals P23 f(X).
```

The audit therefore compares `f(X)` with `P23^-1 f(P23 X)` for every test
record and output component.

## Evaluation protocol

- Checkpoint: the registered `figure2a_psmi` checkpoint, epoch 101.
- Test partition: 803 records from 78 held-out chemical systems.
- Pairing: each original record is matched to exactly one Component-2/3
  exchanged record.
- Parameter handling: the same saved weights and feature scalers are used for
  both inference passes.
- Predictive metrics: MAE, RMSE, and R2 after flattening the three composition
  outputs within each reported phase.
- Equivariance metrics: MAE, RMSE, 95th percentile, and maximum absolute
  difference between the paired predictions.
- Uncertainty: 95% percentile intervals from 10,000 bootstrap resamples of the
  78 complete systems. Tie lines from the same system are never resampled as
  independent observations.

## Results

The original-ordering metrics reproduce the Figure 2a values to the precision
reported in the manuscript.

| Evaluation | Phase | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| Original ordering | Extract | 0.0371 | 0.0566 | 0.9671 |
| Original ordering | Raffinate | 0.0318 | 0.0545 | 0.9784 |
| Components 2/3 exchanged | Extract | 0.0356 | 0.0556 | 0.9682 |
| Components 2/3 exchanged | Raffinate | 0.0312 | 0.0539 | 0.9788 |

| Phase | Equivariance MAE | 95% system-bootstrap CI | RMSE | P95 absolute deviation | Maximum absolute deviation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 0.0110 | 0.0098-0.0123 | 0.0168 | 0.0379 | 0.0783 |
| Extract | 0.0104 | 0.0089-0.0121 | 0.0151 | 0.0344 | 0.0526 |
| Raffinate | 0.0115 | 0.0098-0.0134 | 0.0183 | 0.0426 | 0.0783 |

For overall predictive accuracy, the exchanged-minus-original differences
were -0.0010 for MAE (95% CI: -0.0025 to 0.0006), -0.0008 for RMSE (95% CI:
-0.0032 to 0.0016), and 0.0007 for R2 (95% CI: -0.0015 to 0.0029). Each
interval includes zero. The exchange therefore produced no detectable
systematic loss of predictive accuracy in this paired test.

The audit supports approximate permutation equivariance of the released model;
it does not claim mathematically exact equivariance. It evaluates the behavior
of the complete trained pipeline and is distinct from a causal ablation of the
training augmentation.

## Reproduction

Run from the repository root in the documented `ggnn39` environment:

```bash
python scripts/analysis/evaluate_component_permutation_equivariance.py \
  --device cuda \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 2026
```

Use `--device cpu` on systems without CUDA. This command performs inference
and statistical analysis only.

## Evidence files

| File | Contents |
| --- | --- |
| `results/predictive_metrics.csv` | Original and exchanged predictive metrics by phase |
| `results/equivariance_metrics.csv` | Paired equivariance deviations by phase |
| `results/system_cluster_bootstrap_intervals.csv` | Point estimates and 95% system-level intervals, including paired metric differences |
| `results/paired_predictions.csv` | All 803 records, exchanged molecular inputs and true labels, both prediction orderings, restored outputs, and componentwise deviations |
| `results/experiment_manifest.json` | Checkpoint, input, runtime, split, and output provenance |
| `figures/component_23_permutation_equivariance.pdf` | Vector publication figure |
| `figures/component_23_permutation_equivariance.png` | 300 dpi raster figure |

The plotting code is part of the evaluation script so the numerical tables and
figure are generated from the same paired predictions.

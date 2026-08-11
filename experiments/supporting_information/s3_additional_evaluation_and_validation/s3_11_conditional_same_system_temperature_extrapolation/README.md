# S3.11 Conditional Same-System Temperature Extrapolation

This package contains the code, fixed split, reference checkpoint, numerical
evidence, and publication figure for SI Section S3.11, Table S17, and Figure
S7.

## Scientific protocol

Chemical systems are identified by the order-invariant sorted tuple of the
three canonical SMILES strings. The 32 systems measured at three or more
distinct experimental temperatures are treated as target systems. For each
target system, the lowest and highest temperatures are held out, while the
interior temperature or temperatures remain in the training partition.

- Target chemical systems: 32
- Held-out system-temperature groups: 64
- Held-out experimental tie lines: 623
- Temperature distance to the nearest retained value: 4.95-35.00 K (median
  11.00 K)
- Systems with only one retained interior temperature: 30 of 32
- Synthetic temperature-interpolation records: 0
- Model selection: validation RMSE on background systems only
- Test use: the held-out extremes are excluded from training, early stopping,
  and checkpoint selection

The nearest-observed-temperature reference transfers the phase path measured
at the closest retained temperature of the same chemical system. It is a
deliberately strong within-system baseline, not a chemistry-generalization
model.

## Table S17

| Held-out side | Method | Tie lines | Median gap (K) | MAE | RMSE | R2 | Median tie-line angle (deg) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low temperature | PSMI | 312 | 10.05 | 0.0431 | 0.0674 | 0.8929 | 3.40 |
| Low temperature | Nearest observed temperature | 312 | 10.05 | 0.0280 | 0.0510 | 0.9387 | 1.82 |
| High temperature | PSMI | 311 | 15.00 | 0.0401 | 0.0610 | 0.9237 | 4.57 |
| High temperature | Nearest observed temperature | 311 | 15.00 | 0.0218 | 0.0392 | 0.9684 | 1.22 |
| Combined | PSMI | 623 | 11.00 | 0.0416 | 0.0643 | 0.9096 | 3.76 |
| Combined | Nearest observed temperature | 623 | 11.00 | 0.0249 | 0.0455 | 0.9547 | 1.50 |

These results define a conditional same-system temperature test. They do not
claim simultaneous extrapolation to both unseen chemistry and unseen
temperature. The nearest-temperature reference performs better than PSMI in
this experiment, as reported in Table S17.

## Reproduction

Run commands from the repository root in the documented `ggnn39`
environment.

Regenerate Figure S7 directly from the archived numerical tables:

```bash
python scripts/visualization/plot_same_system_temperature_extrapolation.py
```

Rebuild and validate the split without training or inference:

```bash
python scripts/experiments/run_same_system_temperature_extrapolation.py \
  --split-only \
  --out-dir outputs/same_system_temperature_extrapolation
```

Re-evaluate the distributed checkpoint and write a separate result bundle:

```bash
python scripts/experiments/run_same_system_temperature_extrapolation.py \
  --device cuda \
  --out-dir outputs/same_system_temperature_extrapolation
```

Use `--device cpu` on systems without CUDA. A new training run is initiated
only when `--train-from-scratch` is supplied explicitly.

## Package contents

| Path | Contents |
| --- | --- |
| `figures/figure_s7.pdf` and `figure_s7.png` | Vector and 300 dpi versions of Figure S7 |
| `results/summary.csv` | Table S17 source metrics |
| `results/by_system_temperature.csv` | Metrics for each held-out chemistry-temperature group |
| `results/predictions.csv` | All 623 pointwise predictions and nearest-temperature references |
| `results/evaluation_manifest.json` | Selected-checkpoint and evaluation summary |
| `splits/split_assignments.csv` | Interior-training and extreme-test temperature assignments |
| `splits/target_systems.csv` | Canonical-SMILES inventory for the 32 target systems |
| `splits/split_manifest.json` | Counts, temperature gaps, and split parameters |
| `models/reference_checkpoint/` | Selected checkpoint, feature-group vocabulary, and provenance |

## Checkpoint provenance note

All model-state tensors are preserved from the selected checkpoint. Its
temperature scaler (`302.449066 K`, `9.739730 K`) agrees with the S3.11
training partition reconstructed from the public benchmark workbook. A stale
machine-local dataset reference inherited from the expanded-dataset global
default was removed from the public checkpoint metadata. The original metadata
discrepancy and the sanitized checkpoint digest are recorded explicitly in
`models/reference_checkpoint/provenance.json`.

# Data-Driven Result Package

This package contains the reference data-driven PSMI run used as the source checkpoint and pointwise prediction export for Figure 2a.

## Summary

- Best epoch: 101
- Test MAE: 0.033925
- Test RMSE: 0.055169
- Test R2: 0.974053

## Contents

- `checkpoints/`: validation-selected model state.
- `metrics/`: best validation/test metrics and the epoch-wise metric log.
- `predictions/`: pointwise test-set compositions.
- `artifacts/`: functional-group corpus used by the run.
- `figures/parity/`: phase-specific parity plots.
- `figures/training_curves/`: predictive-performance curves.
- `figures/ternary_phase_diagrams/`: aggregate and per-system phase-diagram visualizations.
- `artifact_manifest.csv`: SHA-256 identity of every packaged artifact.

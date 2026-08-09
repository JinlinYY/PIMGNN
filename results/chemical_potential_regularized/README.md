# Chemical-Potential-Regularized Result Package

This package contains the reference PSMI run trained with chemical-potential regularization.

## Summary

- Best epoch: 29
- Test MAE: 0.034851
- Test RMSE: 0.054381
- Test R2: 0.974788
- Test chemical-potential residual MAE: 0.541074

## Contents

- `checkpoints/`: validation-selected model state.
- `metrics/`: predictive and thermodynamic metrics.
- `predictions/`: pointwise test-set compositions.
- `artifacts/`: functional-group corpus used by the run.
- `figures/parity/`: phase-specific parity plots.
- `figures/training_curves/`: prediction and physics-diagnostic curves.
- `figures/ternary_phase_diagrams/`: aggregate and per-system phase-diagram visualizations.
- `artifact_manifest.csv`: SHA-256 identity of every packaged artifact.

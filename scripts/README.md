# Command-Line Entry Points

- `train.py`: supervised, physics-regularized, and expanded-LLE training.
- `evaluate_checkpoint.py`: deterministic checkpoint evaluation.
- `evaluate_checkpoint_registry.py`: registry-based multi-checkpoint evaluation.
- `fit_nrtl.py`: NRTL parameter fitting for training and diagnostics.
- `analysis/`: dataset, sensitivity, thermodynamic, and result-bundle analysis.
- `data_preparation/`: filtering, split construction, and format conversion.
- `experiments/`: baseline, split-strategy, sensitivity, transfer-learning,
  and SI S3.11 same-system temperature-extrapolation workflows.
- `visualization/`: paper figures, phase diagrams, and attribution plots.

The S3.11 entry points are:

- `experiments/run_same_system_temperature_extrapolation.py`: reconstruct the
  fixed split, evaluate the distributed checkpoint, or explicitly launch a
  new training run.
- `visualization/plot_same_system_temperature_extrapolation.py`: regenerate
  Figure S7 from the archived result tables without model execution.

All commands are executed from the repository root and resolve the local `src/` package automatically.

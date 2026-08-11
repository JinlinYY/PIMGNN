# Command-Line Entry Points

- `train.py`: supervised, physics-regularized, and expanded-LLE training.
- `evaluate_checkpoint.py`: deterministic checkpoint evaluation.
- `evaluate_checkpoint_registry.py`: registry-based multi-checkpoint evaluation.
- `fit_nrtl.py`: NRTL parameter fitting for training and diagnostics.
- `run_temperature_encoding_sensitivity.py`: one-seed controlled temperature-extrapolation training.
- `plot_temperature_encoding_sensitivity.py`: archived multi-seed aggregation, group bootstrap, and Figure S5 generation.
- `analysis/`: dataset, sensitivity, thermodynamic, and result-bundle analysis.
- `data_preparation/`: filtering, split construction, and format conversion.
- `experiments/`: baseline, split-strategy, sensitivity, and transfer-learning workflows.
- `visualization/`: paper figures, phase diagrams, and attribution plots.

All commands are executed from the repository root and resolve the local `src/` package automatically.

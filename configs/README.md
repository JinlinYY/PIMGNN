# Configuration System

PSMI configurations are composed from layered YAML files.

- `data/` defines datasets, filters, split manifests, and thermodynamic parameter files.
- `model/` defines graph encoders, functional-group interactions, scalar inputs, and fusion modes.
- `training/` defines optimization, checkpoint selection, regularization, and evaluation behavior.
- `experiments/` composes complete training protocols.
- `reproduction/` registers published checkpoints for standardized evaluation.

Paths are resolved relative to the repository root. Command-line `KEY=VALUE` overrides are applied after the YAML layers.

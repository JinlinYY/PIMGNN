# PSMI Documentation

This documentation is written for researchers who want to inspect, evaluate,
extend, or deploy PSMI. The executable code, YAML profiles, checkpoint tensor
shapes, and frozen data manifests are authoritative. The documentation explains
how those pieces fit together and explicitly distinguishes maintained training
workflows from compatibility workflows for published checkpoints.

## Start here

| Goal | Recommended document |
| --- | --- |
| Install the research environment | [Installation](getting_started/installation.md) |
| Evaluate a distributed checkpoint | [Quick start](getting_started/quickstart.md) |
| Understand the model and thermodynamic objective | [Scientific model contract](architecture/scientific_model_contract.md) |
| Follow tensors through the network | [Model pipeline](architecture/model_pipeline.md) |
| Understand filtering, splitting, and augmentation | [Data pipeline and split policy](guides/data_pipeline.md) |
| Understand how the baseline table was produced | [Baseline comparison protocol](guides/baseline_comparison.md) |
| Reproduce figures, tables, and checkpoint metrics | [Reproduction guide](guides/reproduction.md) |
| Run an industrial application case | [Application-case workflow](guides/application_case.md) |
| Run the browser interface | [Web application](guides/web_application.md) |
| Create or override YAML profiles | [Configuration reference](reference/configuration.md) |
| Select a compatible checkpoint | [Checkpoint and artifact reference](reference/checkpoints_and_artifacts.md) |
| Match manuscript figures and tables to public artifacts | [Paper-aligned results index](results/README.md) |
| Interpret predictive and physics diagnostics | [Evaluation metrics](results/evaluation_metrics.md) |
| Inspect non-manuscript maintenance summaries | [Auxiliary multi-seed benchmarks](results/multiseed_benchmark.md) |
| Resolve common setup or compatibility failures | [Troubleshooting](troubleshooting.md) |

## Scientific navigation

The [paper-aligned experiment index](../experiments/README.md) maps each main
text and Supporting Information section to its code, source data, archived
metrics, checkpoints, and figures. The [dataset card](../datasets/DATASET_CARD.md)
defines record counts, field aliases, frozen file hashes, filtering, and reuse
conditions. The [published-results guide](../results/README.md) describes the
canonical Figure 2a and the two distributed objective-comparison packages.

## Two supported model tracks

PSMI has two intentionally separate compatibility tracks:

- `sample_major` is the maintained training and extension implementation. New
  experiments should start from `configs/model/psmi_sample_major.yaml`.
- `component_major` reproduces the node ordering embedded in published
  checkpoints. These checkpoints are selected through
  `configs/reproduction/published_checkpoint_registry.json`.

Do not compare or exchange checkpoints across these layouts unless the
checkpoint registry or compatibility loader explicitly declares the required
adaptation. The distinction is explained in the
[scientific model contract](architecture/scientific_model_contract.md).

## Documentation conventions

- Commands assume the current working directory is the repository root.
- Paths use forward slashes and work with Python on Windows, Linux, and macOS.
- `extract` refers to output components `Ex1`, `Ex2`, and `Ex3`.
- `raffinate` refers to output components `Rx1`, `Rx2`, and `Rx3`.
- Dataset counts always refer to unaugmented experimental records unless a
  document explicitly says otherwise.
- Archived metrics are reported as stored. Commands that train or fit
  thermodynamic parameters are clearly identified so they are not confused
  with checkpoint-only evaluation.

## Repository-level references

- [Main project README](../README.md)
- [Command-line entry points](../scripts/README.md)
- [Contribution guide](../CONTRIBUTING.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)
- [Web deployment guide](../Web/PSMI-LLE-web/README.md)

# Quick Start

This guide evaluates distributed artifacts. It does not require fitting NRTL
parameters or training new neural-network weights.

## 1. Confirm the environment

```bash
conda activate ggnn39
python -c "import psmi, torch; print(psmi.__file__); print(torch.__version__)"
```

Run commands from the repository root so relative paths resolve against the
same project tree used by the YAML profiles and checkpoint registries.

## 2. Inspect available manuscript checkpoints

```bash
python scripts/evaluate_checkpoint_registry.py \
  --registry configs/reproduction/published_checkpoint_registry.json \
  --list
```

The registry binds each run identifier to five pieces of provenance:

1. the layered YAML configuration;
2. the checkpoint file;
3. the functional-group vocabulary used when the checkpoint was created;
4. optional reference predictions;
5. explicit compatibility overrides for the published node layout.

This binding is safer than selecting a `.pt` file by filename alone.

## 3. Evaluate the Figure 2a checkpoint

```bash
python scripts/evaluate_checkpoint_registry.py \
  --registry configs/reproduction/published_checkpoint_registry.json \
  --only figure2a_psmi \
  --output-root outputs/checkpoint_evaluation \
  --device cpu \
  --no-plots
```

Remove `--no-plots` to generate the evaluation figures. The command loads a
saved checkpoint, reconstructs the fixed test partition, computes predictions,
and writes metrics and pointwise tables under the selected output root.

For a CUDA device, replace `--device cpu` with `--device cuda` or a specific
device such as `--device cuda:0`.

## 4. Evaluate a checkpoint directly

The lower-level evaluator is useful for a custom output location or an
unregistered compatible checkpoint:

```bash
python scripts/evaluate_checkpoint.py \
  --config configs/experiments/main_benchmark_stage1.yaml \
  --checkpoint experiments/section_3_results/3_1_lle_prediction/main_benchmark/models/figure_2a_psmi/best_model.pt \
  --fg-corpus experiments/section_3_results/3_1_lle_prediction/main_benchmark/artifacts/figure_2a_fg_corpus.json \
  --reference-predictions experiments/section_3_results/3_1_lle_prediction/main_benchmark/data/figure_2a_predictions.csv \
  --output-dir outputs/figure2a_direct \
  --set MIXTURE_NODE_LAYOUT=component_major \
  --allow-input-hash-mismatch \
  --device cpu \
  --no-plots
```

Use the registry command for published checkpoints unless you specifically
need direct control. It already supplies the verified compatibility settings.

## 5. Inspect the archived reference packages

The top-level `results/` directory provides two self-contained packages:

- `results/data_driven/`;
- `results/chemical_potential_regularized/`.

Each package contains a validation-selected checkpoint, pointwise predictions,
metrics, training curves, parity plots, ternary phase diagrams, and an artifact
manifest with byte sizes and SHA-256 hashes. These packages are the quickest
way to inspect archived evidence without executing Python.

The canonical main-text image is `results/figure_2a.png`. Its numerical source
table and checkpoint remain in the paper-aligned experiment directory rather
than being duplicated under several historical names.

## 6. Export the canonical Figure 2a bundle

```bash
python scripts/analysis/export_figure_2a_results.py
```

This exporter organizes existing source predictions, metrics, figures, and
checkpoint artifacts. It is an artifact-management command, not a training
entry point.

## 7. Analyze the bundled industrial application output

```bash
python scripts/run_application_case.py \
  --csv experiments/section_3_results/3_4_industrial_extraction_design/application_workflow/results/application_case_predictions.csv \
  --out_dir outputs/application_case_analysis \
  --analyze_only
```

Analysis-only mode reads an existing prediction table and generates summaries
and visualizations. For new application input, follow the
[application-case workflow](../guides/application_case.md).

## 8. Choose the correct model track

| Task | Model layout | Recommended entry point |
| --- | --- | --- |
| Reproduce Figure 2a or Table 3 | `component_major` | Published checkpoint registry |
| Train a new main-benchmark model | `sample_major` | `scripts/train.py` with an experiment YAML |
| Adapt the maintained model to a new dataset | `sample_major` | New layered YAML profile |
| Run the bundled Web application | Checkpoint-declared compatibility | Web backend configuration |

Never infer layout compatibility from a filename. Use checkpoint metadata,
the registry, or the compatibility loader.

## 9. Next steps

- Read the [reproduction guide](../guides/reproduction.md) for paper-section
  mapping and artifact validation.
- Read the [data pipeline](../guides/data_pipeline.md) before changing filters,
  splits, or augmentation.
- Read the [configuration reference](../reference/configuration.md) before
  starting new training.
- Read [evaluation metrics](../results/evaluation_metrics.md) before comparing
  composition accuracy with thermodynamic diagnostics.

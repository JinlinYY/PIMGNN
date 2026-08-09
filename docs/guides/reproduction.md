# Reproducing Published and Archived Results

## Reproduction levels

The repository supports three distinct activities. State which one was used:

| Level | Weight updates | Typical purpose |
| --- | --- | --- |
| Artifact inspection | None | Read archived metrics, figures, predictions, and hashes |
| Checkpoint evaluation | None | Recompute predictions and metrics from distributed weights |
| Training reproduction | Yes | Re-run supervised, physics, or transfer optimization |

Artifact inspection is the fastest and most deterministic. Checkpoint
evaluation tests the executable model and data contract. Training reproduction
adds optimizer, GPU, and numerical variability and should not be required merely
to inspect reported evidence.

## Paper-aligned directory structure

The authoritative mapping is [experiments/README.md](../../experiments/README.md).
The main branches are:

```text
experiments/
|- section_3_results/
|  |- 3_1_lle_prediction/
|  |- 3_2_molecular_interaction_mechanisms/
|  |- 3_3_binary_solubility_validation/
|  `- 3_4_industrial_extraction_design/
`- supporting_information/
   |- s3_additional_evaluation_and_validation/
   |- s4_web_application/
   |- s5_dataset_construction_and_distribution/
   `- s6_system_classification/
```

Each experiment README states the scientific question, implementation entry
point, archived evidence, and paper location. Generated work should be written
to a new output directory rather than over the distributed reference files.

## Verify the repository before evaluation

```bash
python -m pytest -q
```

Then list the manuscript registry with the
[canonical quick-start command](../getting_started/quickstart.md#2-inspect-available-manuscript-checkpoints).

The registered checkpoint, functional-group corpus, and reference prediction
paths must all exist. The release tests also verify selected frozen SHA-256
identities and result-package manifests.

## Figure 2a

The canonical public image is:

```text
results/figure_2a.png
```

Its source assets are kept with Section 3.1:

```text
experiments/section_3_results/3_1_lle_prediction/main_benchmark/
|- artifacts/figure_2a_fg_corpus.json
|- data/figure_2a_predictions.csv
|- figures/figure_2a_parity.png
`- models/figure_2a_psmi/best_model.pt
```

Evaluate the source checkpoint with the
[canonical quick-start command](../getting_started/quickstart.md#3-evaluate-the-figure-2a-checkpoint).
Set `--output-root outputs/reproduction/figure2a` when a paper-section-specific
output directory is preferred. The registry supplies the `component_major`
override required by this archived checkpoint. Do not replace it with the
maintained sample-major profile when claiming checkpoint reproduction.

Re-export the canonical image from its source location:

```bash
python scripts/analysis/export_figure_2a_results.py
```

## Table 3 objective comparison

Two registered runs provide the data-driven and physics-informed checkpoints:

```bash
python scripts/evaluate_checkpoint_registry.py \
  --registry configs/reproduction/published_checkpoint_registry.json \
  --only paper_table_3 \
  --output-root outputs/reproduction/table3 \
  --device cpu \
  --no-plots
```

`--only` accepts either a run identifier or group and can be repeated. The
reference input tables, checkpoint weights, and parity figures are under
`3_1_2_ablation_analysis/physics_regularization/`.

The top-level `results/data_driven/` and
`results/chemical_potential_regularized/` directories are structured archival
packages. Their manifests verify every distributed file in each package.

## Figure 2b-2e interaction analysis

Attribution tables and publication assets are located under:

```text
experiments/section_3_results/3_2_molecular_interaction_mechanisms/
```

`scripts/explain.py` exposes saliency, integrated gradients, graph-explainer,
and functional-group SHAP-style modes. Explainability methods can be
computationally expensive and method settings affect ranking. When regenerating
an attribution result, record checkpoint, objective, target output, sample
subset, random seed, and method-specific step count.

## Figure 2f and external solubility validation

The binary-solubility validation directory contains the available base,
Compsol, and BigSolDB checkpoints. Conversion, fine-tuning, prediction, and
evaluation entry points are grouped under:

```text
scripts/experiments/transfer_learning/public_release/
```

Read that directory's README before running a transfer workflow. External
datasets can have their own licenses and target conventions.

## Figure 3 industrial cases

The two standardized case-study datasets and Figure 3c/3d assets are under:

```text
experiments/section_3_results/3_4_industrial_extraction_design/
```

For the bundled pointwise prediction table, use analysis-only mode:

```bash
python scripts/run_application_case.py \
  --csv experiments/section_3_results/3_4_industrial_extraction_design/application_workflow/results/application_case_predictions.csv \
  --out_dir outputs/reproduction/application_case \
  --analyze_only
```

## Expanded-LLE adaptation

Three expanded-data checkpoints are registered in
`configs/reproduction/multiseed_checkpoint_registry.json`:

```bash
python scripts/evaluate_checkpoint_registry.py \
  --registry configs/reproduction/multiseed_checkpoint_registry.json \
  --list
```

The corresponding per-seed metrics, predictions, figures, functional-group
corpora, and weights are stored under
`expanded_lle_adaptation/results/multiseed_reference/`.

The expanded profile uses pressure as a third scalar and full-network
supervised fine-tuning. It is a separate protocol from the main benchmark.

## Supplementary analyses from archived predictions

Many SI analyses operate on saved predictions or metric tables and do not need
weight updates. Examples include:

```bash
python scripts/visualization/build_supplementary_error_figures.py \
  --predictions experiments/supporting_information/s3_additional_evaluation_and_validation/s3_1_prediction_error_analysis/results/test_pointwise_predictions.csv \
  --output-dir outputs/reproduction/supplementary_errors

python scripts/analysis/evaluate_thermodynamic_consistency.py \
  --predictions results/chemical_potential_regularized/predictions/test_pointwise_predictions.csv \
  --nrtl-params datasets/parameters/main_benchmark/nrtl_params_all.json \
  --out-dir outputs/reproduction/thermodynamic_audit
```

The consistency script reports sensitivity across residual thresholds rather
than selecting one universal cutoff.

## Training reproduction

Training is configuration-driven:

```bash
python scripts/train.py --config configs/experiments/main_benchmark_stage1.yaml
python scripts/train.py --config configs/experiments/main_benchmark_stage2.yaml
python scripts/train.py --config configs/experiments/expanded_lle_finetune.yaml
```

These profiles write to `results/main_benchmark/` or
`results/transfer_evaluation/` by default. They do not overwrite paper-aligned
archived artifacts unless paths are explicitly changed.

Stage 2 expects the stage-1 output path declared in its YAML. Expanded
fine-tuning expects the stage-2 output. Run them sequentially or override
`LOAD_CKPT_PATH` with a compatible sample-major checkpoint:

```bash
python scripts/train.py \
  --config configs/experiments/main_benchmark_stage2.yaml \
  --set LOAD_CKPT_PATH=outputs/my_stage1/best_model.pt \
  --set OUT_DIR=outputs/my_stage2
```

Published component-major weights are compatibility artifacts and are not a
drop-in initialization for the maintained sample-major training sequence.

## Expected numerical variation

Small differences can arise from GPU kernels, PyTorch/RDKit versions, device,
batch order, and nondeterministic graph operations. A reproduction report
should distinguish:

- exact artifact identity;
- checkpoint re-evaluation on the same inputs;
- new training with the same protocol;
- modified architecture, split, or preprocessing.

If a result differs materially, compare hashes and pointwise predictions before
attributing the difference to stochastic training.

## Reproduction record template

Record:

```text
Git commit:
Environment file and package versions:
Device:
Registry and run id:
Checkpoint SHA-256:
Dataset SHA-256:
Split-manifest SHA-256:
Command:
Runtime overrides:
Output directory:
Observed metrics:
```

This minimal record is usually sufficient to diagnose protocol drift.

# Checkpoints and Artifacts

## Why registries are necessary

A neural-network checkpoint is scientifically meaningful only together with
the data, architecture, feature vocabulary, scaler, and evaluation protocol
that produced it. PSMI registries bind those dependencies to named run ids so a
researcher does not need to infer compatibility from filenames.

## Distributed registries

### Manuscript registry

`configs/reproduction/published_checkpoint_registry.json` contains:

| Run id | Purpose | Layout |
| --- | --- | --- |
| `figure2a_psmi` | Main Figure 2a evaluation | `component_major` |
| `table3_data_driven` | Table 3 data-only ablation | `component_major` |
| `table3_physics_informed` | Table 3 physics-informed ablation | `component_major` |

### Expanded-LLE registry

`configs/reproduction/multiseed_checkpoint_registry.json` contains the
expanded-data checkpoints for seeds 42, 43, and 44. These are separate from the
published main-benchmark compatibility weights.

List the published registry with the
[canonical quick-start command](../getting_started/quickstart.md#2-inspect-available-manuscript-checkpoints).

## Registry fields

| Field | Meaning |
| --- | --- |
| `id` | Stable command-line run identifier |
| `group` | Collection that can be selected with `--only` |
| `seed` | Recorded experiment seed |
| `config` | Layered YAML entry point |
| `checkpoint` | Model weight artifact |
| `fg_corpus` | Functional-group vocabulary required by the weight |
| `reference_predictions` | Frozen pointwise comparison table |
| `set` | Explicit configuration overrides |
| `allow_input_hash_mismatch` | Named compatibility exception for legacy metadata |
| `allow_derived_scalers` | Named compatibility exception for missing scaler metadata |

Compatibility flags apply only to the named registered artifact. They should
not be copied into a general evaluation command without inspecting the reason.

## Checkpoint formats

The compatibility loader recognizes common historical containers, including:

- a dictionary with `state_dict`;
- a nested `state_dict["model"]`;
- a dictionary with `model`;
- a plain model state dictionary.

Maintained checkpoints may also contain architecture settings, feature scalers,
dataset and split hashes, and optimizer provenance. The loader reports any
state-dict adaptation applied for a historical checkpoint.

PyTorch checkpoint files use Python serialization. Load only artifacts from a
trusted source and verify their hashes before use.

## Functional-group corpus

Functional-group token ids depend on a corpus or vocabulary. The corpus must
match the checkpoint. A mismatched corpus can preserve tensor shapes while
changing token meaning, so it is treated as a first-class artifact in the
published registry and result packages.

## Main checkpoint locations

```text
experiments/section_3_results/3_1_lle_prediction/
|- main_benchmark/models/figure_2a_psmi/best_model.pt
`- 3_1_2_ablation_analysis/physics_regularization/models/
   |- data_driven/best_model.pt
   `- physics_informed/best_model.pt
```

Expanded and transfer checkpoints are kept with their respective paper-aligned
experiments instead of a generic top-level `models/` archive.

## Result-package structure

The two top-level objective-comparison packages use the same layout:

```text
<run>/
|- checkpoints/best_model.pt
|- metrics/best_metrics.json
|- metrics/best_metrics.txt
|- metrics/training_metrics_log.csv
|- predictions/test_pointwise_predictions.csv
|- artifacts/functional_group_corpus.json
|- figures/parity/
|- figures/training_curves/
|- figures/ternary_phase_diagrams/
|- artifact_manifest.csv
`- README.md
```

`best_model.pt` is the validation-selected checkpoint. Pointwise predictions
are the basis for aggregate metrics and diagnostic figures. The training log is
provided for trajectory inspection; it does not supersede the selected metrics
record.

## Artifact manifests

Each `artifact_manifest.csv` records:

- package-relative path;
- file size in bytes;
- uppercase SHA-256 digest.

The release test recomputes every listed size and digest. To verify the complete
release:

```bash
python -m pytest -q tests/test_figure_2a_results.py
```

If a result-package file is intentionally replaced, regenerate the manifest and
explain the provenance change. Do not edit only the digest to make a modified
artifact appear original.

## Archived evidence versus new outputs

Paper-aligned experiment directories and top-level result packages contain
reference evidence. New evaluations should use a separate path such as:

```text
outputs/checkpoint_evaluation/<run_id>/
```

New training profiles default to paths under `results/main_benchmark/` or
`results/transfer_evaluation/`. Use a unique output path for each seed and
protocol. Avoid writing exploratory files into a distributed artifact package.

## Safe checkpoint-selection checklist

Before loading a checkpoint:

1. verify the source and SHA-256 digest;
2. identify `sample_major` or `component_major` layout;
3. confirm scalar dimension and temperature encoding;
4. load the matching functional-group corpus;
5. confirm scaler provenance;
6. confirm the dataset and split manifest;
7. inspect compatibility messages;
8. run closure and reference-prediction checks before reporting metrics.

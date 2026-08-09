# Application-Case Workflow

## Scope

`scripts/run_application_case.py` evaluates and visualizes a tabular
application case using a compatible PSMI checkpoint. The current command-line
workflow is designed for pointwise validation against an experimental phase
path: input rows contain measured extract and raffinate compositions or an
already assigned `t` coordinate, and the script writes model predictions beside
the source rows.

For interactive prediction from three molecules and operating conditions, use
the [Web application](web_application.md), which generates a phase-path grid for
the user interface.

## Input workbook

The complete workflow accepts an Excel file. Canonical columns are:

| Column | Required | Meaning |
| --- | --- | --- |
| `system_id` | Recommended | Identifier used to group one chemical system |
| `T` | Yes | Temperature in kelvin |
| `smiles1` | Yes | Component 1 SMILES |
| `smiles2` | Yes | Component 2 SMILES |
| `smiles3` | Yes | Component 3 SMILES |
| `t` | Conditional | Phase-path coordinate in `[0, 1]` |
| `Ex1`-`Ex3` | Conditional | Measured extract composition |
| `Rx1`-`Rx3` | Conditional | Measured raffinate composition |
| `Component 1`-`Component 3` | Optional | Human-readable plot labels |

If `system_id` is absent, the loader assigns `system_id = 1`. This is suitable
only when the file contains one system. If `t` is absent, all six measured
composition columns are required so the PCA-based phase-path coordinate can be
constructed within each `(system_id, T)` group.

The present CLI builds a supervised dataset object whose target tensor contains
`Ex1`-`Rx3`. Therefore, a completely unlabeled Excel file is not the intended
input for this script. Use the Web API or add a dedicated inference-only dataset
adapter for unlabeled batch prediction.

## Component and phase conventions

The component order in SMILES, measured targets, predictions, and plot labels
must match:

```text
smiles1 -> Ex1 and Rx1
smiles2 -> Ex2 and Rx2
smiles3 -> Ex3 and Rx3
```

Do not reorder only the component names or only one phase. If components 2 and
3 are swapped, both phase-composition columns must be swapped as well.

## Supported and experimental paths

The distributed prediction table can be analyzed safely with `--analyze_only`,
as shown below. Published checkpoint reproduction must use
`scripts/evaluate_checkpoint_registry.py`; the registry binds the checkpoint to
its YAML profile, functional-group corpus, scaler provenance, and
component-major compatibility override.

The Excel-to-prediction branch in `scripts/run_application_case.py` is an
experimental validation adapter. It reads checkpoint metadata but does not yet
apply the embedded model configuration when `build_model()` is called, and it
does not accept a functional-group corpus on the command line. Consequently,
it must not be used with the published Figure 2a checkpoint or cited as a
manuscript-reproduction path. A tensor-shape-compatible load alone does not
establish scientific compatibility.

Only use the experimental branch with a checkpoint explicitly created under
the active process configuration and matching functional-group vocabulary.
Before reporting results, compare its predictions with a trusted reference
table for the same checkpoint. A future registry-aware application adapter
should resolve the profile, corpus, scalar dimension, scaler, and node layout
before model construction.

When those compatibility conditions are satisfied, the branch:

1. resolves supported column aliases;
2. canonicalizes the three SMILES;
3. constructs `t` when it is absent;
4. adds a 50-point phase-path grid within each observed `t` range;
5. loads the supplied checkpoint and temperature scaler;
6. evaluates pointwise predictions without weight updates;
7. writes a prediction CSV;
8. generates ternary phase diagrams and summary plots.

## Supported analysis-only workflow

Use a previously generated or distributed prediction table:

```bash
python scripts/run_application_case.py \
  --csv experiments/section_3_results/3_4_industrial_extraction_design/application_workflow/results/application_case_predictions.csv \
  --out_dir outputs/application_case_analysis \
  --analyze_only
```

Analysis-only mode does not load a checkpoint. The input CSV must contain the
fields required by the plotting and summary routines, including system,
temperature, component labels, compositions, and model labels where expected.

## Output interpretation

The main table is:

```text
<out_dir>/application_case_predictions.csv
```

Predicted columns use the `pred_` prefix:

```text
pred_Ex1, pred_Ex2, pred_Ex3,
pred_Rx1, pred_Rx2, pred_Rx3
```

Each phase should be non-negative and sum to approximately one. Compare
prediction and experiment at matching `system_id`, `T`, and `t`. Interpolated
grid rows are intended to smooth the visualized curve; they are not additional
experimental observations.

## Pressure-aware use

The distributed expanded-LLE model supports a third pressure scalar, but the
current application CLI is centered on the two-scalar main-benchmark workflow
and does not expose a complete pressure-aware input contract. Adding a pressure
column alone is insufficient to guarantee correct expanded-model inference.

Pressure-aware deployment should explicitly load the expanded checkpoint,
pressure scaler, `SCALAR_DIM: 3` configuration, and an inference adapter that
passes `P` through the dataset. Treat this as a separate protocol and validate
it against the expanded-LLE archived predictions.

## Quality checks

Before accepting an application result:

- confirm all SMILES parse successfully;
- verify component order against source data;
- confirm temperature units are kelvin;
- inspect the observed and generated `t` range;
- verify extract and raffinate closure;
- compare against experimental points rather than interpolated rows alone;
- record checkpoint hash and node-layout contract;
- avoid extrapolating scientific conclusions beyond the chemical and
  temperature domain represented by the training data.

## Common failures

### Missing composition columns

If `t` is absent, the phase-path constructor needs `Ex1`-`Rx3`. Supply measured
compositions or add a valid `t` column.

### Invalid SMILES

Invalid component strings are canonicalized to empty values and their rows are
removed. Correct the structures in the source workbook and rerun.

### Checkpoint shape mismatch or silent contract mismatch

The selected checkpoint may use a different architecture, layout, scalar
dimension, scaler, or functional-group vocabulary. A successful state-dict
load does not rule out a silent vocabulary or layout mismatch. Use the registry
evaluator for published checkpoints. Do not use the experimental application
adapter until all contracts are applied before model construction.

### Empty or misleading phase diagram

Check that rows share the intended `system_id` and temperature, that `t` spans
more than one value, and that component labels follow the same order as the
composition columns.

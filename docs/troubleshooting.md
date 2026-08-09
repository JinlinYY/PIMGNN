# Troubleshooting

## Import and environment problems

### `ModuleNotFoundError: psmi`

Run from the repository root and install the package in editable mode:

```bash
python -m pip install -e .
python -c "import psmi; print(psmi.__file__)"
```

The scripts also add the local `src/` directory automatically, but editable
installation is recommended for notebooks and external tools.

### RDKit or PyTorch binary import failure

Create a clean Conda environment from `environment.yml` or
`environment-cpu.yml`. Mixing pip CUDA wheels, Conda CUDA libraries, and an
older RDKit build is a common cause of binary incompatibility.

### CUDA is unavailable

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Check the NVIDIA driver and whether the installed PyTorch build includes CUDA.
Use `--device cpu` while diagnosing the GPU environment.

## Data and split problems

### Required Excel column cannot be found

Compare the workbook headers with the canonical schema in the
[data pipeline guide](guides/data_pipeline.md). The loader accepts common
aliases but does not guess arbitrary component or composition names.

### Rows disappear after loading

Likely causes are invalid SMILES, nonnumeric temperature/composition values, or
fewer than six rows in a `(system_id, T)` group. Run the dataset-distribution
analysis and compare raw versus filtered counts.

### Split manifest mismatch

The workbook and manifest do not describe the same filtered systems, or a
filtering setting changed. Verify workbook and manifest hashes, minimum-density
threshold, and SMILES parsing before generating a new split.

### Train/validation/test overlap error

One or more system identifiers occur in multiple manifest partitions. Fix the
manifest; do not bypass the disjointness check.

## Checkpoint problems

### State-dict size or key mismatch

Confirm model layout, architecture switches, scalar dimension, and
functional-group vocabulary. Published weights require the component-major
registry. Maintained training uses sample-major layout.

### Reference prediction or input hash mismatch

First verify that the distributed files are unmodified. Compatibility flags in
the manuscript registry exist for named historical artifacts and should not be
used to suppress an unexplained mismatch in a new checkpoint.

### Missing scaler metadata

Use a registry entry that explicitly allows a documented scaler fallback, or
evaluate a maintained checkpoint with embedded scalers. A scaler fitted on the
test partition invalidates a strict evaluation.

### Missing functional-group corpus

Load the corpus recorded by the registry or result package. Rebuilding a
vocabulary can change token ids and invalidate the checkpoint even when tensor
dimensions match.

## Thermodynamic problems

### NRTL parameter file not found

The public main parameter files are under:

```text
datasets/parameters/main_benchmark/
```

Check the effective `NRTL_TRAIN_PARAMS_PATH` and `NRTL_EVAL_PARAMS_PATH`. Training
and all-system diagnostic stores are intentionally separate.

### Physics metrics are `NaN`

The evaluation may lack an NRTL store, parameter coverage may be zero, or the
selected profile may disable final physics metrics. Inspect `param_cov` before
interpreting residual values.

### Thermodynamic residual differs from an archived value

Confirm predicted compositions, temperature units, GE model, parameter file,
component permutation, diagnostic settings, and coverage. Physics diagnostics
are not determined by the neural checkpoint alone.

## Training problems

### Stage 2 cannot find the stage-1 checkpoint

The public stage-2 YAML expects the default output of stage 1. Run stage 1 first
or override `LOAD_CKPT_PATH` with a compatible sample-major checkpoint:

```bash
python scripts/train.py \
  --config configs/experiments/main_benchmark_stage2.yaml \
  --set LOAD_CKPT_PATH=outputs/stage1/best_model.pt \
  --set OUT_DIR=outputs/stage2
```

### CUDA out of memory

Reduce `BATCH_SIZE_GRAPH`, disable expensive plotting during training, or select
a smaller GPU workload. Record the override because batch size can affect
optimization behavior.

### Test metrics appear during training

The public profiles set `EVALUATE_TEST_DURING_TRAINING: false`. Check CLI
overrides and custom YAML files. Test-set access during checkpoint selection
changes the protocol.

## Result and figure problems

### Reproduced aggregate metric differs slightly

Check package versions and device first. Then compare pointwise predictions and
hashes. Small numerical differences can arise from hardware and library
versions; large differences usually indicate data, split, scaler, layout, or
checkpoint drift.

### Figure differs but numerical data match

Inspect plotting script version, font availability, image size, DPI, axis
limits, and row ordering. Treat a layout-only difference separately from a
prediction difference.

### Artifact manifest test fails

A packaged file was added, removed, or modified without regenerating the
manifest. Determine whether the change is intentional, regenerate from trusted
source artifacts, and document the new provenance.

## Windows-specific problems

### Conda raises a GBK or Unicode error

Use UTF-8 mode for the current shell and run the test command through the named
Conda environment:

```powershell
$env:PYTHONUTF8 = "1"
conda run -n ggnn39 python -m pytest -q
```

### Pytest cannot access its default temporary directory

Choose a writable base directory:

```bash
python -m pytest -q --basetemp outputs/pytest-temp
```

Use a new or disposable output path because pytest manages the selected base
directory.

### Non-ASCII path causes a third-party failure

PSMI paths are resolved with `pathlib`, but some compiled dependencies and
external tools remain sensitive to non-ASCII paths. Move the checkout to a
short English-only path if a failure occurs below Python-level path handling.

## Web application problems

### Frontend opens but predictions fail

Verify that the backend is running, inspect `http://localhost:8000/docs`, check
the browser network request, and confirm the frontend API base URL.

### Backend fails while loading the default checkpoint

Check `Web/PSMI-LLE-web/backend/config.py`, checkpoint existence, model metadata,
and Python environment. Run:

```bash
python -m pytest -q tests/test_web_checkpoint_contract.py Web/PSMI-LLE-web/tests
```

### Port already in use

Select another backend or frontend port and update the frontend API/CORS
configuration consistently. Detailed variables are listed in the Web README.

## What to include in a bug report

Provide:

- Git commit hash;
- operating system and environment file;
- Python and key package versions;
- exact command;
- complete traceback;
- checkpoint and dataset paths plus hashes;
- whether the failure occurs on CPU and GPU;
- smallest input or registry entry that reproduces the issue.

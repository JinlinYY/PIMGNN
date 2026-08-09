# Installation

## Supported environment

The reference GPU environment is `ggnn39`, based on Python 3.9, PyTorch 2.6,
CUDA 12.6, RDKit, and the scientific Python stack. A CPU-only profile is also distributed for
checkpoint inspection, deterministic evaluation, plotting, and most analyses.

The repository includes datasets, fixed split manifests, published
checkpoints, and archived results. A normal checkout therefore does not need to
download model weights from a separate service.

## GPU installation with Conda

From the repository root:

```bash
conda env create -f environment.yml
conda activate ggnn39
python -m pip install -e .
```

Verify the core runtime:

```bash
python -c "import torch, rdkit, pandas; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import psmi; print(psmi.__file__)"
```

`torch.cuda.is_available()` should be `True` when the NVIDIA driver and the
CUDA-enabled PyTorch build are available. PSMI can still be evaluated on CPU
by passing `--device cpu`.

## CPU-only installation

```bash
conda env create -f environment-cpu.yml
conda activate psmi-cpu
python -m pip install -e .
```

Use the CPU environment for repository inspection, data analysis, result
export, tests, and small checkpoint evaluations. Full model training and large
explainability jobs are considerably faster on a CUDA-capable GPU.

## Install into an existing Python 3.9 environment

```bash
python -m pip install -e .
```

Optional dependency groups are available through `pyproject.toml`:

```bash
python -m pip install -e ".[baselines,interactive,dev]"
```

Conda is preferred for a fresh GPU installation because RDKit, PyTorch, CUDA,
and compiled graph dependencies must be binary-compatible. The pinned Conda
profiles are the tested starting point; `requirements.txt` is primarily a
readable package inventory and pip fallback.

Several imported graph-baseline implementations additionally use
`torch_geometric`. It is not pinned in the core environment profiles because
its compiled extensions must match the local PyTorch and CUDA build. Install a
compatible PyTorch Geometric distribution before running CGIB, CIGNN, GLAM, or
other baseline entry points that import it, then verify the baseline's imports
before starting a training job.

The complete ten-model classical comparison also requires XGBoost and
PyTorch-TabNet, which are not part of the core PSMI runtime:

```bash
python -m pip install xgboost pytorch-tabnet
```

Install these packages only after the core environment is working. Record their
versions together with the PyTorch Geometric build when reporting a rerun. The
[baseline comparison guide](../guides/baseline_comparison.md) separates a
core-compatible subset from the complete optional-dependency run.

## Validate the installation

Run the repository tests:

```bash
python -m pytest -q
```

List the distributed manuscript checkpoints without loading model tensors by
following the [canonical quick-start command](quickstart.md#2-inspect-available-manuscript-checkpoints).

Expected registry identifiers are `figure2a_psmi`, `table3_data_driven`, and
`table3_physics_informed`.

## Web application dependencies

The Web application uses a separate FastAPI backend and Vue/Vite frontend.
Install the backend dependencies after activating a PSMI Python environment:

```bash
python -m pip install -r Web/PSMI-LLE-web/requirements.txt
```

Install the frontend with a current Node.js LTS release:

```bash
cd Web/PSMI-LLE-web/frontend
npm install
```

Return to the repository root before running research scripts. Complete Web
startup instructions are in the [Web application guide](../guides/web_application.md).

## Platform notes

### Windows

- PowerShell launch scripts are available under `Web/PSMI-LLE-web/scripts/`.
- If Conda output fails because of the system code page, run the environment's
  `python.exe` directly or set `PYTHONUTF8=1` for the current shell.
- Keep the repository in a path with enough free space for generated figures,
  evaluation tables, and optional training checkpoints.

### Linux and macOS

- Use forward-slash paths as shown in the documentation.
- The CPU profile is portable. The GPU profile requires an NVIDIA-supported
  Linux installation; macOS uses CPU or the locally supported PyTorch backend.
- The Windows `.ps1` launchers are conveniences only. The documented Python,
  Uvicorn, and npm commands are platform-independent.

## Reproducible environment reporting

When reporting a reproduced result, record at least:

- Git commit hash;
- Python, PyTorch, RDKit, NumPy, and pandas versions;
- operating system and compute device;
- checkpoint registry identifier and SHA-256 digest;
- dataset and split-manifest digests;
- command-line overrides supplied through `--set`.

These details distinguish numerical variation from changes in data, split,
architecture, or checkpoint provenance.

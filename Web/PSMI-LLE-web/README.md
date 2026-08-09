# PSMI Ternary LLE Web Application

The PSMI Web application combines a FastAPI inference service with a Vue 3 interface for ternary liquid-liquid-equilibrium prediction. It accepts three component SMILES strings, temperature, optional pressure, and a requested number of displayed tie lines. The response contains extract- and raffinate-phase compositions, molecular descriptors, plot-ready tie-line data, and a ternary phase diagram.

## Application layout

```text
Web/PSMI-LLE-web/
|- backend/                 FastAPI routes, schemas, checkpoint adapter, and plotting
|- frontend/                Vue 3 and Vite interface
|- checkpoints/default/     Default checkpoint, scaler metadata, and FG vocabulary
|- assets/explainability/   Precomputed feature-attribution summaries
|- scripts/                 PowerShell launch scripts
|- tests/                   Backend and checkpoint-contract smoke tests
`- requirements.txt         Web-specific Python dependencies
```

The backend imports the maintained PSMI implementation from the repository-level `src/psmi/` package. Run all commands from the repository root unless a step explicitly changes directory.

## Local deployment

### 1. Prerequisites

- Windows 10/11 with PowerShell 5.1 or PowerShell 7 is the primary documented platform.
- Miniconda or Anaconda is required for the reproducible `ggnn39` environment.
- Node.js must satisfy `^20.19.0` or `>=22.12.0`; Node.js 22 LTS is recommended.
- At least 4 GB of free memory is recommended for CPU inference. An NVIDIA GPU is optional.

Verify the command-line tools:

```powershell
conda --version
node --version
npm --version
```

### 2. Obtain the repository

```powershell
$repositoryUrl = "https://github.com/JinlinYY/PSMI.git"
git clone $repositoryUrl PSMI
cd PSMI
```

The command checks out the repository into a local directory named `PSMI`.

### 3. Create the Python environment

Create the environment once:

```powershell
conda env create -f environment.yml
conda activate ggnn39
python -m pip install -e .
python -m pip install -r Web/PSMI-LLE-web/requirements.txt
```

For an existing environment, synchronize it before installing the Web dependencies:

```powershell
conda env update -n ggnn39 -f environment.yml --prune
conda activate ggnn39
python -m pip install -e .
python -m pip install -r Web/PSMI-LLE-web/requirements.txt
```

Confirm the core imports:

```powershell
python -c "import fastapi, torch, rdkit; print(f'FastAPI {fastapi.__version__}'); print(f'PyTorch {torch.__version__}'); print(f'RDKit {rdkit.__version__}')"
```

### 4. Verify the bundled model artifacts

The bundled default directory should contain the three files below. `best_model.pt` and `fg_corpus.json` are required; `last_model.pt` provides fallback scaler metadata for compatible published checkpoints.

```powershell
Test-Path Web/PSMI-LLE-web/checkpoints/default/best_model.pt
Test-Path Web/PSMI-LLE-web/checkpoints/default/last_model.pt
Test-Path Web/PSMI-LLE-web/checkpoints/default/fg_corpus.json
```

Each command should return `True`. The default checkpoint uses the audited two-scalar `[T, s]` contract. It accepts the API pressure field for schema compatibility but returns `pressure_used: false` because pressure is not an input to this checkpoint.

### 5. Install the frontend dependencies

```powershell
cd Web/PSMI-LLE-web/frontend
npm ci
cd ../../..
```

`npm ci` installs the exact versions recorded in `package-lock.json`. Re-run it after pulling a change to the lockfile.

### 6. Start the backend

Open the first PowerShell terminal at the repository root:

```powershell
conda activate ggnn39
$env:PSMI_WEB_HOST = "127.0.0.1"
$env:PSMI_WEB_PORT = "8000"
$env:PSMI_WEB_DEVICE = "cpu"
Web/PSMI-LLE-web/scripts/run_backend.ps1 -PythonExecutable python
```

The process loads the checkpoint once during startup. Keep this terminal open. A successful start reports a Uvicorn service listening on `http://127.0.0.1:8000`.

If PowerShell script execution is disabled, use:

```powershell
powershell -ExecutionPolicy Bypass -File Web/PSMI-LLE-web/scripts/run_backend.ps1
```

### 7. Check the backend

Open a second PowerShell terminal and request the health endpoint:

```powershell
Invoke-RestMethod http://localhost:8000/health | Format-List
```

The response should include `status: healthy`, `rdkit: True`, and `model_loaded: True`. Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

Run a prediction directly from PowerShell:

```powershell
$body = @{
    smiles1 = "O"
    smiles2 = "CCO"
    smiles3 = "CCCCCC"
    name1 = "Water"
    name2 = "Ethanol"
    name3 = "Hexane"
    temperature = 298.15
    pressure = 101.325
    tie_lines_count = 14
} | ConvertTo-Json

$response = Invoke-RestMethod `
    -Uri http://localhost:8000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

$response.success
$response.data.e_phase
$response.data.r_phase
```

### 8. Start the frontend

Open a third PowerShell terminal at the repository root:

```powershell
Web/PSMI-LLE-web/scripts/run_frontend.ps1
```

The Vite development server listens on `http://localhost:3000`. Its `/api` proxy forwards requests to `http://localhost:8000`, so `VITE_API_BASE` should normally remain unset for the standard two-process deployment.

Open `http://localhost:3000` in a browser. Enter three valid SMILES strings and a temperature, then submit the form. The first request can be slower because RDKit molecular graphs and descriptors are cached on demand.

The development server binds to `127.0.0.1` by default. To test deliberately from another device on a trusted network, run `npm run dev -- --host 0.0.0.0` from `Web/PSMI-LLE-web/frontend` and configure the backend CORS allowlist for the chosen frontend origin. Do not expose the development server directly to an untrusted network.

## GPU inference

Verify CUDA availability inside `ggnn39`:

```powershell
conda activate ggnn39
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If CUDA is available, start the backend with:

```powershell
$env:PSMI_WEB_DEVICE = "cuda"
Web/PSMI-LLE-web/scripts/run_backend.ps1 -PythonExecutable python
```

Use `cpu` when the installed PyTorch build has no compatible CUDA runtime.

## Runtime configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PSMI_WEB_DEVICE` | `cpu` | PyTorch device, normally `cpu` or `cuda`. |
| `PSMI_WEB_MODEL_DIR` | `checkpoints/default` | Directory containing the checkpoint and `fg_corpus.json`. |
| `PSMI_WEB_MODEL_PATH` | `<MODEL_DIR>/best_model.pt` | Exact checkpoint loaded by the backend. |
| `PSMI_WEB_EXPLAIN_DIR` | bundled explainability directory | Directory containing attribution summaries. |
| `PSMI_WEB_HOST` | `0.0.0.0` | Backend bind address; use `127.0.0.1` for local-only access. |
| `PSMI_WEB_PORT` | `8000` | Backend TCP port. |
| `VITE_API_BASE` | `/api` | Frontend API base; `/api` uses the Vite development proxy. |

Set variables in the same terminal before starting the corresponding process. For example:

```powershell
$env:PSMI_WEB_DEVICE = "cpu"
$modelDirectory = (Resolve-Path "models/custom_web_checkpoint").Path
$env:PSMI_WEB_MODEL_DIR = $modelDirectory
$env:PSMI_WEB_MODEL_PATH = Join-Path $modelDirectory "best_model.pt"
$env:PSMI_WEB_EXPLAIN_DIR = Join-Path $modelDirectory "explainability"
Web/PSMI-LLE-web/scripts/run_backend.ps1 -PythonExecutable python
```

A custom checkpoint directory must include the matching `fg_corpus.json`. A three-scalar checkpoint must also contain pressure-scaler metadata (`P_mean` and `P_std`) and architecture provenance identifying `[T, s, P]`.

## macOS and Linux

The PowerShell wrappers are optional. From the repository root, start the backend with:

```bash
conda activate ggnn39
export PYTHONPATH="$PWD/src:$PWD"
export PSMI_WEB_HOST="127.0.0.1"
export PSMI_WEB_PORT="8000"
export PSMI_WEB_DEVICE="cpu"
cd Web/PSMI-LLE-web
python -m backend.main
```

Start the frontend in another terminal:

```bash
cd Web/PSMI-LLE-web/frontend
npm ci
npm run dev
```

## Validation

Backend and checkpoint-contract tests:

```powershell
conda activate ggnn39
$env:PYTHONPATH = "src;Web/PSMI-LLE-web"
python -m pytest tests/test_web_checkpoint_contract.py Web/PSMI-LLE-web/tests -q
```

Frontend production build:

```powershell
cd Web/PSMI-LLE-web/frontend
npm ci
npm run build
```

The build output is written to `Web/PSMI-LLE-web/frontend/dist/` and is intentionally excluded from version control.

## Troubleshooting

### `node` or `npm` is not recognized

Install Node.js 22 LTS, open a new terminal, and verify `node --version` and `npm --version`. An older Node release can fail because the bundled Vite version enforces the engine range in `frontend/package.json`.

### PowerShell reports that script execution is disabled

Use `powershell -ExecutionPolicy Bypass -File <script>` for the current launch, or run the underlying commands manually. A system-wide execution-policy change is not required.

### `ModuleNotFoundError: psmi` or `ModuleNotFoundError: backend`

Run the provided scripts from the repository root. The backend wrapper sets `PYTHONPATH` to the repository `src/` directory and changes into the Web application directory before launching Python.

### The backend reports a missing checkpoint or functional-group corpus

Confirm that `best_model.pt`, `last_model.pt`, and `fg_corpus.json` exist together under `Web/PSMI-LLE-web/checkpoints/default/`. For a custom model, set both `PSMI_WEB_MODEL_DIR` and `PSMI_WEB_MODEL_PATH` to consistent locations.

### CUDA initialization fails

Set `$env:PSMI_WEB_DEVICE = "cpu"` and restart the backend. GPU execution requires a CUDA-enabled PyTorch build compatible with the installed NVIDIA driver.

### Port 8000 or 3000 is already in use

Choose a different backend port, for example `$env:PSMI_WEB_PORT = "8010"`. Then start the frontend with `$env:VITE_API_BASE = "http://127.0.0.1:8010"; Web/PSMI-LLE-web/scripts/run_frontend.ps1`. When the frontend origin differs from `localhost:3000` or `127.0.0.1:3000`, add that origin to the FastAPI CORS allowlist in `backend/main.py`.

### The page opens but predictions fail

Check `http://localhost:8000/health` first, then inspect the backend terminal for the complete exception. Confirm that all three SMILES strings are valid and that temperature and pressure are positive numbers.

### Pressure appears in the form but `pressure_used` is false

This is expected for the bundled published Web checkpoint because its scalar input is `[T, s]`. Load a compatible three-scalar `[T, s, P]` checkpoint with pressure scaler metadata to enable pressure-conditioned inference.

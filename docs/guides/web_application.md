# Web Application

## Purpose

`Web/PSMI-LLE-web/` provides a FastAPI inference backend and a Vue/Vite user
interface for entering three molecular components, selecting operating
conditions, running checkpoint-backed PSMI inference, and visualizing ternary
phase compositions.

The complete platform-specific deployment guide is maintained in
[Web/PSMI-LLE-web/README.md](../../Web/PSMI-LLE-web/README.md). This page explains
how the Web application relates to the research code and artifacts.

## Components

```text
Web/PSMI-LLE-web/
|- backend/                 FastAPI routes, schemas, and predictor adapter
|- frontend/                Vue components and API client
|- checkpoints/default/     Bundled default model artifacts
|- assets/explainability/   Default attribution summary
|- scripts/                 Windows launch helpers
`- tests/                   Backend contract tests
```

The frontend does not run the neural network. It submits validated input to the
backend, which loads the configured checkpoint, constructs molecular and
mixture features, performs inference, and returns plot-ready data.

## Local startup

Install the Python backend dependencies in an activated PSMI environment:

```bash
python -m pip install -r Web/PSMI-LLE-web/requirements.txt
```

Install frontend dependencies:

```bash
cd Web/PSMI-LLE-web/frontend
npm install
```

Start the backend and frontend in separate terminals using the commands in the
Web README. On Windows, the convenience launchers are:

```powershell
Web\PSMI-LLE-web\scripts\run_backend.ps1
Web\PSMI-LLE-web\scripts\run_frontend.ps1
```

Default addresses are:

- user interface: `http://localhost:3000`;
- backend API documentation: `http://localhost:8000/docs`.

## Checkpoint contract

The default checkpoint bundle is under
`Web/PSMI-LLE-web/checkpoints/default/`. Backend configuration in
`Web/PSMI-LLE-web/backend/config.py` resolves the model path and related
artifacts.

When replacing the default checkpoint, verify:

1. node layout;
2. scalar dimension;
3. graph and functional-group settings;
4. checkpoint scaler metadata;
5. state-dict compatibility;
6. input temperature and pressure conventions;
7. output order `[Ex1, Ex2, Ex3, Rx1, Rx2, Rx3]`.

A `.pt` file loading without an exception is not sufficient evidence of
scientific compatibility.

## API verification

After the backend starts, open `/docs` and inspect the live request schema. Use
the health or metadata endpoints described by the Web README before testing
predictions. Confirm that the reported model name is PSMI and that the resolved
checkpoint is the intended public artifact.

## Research-use checks

- Preserve the component order between the request and returned compositions.
- Treat generated phase-path samples as model predictions, not measurements.
- Record the checkpoint digest for reported Web results.
- Validate unfamiliar chemical systems against experimental or independent
  thermodynamic evidence.
- Do not infer pressure sensitivity from a checkpoint that uses only
  temperature and phase-path scalars.

## Development checks

Run the repository-level Web tests:

```bash
python -m pytest -q tests/test_public_web_deployment.py tests/test_web_checkpoint_contract.py
```

Run the Web backend tests from the repository root:

```bash
python -m pytest -q Web/PSMI-LLE-web/tests
```

Build the frontend before distributing a change:

```bash
cd Web/PSMI-LLE-web/frontend
npm run build
```

See the Web README for environment variables, CORS settings, port changes,
production hosting, and detailed troubleshooting.

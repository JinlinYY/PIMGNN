# PSMI Ternary LLE Web Application

The web application combines a Vue 3 frontend with a FastAPI backend. Users provide three component SMILES strings, temperature, optional pressure, and curve settings to generate ternary LLE predictions.

## Layout

- `backend/`: FastAPI routes, schemas, checkpoint adapter, and plot generation.
- `frontend/`: Vue 3 and Vite interface.
- `checkpoints/default/`: default checkpoint and functional-group vocabulary.
- `assets/explainability/`: precomputed attribution summaries.
- `scripts/`: Windows launch scripts.
- `tests/`: backend interface tests.

## Model contract

The backend restores scalar dimension, fusion mode, mixture-node layout, and feature switches from checkpoint provenance. The default historical checkpoint uses `[T, s]`. Checkpoints with the corrected three-scalar contract use `[T, s, P]` and enable pressure normalization.

## Launch

```powershell
Web/PSMI-LLE-web/scripts/run_backend.ps1
Web/PSMI-LLE-web/scripts/run_frontend.ps1
```

Open `http://localhost:3000`; the API schema is available at `http://localhost:8000/docs`.

Runtime variables include `PSMI_WEB_DEVICE`, `PSMI_WEB_MODEL_PATH`, `PSMI_WEB_MODEL_DIR`, `PSMI_WEB_EXPLAIN_DIR`, `PSMI_WEB_HOST`, `PSMI_WEB_PORT`, and `VITE_API_BASE`.

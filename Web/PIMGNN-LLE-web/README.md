# PIMGNN-LLE — Ternary Liquid–Liquid Equilibrium Prediction

Physics-informed molecular graph neural network (PIMGNN) web application for predicting **ternary liquid–liquid equilibrium (LLE)** from SMILES strings and generating ternary phase diagrams.

## Features

- Vue 3 + Element Plus frontend for ternary system input and visualization
- FastAPI backend with pretrained PIMGNN checkpoint inference
- Matplotlib-rendered ternary phase diagrams (binodal curves + tie-lines)
- RDKit-based SMILES validation and 2D structure preview
- Precomputed explainability summaries (global / mixture / functional-group saliency)

## Repository layout

```
.
├── src/                 # Core PIMGNN training & model code
├── web_backend/         # FastAPI inference service
├── web_frontend/        # Vue 3 web UI
├── checkpoints/default/ # Default model weights (best_model.pt, scalers, etc.)
├── assets/explainability/
├── nrtl_param/          # NRTL parameter utilities (optional)
└── requirements.txt
```

## Requirements

- Python 3.9+
- Node.js 18+ (frontend development)
- CPU inference supported; CUDA optional for faster inference

## Quick start

### 1. Backend

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
cd web_backend
python main.py
```

Backend default URL: `http://127.0.0.1:8000`

### 2. Frontend (development)

```bash
cd web_frontend
npm install
npm run dev
```

Frontend default URL: `http://127.0.0.1:3000`

### 3. Production build (optional)

```bash
cd web_frontend
npm run build
```

Serve `web_frontend/dist` with any static file server and proxy `/api` to the backend.

## API overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service & model status |
| `/validate-smiles` | POST | SMILES validation + RDKit SVG |
| `/predict` | POST | LLE prediction + ternary plot (base64 PNG) |
| `/explain/summary` | GET | Precomputed feature-importance summary |

## Model checkpoint

The default inference bundle is in `checkpoints/default/`:

| File | Purpose |
|------|---------|
| `best_model.pt` | Model weights (`model` state dict) |
| `last_model.pt` | Temperature scaler (`T_mean`, `T_std`) — required at runtime |
| `fg_corpus.json` | Functional-group corpus (optional; used when FG mode is enabled) |

To use another checkpoint, update `MODEL_DIR` and `MODEL_PATH` in `web_backend/config.py`.

## Training (optional)

See `src/README.md` for the original training pipeline. Training data is **not** included in this repository due to size/licensing.

## Citation

If you use this software in academic work, please cite the corresponding PIMGNN / LLE paper and acknowledge this repository.

## License

MIT License — see [LICENSE](LICENSE).

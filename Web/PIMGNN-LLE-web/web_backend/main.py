# web_backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import importlib.util
import os
import traceback

import config as C
from schemas.request import PredictRequest
from schemas.response import PredictResponse
from schemas.chem import (
    SmilesValidationRequest,
    SmilesValidationResponse,
    HealthResponse,
    ExplainabilitySummary,
    MolecularDescriptor,
    FeatureImportance,
)
from models.predictor import predictor


def _load_local_module(module_name: str, file_name: str):
    file_path = os.path.join(os.path.dirname(__file__), "utils", file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"Failed to load module: {file_path}")
    loader.exec_module(module)
    return module


chem_utils = _load_local_module("web_backend_chem", "chem.py")
explain_utils = _load_local_module("web_backend_explainability", "explainability.py")


def _load_smiles_utils():
    utils_path = os.path.join(os.path.dirname(__file__), "utils", "smiles_utils.py")
    spec = importlib.util.spec_from_file_location("web_backend_smiles_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"Failed to load smiles_utils from {utils_path}")
    loader.exec_module(module)
    return module


def _load_plot_generator():
    plot_path = os.path.join(os.path.dirname(__file__), "utils", "plot_generator.py")
    spec = importlib.util.spec_from_file_location("web_backend_plot_generator", plot_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"Failed to load plot_generator from {plot_path}")
    loader.exec_module(module)
    return module


SMILES_UTILS = _load_smiles_utils()
PLOT_GENERATOR = _load_plot_generator()

app = FastAPI(title="LLE Prediction API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_components(smiles_list, names):
    components = []
    for index, smiles in enumerate(smiles_list, start=1):
        label = names[index - 1] if names and names[index - 1] else f"Component {index}"
        info = chem_utils.validate_smiles(smiles)
        components.append({
            "index": index,
            "label": label,
            "input_smiles": smiles,
            "canonical_smiles": info["canonical_smiles"],
            "formula": info["formula"],
            "svg": info["svg"],
            "descriptors": info["descriptors"],
        })
    return components


def _build_tie_lines(smiles_list, temperature, tie_lines_count):
    config_module, utils_module, data_module = PLOT_GENERATOR._load_project_modules()
    s1 = utils_module.canonicalize_smiles(smiles_list[0])
    s2 = utils_module.canonicalize_smiles(smiles_list[1])
    s3 = utils_module.canonicalize_smiles(smiles_list[2])
    n_sweep = int(getattr(config_module, "N_SWEEP", 80))
    t_grid, e_pred, r_pred = PLOT_GENERATOR.predict_curve_sweep(
        predictor.model,
        predictor.scaler,
        s1,
        s2,
        s3,
        float(temperature),
        n_sweep=n_sweep,
        config_module=config_module,
        utils_module=utils_module,
        data_module=data_module,
    )
    draw_max = max(1, min(int(tie_lines_count), len(t_grid)))
    idxs = np.linspace(0, len(t_grid) - 1, draw_max, dtype=int)
    tie_lines = []
    for idx in idxs:
        e_row = utils_module.renorm3(e_pred[idx])
        r_row = utils_module.renorm3(r_pred[idx])
        tie_lines.append({
            "t": float(t_grid[idx]),
            "extract": {
                "component1": float(e_row[0]),
                "component2": float(e_row[1]),
                "component3": float(e_row[2]),
            },
            "raffinate": {
                "component1": float(r_row[0]),
                "component2": float(r_row[1]),
                "component3": float(r_row[2]),
            },
        })
    return tie_lines


@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_loaded = predictor.model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        rdkit=True,
        model_loaded=model_loaded,
        model_path=getattr(C, "MODEL_PATH", None),
        device=str(getattr(predictor, "device", "cpu")),
    )


@app.post("/validate-smiles", response_model=SmilesValidationResponse)
async def validate_smiles_endpoint(request: SmilesValidationRequest):
    info = chem_utils.validate_smiles(request.smiles)
    return SmilesValidationResponse(
        input_smiles=info["input_smiles"],
        canonical_smiles=info["canonical_smiles"],
        formula=info["formula"],
        svg=info["svg"],
        descriptors=[MolecularDescriptor(**item) for item in info["descriptors"]],
    )


@app.get("/explain/summary", response_model=ExplainabilitySummary)
async def explain_summary():
    summary = explain_utils.explainability_summary()
    return ExplainabilitySummary(
        source=summary["source"],
        mechanism_notes=summary["mechanism_notes"],
        global_features=[FeatureImportance(**item) for item in summary["global_features"]],
        mixture_features=[FeatureImportance(**item) for item in summary["mixture_features"]],
        atom_features={
            key: [FeatureImportance(**item) for item in values]
            for key, values in summary["atom_features"].items()
        },
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_lle(request: PredictRequest):
    """Predict LLE and return plot plus structured metadata."""
    try:
        smiles_list = [request.smiles1, request.smiles2, request.smiles3]
        names = [request.name1, request.name2, request.name3]
        for i, smiles in enumerate(smiles_list, 1):
            if not SMILES_UTILS.validate_smiles(smiles):
                raise HTTPException(status_code=400, detail=f"Invalid SMILES for component {i}: {smiles}")

        e_compositions, r_compositions = predictor.predict_from_smiles(
            smiles_list,
            request.temperature,
        )

        plot_base64 = PLOT_GENERATOR.generate_ternary_plot(
            predictor.model,
            predictor.scaler,
            smiles_list,
            request.temperature,
            e_compositions,
            r_compositions,
            tie_lines_count=int(getattr(request, "tie_lines_count", 14)),
        )

        components = _build_components(smiles_list, names)
        tie_lines = _build_tie_lines(
            smiles_list,
            request.temperature,
            int(getattr(request, "tie_lines_count", 14)),
        )

        data = {
            "e_phase": {
                "component1": e_compositions[0],
                "component2": e_compositions[1],
                "component3": e_compositions[2],
            },
            "r_phase": {
                "component1": r_compositions[0],
                "component2": r_compositions[1],
                "component3": r_compositions[2],
            },
            "temperature": request.temperature,
            "smiles": smiles_list,
            "names": [
                names[i] or f"Component {i + 1}"
                for i in range(3)
            ],
            "tie_lines_count": int(getattr(request, "tie_lines_count", 14)),
            "components": components,
            "tie_lines": tie_lines,
        }

        return PredictResponse(
            success=True,
            message="Prediction successful",
            data=data,
            plot_base64=plot_base64,
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return PredictResponse(
            success=False,
            message=f"Prediction failed: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

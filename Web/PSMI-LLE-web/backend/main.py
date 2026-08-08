"""FastAPI entry point for the PSMI ternary LLE web application."""

from __future__ import annotations

import traceback
from typing import Sequence

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend import config
from backend.models import ModelPredictor
from backend.schemas.chem import (
    ExplainabilitySummary,
    FeatureImportance,
    HealthResponse,
    MolecularDescriptor,
    SmilesValidationRequest,
    SmilesValidationResponse,
)
from backend.schemas.request import PredictRequest
from backend.schemas.response import PredictResponse
from backend.utils import chem, explainability
from backend.utils.plot_generator import generate_ternary_plot


app = FastAPI(
    title="PSMI Ternary LLE Prediction API",
    description="Physics-informed ternary liquid-liquid equilibrium prediction.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at process startup so requests reuse molecular feature caches.
predictor = ModelPredictor()


def _component_records(smiles_list: Sequence[str], names: Sequence[str | None]) -> list[dict]:
    records = []
    for index, smiles in enumerate(smiles_list, start=1):
        label = names[index - 1] or f"Component {index}"
        info = chem.validate_smiles(smiles)
        records.append(
            {
                "index": index,
                "label": label,
                "input_smiles": smiles,
                "canonical_smiles": info["canonical_smiles"],
                "formula": info["formula"],
                "svg": info["svg"],
                "descriptors": info["descriptors"],
            }
        )
    return records


def _tie_line_records(t_grid: np.ndarray, extract: np.ndarray, raffinate: np.ndarray, count: int) -> list[dict]:
    indices = np.linspace(0, len(t_grid) - 1, max(1, min(count, len(t_grid))), dtype=int)
    records = []
    for index in indices:
        records.append(
            {
                "t": float(t_grid[index]),
                "extract": {f"component{i + 1}": float(value) for i, value in enumerate(extract[index])},
                "raffinate": {f"component{i + 1}": float(value) for i, value in enumerate(raffinate[index])},
            }
        )
    return records


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        rdkit=True,
        model_loaded=predictor.model is not None,
        model_path=str(predictor.model_path),
        device=str(predictor.device),
    )


@app.post("/validate-smiles", response_model=SmilesValidationResponse)
async def validate_smiles_endpoint(request: SmilesValidationRequest) -> SmilesValidationResponse:
    info = chem.validate_smiles(request.smiles)
    return SmilesValidationResponse(
        input_smiles=info["input_smiles"],
        canonical_smiles=info["canonical_smiles"],
        formula=info["formula"],
        svg=info["svg"],
        descriptors=[MolecularDescriptor(**item) for item in info["descriptors"]],
    )


@app.get("/explain/summary", response_model=ExplainabilitySummary)
async def explainability_summary() -> ExplainabilitySummary:
    summary = explainability.explainability_summary()
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
async def predict_lle(request: PredictRequest) -> PredictResponse:
    """Predict a complete LLE curve and return structured tie-line data."""
    try:
        smiles = [request.smiles1, request.smiles2, request.smiles3]
        names = [request.name1, request.name2, request.name3]
        for index, value in enumerate(smiles, start=1):
            if not chem.canonicalize_smiles(value):
                raise HTTPException(status_code=400, detail=f"Invalid SMILES for component {index}: {value}")

        t_grid, extract, raffinate = predictor.predict_curve(
            smiles,
            temperature=request.temperature,
            pressure=request.pressure,
        )
        center = int(np.argmin(np.abs(t_grid - 0.5)))
        display_names = [names[index] or f"Component {index + 1}" for index in range(3)]
        plot = generate_ternary_plot(
            t_grid,
            extract,
            raffinate,
            temperature=request.temperature,
            pressure=request.pressure if predictor.pressure_supported else None,
            labels=display_names,
            tie_lines_count=request.tie_lines_count,
        )
        data = {
            "e_phase": {f"component{i + 1}": float(value) for i, value in enumerate(extract[center])},
            "r_phase": {f"component{i + 1}": float(value) for i, value in enumerate(raffinate[center])},
            "temperature": request.temperature,
            "pressure": request.pressure if predictor.pressure_supported else None,
            "pressure_used": predictor.pressure_supported,
            "smiles": smiles,
            "names": display_names,
            "tie_lines_count": request.tie_lines_count,
            "components": _component_records(smiles, names),
            "tie_lines": _tie_line_records(t_grid, extract, raffinate, request.tie_lines_count),
        }
        return PredictResponse(success=True, message="Prediction successful", data=data, plot_base64=plot)
    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()
        return PredictResponse(success=False, message=f"Prediction failed: {error}")


if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)

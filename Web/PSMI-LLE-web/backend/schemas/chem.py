"""Chemistry and service-status schemas."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MolecularDescriptor(BaseModel):
    name: str
    value: float
    unit: Optional[str] = None


class SmilesValidationRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string to validate")


class SmilesValidationResponse(BaseModel):
    input_smiles: str
    canonical_smiles: str
    formula: str
    svg: str
    descriptors: List[MolecularDescriptor]


class HealthResponse(BaseModel):
    status: str
    rdkit: bool
    model_loaded: bool
    model_path: Optional[str] = None
    device: str = "cpu"


class FeatureImportance(BaseModel):
    name: str
    importance: float


class ExplainabilitySummary(BaseModel):
    source: str
    mechanism_notes: List[str]
    global_features: List[FeatureImportance]
    mixture_features: List[FeatureImportance]
    atom_features: Dict[str, List[FeatureImportance]]

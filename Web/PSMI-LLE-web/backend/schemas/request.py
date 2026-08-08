"""Request schemas for the PSMI prediction API."""

from typing import Optional

from pydantic import BaseModel, Field

from backend.config import DEFAULT_PRESSURE_KPA, DEFAULT_TIE_LINES


class PredictRequest(BaseModel):
    """Input conditions for a ternary LLE prediction."""

    smiles1: str = Field(..., min_length=1, description="SMILES for component 1")
    smiles2: str = Field(..., min_length=1, description="SMILES for component 2")
    smiles3: str = Field(..., min_length=1, description="SMILES for component 3")
    name1: Optional[str] = Field(None, description="Display name for component 1")
    name2: Optional[str] = Field(None, description="Display name for component 2")
    name3: Optional[str] = Field(None, description="Display name for component 3")
    temperature: float = Field(..., gt=0, description="Temperature in K")
    pressure: float = Field(DEFAULT_PRESSURE_KPA, gt=0, description="Pressure in kPa")
    tie_lines_count: int = Field(DEFAULT_TIE_LINES, ge=1, le=100)


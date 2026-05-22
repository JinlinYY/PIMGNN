# web_backend/schemas/request.py
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    smiles1: str = Field(..., description="SMILES for component 1")
    smiles2: str = Field(..., description="SMILES for component 2")
    smiles3: str = Field(..., description="SMILES for component 3")
    name1: Optional[str] = Field(None, description="Display name for component 1")
    name2: Optional[str] = Field(None, description="Display name for component 2")
    name3: Optional[str] = Field(None, description="Display name for component 3")
    temperature: float = Field(..., gt=0, description="Temperature in K")
    tie_lines_count: int = Field(14, ge=1, le=100, description="Number of tie-lines to draw")
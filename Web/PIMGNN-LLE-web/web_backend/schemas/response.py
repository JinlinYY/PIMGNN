# web_backend/schemas/response.py
from pydantic import BaseModel
from typing import List, Dict, Any

class PredictResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any] = None
    plot_base64: str = None
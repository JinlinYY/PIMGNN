"""Response schemas for the PSMI prediction API."""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class PredictResponse(BaseModel):
    """Prediction data and a base64-encoded ternary diagram."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    plot_base64: Optional[str] = None


"""Request/response models. Field ranges reject impossible inputs with a 422."""
from typing import List

from pydantic import BaseModel, Field

NOTE = "Decision-support only. Confirm with a local agronomist before acting."


class CropRequest(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Nitrogen content in soil (kg/ha)")
    P: float = Field(..., ge=0, le=200, description="Phosphorous content in soil (kg/ha)")
    K: float = Field(..., ge=0, le=300, description="Potassium content in soil (kg/ha)")
    temperature: float = Field(..., ge=-10, le=60, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall (mm)")

    model_config = {
        "json_schema_extra": {
            "example": {"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82, "ph": 6.5, "rainfall": 203}
        }
    }


class FertilizerRequest(BaseModel):
    temperature: float = Field(..., ge=-10, le=60, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    moisture: float = Field(..., ge=0, le=100, description="Soil moisture (%)")
    soil_type: str = Field(..., min_length=1, description="See GET /api/v1/options")
    crop_type: str = Field(..., min_length=1, description="See GET /api/v1/options")
    nitrogen: float = Field(..., ge=0, le=200)
    potassium: float = Field(..., ge=0, le=200)
    phosphorous: float = Field(..., ge=0, le=200)

    model_config = {
        "json_schema_extra": {
            "example": {
                "temperature": 26, "humidity": 52, "moisture": 38,
                "soil_type": "Loamy", "crop_type": "Maize",
                "nitrogen": 37, "potassium": 0, "phosphorous": 0,
            }
        }
    }


class Recommendation(BaseModel):
    name: str
    probability: float


class RecommendationResponse(BaseModel):
    recommendation: str
    top_k: List[Recommendation]
    model_version: str
    note: str = NOTE

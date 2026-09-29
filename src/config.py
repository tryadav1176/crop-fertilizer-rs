"""Central configuration. Paths can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "dataset"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))

CROP_CSV_NAME = "Crop_recommendation.csv"
FERT_CSV_NAME = "Fertilizer Prediction.csv"

# Crop model
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_TARGET = "label"

# Fertilizer model (raw CSV headers -> clean internal names)
FERT_RENAME = {
    "Temparature": "temperature",   # (sic) the dataset misspells this
    "Humidity": "humidity",         # raw header has a trailing space; stripped first
    "Moisture": "moisture",
    "Soil Type": "soil_type",
    "Crop Type": "crop_type",
    "Nitrogen": "nitrogen",
    "Potassium": "potassium",
    "Phosphorous": "phosphorous",
    "Fertilizer Name": "fertilizer",
}
FERT_NUMERIC = ["temperature", "humidity", "moisture", "nitrogen", "potassium", "phosphorous"]
FERT_CATEGORICAL = ["soil_type", "crop_type"]
FERT_TARGET = "fertilizer"

RANDOM_STATE = 42
TOP_K = 3

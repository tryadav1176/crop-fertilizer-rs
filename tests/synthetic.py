"""Tiny synthetic datasets with the same headers as the real Kaggle CSVs."""
from pathlib import Path

import numpy as np
import pandas as pd

SOILS = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
CROPS = ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat",
         "Millets", "Oil seeds", "Pulses", "Ground Nuts"]


def make_crop_csv(path: Path, per_class: int = 40) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i, label in enumerate(["rice", "maize", "cotton", "banana", "apple"]):
        centre = np.array([20 + 25 * i, 20 + 15 * i, 20 + 20 * i, 18 + 3 * i, 50 + 8 * i, 5 + 0.5 * i, 60 + 40 * i])
        data = centre + rng.normal(0, 2.0, size=(per_class, 7))
        for r in data:
            rows.append([*r, label])
    cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def make_fertilizer_csv(path: Path, per_class: int = 25) -> None:
    rng = np.random.default_rng(1)
    rows = []
    for i, fert in enumerate(["Urea", "DAP", "28-28", "14-35-14"]):
        for _ in range(per_class):
            rows.append([
                26 + 3 * i + rng.normal(0, 0.5),
                52 + 5 * i + rng.normal(0, 1),
                30 + 8 * i + rng.normal(0, 1),
                SOILS[rng.integers(len(SOILS))],
                CROPS[rng.integers(len(CROPS))],
                10 + 10 * i + rng.normal(0, 1),
                5 * i + rng.normal(0, 1),
                12 * i + rng.normal(0, 1),
                fert,
            ])
    # Mirror the real CSV quirks: 'Temparature' typo and 'Humidity ' trailing space.
    cols = ["Temparature", "Humidity ", "Moisture", "Soil Type", "Crop Type",
            "Nitrogen", "Potassium", "Phosphorous", "Fertilizer Name"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)

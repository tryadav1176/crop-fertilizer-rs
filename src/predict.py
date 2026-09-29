"""Load trained pipelines and produce top-k recommendations."""
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config

logger = logging.getLogger("predict")


class ModelNotFoundError(RuntimeError):
    pass


class Predictor:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / "metadata.json"
        if not meta_path.exists():
            raise ModelNotFoundError(
                f"No trained models in {self.model_dir}. Run: python -m src.train"
            )
        self.metadata: Dict[str, dict] = json.loads(meta_path.read_text())
        self.models = {}
        for key in ("crop", "fertilizer"):
            path = self.model_dir / f"{key}_pipeline.joblib"
            if key in self.metadata and path.exists():
                self.models[key] = joblib.load(path)  # only load files you trained yourself
                saved = self.metadata[key].get("sklearn_version")
                if saved != sklearn.__version__:
                    logger.warning(
                        "%s model trained with scikit-learn %s but %s is installed; retrain.",
                        key, saved, sklearn.__version__,
                    )
        if not self.models:
            raise ModelNotFoundError(f"No model files found in {self.model_dir}.")

    # -- helpers ---------------------------------------------------------
    def loaded(self) -> Dict[str, bool]:
        return {k: k in self.models for k in ("crop", "fertilizer")}

    def version(self, key: str) -> str:
        m = self.metadata[key]
        return f"{m['model_name']}@{m['trained_at']}"

    def options(self) -> Dict[str, List[str]]:
        return self.metadata.get("fertilizer", {}).get("categories", {})

    def _require(self, key: str):
        if key not in self.models:
            raise ModelNotFoundError(f"The {key} model is not loaded.")
        return self.models[key]

    @staticmethod
    def _top_k(pipe, frame: pd.DataFrame, k: int) -> List[Dict[str, object]]:
        proba = pipe.predict_proba(frame)[0]
        order = np.argsort(proba)[::-1][:k]
        return [{"name": str(pipe.classes_[i]), "probability": round(float(proba[i]), 4)} for i in order]

    def _canonical(self, field: str, value: str) -> str:
        allowed = self.options().get(field, [])
        lookup = {a.lower(): a for a in allowed}
        key = value.strip().lower()
        if key not in lookup:
            raise ValueError(f"Unknown {field} '{value}'. Allowed values: {allowed}")
        return lookup[key]

    # -- public API ------------------------------------------------------
    def predict_crop(self, features: Dict[str, float], top_k: int = config.TOP_K):
        pipe = self._require("crop")
        frame = pd.DataFrame([features], columns=config.CROP_FEATURES)
        return self._top_k(pipe, frame, top_k)

    def predict_fertilizer(self, features: Dict[str, object], top_k: int = config.TOP_K):
        pipe = self._require("fertilizer")
        row = dict(features)
        for field in config.FERT_CATEGORICAL:
            row[field] = self._canonical(field, str(row[field]))
        frame = pd.DataFrame([row], columns=config.FERT_NUMERIC + config.FERT_CATEGORICAL)
        return self._top_k(pipe, frame, top_k)

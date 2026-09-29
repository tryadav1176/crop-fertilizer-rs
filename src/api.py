"""FastAPI service.  Run:  uvicorn src.api:app --reload"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.predict import ModelNotFoundError, Predictor
from src.schemas import CropRequest, FertilizerRequest, RecommendationResponse

logger = logging.getLogger("api")
STATIC_DIR = config.ROOT / "static"


def create_app(model_dir: Optional[Path] = None) -> FastAPI:
    model_dir = Path(model_dir) if model_dir else config.MODEL_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = Predictor(model_dir)
            logger.info("Models loaded: %s", app.state.predictor.loaded())
        except ModelNotFoundError as exc:
            logger.error("%s", exc)
            app.state.predictor = None
        yield

    app = FastAPI(
        title="Crop & Fertilizer Recommendation API",
        version="1.0.0",
        description="Top-k crop and fertilizer recommendations from soil and weather inputs.",
        lifespan=lifespan,
    )

    def get_predictor(request: Request) -> Predictor:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(503, "Models not loaded. Run `python -m src.train` first.")
        return predictor

    @app.get("/health", tags=["ops"])
    def health(request: Request):
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(503, "Models not loaded.")
        return {"status": "ok", "models_loaded": predictor.loaded()}

    @app.get("/api/v1/model-info", tags=["ops"])
    def model_info(predictor: Predictor = Depends(get_predictor)):
        return predictor.metadata

    @app.get("/api/v1/options", tags=["meta"])
    def options(predictor: Predictor = Depends(get_predictor)):
        """Allowed categorical values for the fertilizer endpoint."""
        return predictor.options()

    @app.post("/api/v1/predict/crop", response_model=RecommendationResponse, tags=["predict"])
    def predict_crop(body: CropRequest, predictor: Predictor = Depends(get_predictor)):
        try:
            top = predictor.predict_crop(body.model_dump())
        except ModelNotFoundError as exc:
            raise HTTPException(503, str(exc))
        logger.info("crop request=%s top=%s", body.model_dump(), top[0])
        return {"recommendation": top[0]["name"], "top_k": top, "model_version": predictor.version("crop")}

    @app.post("/api/v1/predict/fertilizer", response_model=RecommendationResponse, tags=["predict"])
    def predict_fertilizer(body: FertilizerRequest, predictor: Predictor = Depends(get_predictor)):
        try:
            top = predictor.predict_fertilizer(body.model_dump())
        except ModelNotFoundError as exc:
            raise HTTPException(503, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        logger.info("fertilizer request=%s top=%s", body.model_dump(), top[0])
        return {"recommendation": top[0]["name"], "top_k": top, "model_version": predictor.version("fertilizer")}

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(STATIC_DIR / "index.html")

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()

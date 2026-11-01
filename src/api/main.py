"""FastAPI: churn + uplift predictions and Qini curve endpoint."""
from __future__ import annotations

import logging
import pickle
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import polars as pl
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ChurnPrediction,
    CustomerFeatures,
    FullPrediction,
    HealthResponse,
    ModelInfoResponse,
    QiniPoint,
    UpliftPrediction,
)
from src.config import get_app_config, get_settings
from src.evaluation.uplift_curves import qini_curve

logger = logging.getLogger(__name__)
logging.basicConfig(level=get_settings().log_level)

CHURN_RISK_THRESHOLD = 0.5


class Artifacts:
    def __init__(self, path: Path):
        self.path = path
        self.churn_model = None
        self.uplift_model = None
        self.feature_names: list[str] = []
        self.metadata: dict = {}
        self.qini_curve: pl.DataFrame | None = None

    def load(self) -> "Artifacts":
        model_path = self.path / "models.pkl"
        meta_path = self.path / "metadata.pkl"
        qini_path = self.path / "qini.parquet"

        if model_path.exists():
            with model_path.open("rb") as f:
                payload = pickle.load(f)
            self.churn_model = payload["churn_model"]
            self.uplift_model = payload["uplift_model"]
            self.feature_names = payload["feature_names"]
            logger.info("Loaded churn + uplift models")
        else:
            logger.warning("Model file missing at %s", model_path)

        if meta_path.exists():
            with meta_path.open("rb") as f:
                self.metadata = pickle.load(f)
        if qini_path.exists():
            self.qini_curve = pl.read_parquet(qini_path)
        return self

    @property
    def ready(self) -> bool:
        return self.churn_model is not None and self.uplift_model is not None


@lru_cache(maxsize=1)
def get_artifacts() -> Artifacts:
    settings = get_settings()
    return Artifacts(Path(settings.artifacts_dir)).load()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_artifacts()
    yield


app = FastAPI(title="Churn + Uplift API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=app.version)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(artifacts: Artifacts = Depends(get_artifacts)) -> ModelInfoResponse:
    if not artifacts.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        churn_model=type(artifacts.churn_model).__name__,
        uplift_model=type(artifacts.uplift_model).__name__,
        n_features=len(artifacts.feature_names),
        feature_names=artifacts.feature_names,
        qini_coefficient=artifacts.metadata.get("qini"),
        auuc=artifacts.metadata.get("auuc"),
    )


def _build_features_from_request(req: CustomerFeatures, cfg) -> np.ndarray:
    """Build a single-row feature vector matching training feature_names.

    Numeric: recency, history, mens, womens, newbie
    Categoricals (one-hot): zip_code values, channel values
    """
    cfg_feats = cfg.features
    numeric_vals = [req.recency, req.history, req.mens, req.womens, req.newbie]
    cat_map = {
        "zip_code": ["Rural", "Suburban", "Urban"],
        "channel": ["Multichannel", "Phone", "Web"],
    }
    cat_vals: list[int] = []
    for cat in cfg_feats.categorical:
        for v in cat_map.get(cat, []):
            val = getattr(req, cat, "")
            cat_vals.append(int(val == v))
    return np.array(numeric_vals + cat_vals, dtype=np.float32).reshape(1, -1)


def _segmentation(churn_prob: float, uplift: float) -> str:
    """Classic 4-quadrant segmentation.

    - persuadable: high churn + positive uplift  → target
    - sure_thing: low churn + small uplift       → don't spend
    - lost_cause: high churn + non-positive uplift → don't spend
    - sleeping_dog: low churn + negative uplift  → definitely don't contact
    """
    high_risk = churn_prob >= CHURN_RISK_THRESHOLD
    positive_uplift = uplift > 0
    if high_risk and positive_uplift:
        return "persuadable"
    if not high_risk and positive_uplift:
        return "sure_thing"
    if high_risk and not positive_uplift:
        return "lost_cause"
    return "sleeping_dog"


@app.post("/predict", response_model=FullPrediction)
def predict(
    req: CustomerFeatures, artifacts: Artifacts = Depends(get_artifacts)
) -> FullPrediction:
    if not artifacts.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    cfg = get_app_config()
    X = _build_features_from_request(req, cfg)
    if X.shape[1] != len(artifacts.feature_names):
        raise HTTPException(
            status_code=500,
            detail=f"Feature mismatch: got {X.shape[1]}, expected {len(artifacts.feature_names)}",
        )
    churn_prob = float(artifacts.churn_model.predict_proba(X)[0])
    uplift = float(artifacts.uplift_model.predict_uplift(X)[0])
    segment = _segmentation(churn_prob, uplift)

    return FullPrediction(
        churn=ChurnPrediction(
            churn_probability=churn_prob,
            is_high_risk=churn_prob >= CHURN_RISK_THRESHOLD,
            risk_threshold=CHURN_RISK_THRESHOLD,
        ),
        uplift=UpliftPrediction(
            uplift=uplift,
            segment=segment,
            recommend_targeting=(segment == "persuadable"),
        ),
    )


@app.get("/qini", response_model=list[QiniPoint])
def qini(
    n_points: int = 50, artifacts: Artifacts = Depends(get_artifacts)
) -> list[QiniPoint]:
    if artifacts.qini_curve is None:
        raise HTTPException(status_code=503, detail="Qini curve not available")
    curve = artifacts.qini_curve
    if curve.height > n_points:
        step = max(1, curve.height // n_points)
        curve = curve.gather_every(step)
    return [
        QiniPoint(
            fraction=float(r["fraction"]),
            cumulative_incremental=float(r["cumulative_incremental"]),
        )
        for r in curve.iter_rows(named=True)
    ]

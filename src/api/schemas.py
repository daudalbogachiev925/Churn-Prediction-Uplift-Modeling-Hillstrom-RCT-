"""Pydantic v2 schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class CustomerFeatures(BaseModel):
    recency: int = Field(..., ge=0, description="Months since last purchase")
    history: float = Field(..., ge=0, description="Lifetime spend")
    mens: int = Field(..., ge=0, le=1)
    womens: int = Field(..., ge=0, le=1)
    zip_code: str = Field(..., description="Urban/Suburban/Rural")
    newbie: int = Field(..., ge=0, le=1)
    channel: str = Field(..., description="Phone/Web/Multichannel")


class ChurnPrediction(BaseModel):
    churn_probability: float
    is_high_risk: bool
    risk_threshold: float


class UpliftPrediction(BaseModel):
    uplift: float
    segment: str = Field(..., description="persuadable / sure_thing / lost_cause / sleeping_dog")
    recommend_targeting: bool


class FullPrediction(BaseModel):
    churn: ChurnPrediction
    uplift: UpliftPrediction


class QiniPoint(BaseModel):
    fraction: float
    cumulative_incremental: float


class ModelInfoResponse(BaseModel):
    churn_model: str
    uplift_model: str
    n_features: int
    feature_names: list[str]
    qini_coefficient: float | None = None
    auuc: float | None = None

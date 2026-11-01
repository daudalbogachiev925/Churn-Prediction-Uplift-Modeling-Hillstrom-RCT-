"""T-Learner: fit separate model on treatment and control, uplift = mu_t - mu_c."""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np

from src.config import LightGBMConfig
from src.models.base import BaseUpliftModel

logger = logging.getLogger(__name__)


def _make_model(config: LightGBMConfig) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="binary",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


class TLearner(BaseUpliftModel):
    """Two-model baseline: train outcomes on treated and control separately."""

    def __init__(self, config: LightGBMConfig):
        self.config = config
        self.mu_t: lgb.LGBMRegressor | None = None
        self.mu_c: lgb.LGBMRegressor | None = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "TLearner":
        X_t = X[t == 1]
        y_t = y[t == 1]
        X_c = X[t == 0]
        y_c = y[t == 0]
        if len(y_t) < 10 or len(y_c) < 10:
            raise ValueError("Insufficient samples in one arm")

        self.mu_t = _make_model(self.config).fit(X_t, y_t)
        self.mu_c = _make_model(self.config).fit(X_c, y_c)
        logger.info("T-Learner fitted: treated=%s, control=%s", len(y_t), len(y_c))
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        if self.mu_t is None or self.mu_c is None:
            raise RuntimeError("Model not fitted")
        return self.mu_t.predict(X) - self.mu_c.predict(X)

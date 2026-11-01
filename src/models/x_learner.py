"""X-Learner: fit outcome models on both arms, impute individual effects,
then fit effect models and combine via propensity.

Reference: Künzel et al. (2019) "Metalearners for estimating heterogeneous
treatment effects using machine learning".
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np

from src.config import LightGBMConfig
from src.models.base import BaseUpliftModel

logger = logging.getLogger(__name__)


def _make_regressor(config: LightGBMConfig) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
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


class XLearner(BaseUpliftModel):
    def __init__(self, config: LightGBMConfig):
        self.config = config
        self.mu_t: lgb.LGBMRegressor | None = None
        self.mu_c: lgb.LGBMRegressor | None = None
        self.tau_t: lgb.LGBMRegressor | None = None
        self.tau_c: lgb.LGBMRegressor | None = None
        self.g: lgb.LGBMClassifier | None = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "XLearner":
        X_t, y_t = X[t == 1], y[t == 1]
        X_c, y_c = X[t == 0], y[t == 0]
        if len(y_t) < 10 or len(y_c) < 10:
            raise ValueError("Insufficient samples in one arm")

        self.mu_t = _make_regressor(self.config).fit(X_t, y_t)
        self.mu_c = _make_regressor(self.config).fit(X_c, y_c)

        # imputed individual treatment effects
        d_t = y_t - self.mu_c.predict(X_t)  # treated arm
        d_c = self.mu_t.predict(X_c) - y_c  # control arm

        self.tau_t = _make_regressor(self.config).fit(X_t, d_t)
        self.tau_c = _make_regressor(self.config).fit(X_c, d_c)

        self.g = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            n_jobs=-1, random_state=42, verbose=-1,
        ).fit(X, t)
        logger.info("X-Learner fitted")
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        if any(m is None for m in (self.tau_t, self.tau_c, self.g)):
            raise RuntimeError("Model not fitted")
        g = self.g.predict_proba(X)[:, 1]
        tau_t = self.tau_t.predict(X)
        tau_c = self.tau_c.predict(X)
        return g * tau_t + (1 - g) * tau_c

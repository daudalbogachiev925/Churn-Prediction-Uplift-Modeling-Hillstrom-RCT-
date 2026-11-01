"""LightGBM churn classifier."""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np

from src.config import LightGBMConfig
from src.models.base import BaseChurnModel

logger = logging.getLogger(__name__)


class LightGBMChurnModel(BaseChurnModel):
    def __init__(self, config: LightGBMConfig):
        self.config = config
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMChurnModel":
        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)
        logger.info("Churn model fitted on %s rows, %s features", X.shape[0], X.shape[1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> list[tuple[str, int]]:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        imp = self.model.feature_importances_
        pairs = sorted(zip(feature_names, imp.tolist()), key=lambda x: x[1], reverse=True)
        return pairs

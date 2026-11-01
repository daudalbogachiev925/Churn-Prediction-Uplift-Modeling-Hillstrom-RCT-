"""Feature building: one-hot encode categoricals, scale numerics, produce matrix."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.config import AppConfig, get_app_config

logger = logging.getLogger(__name__)


@dataclass
class FeatureMatrix:
    X: np.ndarray
    y_churn: np.ndarray
    y_conversion: np.ndarray
    y_visit: np.ndarray
    treatment: np.ndarray
    feature_names: list[str]
    row_ids: np.ndarray


def build_feature_matrix(df: pl.DataFrame, cfg: AppConfig | None = None) -> FeatureMatrix:
    cfg = cfg or get_app_config()
    numeric = [c for c in cfg.features.numeric if c in df.columns]
    categorical = [c for c in cfg.features.categorical if c in df.columns]

    X_num = df.select(numeric).to_numpy().astype(np.float64) if numeric else np.zeros((df.height, 0))

    # one-hot encode categoricals (deterministic order)
    cat_frames: list[np.ndarray] = []
    cat_names: list[str] = []
    for c in categorical:
        values = sorted(df[c].unique().to_list())
        for v in values:
            col = (df[c] == v).cast(pl.Int8).to_numpy()
            cat_frames.append(col)
            cat_names.append(f"{c}={v}")

    X_cat = np.column_stack(cat_frames) if cat_frames else np.zeros((df.height, 0))
    X = np.hstack([X_num, X_cat]).astype(np.float32)
    feature_names = numeric + cat_names

    return FeatureMatrix(
        X=X,
        y_churn=df["churn"].to_numpy().astype(np.int8),
        y_conversion=df["conversion"].to_numpy().astype(np.int8),
        y_visit=df["visit"].to_numpy().astype(np.int8),
        treatment=df["is_treatment"].to_numpy().astype(np.int8),
        feature_names=feature_names,
        row_ids=np.arange(df.height),
    )


def split_train_control(X: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X[t == 1], X[t == 0]

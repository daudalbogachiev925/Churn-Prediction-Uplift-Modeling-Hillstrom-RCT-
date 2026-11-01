"""Uplift learner tests: fit, predict shapes, sanity on synthetic HTE data."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import LightGBMConfig
from src.models.t_learner import TLearner
from src.models.x_learner import XLearner


def _synthetic_hte(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic data with known heterogeneous treatment effect.

    x0 in [0, 1] controls treatment effect strength:
        true_uplift(x) = 0.3 * x0
    baseline outcome y0 = 0.1 + 0.4 * x1
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    t = rng.integers(0, 2, n)
    p_y1 = 0.1 + 0.4 * X[:, 1] + 0.3 * X[:, 0] * t
    p_y1 = np.clip(p_y1, 0.0, 1.0)
    y = (rng.random(n) < p_y1).astype(float)
    return X, t, y


def _small_cfg() -> LightGBMConfig:
    return LightGBMConfig(n_estimators=50, learning_rate=0.1, num_leaves=15, min_child_samples=20)


def test_t_learner_fit_predict() -> None:
    X, t, y = _synthetic_hte()
    model = TLearner(_small_cfg())
    model.fit(X, t, y)
    uplift = model.predict_uplift(X)
    assert uplift.shape == (len(y),)
    assert np.isfinite(uplift).all()


def test_x_learner_fit_predict() -> None:
    X, t, y = _synthetic_hte()
    model = XLearner(_small_cfg())
    model.fit(X, t, y)
    uplift = model.predict_uplift(X)
    assert uplift.shape == (len(y),)
    assert np.isfinite(uplift).all()


def test_uplift_correlates_with_truth() -> None:
    """On data where true uplift = 0.3 * x0, predicted uplift should positively
    correlate with x0."""
    X, t, y = _synthetic_hte(n=3000, seed=1)
    model = TLearner(_small_cfg())
    model.fit(X, t, y)
    uplift = model.predict_uplift(X)
    corr = float(np.corrcoef(uplift, X[:, 0])[0, 1])
    assert corr > 0.1, f"Expected positive correlation with true HTE, got {corr}"


def test_insufficient_samples_raises() -> None:
    X = np.random.random((20, 3))
    t = np.array([1] * 5 + [0] * 15)
    y = np.random.random(20)
    model = TLearner(_small_cfg())
    with pytest.raises(ValueError):
        model.fit(X, t, y)

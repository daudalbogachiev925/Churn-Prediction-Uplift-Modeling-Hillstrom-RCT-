"""Feature matrix tests: shape, determinism, one-hot correctness."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.config import AppConfig
from src.features.build import build_feature_matrix


def _make_df(n: int = 100, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    zip_choices = ["Urban", "Suburban", "Rural"]
    chan_choices = ["Phone", "Web", "Multichannel"]
    return pl.DataFrame(
        {
            "recency": rng.integers(1, 13, n).astype(int),
            "history": rng.uniform(0, 1000, n),
            "mens": rng.integers(0, 2, n).astype(int),
            "womens": rng.integers(0, 2, n).astype(int),
            "newbie": rng.integers(0, 2, n).astype(int),
            "zip_code": [zip_choices[i % 3] for i in range(n)],
            "channel": [chan_choices[i % 3] for i in range(n)],
            "segment": ["Mens E-Mail" if i % 2 == 0 else "No E-Mail" for i in range(n)],
            "is_treatment": [1 if i % 2 == 0 else 0 for i in range(n)],
            "is_control": [0 if i % 2 == 0 else 1 for i in range(n)],
            "visit": rng.integers(0, 2, n).astype(int),
            "conversion": rng.integers(0, 2, n).astype(int),
            "spend": rng.uniform(0, 500, n),
            "churn": rng.integers(0, 2, n).astype(int),
        }
    )


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


def test_feature_matrix_shape(cfg: AppConfig) -> None:
    df = _make_df(50)
    fm = build_feature_matrix(df, cfg)
    # 5 numeric + 3 zip one-hot + 3 channel one-hot = 11
    assert fm.X.shape[1] == 11
    assert fm.X.shape[0] == 50
    assert len(fm.feature_names) == 11


def test_feature_names_contain_expected(cfg: AppConfig) -> None:
    df = _make_df(30)
    fm = build_feature_matrix(df, cfg)
    for name in ["recency", "history", "mens", "womens", "newbie"]:
        assert name in fm.feature_names
    assert "zip_code=Urban" in fm.feature_names
    assert "channel=Web" in fm.feature_names


def test_one_hot_correctness(cfg: AppConfig) -> None:
    df = pl.DataFrame(
        {
            "recency": [1, 1],
            "history": [100.0, 200.0],
            "mens": [1, 0],
            "womens": [0, 1],
            "newbie": [0, 0],
            "zip_code": ["Urban", "Rural"],
            "channel": ["Phone", "Web"],
            "segment": ["Mens E-Mail", "No E-Mail"],
            "is_treatment": [1, 0],
            "is_control": [0, 1],
            "visit": [0, 1],
            "conversion": [0, 1],
            "spend": [0.0, 50.0],
            "churn": [1, 0],
        }
    )
    fm = build_feature_matrix(df, cfg)
    name_to_idx = {n: i for i, n in enumerate(fm.feature_names)}
    urban_col = name_to_idx["zip_code=Urban"]
    rural_col = name_to_idx["zip_code=Rural"]
    phone_col = name_to_idx["channel=Phone"]
    web_col = name_to_idx["channel=Web"]
    assert fm.X[0, urban_col] == 1
    assert fm.X[0, rural_col] == 0
    assert fm.X[0, phone_col] == 1
    assert fm.X[0, web_col] == 0
    assert fm.X[1, urban_col] == 0
    assert fm.X[1, rural_col] == 1


def test_deterministic_output(cfg: AppConfig) -> None:
    df = _make_df(40, seed=42)
    fm1 = build_feature_matrix(df, cfg)
    fm2 = build_feature_matrix(df, cfg)
    np.testing.assert_array_equal(fm1.X, fm2.X)
    assert fm1.feature_names == fm2.feature_names


def test_targets_extracted(cfg: AppConfig) -> None:
    df = _make_df(20)
    fm = build_feature_matrix(df, cfg)
    assert fm.y_churn.shape == (20,)
    assert fm.y_conversion.shape == (20,)
    assert fm.treatment.shape == (20,)
    assert set(np.unique(fm.treatment)).issubset({0, 1})

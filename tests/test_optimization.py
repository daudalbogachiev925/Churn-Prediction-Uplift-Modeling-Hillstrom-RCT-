"""Budget optimization tests."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.config import OptimizationConfig
from src.optimization.budget import (
    compare_policies,
    compute_profit_curve,
    optimal_threshold,
)


@pytest.fixture
def cfg() -> OptimizationConfig:
    return OptimizationConfig(
        budget_per_customer=5.0,
        revenue_per_conversion=50.0,
        max_target_fraction=0.5,
    )


def test_profit_curve_monotone_in_n() -> None:
    rng = np.random.default_rng(0)
    uplift = rng.normal(0.05, 0.1, 1000)
    cfg = OptimizationConfig()
    curve = compute_profit_curve(uplift, cfg, n_bins=20)
    assert curve.height > 0
    # n_targeted is increasing
    n_vals = curve["n_targeted"].to_numpy()
    assert (np.diff(n_vals) > 0).all()


def test_optimal_threshold_respects_max_fraction() -> None:
    rng = np.random.default_rng(1)
    uplift = rng.normal(0.1, 0.2, 5000)
    cfg = OptimizationConfig(max_target_fraction=0.3)
    decision = optimal_threshold(uplift, cfg)
    assert decision.fraction_targeted <= 0.35  # allow small bin overshoot
    assert decision.n_targeted > 0


def test_optimal_threshold_positive_uplift_targeting() -> None:
    # all uplifts strongly positive; targeting more should be profitable
    uplift = np.full(1000, 0.5)
    cfg = OptimizationConfig(
        budget_per_customer=1.0, revenue_per_conversion=100.0, max_target_fraction=1.0
    )
    decision = optimal_threshold(uplift, cfg)
    assert decision.expected_profit > 0


def test_compare_policies_columns() -> None:
    rng = np.random.default_rng(2)
    n = 1000
    uplift = rng.normal(0.1, 0.2, n)
    y = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    cfg = OptimizationConfig()
    table = compare_policies(uplift, y, t, cfg)
    assert {"policy", "n_targeted", "incremental_conversions", "roi"}.issubset(
        set(table.columns)
    )
    assert table.height == 3
    assert "target_all" in table["policy"].to_list()
    assert "uplift_optimized" in table["policy"].to_list()

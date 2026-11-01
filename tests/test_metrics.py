"""Metrics tests: classification and experiment analysis."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.experiment import analyze_experiment, power_analysis
from src.evaluation.metrics import all_classification_metrics, ks_statistic, pr_auc, roc_auc
from src.models.cuped import cuped_adjust


def test_roc_auc_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, s) == 1.0


def test_roc_auc_random() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    s = rng.random(1000)
    auc = roc_auc(y, s)
    assert 0.4 < auc < 0.6


def test_pr_auc_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc(y, s) == 1.0


def test_ks_in_range() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = rng.random(500)
    ks = ks_statistic(y, s)
    assert 0.0 <= ks <= 1.0


def test_all_metrics_keys() -> None:
    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.6])
    m = all_classification_metrics(y, s)
    assert set(m.keys()) == {"roc_auc", "pr_auc", "ks"}


def test_cuped_reduces_variance() -> None:
    rng = np.random.default_rng(2)
    x_pre = rng.normal(0, 1, 2000)
    y = 2.0 * x_pre + rng.normal(0, 1, 2000)
    y_adj, theta = cuped_adjust(y, x_pre)
    assert abs(theta - 2.0) < 0.2
    assert y_adj.var() < y.var()


def test_analyze_experiment_known_effect() -> None:
    rng = np.random.default_rng(3)
    n = 5000
    t = rng.integers(0, 2, n)
    baseline = rng.random(n) < 0.1
    treated = rng.random(n) < 0.12
    y = np.where(t == 1, treated, baseline).astype(int)
    res = analyze_experiment(y, t, outcome_name="conversion")
    assert res.n_treatment > 0
    assert res.n_control > 0
    # true effect is 2pp; with n=5000 we expect estimate within ±1pp
    assert 0.005 < res.ate < 0.035
    assert res.p_value < 0.01


def test_power_analysis_sanity() -> None:
    n = power_analysis(baseline_rate=0.1, min_detectable_effect=0.02)
    assert n > 1000
    assert n < 100_000

"""Budget optimization: choose targeting threshold to maximize profit.

Given predicted uplift per customer and unit economics (cost per contact,
revenue per conversion), compute expected profit as a function of the
threshold. The profit-maximizing threshold gives optimal budget allocation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.config import OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class BudgetDecision:
    threshold: float
    n_targeted: int
    fraction_targeted: float
    expected_incremental_conversions: float
    expected_revenue: float
    expected_cost: float
    expected_profit: float
    roi: float


def compute_profit_curve(
    uplift: np.ndarray, cfg: OptimizationConfig, n_bins: int = 100
) -> pl.DataFrame:
    """Profit as a function of threshold quantile (top-k% targeted)."""
    n = len(uplift)
    order = np.argsort(-uplift)
    sorted_uplift = uplift[order]
    step = max(1, n // n_bins)
    rows = []
    for k in range(step, n + 1, step):
        selected = sorted_uplift[:k]
        expected_inc = float(selected.sum())
        revenue = expected_inc * cfg.revenue_per_conversion
        cost = k * cfg.budget_per_customer
        profit = revenue - cost
        rows.append(
            {
                "fraction_targeted": k / n,
                "n_targeted": k,
                "expected_incremental": expected_inc,
                "expected_revenue": revenue,
                "expected_cost": cost,
                "expected_profit": profit,
                "roi": float(profit / cost) if cost > 0 else 0.0,
            }
        )
    return pl.DataFrame(rows)


def optimal_threshold(
    uplift: np.ndarray, cfg: OptimizationConfig, n_bins: int = 100
) -> BudgetDecision:
    """Find threshold that maximizes expected profit subject to max_target_fraction."""
    curve = compute_profit_curve(uplift, cfg, n_bins)
    if curve.height == 0:
        raise RuntimeError("Empty profit curve")
    # enforce max targeting fraction
    feasible = curve.filter(pl.col("fraction_targeted") <= cfg.max_target_fraction)
    if feasible.height == 0:
        feasible = curve.head(1)
    best = feasible.sort("expected_profit", descending=True).head(1).to_dicts()[0]

    # threshold = uplift value at the boundary of targeted set
    n_targeted = int(best["n_targeted"])
    sorted_uplift = np.sort(uplift)[::-1]
    threshold = float(sorted_uplift[n_targeted - 1]) if n_targeted > 0 else 0.0

    return BudgetDecision(
        threshold=threshold,
        n_targeted=n_targeted,
        fraction_targeted=float(best["fraction_targeted"]),
        expected_incremental_conversions=float(best["expected_incremental"]),
        expected_revenue=float(best["expected_revenue"]),
        expected_cost=float(best["expected_cost"]),
        expected_profit=float(best["expected_profit"]),
        roi=float(best["roi"]),
    )


def compare_policies(
    uplift: np.ndarray,
    actual_outcome: np.ndarray,
    treatment: np.ndarray,
    cfg: OptimizationConfig,
) -> pl.DataFrame:
    """Compare policies: target-all, random-k, uplift-based, churn-only baseline.

    Returns table with realized incremental conversions and ROI for each policy
    based on ground-truth outcomes (only on historical RCT).
    """
    n = len(uplift)
    results = []

    def _evaluate(mask: np.ndarray, name: str) -> dict:
        n_sel = int(mask.sum())
        if n_sel == 0:
            return {
                "policy": name,
                "n_targeted": 0,
                "incremental_conversions": 0.0,
                "revenue": 0.0,
                "cost": 0.0,
                "profit": 0.0,
                "roi": 0.0,
            }
        y_sel = actual_outcome[mask]
        t_sel = treatment[mask]
        n_t = int((t_sel == 1).sum())
        n_c = int((t_sel == 0).sum())
        if n_t == 0 or n_c == 0:
            inc = 0.0
        else:
            inc = n_t * (y_sel[t_sel == 1].mean() - y_sel[t_sel == 0].mean())
        revenue = inc * cfg.revenue_per_conversion
        cost = n_sel * cfg.budget_per_customer
        profit = revenue - cost
        return {
            "policy": name,
            "n_targeted": n_sel,
            "incremental_conversions": float(inc),
            "revenue": float(revenue),
            "cost": float(cost),
            "profit": float(profit),
            "roi": float(profit / cost) if cost > 0 else 0.0,
        }

    # 1. Target everyone
    results.append(_evaluate(np.ones(n, dtype=bool), "target_all"))

    # 2. Random targeting at 30%
    rng = np.random.default_rng(42)
    random_mask = rng.random(n) < 0.3
    results.append(_evaluate(random_mask, "random_30pct"))

    # 3. Uplift-based targeting (optimal threshold)
    decision = optimal_threshold(uplift, cfg)
    uplift_mask = uplift >= decision.threshold
    results.append(_evaluate(uplift_mask, "uplift_optimized"))

    # 4. Baseline: target top by raw conversion propensity (if available as uplift-less score)
    # we use uplift-based as proxy here; in production this would be churn score
    return pl.DataFrame(results)

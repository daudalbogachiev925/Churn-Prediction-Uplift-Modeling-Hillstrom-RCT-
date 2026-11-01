"""Qini curve, Qini coefficient, AUUC computation.

Qini curve: for top-k% by predicted uplift, plot cumulative incremental
conversions. Qini coefficient = area between Qini curve and random baseline,
normalized by perfect-model area.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def qini_curve(
    y: np.ndarray, t: np.ndarray, uplift: np.ndarray, n_bins: int = 100
) -> pl.DataFrame:
    """Compute Qini curve points.

    At each decile k% we take top-k% by predicted uplift and compute:
        incremental = N_t(k) * (Y_t(k)/N_t(k) - Y_c(k)/N_c(k))
    where N_t, N_c are counts in treated/control among selected.
    """
    order = np.argsort(-uplift)
    y_sorted = y[order]
    t_sorted = t[order]
    n = len(y)
    step = max(1, n // n_bins)
    rows = []
    cum_t = 0
    cum_c = 0
    sum_y_t = 0.0
    sum_y_c = 0.0
    for k in range(step, n + 1, step):
        chunk_y = y_sorted[k - step:k]
        chunk_t = t_sorted[k - step:k]
        cum_t += int((chunk_t == 1).sum())
        cum_c += int((chunk_t == 0).sum())
        sum_y_t += float(chunk_y[chunk_t == 1].sum())
        sum_y_c += float(chunk_y[chunk_t == 0].sum())
        if cum_t == 0 or cum_c == 0:
            continue
        inc = cum_t * (sum_y_t / cum_t - sum_y_c / cum_c)
        rows.append(
            {
                "fraction": k / n,
                "cumulative_incremental": inc,
                "n_treated": cum_t,
                "n_control": cum_c,
            }
        )
    return pl.DataFrame(rows)


def qini_coefficient(
    y: np.ndarray, t: np.ndarray, uplift: np.ndarray, n_bins: int = 100
) -> float:
    """Normalized Qini coefficient in [0, 1]. Higher = better targeting.

    The Qini coefficient equals area between Qini curve and random-targeting
    line, divided by the area of a perfect model (all incrementals ranked first).
    """
    curve = qini_curve(y, t, uplift, n_bins)
    if curve.height < 2:
        return 0.0
    x = curve["fraction"].to_numpy()
    y_vals = curve["cumulative_incremental"].to_numpy()

    # random-targeting baseline: straight line from origin to full-population value
    total_t = int((t == 1).sum())
    total_c = int((t == 0).sum())
    if total_t == 0 or total_c == 0:
        return 0.0
    full_incremental = total_t * (
        float(y[t == 1].mean()) - float(y[t == 0].mean())
    )
    baseline = full_incremental * x

    # area between curve and baseline using trapezoid
    area_model = float(np.trapz(y_vals, x))
    area_baseline = float(np.trapz(baseline, x))
    area_diff = area_model - area_baseline

    # perfect model: all positives in treated sorted first, all negatives last
    perfect_uplift = np.where(t == 1, y + 0.5, y - 0.5)
    perfect_curve = qini_curve(y, t, perfect_uplift, n_bins)
    if perfect_curve.height < 2:
        return 0.0
    px = perfect_curve["fraction"].to_numpy()
    py = perfect_curve["cumulative_incremental"].to_numpy()
    area_perfect = float(np.trapz(py, px)) - area_baseline

    if area_perfect <= 0:
        return 0.0
    return float(area_diff / area_perfect)


def auuc(
    y: np.ndarray, t: np.ndarray, uplift: np.ndarray, n_bins: int = 100
) -> float:
    """Area Under Uplift Curve (unnormalized)."""
    curve = qini_curve(y, t, uplift, n_bins)
    if curve.height < 2:
        return 0.0
    x = curve["fraction"].to_numpy()
    y_vals = curve["cumulative_incremental"].to_numpy()
    return float(np.trapz(y_vals, x))


def uplift_at_k(
    y: np.ndarray, t: np.ndarray, uplift: np.ndarray, k: float = 0.1
) -> dict[str, float]:
    """Business metrics at top-k% by predicted uplift.

    Returns counts, conversion rates, and ROI assuming constant unit economics.
    """
    n = len(uplift)
    top_n = max(1, int(n * k))
    order = np.argsort(-uplift)[:top_n]
    y_top = y[order]
    t_top = t[order]
    n_t = int((t_top == 1).sum())
    n_c = int((t_top == 0).sum())
    if n_t == 0 or n_c == 0:
        return {
            "n_targeted": float(top_n),
            "conversion_rate_treated": float("nan"),
            "conversion_rate_control": float("nan"),
            "incremental_conversions": 0.0,
            "uplift_pp": 0.0,
        }
    cr_t = float(y_top[t_top == 1].mean())
    cr_c = float(y_top[t_top == 0].mean())
    incremental = n_t * (cr_t - cr_c)
    return {
        "n_targeted": float(top_n),
        "conversion_rate_treated": cr_t,
        "conversion_rate_control": cr_c,
        "incremental_conversions": float(incremental),
        "uplift_pp": float((cr_t - cr_c) * 100.0),
    }

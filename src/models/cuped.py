"""CUPED: variance reduction for treatment effect via pre-period covariate.

Given pre-period covariate X, adjusted outcome Y_adj = Y - theta * (X - mean(X)),
with theta = Cov(Y, X) / Var(X). Reduces variance without bias.
"""
from __future__ import annotations

import numpy as np


def cuped_adjust(
    y: np.ndarray, x_pre: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return (y_adjusted, theta)."""
    cov = np.cov(y, x_pre, ddof=1)[0, 1]
    var = np.var(x_pre, ddof=1)
    if var == 0:
        return y.copy(), 0.0
    theta = float(cov / var)
    y_adj = y - theta * (x_pre - x_pre.mean())
    return y_adj, theta


def ate_cuped(
    y: np.ndarray, t: np.ndarray, x_pre: np.ndarray
) -> tuple[float, float, float]:
    """Compute CUPED ATE with 95% CI via normal approximation.

    Returns: (ate, ci_lower, ci_upper)
    """
    y_adj, _ = cuped_adjust(y, x_pre)
    n_t = (t == 1).sum()
    n_c = (t == 0).sum()
    mean_t = y_adj[t == 1].mean()
    mean_c = y_adj[t == 0].mean()
    var_t = y_adj[t == 1].var(ddof=1)
    var_c = y_adj[t == 0].var(ddof=1)
    ate = float(mean_t - mean_c)
    se = float(np.sqrt(var_t / n_t + var_c / n_c))
    ci_lo = ate - 1.96 * se
    ci_hi = ate + 1.96 * se
    return ate, float(ci_lo), float(ci_hi)

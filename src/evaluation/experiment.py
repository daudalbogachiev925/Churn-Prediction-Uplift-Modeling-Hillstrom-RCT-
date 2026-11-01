"""A/B experiment analysis: ATE, CUPED, lift confidence intervals."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.models.cuped import cuped_adjust

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    outcome: str
    n_treatment: int
    n_control: int
    mean_treatment: float
    mean_control: float
    ate: float
    ate_cuped: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    relative_lift: float


def analyze_experiment(
    y: np.ndarray, t: np.ndarray, x_pre: np.ndarray | None = None, outcome_name: str = "conversion"
) -> ExperimentResult:
    """Standard analysis: difference in means + Welch t-test + CUPED adjustment."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.int64)
    y_t = y[t == 1]
    y_c = y[t == 0]
    n_t, n_c = len(y_t), len(y_c)
    if n_t < 2 or n_c < 2:
        raise ValueError("Insufficient samples per arm")

    mean_t = float(y_t.mean())
    mean_c = float(y_c.mean())
    ate = mean_t - mean_c
    var_t = float(y_t.var(ddof=1))
    var_c = float(y_c.var(ddof=1))
    se = float(np.sqrt(var_t / n_t + var_c / n_c))

    # Welch's t-test
    t_stat = ate / se if se > 0 else 0.0
    df = (
        (var_t / n_t + var_c / n_c) ** 2
        / ((var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1))
    )
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))

    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    relative_lift = (ate / mean_c) if mean_c > 0 else float("nan")

    ate_cuped_val = float("nan")
    if x_pre is not None and len(x_pre) == len(y):
        y_adj, _ = cuped_adjust(y, x_pre)
        mean_t_adj = float(y_adj[t == 1].mean())
        mean_c_adj = float(y_adj[t == 0].mean())
        ate_cuped_val = mean_t_adj - mean_c_adj

    return ExperimentResult(
        outcome=outcome_name,
        n_treatment=n_t,
        n_control=n_c,
        mean_treatment=mean_t,
        mean_control=mean_c,
        ate=ate,
        ate_cuped=ate_cuped_val,
        se=se,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=p_value,
        relative_lift=float(relative_lift),
    )


def power_analysis(
    baseline_rate: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
    treatment_fraction: float = 0.5,
) -> int:
    """Required sample size per arm for given MDE (absolute rate difference).

    Uses normal approximation: n = (z_alpha/2 + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / mde^2
    adjusted for unequal split.
    """
    p1 = baseline_rate
    p2 = baseline_rate + min_detectable_effect
    if p2 >= 1.0 or p2 <= 0.0:
        raise ValueError("Invalid MDE: p2 out of (0, 1)")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    var = p1 * (1 - p1) + p2 * (1 - p2)
    n_per_arm = ((z_alpha + z_beta) ** 2 * var) / (min_detectable_effect ** 2)
    # adjust for treatment_fraction (default 50/50)
    n_total = n_per_arm / (treatment_fraction * (1 - treatment_fraction))
    return int(np.ceil(n_total))

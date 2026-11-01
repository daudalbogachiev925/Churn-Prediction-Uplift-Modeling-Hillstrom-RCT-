"""Train churn model + uplift model, run CV, save artifacts."""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

from src.config import ensure_dirs, get_app_config, get_settings
from src.data.loader import load_processed
from src.evaluation.experiment import analyze_experiment
from src.evaluation.metrics import all_classification_metrics
from src.evaluation.uplift_curves import (
    auuc,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
)
from src.features.build import build_feature_matrix
from src.models.churn import LightGBMChurnModel
from src.models.t_learner import TLearner
from src.models.x_learner import XLearner
from src.optimization.budget import compare_policies, optimal_threshold

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("churn.train")


def main() -> int:
    ensure_dirs()
    cfg = get_app_config()

    logger.info("Loading processed data...")
    df = load_processed(cfg)
    logger.info("Loaded %s rows", df.height)

    # overall experiment view
    exp = analyze_experiment(
        df["conversion"].to_numpy(),
        df["is_treatment"].to_numpy(),
        x_pre=df["history"].to_numpy(),
        outcome_name="conversion",
    )
    logger.info("Overall experiment: ATE=%.4f (CI [%.4f, %.4f], p=%.4g), CUPED=%.4f",
                exp.ate, exp.ci_lower, exp.ci_upper, exp.p_value, exp.ate_cuped)

    fm = build_feature_matrix(df, cfg)
    logger.info("Feature matrix: %s rows x %s features", fm.X.shape[0], fm.X.shape[1])

    # Churn model CV
    logger.info("Churn model: 5-fold CV")
    skf = StratifiedKFold(n_splits=cfg.cv.n_splits, shuffle=True, random_state=42)
    churn_metrics = {"roc_auc": [], "pr_auc": [], "ks": []}
    for fold, (tr_idx, va_idx) in enumerate(skf.split(fm.X, fm.y_churn)):
        model = LightGBMChurnModel(cfg.models.churn_lightgbm)
        model.fit(fm.X[tr_idx], fm.y_churn[tr_idx])
        scores = model.predict_proba(fm.X[va_idx])
        m = all_classification_metrics(fm.y_churn[va_idx], scores)
        for k, v in m.items():
            churn_metrics[k].append(v)
    churn_summary = {k: float(np.nanmean(v)) for k, v in churn_metrics.items()}
    logger.info("Churn CV metrics: %s", churn_summary)

    # Final churn model on all data
    churn_final = LightGBMChurnModel(cfg.models.churn_lightgbm)
    churn_final.fit(fm.X, fm.y_churn)
    logger.info("Churn model fitted. Top features: %s",
                churn_final.feature_importance(fm.feature_names)[:5])

    # Uplift: T-Learner vs X-Learner on full data
    logger.info("Training T-Learner...")
    t_learner = TLearner(cfg.models.uplift_lightgbm)
    t_learner.fit(fm.X, fm.treatment, fm.y_conversion.astype(float))
    t_uplift = t_learner.predict_uplift(fm.X)
    t_qini = qini_coefficient(fm.y_conversion, fm.treatment, t_uplift)
    t_auuc = auuc(fm.y_conversion, fm.treatment, t_uplift)
    logger.info("T-Learner Qini=%.4f, AUUC=%.2f", t_qini, t_auuc)

    logger.info("Training X-Learner...")
    x_learner = XLearner(cfg.models.uplift_lightgbm)
    x_learner.fit(fm.X, fm.treatment, fm.y_conversion.astype(float))
    x_uplift = x_learner.predict_uplift(fm.X)
    x_qini = qini_coefficient(fm.y_conversion, fm.treatment, x_uplift)
    x_auuc = auuc(fm.y_conversion, fm.treatment, x_uplift)
    logger.info("X-Learner Qini=%.4f, AUUC=%.2f", x_qini, x_auuc)

    best_uplift = t_uplift if t_qini >= x_qini else x_uplift
    best_model = t_learner if t_qini >= x_qini else x_learner
    best_name = "T-Learner" if t_qini >= x_qini else "X-Learner"
    best_qini = max(t_qini, x_qini)
    best_auuc = max(t_auuc, x_auuc)
    logger.info("Best uplift model: %s (Qini=%.4f)", best_name, best_qini)

    # Business metrics at top 10%
    k_metrics = uplift_at_k(fm.y_conversion, fm.treatment, best_uplift, k=0.1)
    logger.info("Uplift@10%%: %s", k_metrics)

    # Budget optimization
    decision = optimal_threshold(best_uplift, cfg.optimization)
    logger.info("Optimal budget decision: target %.1f%% (n=%s), profit=%.2f, ROI=%.2f",
                decision.fraction_targeted * 100,
                decision.n_targeted,
                decision.expected_profit,
                decision.roi)

    # Compare policies
    policy_table = compare_policies(
        best_uplift, fm.y_conversion, fm.treatment, cfg.optimization
    )
    logger.info("Policy comparison:\n%s", policy_table)

    # Save artifacts
    out_dir = Path(get_settings().artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "models.pkl").open("wb") as f:
        pickle.dump(
            {
                "churn_model": churn_final,
                "uplift_model": best_model,
                "feature_names": fm.feature_names,
            },
            f,
        )
    with (out_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(
            {
                "churn_cv_metrics": churn_summary,
                "qini": best_qini,
                "auuc": best_auuc,
                "best_uplift_model": best_name,
                "uplift_at_10pct": k_metrics,
                "optimal_decision": decision.__dict__,
            },
            f,
        )
    qini_curve(fm.y_conversion, fm.treatment, best_uplift).write_parquet(
        out_dir / "qini.parquet"
    )
    logger.info("Artifacts saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

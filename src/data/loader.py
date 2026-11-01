"""Persistence helpers and churn label construction."""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.config import AppConfig

logger = logging.getLogger(__name__)


def add_churn_label(df: pl.DataFrame, cfg: AppConfig) -> pl.DataFrame:
    """Construct a churn proxy: customer with high recency AND low history.

    Rationale: Hillstrom has no explicit churn label; we operationalize churn as
    "customer hasn't purchased in a while and has little lifetime history".
    This is a common industry approach for proxy labeling.
    """
    churn = (
        (pl.col("recency") > cfg.churn.churn_threshold_recency)
        & (pl.col("history") < cfg.churn.churn_threshold_history)
    ).cast(pl.Int8)
    return df.with_columns(churn.alias("churn"))


def persist_processed(df: pl.DataFrame, cfg: AppConfig) -> Path:
    out_path = Path(cfg.data.processed_dir) / "hillstrom_clean.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info("Persisted processed: %s rows -> %s", df.height, out_path)
    return out_path


def load_processed(cfg: AppConfig) -> pl.DataFrame:
    path = Path(cfg.data.processed_dir) / "hillstrom_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found at {path}. Run download_data.py first.")
    return pl.read_parquet(path)

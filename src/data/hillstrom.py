"""Hillstrom MineThatData E-Mail Analytics RCT loader.

The raw CSV has no header; we assign column names manually and standardize
to snake_case.
"""
from __future__ import annotations

import logging
from pathlib import Path

import httpx
import polars as pl

from src.config import AppConfig

logger = logging.getLogger(__name__)

COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]


def download_raw(cfg: AppConfig) -> Path:
    out_dir = Path(cfg.data.raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hillstrom.csv"
    if out_path.exists():
        logger.info("Skip existing: %s", out_path)
        return out_path
    logger.info("Downloading %s", cfg.data.raw_url)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(cfg.data.raw_url)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    logger.info("Saved %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def load_raw(cfg: AppConfig) -> pl.DataFrame:
    path = Path(cfg.data.raw_dir) / "hillstrom.csv"
    if not path.exists():
        path = download_raw(cfg)
    df = pl.read_csv(path, has_header=False, new_columns=COLUMNS, try_parse_dates=False)
    # drop history_segment (bucketed, redundant with history)
    df = df.drop("history_segment")
    # normalize segment values
    df = df.with_columns(
        pl.col("segment").str.replace_all(r"\s+", " ").str.strip_chars().alias("segment")
    )
    # binary treatment flag
    df = df.with_columns(
        (pl.col("segment") == cfg.data.treatment_value).cast(pl.Int8).alias("is_treatment"),
        (pl.col("segment") == cfg.data.control_value).cast(pl.Int8).alias("is_control"),
    )
    logger.info("Loaded Hillstrom: %s rows, segments=%s",
                df.height, df["segment"].value_counts().to_dicts())
    return df

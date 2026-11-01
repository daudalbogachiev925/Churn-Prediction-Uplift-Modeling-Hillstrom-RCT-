"""Download Hillstrom RCT and persist processed parquet with churn label."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dirs, get_app_config, get_settings
from src.data.hillstrom import download_raw, load_raw
from src.data.loader import add_churn_label, persist_processed

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("churn.download")


def main() -> int:
    ensure_dirs()
    cfg = get_app_config()
    download_raw(cfg)
    df = load_raw(cfg)
    df = add_churn_label(df, cfg)
    logger.info("Churn rate: %.3f", float(df["churn"].mean()))
    persist_processed(df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Configuration loader: pydantic-settings + YAML."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    raw_url: str = (
        "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    )
    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    treatment_col: str = "segment"
    treatment_value: str = "Mens E-Mail"
    control_value: str = "No E-Mail"
    outcome_col: str = "conversion"


class ChurnConfig(BaseModel):
    churn_threshold_recency: int = 10
    churn_threshold_history: float = 200.0


class FeaturesConfig(BaseModel):
    numeric: list[str] = Field(default_factory=lambda: ["recency", "history", "mens", "womens", "newbie"])
    categorical: list[str] = Field(default_factory=lambda: ["zip_code", "channel"])


class LightGBMConfig(BaseModel):
    n_estimators: int = 500
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_child_samples: int = 30
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1


class ModelsConfig(BaseModel):
    churn_lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    uplift_lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)


class CVConfig(BaseModel):
    n_splits: int = 5


class OptimizationConfig(BaseModel):
    budget_per_customer: float = 5.0
    revenue_per_conversion: float = 50.0
    max_target_fraction: float = 0.5


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    churn: ChurnConfig = Field(default_factory=ChurnConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    config_path: str = "./configs/config.yaml"

    def load_app_config(self) -> AppConfig:
        path = Path(self.config_path)
        if not path.exists():
            return AppConfig()
        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return AppConfig(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    return get_settings().load_app_config()


def ensure_dirs() -> None:
    settings = get_settings()
    for p in (settings.data_dir, settings.artifacts_dir):
        Path(p).mkdir(parents=True, exist_ok=True)
    cfg = get_app_config()
    Path(cfg.data.raw_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data.processed_dir).mkdir(parents=True, exist_ok=True)

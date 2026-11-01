"""FastAPI tests with mocked artifacts."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient


class _DummyChurn:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # higher recency -> higher churn prob
        p = np.clip(X[:, 0] / 12.0, 0.05, 0.95)
        return p


class _DummyUplift:
    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], 0.05)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    feature_names = [
        "recency", "history", "mens", "womens", "newbie",
        "zip_code=Rural", "zip_code=Suburban", "zip_code=Urban",
        "channel=Multichannel", "channel=Phone", "channel=Web",
    ]
    with (tmp_path / "models.pkl").open("wb") as f:
        pickle.dump(
            {
                "churn_model": _DummyChurn(),
                "uplift_model": _DummyUplift(),
                "feature_names": feature_names,
            },
            f,
        )
    with (tmp_path / "metadata.pkl").open("wb") as f:
        pickle.dump({"qini": 0.35, "auuc": 120.0}, f)
    qini_curve = pl.DataFrame(
        {
            "fraction": np.linspace(0.1, 1.0, 10),
            "cumulative_incremental": np.linspace(5.0, 50.0, 10),
            "n_treated": np.arange(10) * 10 + 100,
            "n_control": np.arange(10) * 10 + 100,
        }
    )
    qini_curve.write_parquet(tmp_path / "qini.parquet")

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from src.config import get_settings, get_app_config
    from src.api import main as api_main

    get_settings.cache_clear()
    get_app_config.cache_clear()
    api_main.get_artifacts.cache_clear()

    with TestClient(api_main.app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200


def test_model_info(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert data["n_features"] == 11
    assert data["qini_coefficient"] == pytest.approx(0.35)


def test_predict_persuadable(client: TestClient) -> None:
    """High recency (churn risk) + positive uplift => persuadable."""
    body = {
        "recency": 12,
        "history": 100.0,
        "mens": 1,
        "womens": 0,
        "zip_code": "Urban",
        "newbie": 0,
        "channel": "Web",
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["churn"]["is_high_risk"] is True
    assert data["uplift"]["segment"] == "persuadable"
    assert data["uplift"]["recommend_targeting"] is True


def test_predict_sure_thing(client: TestClient) -> None:
    body = {
        "recency": 1,
        "history": 800.0,
        "mens": 1,
        "womens": 1,
        "zip_code": "Suburban",
        "newbie": 0,
        "channel": "Phone",
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["churn"]["is_high_risk"] is False
    assert data["uplift"]["segment"] == "sure_thing"


def test_predict_invalid_input(client: TestClient) -> None:
    body = {"recency": -5, "history": 100.0, "mens": 1, "womens": 0,
            "zip_code": "Urban", "newbie": 0, "channel": "Web"}
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_qini_endpoint(client: TestClient) -> None:
    r = client.get("/qini")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "fraction" in data[0]
    assert "cumulative_incremental" in data[0]

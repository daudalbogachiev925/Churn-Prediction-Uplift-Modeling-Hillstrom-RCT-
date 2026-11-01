# Churn Prediction + Uplift Modeling (Hillstrom RCT)

End-to-end система: предсказание оттока + uplift-моделирование для оптимизации
маркетинговых кампаний по удержанию. Не «кому плохо», а «на кого кампания
реально повлияет».

## Бизнес-задача
- Компания тратит бюджет на удержание всех клиентов с высоким риском оттока.
- Но большинство из них ушли бы и без кампании (sure things), либо уйдут
  несмотря на неё (lost causes).
- **Цель**: максимизировать incremental conversions на единицу бюджета
  через uplift-моделирование и оптимизацию порогов по Qini-кривой.

## Данные
**Hillstrom MineThatData E-Mail Analytics Dataset** (2008) — классический
RCT-датасет для uplift-моделирования:
- 64 000 клиентов, 3 группы: `Mens E-Mail`, `Womens E-Mail`, `No E-Mail` (control)
- Признаки: recency, history, mens/womens, zip_code, newbie, channel
- Таргеты: visit, conversion, spend
- Источник: http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

Дополнительно: синтетическая churn-разметка на основе recency + spend
для задачи классического churn prediction.

## Стек
Python 3.11, Polars, scikit-learn, LightGBM, CatBoost, FastAPI, Pydantic v2,
MLflow (опционально), Docker.

## Модели
- **Churn**: LightGBM classifier (predict probability of churn)
- **Uplift T-Learner**: две модели (treatment/control), разница предсказаний
- **Uplift X-Learner**: использует обе модели + imputed effects, лучше при
  несбалансированных группах
- **CUPED**: снижение дисперсии оценки эффекта через ковариату (pre-period spend)

## Метрики
- Churn: ROC-AUC, PR-AUC, KS
- Uplift: Qini coefficient, Qini curve, AUUC, uplift@k%
- Бизнес: incremental conversions, ROI, cost per incremental conversion

## Быстрый старт
```bash
make setup
make download
make train
make serve
curl "http://localhost:8000/uplift?customer_id=12345"



---

## `pyproject.toml`

```toml
[project]
name = "churn-uplift-modeling"
version = "0.1.0"
description = "Churn prediction + uplift modeling on Hillstrom RCT"
requires-python = ">=3.11"
dependencies = [
    "polars>=0.20.0",
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.4",
    "lightgbm>=4.3",
    "catboost>=1.2",
    "scipy>=1.12",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "httpx>=0.27",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.3",
    "mypy>=1.9",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

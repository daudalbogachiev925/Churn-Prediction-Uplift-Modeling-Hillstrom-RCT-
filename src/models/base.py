"""Base interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseChurnModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseChurnModel":
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


class BaseUpliftModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "BaseUpliftModel":
        ...

    @abstractmethod
    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        ...

"""TabPFN v3 regression wrapper for TabTune pipeline compatibility.

Mirrors `models/regression/tabpfnv26/regressor.py`: a thin sklearn-friendly
adapter over the vendored v3 regressor that normalizes inputs (sparse -> dense,
DataFrame/Series -> ndarray) before delegating to the upstream estimator.

Inputs:
    X: array-like / DataFrame / sparse matrix, shape (n_samples, n_features)
    y: array-like / Series / 1-col DataFrame, shape (n_samples,)
Outputs:
    predict(X) -> np.ndarray, shape (n_samples,)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from tabtune.models.tabpfnv3 import TabPFNv3Regressor

logger = logging.getLogger(__name__)


class TabPFNv3RegressorWrapper(TabPFNv3Regressor):
    """Wrapper for the TabPFN v3 regressor — ensures TabTune pipeline compatibility."""

    def __init__(self, tuning_strategy: str = "inference", **kwargs):
        if tuning_strategy not in ("inference", "finetune"):
            raise ValueError(
                f"Regression supports 'inference' or 'finetune'. Got: '{tuning_strategy}'"
            )
        # Drop TabTune-orchestration-only keys the upstream estimator never sees.
        filtered = {
            k: v for k, v in kwargs.items() if k not in ("task_type", "tuning_strategy")
        }
        super().__init__(**filtered)
        self.tuning_strategy = tuning_strategy
        # Pipeline introspects `.model`; point it at self (estimator is its own model).
        self.model = self

    @staticmethod
    def _densify(X):
        if issparse(X):
            return X.toarray()
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if hasattr(X[col], "sparse") and X[col].sparse is not None:
                    X = X.copy()
                    X[col] = X[col].sparse.to_dense()
        return X

    def fit(self, X, y):
        X = self._densify(X)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y).flatten()
        return super().fit(X, y)

    def predict(self, X):
        X = self._densify(X)
        return super().predict(X)

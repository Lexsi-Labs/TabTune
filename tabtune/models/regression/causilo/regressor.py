"""Causilo regression wrapper for TabTune pipeline compatibility.

Mirrors the other regression wrappers: a thin sklearn-friendly adapter that
normalises inputs (sparse to dense, DataFrame/Series to ndarray) before
delegating to the vendored estimator, and points ``.model`` at the object the
pipeline introspects.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from tabtune.models.causilo.tabtune_support import CausiloTabTuneRegressor

logger = logging.getLogger(__name__)


class CausiloRegressorWrapper(CausiloTabTuneRegressor):
    """Wrapper for the Causilo regressor — ensures TabTune pipeline compatibility."""

    def __init__(self, tuning_strategy: str = "inference", **kwargs):
        if tuning_strategy not in ("inference", "finetune", "peft"):
            raise ValueError(
                f"Regression supports 'inference', 'finetune' or 'peft'. Got: {tuning_strategy!r}"
            )
        filtered = {
            k: v for k, v in kwargs.items() if k not in ("task_type", "tuning_strategy")
        }
        super().__init__(tuning_strategy=tuning_strategy, **filtered)

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
        return super().fit(self._densify(X), y)

    def predict(self, X, **kwargs):
        return super().predict(self._densify(X), **kwargs)

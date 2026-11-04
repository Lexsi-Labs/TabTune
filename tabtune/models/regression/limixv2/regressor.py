"""LimiX-2 regression wrapper for the TabTune pipeline."""
from __future__ import annotations

import pandas as pd
from scipy.sparse import issparse

from tabtune.models.limixv2.tabtune_support import LimiXV2Regressor


class LimiXV2RegressorWrapper(LimiXV2Regressor):

    @staticmethod
    def _dense(X):
        if issparse(X):
            return X.toarray()
        if isinstance(X, pd.DataFrame):
            sparse_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.SparseDtype)]
            if sparse_cols:
                X = X.copy()
                for col in sparse_cols:
                    X[col] = X[col].sparse.to_dense()
        return X

    def fit(self, X, y):
        if self.tuning_strategy not in ("inference", "finetune", "peft"):
            raise ValueError(
                "Regression supports 'inference', 'finetune' or 'peft'. "
                f"Got: {self.tuning_strategy!r}"
            )
        return super().fit(self._dense(X), y)

    def predict(self, X):
        return super().predict(self._dense(X))

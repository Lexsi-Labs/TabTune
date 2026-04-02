"""TabPFNv26 regression wrapper for TabTune pipeline compatibility."""
import numpy as np
import pandas as pd
import logging
from scipy.sparse import issparse
from tabtune.models.tabpfnv26.regressor import TabPFNRegressor as TabPFNv26Regressor

logger = logging.getLogger(__name__)


class TabPFNv26RegressorWrapper(TabPFNv26Regressor):
    """Wrapper for TabPFNv26 Regressor — ensures TabTune pipeline compatibility."""

    def __init__(self, tuning_strategy='inference', **kwargs):
        if tuning_strategy not in ("inference", "finetune"):
            raise ValueError(
                f"Regression supports 'inference' or 'finetune'. Got: '{tuning_strategy}'"
            )
        filtered = {k: v for k, v in kwargs.items()
                    if k not in ('task_type', 'tuning_strategy')}
        super().__init__(**filtered)
        self.tuning_strategy = tuning_strategy
        self.model = self

    def fit(self, X, y):
        if issparse(X):
            X = X.toarray()
        elif isinstance(X, pd.DataFrame):
            for col in X.columns:
                if hasattr(X[col], 'sparse') and X[col].sparse:
                    X = X.copy()
                    X[col] = X[col].sparse.to_dense()
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            y = y.values
        y = np.array(y).flatten()
        return super().fit(X, y)

    def predict(self, X):
        if issparse(X):
            X = X.toarray()
        elif isinstance(X, pd.DataFrame):
            for col in X.columns:
                if hasattr(X[col], 'sparse') and X[col].sparse:
                    X = X.copy()
                    X[col] = X[col].sparse.to_dense()
        return super().predict(X)

"""Causilo preprocessor for TabTune's DataProcessor.

Causilo does its own feature handling: missing values, categorical columns and
normalisation are all handled inside ``PreparedDataset.prepare``, per member of
its permutation ensemble. Imputing or one-hot encoding here would therefore
destroy information the model is built to use - a category encoded as a float
before it reaches the model can no longer be recognised as a category.

So this preprocessor keeps the feature frame intact and does exactly two things
TabTune's pipeline needs: it preserves categorical dtypes through the frame
round-trip, and it label-encodes the target for classification so ``classes_``
is contiguous. Regression targets pass through as floats.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class CausiloPreprocessor(BaseEstimator, TransformerMixin):
    """Pass features through untouched; encode the target for classification."""

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.label_encoder_ = LabelEncoder()
        self.feature_names_: list | None = None
        self.categorical_cols_: list = []
        self.numerical_cols_: list = []

    def fit(self, X, y=None):
        logger.info("[CausiloPreprocessor] Fitting (features pass through untouched)")
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()
            self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        if y is not None and self.task_type == "classification":
            self.label_encoder_.fit(np.asarray(y).ravel())
        return self

    def transform(self, X, y=None):
        """Return the features as given, as a DataFrame where possible.

        Causilo reads pandas categorical dtype as categories and object columns
        of strings as categories, so the frame is returned rather than a numpy
        array whenever one came in.

        Returns ``X`` alone when ``y`` is None and ``(X, y)`` otherwise, which
        is the contract ``DataProcessor.transform`` calls this with.
        """
        if isinstance(X, pd.DataFrame):
            features = X
        elif hasattr(X, "toarray"):
            features = X.toarray()
        else:
            features = np.asarray(X)
        if y is None:
            return features
        return features, self.transform_target(y)

    def transform_target(self, y):
        """Encode classification labels; leave regression targets as floats."""
        values = np.asarray(y).ravel()
        if self.task_type != "classification":
            return values.astype(float)
        return self.label_encoder_.transform(values)

    def inverse_transform_target(self, y):
        if self.task_type != "classification":
            return np.asarray(y).ravel()
        return self.label_encoder_.inverse_transform(np.asarray(y).ravel().astype(int))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

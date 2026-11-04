"""TabLDM preprocessor for TabTune's DataProcessor.

TabLDM does its own feature handling. ``TransformToNumerical`` detects and
ordinal-encodes string, object, category and boolean columns, gives missing
categorical values a category of their own, mean-imputes missing numerics, and
then ``EnsembleGenerator`` clips outliers and applies a different normalisation
per ensemble member. Imputing or one-hot encoding here would destroy exactly
the signals those steps read: a category encoded as a float before it reaches
the model can no longer be recognised as a category, and an imputed NaN can no
longer be recognised as missing.

So this preprocessor keeps the feature frame intact and does the two things
TabTune's pipeline needs: it preserves column dtypes through the frame
round-trip, and it label-encodes the target for classification so ``classes_``
is contiguous. Regression targets pass through as floats - TabLDM standardises
them itself with its own ``y_scaler_`` and predicts in the original space.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class TabLDMPreprocessor(BaseEstimator, TransformerMixin):
    """Pass features through untouched; encode the target for classification."""

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.label_encoder_ = LabelEncoder()
        self.feature_names_: list | None = None
        self.categorical_cols_: list = []
        self.numerical_cols_: list = []

    def fit(self, X, y=None):
        logger.info("[TabLDMPreprocessor] Fitting (features pass through untouched)")
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()
            self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        if y is not None and self.task_type == "classification":
            self.label_encoder_.fit(np.asarray(y).ravel())
        return self

    def transform(self, X, y=None):
        """Return the features as given, as a DataFrame where possible.

        TabLDM reads pandas categorical dtype and object columns of strings as
        categories, so the frame is returned rather than a numpy array whenever
        one came in - in a numpy array every column shares one dtype and integer
        columns are treated as numerical.

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

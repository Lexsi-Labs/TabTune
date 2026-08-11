"""
xRFM model-aware preprocessor for TabTune.

xRFM performs its own feature encoding internally (one-hot categoricals +
standardised numerics, per the upstream README), so at inference time TabTune
hands raw mixed-type frames straight to the wrapper.  This preprocessor exists
for the same two reasons as every other TabTune model preprocessor:

1.  it fits a ``LabelEncoder`` on the target so ``TabularPipeline.evaluate``
    and the probability-alignment code can map predictions back to the
    original label space; and
2.  it produces a clean **numeric** feature matrix (one-hot categoricals +
    standardised numerics) for any consumer that needs processed arrays.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from ..models.xrfm.preprocessing import XRFMFeatureEncoder

logger = logging.getLogger(__name__)


class XRFMPreprocessor(BaseEstimator, TransformerMixin):
    """xRFM-style preprocessing (README recipe) that also fits the label encoder."""

    def __init__(self, task_type: str = "classification", categorical_encoding: str = "onehot"):
        self.task_type = task_type
        self.categorical_encoding = categorical_encoding

        self.feature_encoder_ = None
        self.label_encoder_ = None

    def fit(self, X, y=None):
        logger.info("[XRFM] Fitting XRFM preprocessor (task=%s)", self.task_type)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_encoder_ = XRFMFeatureEncoder(categorical_encoding=self.categorical_encoding)
        self.feature_encoder_.fit(X)

        if self.task_type == "classification" and y is not None:
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
        return self

    def transform(self, X, y=None):
        X_final = self.feature_encoder_.transform(X)

        if y is not None:
            if self.task_type == "classification" and self.label_encoder_ is not None:
                y_final = self.label_encoder_.transform(y)
            else:
                y_final = np.asarray(y, dtype=float).ravel()
            return X_final, y_final
        return X_final

    def get_summary(self):
        enc = self.feature_encoder_
        return {
            "XRFM Preprocessing": {
                "description": "One-hot encoded categoricals (not standardised) + "
                "mean-imputed, standardised numerics (upstream xRFM README recipe). "
                "Raw frames are handed to the wrapper at inference; it applies the "
                "same encoding internally.",
                "details": [
                    f"{len(enc.categorical_cols_)} categorical, "
                    f"{len(enc.numerical_cols_)} numerical columns -> "
                    f"{enc.n_output_features_} float32 features.",
                ],
            }
        }

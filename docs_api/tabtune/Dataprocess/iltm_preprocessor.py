"""
iLTM model-aware preprocessor for TabTune.

iLTM handles feature scaling, categorical features and missing values
**internally** (robust RealMLP-style preprocessing + optional tree
embeddings), so at inference time TabTune hands raw mixed-type frames straight
to the vendored engine.  This preprocessor therefore stays deliberately
minimal and exists for the two reasons every other raw-frame TabTune
preprocessor (TabFM / XRFM) does:

1.  it fits a ``LabelEncoder`` on the target so ``TabularPipeline.evaluate``
    and the probability-alignment code can map predictions back to the
    original label space; and
2.  it produces a clean **numeric** ``float32`` feature matrix (via the shared
    :class:`ILTMEpisodeFeaturizer`) that the ``TuningManager`` uses to build
    episodic support/query tensors during fine-tuning.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from ..models.iltm.episode_features import ILTMEpisodeFeaturizer

logger = logging.getLogger(__name__)


class ILTMPreprocessor(BaseEstimator, TransformerMixin):
    """Minimal iLTM preprocessing: label bookkeeping + FT-ready numerics."""

    def __init__(self, task_type: str = "classification", clip_sigma: float = 4.0):
        self.task_type = task_type
        self.clip_sigma = clip_sigma

        self.featurizer_ = None
        self.label_encoder_ = None
        self.categorical_cols_ = []
        self.numerical_cols_ = []
        self.all_feature_cols_ = []

    def fit(self, X, y=None):
        logger.info("[ILTM] Fitting iLTM preprocessor (task=%s)", self.task_type)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.all_feature_cols_ = X.columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        self.featurizer_ = ILTMEpisodeFeaturizer(clip_sigma=self.clip_sigma)
        self.featurizer_.fit(X)

        if self.task_type == "classification" and y is not None:
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.all_feature_cols_ or None)

        X_final = self.featurizer_.transform(X)

        if y is not None:
            if self.task_type == "classification" and self.label_encoder_ is not None:
                y_final = self.label_encoder_.transform(y)
            else:
                y_final = np.asarray(y, dtype=float).ravel()
            return X_final, y_final
        return X_final

    def get_summary(self):
        return {
            "iLTM Preprocessing": {
                "description": "Raw frames go straight to the vendored iLTM engine "
                "(it scales features, encodes categoricals and imputes missing "
                "values internally). TabTune only ordinal-encodes + standardises "
                f"(clip ±{self.clip_sigma}σ) to build fine-tuning episodes.",
                "details": [
                    f"{len(self.categorical_cols_)} categorical, "
                    f"{len(self.numerical_cols_)} numerical columns.",
                ],
            }
        }

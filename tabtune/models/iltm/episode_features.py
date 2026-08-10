"""
Numeric episode featurizer shared by the iLTM classification and regression
TabTune wrappers.

iLTM handles feature scaling, categorical features and missing values
internally at inference time (raw mixed-type frames go straight to the
vendored engine).  For **episodic fine-tuning** of the hypernetwork the
TuningManager needs a plain numeric ``float32`` matrix to build support/query
tensors from, exactly like the TabFM episode path.  This featurizer produces
that matrix with the same light recipe iLTM's own robust preprocessing is
built on: ordinal-encoded categoricals, mean-imputed + standardised numerics
and a +-clip against outliers.  The model's InitialTransformationBlock
(random features -> PCA -> normalisation) is data-dependent and recomputed
inside every forward pass, so this light TabTune-side encoding is all that is
required.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


class ILTMEpisodeFeaturizer:
    """Ordinal-encode categoricals, impute + standardise, clip, cast float32."""

    def __init__(self, clip_sigma: float = 4.0):
        self.clip_sigma = clip_sigma
        self.all_feature_cols_ = []
        self.categorical_cols_ = []
        self.ordinal_encoder_ = None
        self.imputer_ = None
        self.scaler_ = None
        self._is_fitted = False

    def fit(self, X):
        X = self._as_frame(X)
        self.all_feature_cols_ = X.columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()

        X_num = self._to_numeric(X, fit=True)
        self.imputer_ = SimpleImputer(strategy="mean")
        X_imp = self.imputer_.fit_transform(X_num)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_imp)
        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("ILTMEpisodeFeaturizer must be fitted before transform().")
        X = self._as_frame(X)
        X_num = self._to_numeric(X, fit=False)
        X_scaled = self.scaler_.transform(self.imputer_.transform(X_num))
        if self.clip_sigma and self.clip_sigma > 0:
            X_scaled = np.clip(X_scaled, -self.clip_sigma, self.clip_sigma)
        return np.nan_to_num(X_scaled).astype(np.float32)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X), columns=self.all_feature_cols_ or None)

    def _to_numeric(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        X = X.copy()
        if self.categorical_cols_:
            if fit:
                self.ordinal_encoder_ = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                X[self.categorical_cols_] = self.ordinal_encoder_.fit_transform(
                    X[self.categorical_cols_].astype(str)
                )
            else:
                X[self.categorical_cols_] = self.ordinal_encoder_.transform(
                    X[self.categorical_cols_].astype(str)
                )
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X.to_numpy(dtype=float)

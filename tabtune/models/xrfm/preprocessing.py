"""
Feature encoding shared by the XRFM classification and regression wrappers.

xRFM is a kernel method that consumes dense ``float32`` matrices.  Following
the upstream README's recommended preprocessing:

* **numerical** columns are mean-imputed and standardised (``StandardScaler``);
* **categorical** columns are one-hot encoded (``handle_unknown='ignore'``) and
  are deliberately **not** standardised (identity encoding vectors);
* an ``'ordinal'`` mode is also available for very high-cardinality data.

The encoder is fully picklable (no lambdas / open handles) so the whole
pipeline survives ``joblib.dump``.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


class XRFMFeatureEncoder(BaseEstimator, TransformerMixin):
    """Encode a raw mixed-type frame into the dense float32 matrix xRFM expects."""

    def __init__(self, categorical_encoding: str = "onehot", clip_sigma: float = 4.0):
        if categorical_encoding not in ("onehot", "ordinal"):
            raise ValueError(
                f"XRFM categorical_encoding must be 'onehot' or 'ordinal', got '{categorical_encoding}'."
            )
        self.categorical_encoding = categorical_encoding
        self.clip_sigma = clip_sigma

        self.numerical_cols_ = []
        self.categorical_cols_ = []
        self.all_feature_cols_ = []
        self.imputer_ = None
        self.scaler_ = None
        self.cat_encoder_ = None
        self.n_output_features_ = None

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.all_feature_cols_ = X.columns.tolist()
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()

        if self.numerical_cols_:
            self.imputer_ = SimpleImputer(strategy="mean")
            X_num = self.imputer_.fit_transform(X[self.numerical_cols_])
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_num)

        if self.categorical_cols_:
            if self.categorical_encoding == "onehot":
                self.cat_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                self.cat_encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.cat_encoder_.fit(X[self.categorical_cols_].astype(str))

        self.n_output_features_ = self.transform(X).shape[1]
        logger.debug(
            "[XRFM] Feature encoder fitted: %d numerical, %d categorical -> %d features",
            len(self.numerical_cols_), len(self.categorical_cols_), self.n_output_features_,
        )
        return self

    def transform(self, X):
        X = self._as_frame(X)
        blocks = []
        if self.numerical_cols_:
            X_num = self.imputer_.transform(X[self.numerical_cols_])
            X_num = self.scaler_.transform(X_num)
            if self.clip_sigma and self.clip_sigma > 0:
                X_num = np.clip(X_num, -self.clip_sigma, self.clip_sigma)
            blocks.append(X_num)
        if self.categorical_cols_:
            X_cat = self.cat_encoder_.transform(X[self.categorical_cols_].astype(str))
            blocks.append(np.asarray(X_cat, dtype=float))
        if not blocks:
            raise ValueError("XRFMFeatureEncoder received a frame with no usable columns.")
        X_out = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
        return np.nan_to_num(X_out).astype(np.float32)

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        cols = self.all_feature_cols_ or None
        arr = np.asarray(X)
        if cols is not None and arr.shape[1] != len(cols):
            cols = None
        return pd.DataFrame(arr, columns=cols)

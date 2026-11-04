"""
TabFM (Google) model-aware preprocessor for TabTune.

TabFM performs its own ordinal encoding + numeric scaling internally, so at
**inference** time TabTune hands raw mixed-type frames straight to the upstream
estimator (best zero-shot accuracy).  This preprocessor exists for two reasons
that mirror every other TabTune TFM preprocessor:

1.  it fits a ``LabelEncoder`` on the target so ``TabularPipeline.evaluate`` and
    the probability-alignment code can map predictions back to the original
    label space; and
2.  it produces a clean **numeric** feature matrix (ordinal-encoded categoricals
    + standardised numerics + outlier clipping) that the ``TuningManager`` uses
    to build episodic support/query tensors during fine-tuning -- the same
    convention TabTune already uses for TabICL / TabDPT / Mitra.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


class TabFMPreprocessor(BaseEstimator, TransformerMixin):
    """Light, TabFM-style preprocessing that also produces FT-ready numerics."""

    def __init__(self, task_type: str = "classification", clip_sigma: float = 4.0):
        self.task_type = task_type
        self.clip_sigma = clip_sigma

        self.categorical_cols_ = []
        self.numerical_cols_ = []
        self.all_feature_cols_ = []
        self.ordinal_encoder_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.label_encoder_ = None

    def fit(self, X, y=None):
        logger.info("[TabFM] Fitting TabFM preprocessor (task=%s)", self.task_type)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.all_feature_cols_ = X.columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        X_num = self._to_numeric(X, fit=True)

        self.imputer_ = SimpleImputer(strategy="mean")
        X_imp = self.imputer_.fit_transform(X_num)

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_imp)

        if self.task_type == "classification" and y is not None:
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.all_feature_cols_ or None)

        X_num = self._to_numeric(X, fit=False)
        X_imp = self.imputer_.transform(X_num)
        X_scaled = self.scaler_.transform(X_imp)
        # Clip outliers (TabFM-style robustness) then de-NaN to float32.
        if self.clip_sigma and self.clip_sigma > 0:
            X_scaled = np.clip(X_scaled, -self.clip_sigma, self.clip_sigma)
        X_final = np.nan_to_num(X_scaled).astype(np.float32)

        if y is not None:
            if self.task_type == "classification" and self.label_encoder_ is not None:
                y_final = self.label_encoder_.transform(y)
            else:
                y_final = np.asarray(y, dtype=float).ravel()
            return X_final, y_final
        return X_final

    def _to_numeric(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        """Ordinal-encode categoricals, keep numerics; return a numeric frame."""
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
        # Coerce any residual object columns.
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X.to_numpy(dtype=float)

    def get_summary(self):
        return {
            "TabFM Preprocessing": {
                "description": "Ordinal-encoded categoricals, standardised numerics, "
                f"clipped at ±{self.clip_sigma}σ. Raw frames are used at inference; "
                "these numerics feed episodic fine-tuning.",
                "details": [
                    f"{len(self.categorical_cols_)} categorical, "
                    f"{len(self.numerical_cols_)} numerical columns.",
                ],
            }
        }

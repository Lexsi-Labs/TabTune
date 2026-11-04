import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)


def _fix_dtypes(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """
    Standardizes input to a pandas DataFrame with corrected types.

    Inspired by TabPFN's internal `_fix_dtypes` + type handling docs:
      * Always convert to a pandas DataFrame
      * Normalize numeric / boolean columns to float64
      * Normalize string-like columns to plain Python objects and ensure
        missing values are regular `np.nan` (so sklearn encoders behave).
    """
    # Ensure DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Work on a copy to avoid mutating caller data
    X = X.copy()
    X = X.convert_dtypes()

    # Normalize string-like columns (string / object) and force NaNs to np.nan
    string_like_cols = X.select_dtypes(include=["string", "object"]).columns
    if len(string_like_cols) > 0:
        # Convert to Python objects and replace pandas.NA / None with np.nan
        X[string_like_cols] = X[string_like_cols].astype("object")
        X[string_like_cols] = X[string_like_cols].where(
            ~X[string_like_cols].isna(), np.nan
        )

    # Treat numeric + boolean columns as numeric features and cast to float64
    numeric_cols = X.select_dtypes(include=["number", "boolean", "bool"]).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].astype("float64")

    return X


def _infer_categorical_columns(
    X: pd.DataFrame,
    *,
    max_unique_for_categorical: int,
    min_unique_for_numerical: int,
    min_samples_for_inference: int,
) -> List[str]:
    """
    Infer which columns should be treated as categorical.

    Mirrors the heuristics described in the TabPFN docs: 
      * Columns with dtype 'category', 'string', 'object' or 'bool' are
        always considered categorical.
      * For datasets with at least `min_samples_for_inference` samples,
        numeric columns are also considered categorical when:
            - unique values ≤ `max_unique_for_categorical`, or
            - unique values  < `min_unique_for_numerical`.

    The returned list preserves the original column order.
    """
    # "Obvious" categoricals by dtype
    categorical = set(
        X.select_dtypes(
            include=["category", "string", "object", "boolean", "bool"]
        ).columns
    )

    n_samples = X.shape[0]

    # Only apply numeric-heuristics on sufficiently large datasets
    if n_samples >= min_samples_for_inference:
        for col in X.columns:
            if col in categorical:
                continue
            series = X[col]
            # Ignore all-NaN columns
            n_unique = series.nunique(dropna=True)
            if n_unique == 0:
                continue
            if n_unique <= max_unique_for_categorical or n_unique < min_unique_for_numerical:
                categorical.add(col)

    # Preserve original order
    return [col for col in X.columns if col in categorical]


def _get_ordinal_encoder() -> OrdinalEncoder:
    """
    Creates the OrdinalEncoder configuration used for categorical features.

    Note: we no longer wrap this in a ColumnTransformer to avoid the
    column-reordering bug that affects missing-value handling and feature
    attribution in the TabPFN pipeline. 
    """
    return OrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,
    )


class TabPFNPreprocessor(BaseEstimator, TransformerMixin):
    """
    Initial preprocessing (dtype fixing, categorical inference, ordinal encoding)
    required by the TabPFN model before any additional scaling.

    Key behaviors aligned with TabPFN's preprocessing docs: 
      * Robust dtype normalization (DataFrame, float64 numerics)
      * Heuristic inference of categorical features via unique-value rules
      * Ordinal encoding of categoricals, with original column order preserved
    """

    def __init__(
        self,
        max_unique_for_categorical: int = 30,
        min_unique_for_numerical: int = 4,
        min_samples_for_inference: int = 100,
    ):
        # Heuristic thresholds (match TabPFN defaults)
        self.max_unique_for_categorical = max_unique_for_categorical
        self.min_unique_for_numerical = min_unique_for_numerical
        self.min_samples_for_inference = min_samples_for_inference

        # Fitted attributes (set during fit)
        self.encoder_: OrdinalEncoder | None = None
        self.original_columns_: List[str] | None = None
        self.original_dtypes_: Dict | None = None
        self.categorical_columns_: List[str] | None = None
        self.numeric_columns_: List[str] | None = None
        self._is_fitted: bool = False

    def fit(self, X, y=None):
        logger.info("Fitting TabPFN Preprocessor")
        X_fixed = _fix_dtypes(X)

        # Keep track of original schema
        self.original_columns_ = list(X_fixed.columns)
        self.original_dtypes_ = X_fixed.dtypes.to_dict()

        # Infer categorical vs numeric columns
        self.categorical_columns_ = _infer_categorical_columns(
            X_fixed,
            max_unique_for_categorical=self.max_unique_for_categorical,
            min_unique_for_numerical=self.min_unique_for_numerical,
            min_samples_for_inference=self.min_samples_for_inference,
        )
        self.numeric_columns_ = [
            col for col in self.original_columns_ if col not in self.categorical_columns_
        ]

        # Fit an OrdinalEncoder on categorical columns (if any)
        if self.categorical_columns_:
            self.encoder_ = _get_ordinal_encoder()
            self.encoder_.fit(X_fixed[self.categorical_columns_])
        else:
            self.encoder_ = None

        self._is_fitted = True
        logger.info("TabPFN Preprocessor fitted successfully.")
        return self

    def transform(self, X, y=None):
        if not self._is_fitted:
            raise RuntimeError("You must call fit() before calling transform().")

        X_fixed = _fix_dtypes(X)

        # Make sure we have exactly the same columns, in the same order as during fit
        X_fixed = X_fixed.reindex(columns=self.original_columns_)

        n_samples = X_fixed.shape[0]
        data: Dict[str, np.ndarray] = {}

        # 1) Encode categorical columns back into their original positions
        if self.categorical_columns_:
            if self.encoder_ is None:
                raise RuntimeError(
                    "Internal error: encoder_ is None but categorical_columns_ is not empty."
                )
            X_cat = X_fixed[self.categorical_columns_]
            X_cat_enc = self.encoder_.transform(X_cat)

            for j, col in enumerate(self.categorical_columns_):
                data[col] = X_cat_enc[:, j]

        # 2) Pass numeric columns through (already float64 from _fix_dtypes)
        for col in self.numeric_columns_:
            series = X_fixed[col]
            if series.isna().all():
                # Column is completely missing at transform time → fill with NaNs
                data[col] = np.full(shape=n_samples, fill_value=np.nan, dtype=np.float64)
            else:
                data[col] = series.to_numpy(dtype=np.float64, copy=False)

        # Build final DataFrame with stable column order
        X_final = pd.DataFrame(data, index=X_fixed.index)[self.original_columns_]

        if y is not None:
            return X_final, y
        return X_final

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    # --- UPDATED SUMMARY ---
    def get_summary(self) -> dict:
        """
        Returns a rich dictionary with column-level details for each processing step.
        """
        if not self._is_fitted:
            return {"error": "Preprocessor has not been fitted yet."}

        original_columns = self.original_columns_ or []
        original_dtypes = self.original_dtypes_ or {}
        categorical_cols = set(self.categorical_columns_ or [])
        numeric_cols = set(self.numeric_columns_ or [])

        # Per-column breakdown
        columns_summary: dict[str, dict] = {}
        for col in original_columns:
            role = "categorical" if col in categorical_cols else "numeric"
            transformations = ["convert_to_float64"]
            if role == "categorical":
                transformations.insert(0, "ordinal_encode")

            columns_summary[col] = {
                "original_dtype": str(original_dtypes.get(col)),
                "role": role,
                "transformations": transformations,
            }

        summary = {
            "Data Type Standardization": {
                "description": (
                    "Ensures input is a pandas DataFrame and normalizes dtypes. "
                    "Numeric and boolean columns are cast to float64."
                ),
                "details": [
                    f"{len(original_columns)} columns after dtype normalization.",
                    "Numeric / boolean columns use float64; "
                    "string / object / categorical columns stay as Python objects "
                    "until encoding.",
                ],
            },
            "Categorical Inference": {
                "description": (
                    "Automatically infers which features are treated as categorical "
                    "using TabPFN-style unique-value heuristics."
                ),
                "details": [
                    f"Inferred {len(categorical_cols)} categorical columns: "
                    f"{sorted(categorical_cols)}",
                    f"{len(numeric_cols)} columns treated as numeric: "
                    f"{sorted(numeric_cols)}",
                    (
                        "Numeric columns become categorical when the dataset has at "
                        f"least {self.min_samples_for_inference} samples and the "
                        f"number of unique values is ≤ {self.max_unique_for_categorical} "
                        f"or < {self.min_unique_for_numerical}."
                    ),
                ],
            },
            "Feature Encoding": {
                "description": (
                    "Applies sklearn.OrdinalEncoder to categorical features and passes "
                    "numeric features through unchanged, while preserving original "
                    "column order (no ColumnTransformer reordering)."
                ),
                "details": [
                    (
                        "OrdinalEncoder configuration: categories='auto', dtype=float64, "
                        "handle_unknown='use_encoded_value', unknown_value=-1, "
                        "encoded_missing_value=np.nan."
                    ),
                    (
                        "Encoded categorical values are written back into their original "
                        "columns so downstream steps can rely on stable feature "
                        "positions."
                    ),
                ],
            },
            "Columns": columns_summary,
        }

        return summary

"""Learn a feature schema once and reuse its categories and column selection."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class FeatureEncoder:
    """Persist the input schema and ordinal encoder.

    ``categories`` and ``continuous`` are input-column indices, not category
    values. Encoded columns are ordered categorical first, then continuous;
    ``retained`` is a varying-feature mask in that encoded order. These field
    names are retained for compatibility with saved fitted states.
    """

    names: tuple | None
    width: int
    categories: tuple[int, ...]
    continuous: tuple[int, ...]
    encoder: OrdinalEncoder | None
    retained: np.ndarray

    @classmethod
    def fit(cls, table) -> tuple["FeatureEncoder", np.ndarray]:
        """Infer column kinds and return the schema plus nonconstant encoded features."""
        is_dataframe = isinstance(table, pd.DataFrame)
        array = np.asarray(table)
        if array.ndim != 2 or min(array.shape) == 0:
            raise ValueError("Features must be a nonempty two-dimensional table")
        categorical_indices = []
        for index in range(array.shape[1]):
            dtype = table.dtypes.iloc[index] if is_dataframe else array.dtype
            # Object arrays share one dtype; infer each column without treating
            # numeric-looking strings as continuous measurements.
            if not is_dataframe and pd.api.types.is_object_dtype(dtype):
                inferred = pd.api.types.infer_dtype(array[:, index], skipna=True)
                if inferred in {"integer", "floating", "mixed-integer-float", "decimal"}:
                    continue
            if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
                    raise TypeError("Date and duration features must be encoded before fitting")
                categorical_indices.append(index)
        continuous_indices = tuple(i for i in range(array.shape[1]) if i not in categorical_indices)
        schema = cls(
            names=tuple(table.columns) if is_dataframe else None,
            width=array.shape[1],
            categories=tuple(categorical_indices),
            continuous=continuous_indices,
            encoder=None,
            retained=np.ones(array.shape[1], dtype=bool),
        )
        if categorical_indices:
            schema.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
                dtype=np.float64,
            )
            schema.encoder.fit(schema._categorical(table))
        numeric = schema.encode(table)
        if len(numeric) > 1:
            schema.retained = np.array([np.unique(column).size > 1 for column in numeric.T])
        if not schema.retained.any():
            raise ValueError("At least one varying feature is required")
        return schema, numeric[:, schema.retained]

    def _categorical(self, table) -> np.ndarray:
        """Unify pandas/NumPy missing markers before ordinal encoding."""
        values = np.asarray(table, dtype=object)[:, self.categories].copy()
        values[pd.isna(values)] = np.nan
        return values

    def encode(self, table) -> np.ndarray:
        """Validate the input schema and return categorical-first numeric columns.

        Unknown categories become NaN so the model's missing-value embedding
        handles them. Constant-column removal is applied by ``transform``.
        """
        if isinstance(table, pd.DataFrame):
            if self.names is not None and tuple(table.columns) != self.names:
                raise ValueError("Prediction columns must match fit columns in name and order")
            numerical = table.iloc[:, list(self.continuous)].to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            values = np.asarray(table)
            if values.ndim != 2 or values.shape[1] != self.width:
                raise ValueError(f"Expected {self.width} feature columns")
            numerical = values[:, self.continuous]
            if pd.api.types.is_object_dtype(numerical.dtype):
                numerical = np.where(pd.isna(numerical), np.nan, numerical)
            numerical = np.asarray(numerical, dtype=np.float64)
        if np.asarray(table).shape[1] != self.width:
            raise ValueError(f"Expected {self.width} feature columns")
        parts = [self.encoder.transform(self._categorical(table))] if self.encoder is not None else []
        parts.append(numerical)
        result = np.concatenate(parts, axis=1)
        if np.isinf(result).any():
            raise ValueError("Infinite feature values are not supported")
        return result

    def transform(self, table) -> np.ndarray:
        """Apply the fitted mappings and varying-feature mask without refitting."""
        return self.encode(table)[:, self.retained]

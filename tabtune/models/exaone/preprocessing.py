"""Numeric preprocessing for classification tables."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import QuantileTransformer


@dataclass(frozen=True)
class PreparedBatch:
    """A transformed table and its relationship to the source table."""

    values: np.ndarray
    missing_mask: np.ndarray
    source_columns: np.ndarray


@dataclass(frozen=True)
class _FittedState:
    input_width: int
    fill_values: np.ndarray
    columns: np.ndarray
    centers: np.ndarray
    divisors: np.ndarray
    all_missing: np.ndarray
    quantile_mapper: QuantileTransformer | None


class TabularPreprocessor:
    """Fit and apply numeric classification-table preprocessing."""

    def __init__(
        self,
        feature_limit: int,
        *,
        gaussianize: bool = False,
        feature_count_rescale: bool = False,
        fixed_columns: Sequence[int] | None = None,
    ) -> None:
        if isinstance(feature_limit, (bool, np.bool_)) or not isinstance(
            feature_limit, Integral
        ):
            raise ValueError("feature_limit must be a positive integer")
        if feature_limit <= 0:
            raise ValueError("feature_limit must be a positive integer")

        checked_columns: tuple[int, ...] | None = None
        if fixed_columns is not None:
            try:
                candidates = tuple(fixed_columns)
            except TypeError as exc:
                raise ValueError("fixed_columns must be an integer sequence") from exc
            if not candidates:
                raise ValueError("fixed_columns cannot be empty")
            previous = -1
            normalized: list[int] = []
            for candidate in candidates:
                if isinstance(candidate, (bool, np.bool_)) or not isinstance(
                    candidate, Integral
                ):
                    raise ValueError("fixed_columns entries must be integers")
                column = int(candidate)
                if column <= previous:
                    raise ValueError("fixed_columns must be strictly increasing")
                normalized.append(column)
                previous = column
            if len(normalized) > int(feature_limit):
                raise ValueError("fixed_columns exceeds feature_limit")
            checked_columns = tuple(normalized)

        self.feature_limit = int(feature_limit)
        self.gaussianize = bool(gaussianize)
        self.feature_count_rescale = bool(feature_count_rescale)
        self.fixed_columns = checked_columns
        self._state: _FittedState | None = None

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    @property
    def input_feature_count(self) -> int:
        return self._require_state().input_width

    @property
    def selected_columns(self) -> np.ndarray:
        return self._require_state().columns.copy()

    def fit(self, features: np.ndarray, targets: Any) -> TabularPreprocessor:
        matrix = self._validate_features(features, allow_zero_rows=False)
        labels = self._validate_targets(targets, matrix.shape[0])

        width = matrix.shape[1]
        if self.fixed_columns is not None:
            columns = np.asarray(self.fixed_columns, dtype=np.int64)
            if columns[-1] >= width:
                raise ValueError("fixed column is outside the feature matrix")
        else:
            columns = np.arange(width, dtype=np.int64)

        working = self._floating_copy(matrix)
        missing = np.isnan(working)
        observed_counts = np.sum(~missing, axis=0)
        sums = np.sum(np.where(missing, 0, working), axis=0)
        fill_values = np.zeros(width, dtype=working.dtype)
        np.divide(sums, observed_counts, out=fill_values, where=observed_counts != 0)
        if np.any(missing):
            working[missing] = np.broadcast_to(fill_values, working.shape)[missing]

        if self.fixed_columns is None and width > self.feature_limit:
            try:
                selector = SelectKBest(score_func=f_classif, k=self.feature_limit)
                selector.fit(working, labels)
                columns = selector.get_support(indices=True).astype(np.int64, copy=False)
            except Exception as exc:
                raise ValueError("targets cannot be used for feature selection") from exc

        selected = working[:, columns].copy()
        all_missing = observed_counts[columns] == 0
        mapper: QuantileTransformer | None = None
        if self.gaussianize:
            mapper = QuantileTransformer(
                n_quantiles=min(matrix.shape[0], 1000),
                output_distribution="normal",
                random_state=0,
            )
            selected = mapper.fit_transform(selected)
            if np.any(all_missing):
                selected[:, all_missing] = 0

        centers = np.mean(selected, axis=0, dtype=selected.dtype)
        scales = np.std(selected, axis=0, dtype=selected.dtype)
        substitute = np.asarray(1e-6, dtype=scales.dtype)
        divisors = np.where(scales == 0, substitute, scales)

        # Commit only after every validation and fitting operation has succeeded.
        self._state = _FittedState(
            input_width=width,
            fill_values=fill_values.copy(),
            columns=columns.copy(),
            centers=np.asarray(centers).copy(),
            divisors=np.asarray(divisors).copy(),
            all_missing=all_missing.copy(),
            quantile_mapper=mapper,
        )
        return self

    def transform(self, features: np.ndarray) -> PreparedBatch:
        state = self._require_state()
        matrix = self._validate_features(features, allow_zero_rows=True)
        if matrix.shape[1] != state.input_width:
            raise ValueError("feature width differs from fitted width")

        working = self._floating_copy(matrix)
        missing_full = np.isnan(working)
        if np.any(missing_full):
            working[missing_full] = np.broadcast_to(
                state.fill_values, working.shape
            )[missing_full]

        missing = missing_full[:, state.columns].copy()
        selected = working[:, state.columns].copy()
        if state.quantile_mapper is not None and selected.shape[0] != 0:
            selected = state.quantile_mapper.transform(selected)
            if np.any(state.all_missing):
                selected[:, state.all_missing] = 0

        values = (selected - state.centers) / state.divisors
        if self.feature_count_rescale:
            factor = np.asarray(
                self.feature_limit / state.columns.size, dtype=values.dtype
            )
            values = values * factor
        values = np.array(values, copy=True)
        values[missing] = np.nan

        return PreparedBatch(
            values=values,
            missing_mask=missing,
            source_columns=state.columns.copy(),
        )

    def fit_transform(self, features: np.ndarray, targets: Any) -> PreparedBatch:
        self.fit(features, targets)
        return self.transform(features)

    def _require_state(self) -> _FittedState:
        if self._state is None:
            raise RuntimeError("preprocessor has not been fitted")
        return self._state

    @staticmethod
    def _floating_copy(matrix: np.ndarray) -> np.ndarray:
        if matrix.dtype == np.float32 or matrix.dtype == np.float64:
            return matrix.copy()
        if np.issubdtype(matrix.dtype, np.floating):
            return matrix.copy()
        return matrix.astype(np.float64, copy=True)

    @staticmethod
    def _validate_features(
        features: np.ndarray, *, allow_zero_rows: bool
    ) -> np.ndarray:
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a NumPy array")
        dtype = features.dtype
        supported = np.issubdtype(dtype, np.bool_) or (
            np.issubdtype(dtype, np.number)
            and not np.issubdtype(dtype, np.complexfloating)
        )
        if not supported:
            raise TypeError("features must have a real numeric dtype")
        if features.ndim != 2:
            raise ValueError("features must be rank two")
        if features.shape[1] == 0 or (features.shape[0] == 0 and not allow_zero_rows):
            raise ValueError("features has an empty axis")
        if np.any(np.isinf(features)):
            raise ValueError("features cannot contain infinity")
        return features

    @staticmethod
    def _validate_targets(targets: Any, row_count: int) -> np.ndarray:
        try:
            labels = np.asarray(targets)
        except Exception as exc:
            raise ValueError("targets must be one-dimensional") from exc
        if labels.ndim != 1 or labels.shape[0] != row_count:
            raise ValueError("targets must have one entry per row")
        if TabularPreprocessor._contains_missing_target(labels):
            raise ValueError("targets cannot contain missing values")
        try:
            if np.unique(labels).size < 2:
                raise ValueError("targets must contain at least two classes")
        except TypeError as exc:
            raise ValueError("targets must contain comparable classes") from exc
        return labels.copy()

    @staticmethod
    def _contains_missing_target(labels: np.ndarray) -> bool:
        if np.issubdtype(labels.dtype, np.inexact):
            return bool(np.any(np.isnan(labels)))
        if np.issubdtype(labels.dtype, np.datetime64) or np.issubdtype(
            labels.dtype, np.timedelta64
        ):
            return bool(np.any(np.isnat(labels)))
        if labels.dtype.kind != "O":
            return False
        for value in labels:
            if value is None:
                return True
            try:
                unequal_to_self = value != value
                if isinstance(unequal_to_self, (bool, np.bool_)) and unequal_to_self:
                    return True
            except Exception:
                pass
        return False


__all__ = ["PreparedBatch", "TabularPreprocessor"]

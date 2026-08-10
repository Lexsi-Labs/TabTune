"""Error-correcting output codes for many-class classification."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class ECOCCodec:
    """Construct, encode with, and decode a fixed ECOC codebook."""

    def __init__(
        self,
        class_count: int,
        symbol_count: int,
        *,
        redundancy: int = 4,
        strategy: str = "rest",
        aggregation: str = "log_likelihood",
        retries: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        self._require_int("class_count", class_count)
        self._require_int("symbol_count", symbol_count)
        self._require_int("redundancy", redundancy)
        self._require_int("retries", retries)
        if seed is not None:
            self._require_int("seed", seed)

        if symbol_count < 2:
            raise ValueError("symbol_count must be at least 2")
        if class_count <= symbol_count:
            raise ValueError("class_count must exceed symbol_count")
        if redundancy < 1:
            raise ValueError("redundancy must be positive")
        if retries < 1:
            raise ValueError("retries must be positive")
        if seed is not None and not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed is outside the RandomState seed range")
        if strategy not in ("rest", "balanced"):
            raise ValueError("strategy must be 'rest' or 'balanced'")
        if aggregation not in ("log_likelihood", "average"):
            raise ValueError("unsupported aggregation")

        self._class_count = class_count
        self._symbol_count = symbol_count
        self._redundancy = redundancy
        self._strategy = strategy
        self._aggregation = aggregation
        self._retries = retries
        self._seed = seed
        self._rest_symbol = symbol_count - 1 if strategy == "rest" else None
        self._n_rows = self._row_count(class_count, symbol_count, redundancy)

        parent = np.random.RandomState(seed)
        attempts = retries if class_count <= 200 else 1
        best_matrix = None
        best_quality = None
        for _ in range(attempts):
            child_seed = int(parent.randint(0, 2**31 - 1))
            child = np.random.RandomState(child_seed)
            candidate = self._make_candidate(child)
            if class_count > 200:
                best_matrix = candidate
                break
            quality = self._quality(candidate)
            if best_quality is None or quality > best_quality:
                best_matrix = candidate
                best_quality = quality

        # Generation always produces a candidate because retries is positive.
        self._codebook = np.ascontiguousarray(best_matrix, dtype=np.int64)
        self._codebook.setflags(write=False)

    @staticmethod
    def _require_int(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a Python integer")

    @staticmethod
    def _row_count(class_count: int, symbol_count: int, redundancy: int) -> int:
        # Compute the ceiling logarithm with integers, avoiding rounding at powers.
        level = 0
        capacity = 1
        target = max(class_count, 2)
        while capacity < target:
            capacity *= symbol_count
            level += 1
        coverage = math.ceil(class_count / (symbol_count - 1))
        base = max(level, coverage)
        cap = max(base, 4 * max(level, 1))
        return max(base, min(base * max(redundancy, 1), cap))

    def _make_candidate(self, random: np.random.RandomState) -> np.ndarray:
        if self._strategy == "rest":
            return self._make_rest_candidate(random)
        return self._make_balanced_candidate(random)

    def _make_rest_candidate(self, random: np.random.RandomState) -> np.ndarray:
        matrix = np.full(
            (self._n_rows, self._class_count),
            self._symbol_count - 1,
            dtype=np.int64,
        )
        active_counts = np.zeros(self._class_count, dtype=np.int64)
        width = min(self._symbol_count - 1, self._class_count)
        for row in range(self._n_rows):
            priorities = active_counts + random.uniform(0.0, 0.1, self._class_count)
            chosen = np.argsort(priorities, kind="stable")[:width]
            symbols = random.permutation(self._symbol_count - 1)[:width]
            matrix[row, chosen] = symbols
            active_counts[chosen] += 1
        return matrix

    def _make_balanced_candidate(self, random: np.random.RandomState) -> np.ndarray:
        matrix = np.empty((self._n_rows, self._class_count), dtype=np.int64)
        focused_rows = min(self._class_count, self._n_rows)
        all_classes = np.arange(self._class_count)
        for row in range(focused_rows):
            matrix[row, row] = 0
            others = np.concatenate((all_classes[:row], all_classes[row + 1 :])).copy()
            random.shuffle(others)
            for symbol, group in enumerate(
                np.array_split(others, self._symbol_count - 1), start=1
            ):
                matrix[row, group] = symbol
        for row in range(focused_rows, self._n_rows):
            shuffled = all_classes.copy()
            random.shuffle(shuffled)
            for symbol, group in enumerate(np.array_split(shuffled, self._symbol_count)):
                matrix[row, group] = symbol
        return matrix

    @staticmethod
    def _quality(matrix: np.ndarray) -> tuple[int, float]:
        class_count = matrix.shape[1]
        counts = []
        for left in range(class_count - 1):
            distances = np.count_nonzero(
                matrix[:, left, None] != matrix[:, left + 1 :], axis=0
            )
            counts.extend(distances.tolist())
        values = np.asarray(counts, dtype=np.float64)
        return int(values.min()), float(values.mean())

    @property
    def class_count(self) -> int:
        return self._class_count

    @property
    def symbol_count(self) -> int:
        return self._symbol_count

    @property
    def redundancy(self) -> int:
        return self._redundancy

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def aggregation(self) -> str:
        return self._aggregation

    @property
    def retries(self) -> int:
        return self._retries

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def rest_symbol(self) -> Optional[int]:
        return self._rest_symbol

    @property
    def codebook(self) -> np.ndarray:
        return self._codebook.copy(order="C")

    def encode(self, labels: object) -> np.ndarray:
        try:
            values = np.asarray(labels)
        except Exception as exc:
            raise TypeError("labels must be an array-like collection of integers") from exc
        if values.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.bool_
        ):
            raise TypeError("labels must have an integer dtype")
        if values.size and (np.any(values < 0) or np.any(values >= self._class_count)):
            raise ValueError("label is outside the class range")
        labels_i = np.asarray(values, dtype=np.intp)
        return np.ascontiguousarray(self._codebook[:, labels_i], dtype=np.int64)

    def decode(self, row_probabilities: object) -> np.ndarray:
        try:
            probabilities = np.asarray(row_probabilities)
        except Exception as exc:
            raise TypeError("row probabilities must form a real array") from exc
        if probabilities.ndim != 3 or probabilities.shape[0] != self._n_rows or probabilities.shape[2] != self._symbol_count:
            raise ValueError(
                "row probabilities must have shape (n_rows, query_count, symbol_count)"
            )
        if not np.issubdtype(probabilities.dtype, np.number) or np.issubdtype(
            probabilities.dtype, np.complexfloating
        ):
            raise TypeError("row probabilities must be real numbers")
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("row probabilities must be finite")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("row probabilities must lie in [0, 1]")
        sums = probabilities.sum(axis=2)
        if not np.all(np.isclose(sums, 1.0, rtol=0.0, atol=1e-7)):
            raise ValueError("each symbol vector must sum to one")

        query_count = probabilities.shape[1]
        gathered = np.empty(
            (self._n_rows, query_count, self._class_count), dtype=np.float64
        )
        for row in range(self._n_rows):
            gathered[row] = probabilities[row][:, self._codebook[row]]

        if self._strategy == "rest":
            active = self._codebook != self._rest_symbol
        else:
            active = np.ones_like(self._codebook, dtype=bool)
        divisors = active.sum(axis=0).astype(np.float64)

        if self._aggregation == "log_likelihood":
            logged = np.log(np.clip(gathered, 1e-12, 1.0))
            scores = np.sum(logged * active[:, None, :], axis=0) / divisors
            scores -= np.max(scores, axis=1, keepdims=True)
            result = np.exp(scores)
            result /= result.sum(axis=1, keepdims=True)
        else:
            result = np.sum(gathered * active[:, None, :], axis=0) / divisors
            totals = result.sum(axis=1, keepdims=True)
            zero = totals[:, 0] == 0.0
            if np.any(zero):
                result[zero] = 1.0 / self._class_count
            nonzero = ~zero
            if np.any(nonzero):
                result[nonzero] /= np.maximum(1.0, totals[nonzero])

        return np.ascontiguousarray(result, dtype=np.float64)

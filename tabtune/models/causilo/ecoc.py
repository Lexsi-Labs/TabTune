"""Balanced active/rest codes and streaming class-probability reconstruction."""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


def _row_budget(classes: int, symbols: int) -> int:
    """Allocate redundant coverage for the candidate-search path."""
    coverage = (classes + symbols - 2) // (symbols - 1)
    digits, representable = 0, 1
    while representable < classes:
        digits += 1
        representable *= symbols
    minimum = max(digits, coverage)
    return min(4 * minimum, max(minimum, 4 * digits))


def _active_schedules(classes: int, symbols: int, rng: np.random.Generator) -> np.ndarray:
    """Build fifty balanced schedules together, without assigning output symbols."""
    schedules = np.zeros((50, _row_budget(classes, symbols), classes), dtype=bool)
    uses = np.zeros((len(schedules), classes), dtype=np.int64)
    trials = np.arange(len(schedules))[:, None]
    labels = np.broadcast_to(np.arange(classes), uses.shape)
    for row in range(schedules.shape[1]):
        # A stable usage sort preserves the shuffled order within each tie.
        shuffled = rng.permuted(labels, axis=1)
        counts = np.take_along_axis(uses, shuffled, axis=1)
        ranks = np.argsort(counts, axis=1, kind="stable")[:, : symbols - 1]
        selected = np.take_along_axis(shuffled, ranks, axis=1)
        schedules[trials, row, selected] = True
        uses[trials, selected] += 1
    return schedules


def _partition_codes(classes: int, symbols: int, rng: np.random.Generator) -> np.ndarray:
    """Partition one shuffled class order and spread the final row's overlap.

    Complete groups are disjoint. Padding draws from previous groups in round-
    robin order, minimizing repeated class pairs in the final group. With a
    ten-symbol head and more than 200 classes, no pair is active together twice.
    Padding classes also receive a different symbol on their second appearance.
    """
    width = symbols - 1
    full, remainder = divmod(classes, width)
    codes = np.full((full + bool(remainder), classes), width, dtype=np.int64)
    order = rng.permutation(classes)
    blocks = order[: full * width].reshape(full, width)
    for row, members in enumerate(blocks):
        codes[row, members] = rng.permutation(width)
    if remainder:
        padding = width - remainder
        group_order = rng.permutation(full)
        slots = np.arange(padding)
        donors = blocks[group_order[slots % full], slots // full]
        members = np.concatenate((order[full * width :], donors))
        assigned = rng.permutation(width)
        previous = codes[group_order[slots % full], donors]
        # Each donor forbids one cyclic shift. Fewer than width donors means
        # at least one shift gives every repeated class a different symbol.
        for shift in rng.permutation(width):
            shifted = (assigned + shift) % width
            if np.all(shifted[remainder:] != previous):
                codes[full, members] = shifted
                break
    return codes


def _separation(active: np.ndarray) -> tuple[int, float]:
    """Score pairwise Hamming distance using active-set co-occurrence counts.

    Active symbols in one row are distinct. Two class codes therefore differ
    exactly when at least one class is active, regardless of symbol permutation.
    """
    active = active.astype(np.int64)
    appearances = active.sum(axis=0)
    both_active = active.T @ active
    distance = appearances[:, None] + appearances[None, :] - both_active
    pairs = distance[np.triu_indices(active.shape[1], k=1)]
    return int(pairs.min()), float(pairs.mean())


def _search_codes(classes: int, symbols: int, rng: np.random.Generator) -> np.ndarray:
    """Select an active schedule by distance, then label only the winning schedule."""
    active = max(_active_schedules(classes, symbols, rng), key=_separation)
    codes = np.full(active.shape, symbols - 1, dtype=np.int64)
    # Distinct active symbols make Hamming scores independent of their order.
    labels = np.broadcast_to(np.arange(symbols - 1), (len(active), symbols - 1))
    codes[active] = rng.permuted(labels, axis=1).ravel()
    return codes


@dataclass(frozen=True)
class ECOCCodec:
    """Immutable, seeded output codes with the fixed Causilo many-class recipe."""

    class_count: int
    symbol_count: int
    random_state: int
    codebook: np.ndarray = field(init=False, repr=False)
    rest_symbol: int = field(init=False)

    def __post_init__(self) -> None:
        parameters = (self.class_count, self.symbol_count, self.random_state)
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in parameters
        ):
            raise ValueError("Class count, symbol count and random_state must be integers")
        if self.symbol_count < 2:
            raise ValueError("symbol_count must be at least two")
        if self.class_count <= self.symbol_count:
            raise ValueError("ECOC requires more classes than output symbols")
        if not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must fit in an unsigned 32-bit integer")

        classes, symbols = int(self.class_count), int(self.symbol_count)
        object.__setattr__(self, "rest_symbol", symbols - 1)
        rng = np.random.default_rng(int(self.random_state))
        if classes > 200:
            chosen = _partition_codes(classes, symbols, rng)
        else:
            chosen = _search_codes(classes, symbols, rng)
        chosen.setflags(write=False)
        object.__setattr__(self, "codebook", chosen)

    def __setstate__(self, state: dict) -> None:
        """Unpickling restores the fitted schedule rather than making a new one."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self.codebook.setflags(write=False)

    def encode(self, labels: np.ndarray) -> np.ndarray:
        indices = np.asarray(labels)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("ECOC labels must be a one-dimensional integer array")
        if np.any((indices < 0) | (indices >= self.class_count)):
            raise ValueError("ECOC label is outside the class range")
        return np.ascontiguousarray(np.take(self.codebook, indices, axis=1))

    def decode(self, row_probabilities: np.ndarray) -> np.ndarray:
        """Decode a dense (code rows, query rows, output symbols) array."""
        rows = np.asarray(row_probabilities)
        if rows.ndim != 3 or rows.shape[0] != len(self.codebook) or rows.shape[2] != self.symbol_count:
            raise ValueError("ECOC probabilities must have shape (code rows, query rows, output symbols)")
        return self.decode_rows(rows)

    def decode_rows(self, row_probabilities: Iterable[np.ndarray]) -> np.ndarray:
        """Accumulate only active log probabilities, consuming one code row at a time.

        Auxiliary storage is O(query rows * classes), not O(code rows * query
        rows * classes). Rest probabilities do not contribute to class scores.
        """
        scores = None
        appearances = np.zeros(self.class_count, dtype=np.int64)
        consumed = 0
        for index, values in enumerate(row_probabilities):
            if index >= len(self.codebook):
                raise ValueError("ECOC probabilities have too many code rows")
            probabilities = np.asarray(values, dtype=np.float64)
            if probabilities.ndim != 2 or probabilities.shape[1] != self.symbol_count:
                raise ValueError("Each ECOC probability row must have shape (query rows, output symbols)")
            if scores is None:
                scores = np.zeros((len(probabilities), self.class_count), dtype=np.float64)
            elif len(probabilities) != len(scores):
                raise ValueError("ECOC probability rows must have the same query count")
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                raise ValueError("ECOC probabilities must be finite and nonnegative")
            # Scaling before summing also accepts unnormalized finite weights
            # whose direct sum would overflow. Normal probabilities are unchanged.
            largest = probabilities.max(axis=1, keepdims=True)
            if np.any(largest <= 0):
                raise ValueError("ECOC probability rows must have positive sums")
            scaled = probabilities / largest
            normalized = scaled / scaled.sum(axis=1, keepdims=True)
            classes = np.flatnonzero(self.codebook[index] != self.rest_symbol)
            symbols = self.codebook[index, classes]
            scores[:, classes] += np.log(np.maximum(normalized[:, symbols], 1e-12))
            appearances[classes] += 1
            consumed += 1
        if consumed != len(self.codebook):
            raise ValueError("ECOC probabilities have too few code rows")
        scores /= appearances
        scores -= scores.max(axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=1, keepdims=True)
        return scores

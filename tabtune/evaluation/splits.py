"""Distribution-shift-aware cross-validation splitters.

Why this exists
---------------
TabTune's evaluation surface was entirely IID: ``train_test_split``, ``KFold``
and ``StratifiedKFold``. Every published evaluation of tabular foundation
models under distribution shift reaches the same conclusion - they degrade
systematically, and the ranking under an IID split does not predict the ranking
under a temporal or grouped one.

A model selected on a random split and deployed against next quarter's data has
been validated against the wrong question. These splitters ask the right one.

All classes follow the scikit-learn splitter protocol (``split``,
``get_n_splits``), so they drop into ``cross_validate``, ``GridSearchCV`` and
TabTune's own leaderboard without adaptation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TemporalSplit",
    "GroupedSplit",
    "StratifiedGroupedSplit",
    "resolve_split",
    "SPLIT_REGISTRY",
]


def _as_1d(values: Any, name: str) -> np.ndarray:
    """Coerce a column-like input to a 1-D numpy array."""
    if values is None:
        raise ValueError(f"{name} is required but was None")
    array = np.asarray(getattr(values, "values", values))
    if array.ndim != 1:
        array = array.ravel()
    return array


def _column(X: Any, column: str | None, explicit: Any, name: str) -> np.ndarray:
    """Resolve a split key from either an explicit array or a column of ``X``."""
    if explicit is not None:
        return _as_1d(explicit, name)
    if column is None:
        raise ValueError(
            f"{name} could not be resolved: pass either the array directly or "
            f"set the corresponding column name on the splitter"
        )
    if not hasattr(X, "columns"):
        raise TypeError(
            f"{name} was specified by column name {column!r}, which requires X "
            f"to be a pandas DataFrame; got {type(X).__name__}"
        )
    if column not in X.columns:
        raise KeyError(
            f"Column {column!r} not found in X. Available columns: "
            f"{list(X.columns)[:20]}{'...' if X.shape[1] > 20 else ''}"
        )
    return _as_1d(X[column], name)


class TemporalSplit:
    """Forward-chaining splits that never train on the future.

    Fold ``i`` trains on the earliest ``i + 1`` blocks and tests on block
    ``i + 1``, so the test set is always strictly newer than the training set.
    This is the split that matters for any model that will be deployed against
    data arriving after it was fitted.

    Args:
        n_splits: Number of folds.
        time_col: Column of ``X`` holding the ordering key. Values only need to
            be sortable, not datetimes.
        gap: Rows skipped between train and test, to model the lag between
            observing a label and being able to use it.
        max_train_size: Cap on training rows per fold. ``None`` grows the
            training window with each fold (expanding); a value makes it a
            sliding window.
        test_size: Fixed test-block size. ``None`` divides the data evenly.

    Example:
        >>> import pandas as pd
        >>> X = pd.DataFrame({"t": range(10), "f": range(10)})
        >>> splitter = TemporalSplit(n_splits=3, time_col="t")
        >>> for train_idx, test_idx in splitter.split(X):
        ...     print(len(train_idx), len(test_idx))
        4 2
        6 2
        8 2
    """

    #: Marks this splitter as producing non-IID folds, used by shift reporting.
    shift_type = "temporal"

    def __init__(
        self,
        n_splits: int = 5,
        *,
        time_col: str | None = None,
        gap: int = 0,
        max_train_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        self.n_splits = n_splits
        self.time_col = time_col
        self.gap = gap
        self.max_train_size = max_train_size
        self.test_size = test_size

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of folds (scikit-learn protocol)."""
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: Any = None, *, times: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_index, test_index)`` pairs ordered in time.

        Args:
            X: Feature matrix.
            y: Ignored; present for protocol compatibility.
            groups: Ignored.
            times: Explicit ordering key, overriding ``time_col``.

        Yields:
            Index arrays into ``X``.

        Raises:
            ValueError: If there are too few rows to build the requested folds.
        """
        n_samples = len(X)
        if self.time_col is not None or times is not None:
            keys = _column(X, self.time_col, times, "times")
            order = np.argsort(keys, kind="stable")
        else:
            # No key given: trust the existing row order, which is the common
            # case for already-sorted event logs.
            logger.debug("[TemporalSplit] No time column given; using row order.")
            order = np.arange(n_samples)

        test_size = self.test_size or (n_samples // (self.n_splits + 1))
        if test_size < 1:
            raise ValueError(
                f"Not enough rows ({n_samples}) for {self.n_splits} temporal folds; "
                f"each fold needs at least one test row. Reduce n_splits."
            )

        for fold in range(self.n_splits):
            test_start = n_samples - (self.n_splits - fold) * test_size
            test_stop = test_start + test_size
            train_stop = test_start - self.gap
            if train_stop <= 0:
                raise ValueError(
                    f"Not enough rows ({n_samples}) for {self.n_splits} temporal "
                    f"folds with gap={self.gap}. Reduce n_splits, test_size or gap."
                )
            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_stop - self.max_train_size)
            yield order[train_start:train_stop], order[test_start:test_stop]


class GroupedSplit:
    """Leave-groups-out splits: no group appears in both train and test.

    Use when rows cluster into units that must not leak across the split -
    patients, customers, sites, devices. An IID split of such data reports
    optimistic scores because the model has memorised the unit rather than
    learned the signal.

    Args:
        n_splits: Number of folds. Capped at the number of distinct groups.
        group_col: Column of ``X`` holding the group key.
        shuffle: Shuffle group assignment before partitioning.
        random_state: Seed used when ``shuffle`` is set.

    Example:
        >>> import pandas as pd
        >>> X = pd.DataFrame({"g": list("aabbcc"), "f": range(6)})
        >>> splitter = GroupedSplit(n_splits=3, group_col="g")
        >>> all(set(X.g[tr]).isdisjoint(set(X.g[te])) for tr, te in splitter.split(X))
        True
    """

    shift_type = "grouped"

    def __init__(
        self,
        n_splits: int = 5,
        *,
        group_col: str | None = None,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.group_col = group_col
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of folds (scikit-learn protocol)."""
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_index, test_index)`` pairs with disjoint groups.

        Args:
            X: Feature matrix.
            y: Ignored.
            groups: Explicit group labels, overriding ``group_col``.

        Yields:
            Index arrays into ``X``.

        Raises:
            ValueError: If there are fewer distinct groups than folds.
        """
        keys = _column(X, self.group_col, groups, "groups")
        unique = np.unique(keys)
        if len(unique) < self.n_splits:
            raise ValueError(
                f"GroupedSplit needs at least n_splits={self.n_splits} distinct "
                f"groups, found {len(unique)}. Reduce n_splits or check "
                f"group_col={self.group_col!r}."
            )

        order = np.arange(len(unique))
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(order)

        folds = np.array_split(order, self.n_splits)
        indices = np.arange(len(keys))
        for fold in folds:
            held_out = set(unique[fold].tolist())
            mask = np.array([k in held_out for k in keys])
            yield indices[~mask], indices[mask]


class StratifiedGroupedSplit(GroupedSplit):
    """Grouped splits that also try to balance the class distribution.

    Groups are assigned to folds greedily, always placing the next group in
    whichever fold currently has the fewest samples of that group's majority
    class. The result is not exactly stratified - it cannot be, since groups are
    atomic - but it avoids the pathological case where an entire minority class
    lands in one fold.

    Args:
        n_splits: Number of folds.
        group_col: Column of ``X`` holding the group key.
        random_state: Seed used to break ties deterministically.
    """

    shift_type = "grouped"

    def __init__(
        self,
        n_splits: int = 5,
        *,
        group_col: str | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_splits, group_col=group_col, shuffle=False, random_state=random_state)

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_index, test_index)`` pairs balancing classes per fold."""
        if y is None:
            logger.debug(
                "[StratifiedGroupedSplit] No y given; falling back to GroupedSplit."
            )
            yield from super().split(X, y, groups)
            return

        keys = _column(X, self.group_col, groups, "groups")
        labels = _as_1d(y, "y")
        unique = np.unique(keys)
        if len(unique) < self.n_splits:
            raise ValueError(
                f"StratifiedGroupedSplit needs at least n_splits={self.n_splits} "
                f"distinct groups, found {len(unique)}."
            )

        # Majority class and size per group, largest groups placed first so the
        # greedy assignment has room to correct itself later.
        summary = []
        for group in unique:
            mask = keys == group
            group_labels = labels[mask]
            values, counts = np.unique(group_labels, return_counts=True)
            summary.append((group, int(mask.sum()), values[int(np.argmax(counts))]))
        summary.sort(key=lambda item: (-item[1], str(item[0])))

        fold_counts: list[dict[Any, int]] = [dict() for _ in range(self.n_splits)]
        fold_sizes = [0] * self.n_splits
        assignment: dict[Any, int] = {}
        for group, size, majority in summary:
            target = min(
                range(self.n_splits),
                key=lambda i: (fold_counts[i].get(majority, 0), fold_sizes[i], i),
            )
            assignment[group] = target
            fold_counts[target][majority] = fold_counts[target].get(majority, 0) + size
            fold_sizes[target] += size

        indices = np.arange(len(keys))
        fold_of_row = np.array([assignment[k] for k in keys])
        for fold in range(self.n_splits):
            mask = fold_of_row == fold
            yield indices[~mask], indices[mask]


#: Name -> splitter class, used by the CLI and config files.
SPLIT_REGISTRY: dict[str, type] = {
    "temporal": TemporalSplit,
    "grouped": GroupedSplit,
    "stratified_grouped": StratifiedGroupedSplit,
}


def resolve_split(spec: Any, **kwargs: Any):
    """Coerce a split specification into a splitter instance.

    Accepts an existing splitter (returned unchanged), a registry name, or
    ``None`` for stratified/plain K-fold depending on ``task_type``.

    Args:
        spec: ``None``, a name from :data:`SPLIT_REGISTRY`, or a splitter.
        **kwargs: Forwarded to the splitter constructor. ``task_type`` and
            ``n_splits`` are consumed here when ``spec`` is ``None``.

    Returns:
        A scikit-learn-compatible splitter.

    Raises:
        ValueError: On an unknown split name.
    """
    if spec is None:
        from sklearn.model_selection import KFold, StratifiedKFold

        task_type = kwargs.pop("task_type", "classification")
        n_splits = kwargs.pop("n_splits", 5)
        random_state = kwargs.pop("random_state", 42)
        cls = StratifiedKFold if task_type == "classification" else KFold
        return cls(n_splits=n_splits, shuffle=True, random_state=random_state)

    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in SPLIT_REGISTRY:
            raise ValueError(
                f"Unknown split {spec!r}. Available: {sorted(SPLIT_REGISTRY)}, "
                f"or pass a scikit-learn splitter instance."
            )
        kwargs.pop("task_type", None)
        return SPLIT_REGISTRY[key](**kwargs)

    if hasattr(spec, "split") and hasattr(spec, "get_n_splits"):
        return spec

    raise TypeError(
        f"split must be None, a name from {sorted(SPLIT_REGISTRY)}, or a "
        f"scikit-learn splitter; got {type(spec).__name__}"
    )

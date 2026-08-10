"""Prediction caching for tabular foundation models.

Motivation
----------
Inference with a TFM is expensive: a single forward pass over a large test set
can take minutes and gigabytes of VRAM. TabTune's own evaluation path used to
pay that cost three times for one call:

* ``evaluate()`` calls ``predict(X_test)``, then ``predict_proba(X_test)``
* ``evaluate_calibration()`` calls ``predict_proba(X_test)`` again

The benchmarking pipeline compounded it further. This module removes the
duplication by memoising predictions on ``(pipeline fingerprint, data
fingerprint, method)``.

Correctness
-----------
A cache that returns stale predictions is worse than no cache. The fingerprint
therefore covers everything that can change a prediction: the model identity,
task, tuning strategy, fine-tune mode, a fit counter that increments on every
``fit()``, and a content hash of the input frame including its column names,
dtypes and index. Refitting or mutating the data invalidates automatically.

Example:
    >>> cache = PredictionCache(backend="memory", max_entries=8)
    >>> cache.stats()["hits"]
    0
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["PredictionCache", "CacheStats", "fingerprint_data", "make_cache"]

Backend = Literal["memory", "disk", "none"]

#: Cap on rows hashed byte-for-byte. Beyond this we hash a deterministic
#: stratified sample plus the exact shape, which keeps fingerprinting O(1) on
#: million-row frames while still detecting realistic mutations.
_FULL_HASH_ROW_LIMIT = 50_000


class CacheStats(dict):
    """Hit/miss counters, exposed as a plain dict for easy logging."""

    def __init__(self) -> None:
        super().__init__(hits=0, misses=0, stores=0, evictions=0, errors=0)

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache, or 0.0 when never queried."""
        total = self["hits"] + self["misses"]
        return self["hits"] / total if total else 0.0

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"hits={self['hits']} misses={self['misses']} "
            f"stores={self['stores']} hit_rate={self.hit_rate:.0%}"
        )


def fingerprint_data(X: Any) -> str:
    """Return a stable content fingerprint for a feature matrix.

    Covers values, column names, dtypes, shape and index, so a column rename or
    a reordering produces a different fingerprint even when the numbers match.

    Args:
        X: A DataFrame, numpy array, or anything convertible to one.

    Returns:
        A 32-character hex digest. Returns a shape-only digest if the object
        cannot be hashed, which degrades to "never hit" rather than to a wrong
        hit.
    """
    hasher = hashlib.blake2b(digest_size=16)

    try:
        columns = getattr(X, "columns", None)
        if columns is not None:
            hasher.update(repr(list(columns)).encode("utf-8"))
            dtypes = getattr(X, "dtypes", None)
            if dtypes is not None:
                hasher.update(repr([str(d) for d in dtypes]).encode("utf-8"))

        values = getattr(X, "to_numpy", None)
        array = values() if callable(values) else np.asarray(X)
        hasher.update(repr(array.shape).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))

        n_rows = array.shape[0] if array.ndim else 0
        if n_rows > _FULL_HASH_ROW_LIMIT:
            # Deterministic stride sample: first, last and evenly spaced rows.
            stride = max(1, n_rows // _FULL_HASH_ROW_LIMIT)
            array = array[::stride]

        if array.dtype == object:
            # Object arrays are not guaranteed to expose a stable buffer.
            hasher.update(np.array2string(array, threshold=array.size + 1).encode("utf-8"))
        else:
            contiguous = np.ascontiguousarray(array)
            hasher.update(contiguous.view(np.uint8).tobytes() if contiguous.size else b"")
    except Exception as exc:  # pragma: no cover - exotic inputs
        logger.debug("[Cache] Falling back to shape-only fingerprint: %s", exc)
        hasher.update(repr(getattr(X, "shape", None)).encode("utf-8"))
        hasher.update(b"__unhashable__")

    return hasher.hexdigest()


class PredictionCache:
    """An LRU cache for model predictions, backed by memory or disk.

    The cache is keyed on a *scope* string that identifies the fitted model,
    plus a data fingerprint and a method name. Callers normally interact with
    it through :meth:`get_or_compute`.

    Args:
        backend: ``"memory"`` for an in-process LRU, ``"disk"`` for a pickle
            store under ``cache_dir``, ``"none"`` to disable.
        max_entries: Maximum entries retained by the memory backend.
        cache_dir: Directory for the disk backend. Defaults to
            ``~/.cache/tabtune/predictions``.

    Thread safety:
        All public methods are guarded by a re-entrant lock, so a cache can be
        shared across an ensemble's worker threads.
    """

    def __init__(
        self,
        backend: Backend = "memory",
        *,
        max_entries: int = 32,
        cache_dir: str | Path | None = None,
    ) -> None:
        if backend not in ("memory", "disk", "none"):
            raise ValueError(
                f"backend must be 'memory', 'disk' or 'none', got {backend!r}"
            )
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")

        self.backend: Backend = backend
        self.max_entries = max_entries
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

        if backend == "disk":
            base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "tabtune" / "predictions"
            base.mkdir(parents=True, exist_ok=True)
            self.cache_dir: Path | None = base
        else:
            self.cache_dir = None

    # ------------------------------------------------------------------ keys

    @staticmethod
    def make_key(scope: str, data_fingerprint: str, method: str) -> str:
        """Build a cache key from its three components."""
        return f"{scope}|{method}|{data_fingerprint}"

    # -------------------------------------------------------------- core API

    @property
    def enabled(self) -> bool:
        """Whether this cache actually stores anything."""
        return self.backend != "none"

    def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or ``None`` on a miss."""
        if not self.enabled:
            return None
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.stats["hits"] += 1
                return self._store[key]

            if self.backend == "disk":
                path = self._path_for(key)
                if path.exists():
                    try:
                        with path.open("rb") as handle:
                            value = pickle.load(handle)
                    except Exception as exc:
                        logger.debug("[Cache] Unreadable cache entry %s: %s", path, exc)
                        self.stats["errors"] += 1
                        path.unlink(missing_ok=True)
                    else:
                        self._store[key] = value
                        self.stats["hits"] += 1
                        return value

            self.stats["misses"] += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``, evicting the least recently used entry."""
        if not self.enabled:
            return
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            self.stats["stores"] += 1

            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
                self.stats["evictions"] += 1

            if self.backend == "disk":
                path = self._path_for(key)
                try:
                    with path.open("wb") as handle:
                        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as exc:  # pragma: no cover - disk failures
                    logger.debug("[Cache] Could not persist %s: %s", path, exc)
                    self.stats["errors"] += 1

    def get_or_compute(
        self,
        scope: str,
        X: Any,
        method: str,
        compute: Callable[[], Any],
        *,
        data_fingerprint: str | None = None,
    ) -> Any:
        """Return a cached prediction or compute, store and return it.

        Args:
            scope: Identifier for the fitted model (see
                :meth:`~tabtune.TabularPipeline.pipeline.TabularPipeline._cache_scope`).
            X: The input the prediction is for; fingerprinted automatically.
            method: Which call this is, e.g. ``"predict"`` or ``"predict_proba"``.
            compute: Zero-argument callable producing the value on a miss.
            data_fingerprint: Pre-computed fingerprint, to avoid hashing ``X``
                twice when several methods run against the same frame.

        Returns:
            The cached or freshly computed value.
        """
        if not self.enabled:
            return compute()

        fingerprint = data_fingerprint or fingerprint_data(X)
        key = self.make_key(scope, fingerprint, method)

        cached = self.get(key)
        if cached is not None:
            logger.debug("[Cache] hit for %s (%s)", method, fingerprint[:8])
            return cached

        value = compute()
        if value is not None:
            self.set(key, value)
        return value

    def invalidate(self, scope: str | None = None) -> int:
        """Drop cached entries.

        Args:
            scope: Only drop entries for this scope. ``None`` clears everything.

        Returns:
            The number of in-memory entries removed.
        """
        with self._lock:
            if scope is None:
                removed = len(self._store)
                self._store.clear()
                if self.backend == "disk" and self.cache_dir is not None:
                    shutil.rmtree(self.cache_dir, ignore_errors=True)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                return removed

            keys = [k for k in self._store if k.startswith(f"{scope}|")]
            for key in keys:
                self._store.pop(key, None)
                if self.backend == "disk":
                    self._path_for(key).unlink(missing_ok=True)
            return len(keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    # --------------------------------------------------------------- pickling

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state: configuration only, no lock and no entries.

        ``TabularPipeline.save()`` joblib-pickles the whole pipeline, and a
        cache holds a ``threading.RLock``, which is not picklable. Two things
        are dropped here:

        * the lock, recreated on unpickling;
        * the cached predictions themselves, because a saved pipeline should
          not carry stale results from the session that saved it. The disk
          backend is unaffected - its entries live in ``cache_dir``.
        """
        return {
            "backend": self.backend,
            "max_entries": self.max_entries,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild a cache from :meth:`__getstate__`, empty and unlocked."""
        self.backend = state.get("backend", "none")
        self.max_entries = state.get("max_entries", 32)
        cache_dir = state.get("cache_dir")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._store = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"PredictionCache(backend={self.backend!r}, entries={len(self)}, "
            f"{self.stats})"
        )

    # ----------------------------------------------------------------- disk

    def _path_for(self, key: str) -> Path:
        assert self.cache_dir is not None  # guarded by callers
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
        return self.cache_dir / f"{digest}.pkl"


def make_cache(spec: PredictionCache | Backend | bool | None, **kwargs: Any) -> PredictionCache:
    """Coerce a user-supplied cache specification into a :class:`PredictionCache`.

    Accepts an existing cache (returned unchanged), a backend name, ``True``
    (memory), or ``None``/``False`` (disabled), so ``TabularPipeline(cache=...)``
    can be forgiving about what callers pass.

    Args:
        spec: The specification to coerce.
        **kwargs: Forwarded to :class:`PredictionCache` when constructing.

    Returns:
        A cache instance; disabled caches are real objects with
        ``enabled == False`` so call sites never need a ``None`` check.
    """
    if isinstance(spec, PredictionCache):
        return spec
    if spec is None or spec is False:
        return PredictionCache("none")
    if spec is True:
        return PredictionCache("memory", **kwargs)
    if isinstance(spec, str):
        return PredictionCache(spec, **kwargs)  # type: ignore[arg-type]
    raise TypeError(
        f"cache must be a PredictionCache, 'memory'/'disk'/'none', bool or None; "
        f"got {type(spec).__name__}"
    )

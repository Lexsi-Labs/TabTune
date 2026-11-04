"""Prediction caching.

Tabular foundation models are expensive to query, and TabTune's own evaluation
path used to run three full forward passes for one ``evaluate()`` call. Enabling
a cache collapses that to one:

    >>> from tabtune import TabularPipeline
    >>> pipe = TabularPipeline("TabICLv2", cache="memory")   # doctest: +SKIP
    >>> pipe.fit(X_train, y_train)                           # doctest: +SKIP
    >>> pipe.evaluate(X_test, y_test)                        # doctest: +SKIP
    >>> pipe.cache.stats                                     # doctest: +SKIP
    hits=2 misses=1 stores=1 hit_rate=67%

Cache entries are keyed on a fingerprint covering the fitted model *and* the
input data, so refitting or changing the data invalidates automatically.
"""

from __future__ import annotations

from .prediction_cache import (
    CacheStats,
    PredictionCache,
    fingerprint_data,
    make_cache,
)

__all__ = ["PredictionCache", "CacheStats", "fingerprint_data", "make_cache"]

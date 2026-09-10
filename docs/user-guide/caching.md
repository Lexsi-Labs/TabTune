# Prediction Caching

*New in TabTune 0.2.0.*

Tabular foundation models are expensive to query, and TabTune's own evaluation path used to
run three full forward passes for one `evaluate()` call. Enabling a cache collapses that to
one.

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline("TabICLv2", cache="memory")   # or "disk"
pipeline.fit(X_train, y_train)
pipeline.evaluate(X_test, y_test)

print(pipeline.cache.stats)
# hits=2 misses=1 stores=1 hit_rate=67%
```

---

## 1. Backends

| `cache` value | Backend | Lifetime |
|---|---|---|
| `None` (default) | disabled | — |
| `"memory"` | in-process LRU | the process |
| `"disk"` | on-disk store | across processes |
| a `PredictionCache` | your own instance | yours |

```python
from tabtune.caching import PredictionCache

cache = PredictionCache("disk", max_entries=128, cache_dir="./.tabtune-cache")
pipeline = TabularPipeline("TabICLv2", cache=cache)
```

---

## 2. Invalidation is automatic

Entries are keyed on a fingerprint covering the **fitted model** *and* the **input data**,
so refitting or changing the data invalidates automatically. There is no stale-cache failure
mode to reason about.

```python
from tabtune.caching import fingerprint_data

fingerprint_data(X_test)   # stable hash of the frame
```

---

## 3. Statistics

```python
pipeline.cache.stats
# CacheStats: hits=2 misses=1 stores=1 hit_rate=67%

pipeline.cache.stats["hits"]
pipeline.cache.stats.hit_rate
len(pipeline.cache)
```

---

## 4. Clearing

```python
pipeline.clear_cache()             # entries removed for this pipeline, returns the count
pipeline.cache.invalidate()        # everything
pipeline.cache.invalidate(scope)   # one scope
```

---

## 5. Where it pays off

Disk caching pays off most where the same test rows are queried repeatedly:

- **Leaderboard runs** — many configurations, one test set
- **Shift-evaluation sweeps** — folds re-query overlapping rows
- **Uncertainty work** — `uncertainty_report`, recalibration and conformal calibration all
  read `predict_proba` on the same frames
- **Notebook iteration** — re-running a cell after a `predict` costs nothing

```python
pipeline = TabularPipeline("TabICLv2", cache="disk")
```

---

## 6. Serialization

`PredictionCache` implements `__getstate__` / `__setstate__`, so a pipeline carrying a cache
still pickles. In-memory entries are not written into the pipeline artefact — `save()` stays
the size it was.

---

## 7. Lower-level API

```python
from tabtune.caching import PredictionCache, make_cache, CacheStats

cache = make_cache("memory")           # accepts str | bool | None | PredictionCache

value = cache.get_or_compute(
    scope="my-model",
    X=X_test,
    method="predict_proba",
    compute=lambda: expensive_call(X_test),
)

key = PredictionCache.make_key(scope, data_fingerprint, method)
cache.get(key)
cache.set(key, value)
```

---

## See Also

- [Shift-Aware Evaluation](shift-evaluation.md)
- [Model Comparison](leaderboard.md)
- [API: TabularPipeline](../api/pipeline.md)

"""Tests for the prediction cache.

The cache exists to remove redundant forward passes, but a cache that returns a
stale prediction is worse than no cache at all. Most of these tests are
therefore about *invalidation*: any change to the data or to the fitted model
must produce a miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabtune.caching import PredictionCache, fingerprint_data, make_cache

pytestmark = pytest.mark.unit


@pytest.fixture
def frame():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})


# --------------------------------------------------------------- fingerprints


def test_fingerprint_is_stable_across_copies(frame):
    assert fingerprint_data(frame) == fingerprint_data(frame.copy())


def test_fingerprint_changes_with_values(frame):
    other = frame.copy()
    other.loc[0, "a"] = 99.0
    assert fingerprint_data(frame) != fingerprint_data(other)


def test_fingerprint_changes_with_column_names(frame):
    assert fingerprint_data(frame) != fingerprint_data(frame.rename(columns={"a": "A"}))


def test_fingerprint_changes_with_column_order(frame):
    assert fingerprint_data(frame) != fingerprint_data(frame[["b", "a"]])


def test_fingerprint_changes_with_dtype(frame):
    other = frame.copy()
    other["a"] = other["a"].astype("float32")
    assert fingerprint_data(frame) != fingerprint_data(other)


def test_fingerprint_handles_numpy_arrays():
    a = np.arange(12).reshape(4, 3)
    assert fingerprint_data(a) == fingerprint_data(a.copy())
    assert fingerprint_data(a) != fingerprint_data(a + 1)


def test_fingerprint_never_raises_on_exotic_input():
    class Weird:
        shape = (3, 2)

    assert isinstance(fingerprint_data(Weird()), str)


def test_large_frames_are_subsampled_but_still_shape_sensitive():
    big = pd.DataFrame({"a": np.arange(120_000, dtype=float)})
    assert fingerprint_data(big) == fingerprint_data(big.copy())
    assert fingerprint_data(big) != fingerprint_data(big.iloc[:-1])


# ------------------------------------------------------------------- behaviour


def test_repeated_calls_compute_once(frame):
    cache = PredictionCache("memory")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return np.array([1, 2, 3])

    for _ in range(4):
        result = cache.get_or_compute("scope", frame, "predict", compute)
        assert np.array_equal(result, np.array([1, 2, 3]))

    assert calls["n"] == 1
    assert cache.stats["hits"] == 3
    assert cache.stats.hit_rate == pytest.approx(0.75)


def test_different_methods_do_not_collide(frame):
    cache = PredictionCache("memory")
    cache.get_or_compute("scope", frame, "predict", lambda: "labels")
    proba = cache.get_or_compute("scope", frame, "predict_proba", lambda: "probs")
    assert proba == "probs"


def test_different_scopes_do_not_collide(frame):
    cache = PredictionCache("memory")
    assert cache.get_or_compute("fit1", frame, "predict", lambda: "a") == "a"
    assert cache.get_or_compute("fit2", frame, "predict", lambda: "b") == "b"


def test_changed_data_is_a_miss(frame):
    cache = PredictionCache("memory")
    cache.get_or_compute("scope", frame, "predict", lambda: "first")
    changed = frame.copy()
    changed.loc[0, "a"] = 42.0
    assert cache.get_or_compute("scope", changed, "predict", lambda: "second") == "second"


def test_disabled_cache_always_computes(frame):
    cache = PredictionCache("none")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return 1

    for _ in range(3):
        cache.get_or_compute("scope", frame, "predict", compute)
    assert calls["n"] == 3
    assert not cache.enabled


def test_lru_eviction_respects_max_entries(frame):
    cache = PredictionCache("memory", max_entries=2)
    for i in range(5):
        cache.get_or_compute(f"scope{i}", frame, "predict", lambda i=i: i)
    assert len(cache) == 2
    assert cache.stats["evictions"] == 3


def test_invalidate_by_scope(frame):
    cache = PredictionCache("memory")
    cache.get_or_compute("keep", frame, "predict", lambda: 1)
    cache.get_or_compute("drop", frame, "predict", lambda: 2)
    assert cache.invalidate("drop") == 1
    assert len(cache) == 1


def test_invalidate_everything(frame):
    cache = PredictionCache("memory")
    cache.get_or_compute("a", frame, "predict", lambda: 1)
    cache.get_or_compute("b", frame, "predict", lambda: 2)
    assert cache.invalidate() == 2
    assert len(cache) == 0


def test_none_results_are_not_cached(frame):
    """A model that returns None should be retried, not memoised as a miss."""
    cache = PredictionCache("memory")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return None

    cache.get_or_compute("scope", frame, "predict", compute)
    cache.get_or_compute("scope", frame, "predict", compute)
    assert calls["n"] == 2


# ------------------------------------------------------------------ disk mode


def test_disk_backend_persists_across_instances(tmp_path, frame):
    first = PredictionCache("disk", cache_dir=tmp_path)
    first.get_or_compute("scope", frame, "predict", lambda: np.array([7, 8]))

    second = PredictionCache("disk", cache_dir=tmp_path)
    result = second.get_or_compute(
        "scope", frame, "predict", lambda: pytest.fail("should have hit disk")
    )
    assert np.array_equal(result, np.array([7, 8]))


def test_disk_invalidate_clears_files(tmp_path, frame):
    cache = PredictionCache("disk", cache_dir=tmp_path)
    cache.get_or_compute("scope", frame, "predict", lambda: 1)
    assert list(tmp_path.glob("*.pkl"))
    cache.invalidate()
    assert not list(tmp_path.glob("*.pkl"))


# ----------------------------------------------------------------- coercion


@pytest.mark.parametrize(
    "spec,enabled,backend",
    [
        (None, False, "none"),
        (False, False, "none"),
        (True, True, "memory"),
        ("memory", True, "memory"),
        ("none", False, "none"),
    ],
)
def test_make_cache_coerces_specs(spec, enabled, backend):
    cache = make_cache(spec)
    assert cache.enabled is enabled
    assert cache.backend == backend


def test_make_cache_passes_instances_through():
    existing = PredictionCache("memory")
    assert make_cache(existing) is existing


def test_make_cache_rejects_nonsense():
    with pytest.raises(TypeError):
        make_cache(3.14)


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        PredictionCache("redis")


def test_invalid_max_entries_is_rejected():
    with pytest.raises(ValueError, match="max_entries"):
        PredictionCache("memory", max_entries=0)


# ------------------------------------------------------------------ pickling


def test_cache_survives_pickling(frame):
    """TabularPipeline.save() joblib-pickles the whole pipeline, cache included.

    A cache holds a threading.RLock, which is not picklable; without explicit
    __getstate__ the save path raised "cannot pickle '_thread.RLock' object".
    """
    import pickle

    cache = PredictionCache("memory", max_entries=7)
    cache.get_or_compute("scope", frame, "predict", lambda: np.array([1, 2, 3]))
    assert len(cache) == 1

    restored = pickle.loads(pickle.dumps(cache))
    assert restored.backend == "memory"
    assert restored.max_entries == 7
    # Configuration survives; entries deliberately do not, so a reloaded
    # pipeline never serves predictions cached in the session that saved it.
    assert len(restored) == 0
    assert restored.stats["hits"] == 0

    # And it is functional after the round trip - the lock was rebuilt.
    assert restored.get_or_compute("scope", frame, "predict", lambda: "fresh") == "fresh"


def test_disabled_cache_survives_pickling():
    import pickle

    restored = pickle.loads(pickle.dumps(make_cache(None)))
    assert restored.enabled is False


def test_disk_cache_keeps_its_directory_across_pickling(tmp_path, frame):
    import pickle

    cache = PredictionCache("disk", cache_dir=tmp_path)
    cache.get_or_compute("scope", frame, "predict", lambda: np.array([9]))

    restored = pickle.loads(pickle.dumps(cache))
    assert restored.cache_dir == tmp_path
    # Disk entries are external to the object, so they are still readable.
    result = restored.get_or_compute(
        "scope", frame, "predict", lambda: pytest.fail("should have hit disk")
    )
    assert np.array_equal(result, np.array([9]))

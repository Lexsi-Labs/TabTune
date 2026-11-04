"""Time series forecasting with foundation models.

    >>> from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema
    >>> schema = TimeSeriesSchema(target="sales", item_id="store")                       # doctest: +SKIP
    >>> pipe = TimeSeriesPipeline("Chronos", forecast_params={"prediction_length": 14})  # doctest: +SKIP
    >>> pipe.fit(history_df, schema).predict().to_pandas()                               # doctest: +SKIP

The pipeline, schema and result types are pandas/numpy only. A model backend
(and torch) is imported when a pipeline is fitted, never at import time.
"""

from __future__ import annotations

from typing import Any

from .forecast import ForecastResult
from .pipeline import TimeSeriesPipeline
from .schema import TimeSeriesPanel, TimeSeriesSchema
from .tasks import AnomalyResult, EmbeddingResult, ImputationResult

_LAZY = {
    "TimeSeriesLeaderboard": (".leaderboard", "TimeSeriesLeaderboard"),
    "TimeSeriesEnsemble": (".ensemble", "TimeSeriesEnsemble"),
    "TimeSeriesBenchmark": (".benchmark", "TimeSeriesBenchmark"),
    "make_panel": (".data", "make_panel"),
    "make_anomalous_panel": (".data", "make_anomalous_panel"),
    "split_horizon": (".data", "split_horizon"),
}

__all__ = [
    "TimeSeriesPipeline",
    "TimeSeriesSchema",
    "TimeSeriesPanel",
    "ForecastResult",
    "AnomalyResult",
    "ImputationResult",
    "EmbeddingResult",
    *_LAZY,
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

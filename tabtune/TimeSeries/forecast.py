"""Output contract for time series pipelines.

The pipeline turns an adapter's raw arrays into a :class:`ForecastResult`.
The long-format :meth:`ForecastResult.to_pandas` view is the stable public
layout; it carries an explicit ``target`` column.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["ForecastResult", "future_timestamps"]


def future_timestamps(last: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    """Return the ``horizon`` timestamps that follow ``last`` on a ``freq`` grid."""
    return pd.date_range(start=last, periods=horizon + 1, freq=freq)[1:]


@dataclass(frozen=True, eq=False)
class ForecastResult:
    """Point and quantile forecasts for a panel of series.

    The arrays are read-only; use :meth:`to_pandas` or ``np.array(result.point)``
    for a writable copy. A row is one ``(item, target)`` pair: a univariate
    forecast has one row per item, a multivariate one consecutive rows per item.

    Attributes:
        item_ids: One identifier per row of the arrays.
        timestamps: ``[row, horizon]`` forecast timestamps.
        point: ``[row, horizon]`` point forecast. Which statistic it is
            (mean or median of the predictive distribution) depends on the
            model and is recorded in ``metadata``.
        quantiles: ``[row, horizon, quantile]`` forecasts, or ``None`` when
            no quantiles were requested.
        quantile_levels: The level of each entry on the last axis of
            ``quantiles``.
        targets: Name of the forecast target column of each row.
        item_id_column: Item column name from the schema, or ``None`` for a
            single series.
        timestamp_column: Timestamp column name from the schema.
        metadata: Provenance: model, checkpoint, device, strategy and whether
            any training occurred.
    """

    item_ids: tuple[Hashable, ...]
    timestamps: np.ndarray
    point: np.ndarray
    quantiles: np.ndarray | None
    quantile_levels: tuple[float, ...]
    targets: tuple[str, ...]
    item_id_column: str | None = None
    timestamp_column: str = "timestamp"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("timestamps", "point", "quantiles"):
            array = getattr(self, name)
            if array is not None:
                array = np.array(array)
                array.setflags(write=False)
                object.__setattr__(self, name, array)

    @property
    def target(self) -> str:
        """The single forecast target.

        Kept for univariate results.
        For multivariate result read :attr:`targets` instead.
        """
        names = dict.fromkeys(self.targets)
        if len(names) != 1:
            raise AttributeError(
                f"this forecast covers {len(names)} targets ({list(names)}); "
                f"use .targets, or .to_pandas() which names the target per row"
            )
        return next(iter(names))

    @property
    def prediction_length(self) -> int:
        """Forecast horizon in steps."""
        return int(self.point.shape[1])

    def to_pandas(self) -> pd.DataFrame:
        """Return the forecasts as a long frame, one row per (item, target, timestamp).

        Columns: the item column (panels only), the timestamp column,
        ``target``, ``point``, then one column per quantile level named by its
        string value (``"0.1"``, ``"0.5"``...).
        """
        horizon = self.prediction_length
        data: dict[str, Any] = {}
        if self.item_id_column is not None:
            data[self.item_id_column] = pd.Index(self.item_ids).repeat(horizon)
        data[self.timestamp_column] = pd.to_datetime(self.timestamps.ravel())
        data["target"] = pd.Index(self.targets).repeat(horizon)
        data["point"] = self.point.ravel()
        if self.quantiles is not None:
            for i, level in enumerate(self.quantile_levels):
                data[str(level)] = self.quantiles[:, :, i].ravel()
        return pd.DataFrame(data)

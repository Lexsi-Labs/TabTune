"""Rolling-origin windows cut from a validated panel.

A window moves every series' forecast origin back in time: its context is
the history before the origin and its actuals are the ``horizon`` values
after it. Window 0 ends at the last observation, window 1 ``step`` steps
earlier, and so on. Known covariates observed in the history become the
window's future covariates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .schema import TimeSeriesPanel

__all__ = ["Window", "rolling_windows", "cut_panel"]


@dataclass(frozen=True, eq=False)
class Window:
    """One forecast origin per series.

    Attributes:
        index: 0 for the most recent window, counting back in time.
        context: The history before each origin.
        actual: ``[row, horizon]`` observed values after each origin.
    """

    index: int
    context: TimeSeriesPanel
    actual: np.ndarray


def cut_panel(
    panel: TimeSeriesPanel,
    ends: Sequence[int],
    horizon: int,
    known_covariates: Sequence[str] = (),
) -> TimeSeriesPanel:
    """Return ``panel`` with row ``i`` truncated to its first ``ends[i]`` observations.

    The forecast origin moves back accordingly, and the next ``horizon``
    values of each known covariate become its future covariates.
    """
    offset = to_offset(panel.freq)
    values, lasts, past, future = [], [], [], []
    for row, end in enumerate(ends):
        series = panel.values[row]
        values.append(series[:end])
        removed = len(series) - end
        last = panel.last_timestamps[row]
        lasts.append(last - removed * offset if removed else last)
        covariates = panel.past_covariates[row]
        past.append({name: arr[:end] for name, arr in covariates.items()})
        future.append(
            {name: covariates[name][end : end + horizon] for name in known_covariates if name in covariates}
        )
    return replace(
        panel,
        values=tuple(values),
        last_timestamps=tuple(pd.Timestamp(t) for t in lasts),
        past_covariates=tuple(past),
        future_covariates=tuple(future),
    )


def rolling_windows(
    panel: TimeSeriesPanel,
    *,
    horizon: int,
    windows: int = 1,
    step: int | None = None,
    min_context: int = 1,
    known_covariates: Sequence[str] = (),
) -> list[Window]:
    """Cut ``windows`` evaluation windows from the end of every series.

    Items whose history is too short for a window (fewer than
    ``min_context`` observations before the origin, or no observed value)
    are left out of that window.

    Raises:
        ValueError: If no series is long enough for even the most recent window.
    """
    if horizon < 1 or windows < 1:
        raise ValueError("horizon and windows must be positive")
    step = step or horizon
    out: list[Window] = []
    for w in range(windows):
        keep: list[int] = []
        ends: list[int] = []
        for rows in panel.item_groups():
            end = len(panel.values[rows[0]]) - horizon - w * step
            if end < min_context:
                continue
            if any(not np.isfinite(panel.values[r][:end]).any() for r in rows):
                continue
            keep.extend(rows)
            ends.extend([end] * len(rows))
        if not keep:
            break
        sub = _select(panel, keep)
        context = cut_panel(sub, ends, horizon, known_covariates)
        actual = np.stack([sub.values[i][e : e + horizon] for i, e in enumerate(ends)])
        out.append(Window(index=w, context=context, actual=actual))
    if not out:
        raise ValueError(
            f"No series is long enough for a {horizon}-step backtest window with at least "
            f"{min_context} observation(s) of context."
        )
    return out


def _select(panel: TimeSeriesPanel, rows: Sequence[int]) -> TimeSeriesPanel:
    def pick(block: tuple) -> tuple:
        return tuple(block[i] for i in rows)

    return replace(
        panel,
        item_ids=pick(panel.item_ids),
        values=pick(panel.values),
        last_timestamps=pick(panel.last_timestamps),
        target_names=pick(panel.target_names),
        past_covariates=pick(panel.past_covariates),
        future_covariates=pick(panel.future_covariates),
    )

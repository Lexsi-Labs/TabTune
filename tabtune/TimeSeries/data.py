"""Synthetic panels and train/test splits for long-format time series frames.

The synthetic panels are seeded and have known structure (seasonality,
anomalies, covariate effects) for trying and testing pipeline features offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .schema import TimeSeriesSchema

__all__ = ["make_panel", "make_anomalous_panel", "split_horizon", "SYNTHETIC_KINDS"]

SYNTHETIC_KINDS = ("seasonal", "trend", "random_walk", "intermittent", "multi_seasonal", "noise")


def make_panel(
    n_series: int = 3,
    length: int = 200,
    *,
    freq: str = "h",
    kind: str = "seasonal",
    season: int = 24,
    noise: float = 0.1,
    seed: int = 0,
    start: str = "2024-01-01",
    covariates: Sequence[str] = (),
    n_targets: int = 1,
) -> pd.DataFrame:
    """Return a long frame with ``item_id``, ``timestamp``, the target(s) and covariates.

    Args:
        n_series: Number of items.
        length: Observations per item.
        freq: Pandas frequency of the grid.
        kind: One of :data:`SYNTHETIC_KINDS`.
        season: Seasonal period in steps.
        noise: Standard deviation of the noise relative to the seasonal amplitude.
        seed: RNG seed; identical arguments give identical frames.
        start: First timestamp.
        covariates: Covariate columns to add. ``"promo"`` is a 0/1 indicator
            that lifts the target; any other name is a smooth driver the
            target depends on linearly.
        n_targets: Target columns: ``target``, or ``target_0 ...`` whose extra
            columns are noisy lagged copies of the first.
    """
    if kind not in SYNTHETIC_KINDS:
        raise ValueError(f"kind must be one of {SYNTHETIC_KINDS}, got {kind!r}")
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(start=start, periods=length, freq=freq)
    t = np.arange(length, dtype=float)
    frames = []
    for i in range(n_series):
        level = 10.0 + 5.0 * i
        amplitude = 2.0 + rng.uniform(0.0, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        seasonal = amplitude * np.sin(2 * np.pi * t / season + phase)
        if kind == "seasonal":
            signal = level + seasonal
        elif kind == "multi_seasonal":
            signal = level + seasonal + 0.5 * amplitude * np.sin(2 * np.pi * t / (season * 7) + phase)
        elif kind == "trend":
            signal = level + 0.05 * (i + 1) * t + seasonal
        elif kind == "random_walk":
            signal = level + np.cumsum(rng.normal(0.0, 0.5, length))
        elif kind == "intermittent":
            signal = (rng.poisson(3.0, length) * (rng.uniform(size=length) < 0.3)).astype(float)
        else:
            signal = np.full(length, level)
        values = signal + (0.0 if kind == "intermittent" else rng.normal(0.0, noise * amplitude, length))
        frame = pd.DataFrame({"item_id": f"series_{i}", "timestamp": stamps})
        for name in covariates:
            if name == "promo":
                driver = (rng.uniform(size=length) < 0.15).astype(float)
                values = values + 3.0 * driver
            else:
                driver = np.cos(2 * np.pi * t / (season * 2) + i)
                values = values + 1.5 * driver
            frame[name] = driver
        if n_targets == 1:
            frame["target"] = values
        else:
            for j in range(n_targets):
                frame[f"target_{j}"] = np.roll(values, j) + rng.normal(0.0, noise, length) * j
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_anomalous_panel(
    n_series: int = 2,
    length: int = 300,
    *,
    n_anomalies: int = 4,
    magnitude: float = 6.0,
    seed: int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Seasonal panel with point anomalies and a boolean ``is_anomaly`` column.

    Anomalies are placed in the last two thirds of each series, so the first
    third can serve as the clean reference period.
    """
    frame = make_panel(n_series, length, seed=seed, **kwargs)
    rng = np.random.default_rng(seed + 1)
    frame["is_anomaly"] = False
    for _, idx in frame.groupby("item_id").indices.items():
        idx = np.asarray(idx)
        positions = rng.choice(np.arange(length // 3, length), size=n_anomalies, replace=False)
        scale = float(np.std(frame.loc[idx, "target"]))
        rows = idx[positions]
        signs = rng.choice([-1.0, 1.0], size=n_anomalies)
        frame.loc[rows, "target"] = frame.loc[rows, "target"].to_numpy() + signs * magnitude * scale
        frame.loc[rows, "is_anomaly"] = True
    return frame


def split_horizon(
    df: pd.DataFrame, schema: TimeSeriesSchema, prediction_length: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long frame into history and the last ``prediction_length`` rows of every item.

    Returns:
        ``(history, actual)``: pass ``history`` to ``fit`` and ``actual`` to
        ``evaluate``. Known covariates for the horizon are in ``actual``.
    """
    keys = [c for c in (schema.item_id, schema.timestamp) if c is not None]
    ordered = df.sort_values(keys, kind="mergesort")
    if schema.item_id is None:
        position = pd.Series(np.arange(len(ordered))[::-1], index=ordered.index)
    else:
        position = ordered.groupby(schema.item_id, sort=False).cumcount(ascending=False)
    is_future = position < prediction_length
    return (
        ordered[~is_future].reset_index(drop=True),
        ordered[is_future].reset_index(drop=True),
    )

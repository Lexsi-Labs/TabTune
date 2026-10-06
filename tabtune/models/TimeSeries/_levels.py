"""Quantile levels for models that predict a fixed grid of quantiles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..._internal.deprecation import warn_once
from ...registry.errors import ConfigError

__all__ = ["exact_levels", "select_levels", "warn_clamped_levels"]


def select_levels(
    grid: np.ndarray, native: Sequence[float], requested: Sequence[float], *, model: str = "the model"
) -> np.ndarray:
    """Quantiles at ``requested`` levels from a ``[..., len(native)]`` grid.

    Levels between two native ones are interpolated linearly in the level;
    levels outside the native range take the nearest native quantile, with a
    warning.
    """
    native = np.asarray(native, dtype=float)
    requested = np.asarray(requested, dtype=float)
    outside = requested[(requested < native[0] - 1e-9) | (requested > native[-1] + 1e-9)]
    if outside.size:
        warn_once(
            f"{model} predicts quantiles {native[0]:g} to {native[-1]:g}; requested level(s) "
            f"{sorted(outside.tolist())} take the nearest of them. Calibrate the pipeline for "
            f"wider intervals.",
            UserWarning,
            key=f"ts-levels-outside:{model}:{sorted(outside.tolist())}",
        )
    clipped = np.clip(requested, native[0], native[-1])
    upper = np.clip(np.searchsorted(native, clipped, side="left"), 1, len(native) - 1)
    lower = upper - 1
    span = native[upper] - native[lower]
    weight = np.where(span > 0, (clipped - native[lower]) / np.where(span > 0, span, 1.0), 0.0)
    weight = np.clip(weight, 0.0, 1.0)
    if len(native) == 1:
        return np.repeat(grid[..., :1], len(requested), axis=-1)
    return grid[..., lower] * (1 - weight) + grid[..., upper] * weight


def exact_levels(
    native: Sequence[float], requested: Sequence[float], *, model: str
) -> list[int]:
    """Column indices of ``requested`` in ``native``, refusing anything off the grid.

    Levels are matched to six decimal places.
    """
    grid = [float(q) for q in native]
    lookup = {round(q, 6): index for index, q in enumerate(grid)}
    columns, missing = [], []
    for level in requested:
        index = lookup.get(round(float(level), 6))
        if index is None:
            missing.append(float(level))
        else:
            columns.append(index)
    if missing:
        raise ConfigError(
            f"{model} predicts only the quantile levels {grid}, so it cannot serve "
            f"{missing}. TabTune will not interpolate a level the model never produced: "
            f"pick levels from that list, calibrate the pipeline for other intervals, or "
            f"use a model with a finer grid."
        )
    return columns


def warn_clamped_levels(
    native: Sequence[float], requested: Sequence[float], *, model: str
) -> None:
    """Warn that levels outside ``native`` will be clamped to its ends."""
    grid = [float(q) for q in native]
    if not grid or not requested:
        return
    low, high = min(grid), max(grid)
    outside = sorted(float(q) for q in requested if q < low or q > high)
    if outside:
        warn_once(
            f"{model} predicts quantiles {low:g} to {high:g}; requested level(s) "
            f"{outside} take the nearest of them. Calibrate the pipeline for wider "
            f"intervals.",
            UserWarning,
            key=f"ts-levels-outside:{model}:{outside}",
        )

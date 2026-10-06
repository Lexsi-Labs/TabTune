"""Conformal calibration of forecast quantiles.

The calibrator is fitted on backtest forecasts of a frozen model and the
values that followed them, and keeps one threshold per horizon step because
errors grow with the horizon. Methods (Romano et al., 2019; Stankeviciute et
al., 2021):

* ``"cqr"``: conformalised quantile regression. The model's central interval
  for each level is widened (or narrowed) by the conformal quantile of
  ``max(q_lo - y, y - q_hi)``. It keeps the model's adaptive shape.
* ``"absolute"``: symmetric intervals ``point +/- Q`` from ``|y - point|``.
  Works for point-only models.
* ``"signed"``: asymmetric intervals from the lower and upper conformal
  quantiles of ``y - point``. Also works for point-only models.

Split conformal gives marginal coverage of at least the nominal level when
calibration and test forecasts are exchangeable. Time series are not, so the
coverage is approximate and should be checked with ``evaluate``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConformalCalibrator", "conformal_quantile", "METHODS"]

METHODS = ("cqr", "absolute", "signed")


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """The ``ceil((n + 1) * level)``-th smallest finite score, ``+inf`` if ``n`` is too small."""
    scores = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    n = scores.size
    k = math.ceil((n + 1) * level)
    if n == 0 or k > n:
        return float("inf")
    if k < 1:
        return float("-inf")
    return float(scores[k - 1])


def _lower_quantile(scores: np.ndarray, level: float) -> float:
    """The ``floor((n + 1) * level)``-th smallest finite score, ``-inf`` if there is none."""
    scores = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    k = math.floor((scores.size + 1) * level)
    if scores.size == 0 or k < 1:
        return float("-inf")
    return float(scores[min(k, scores.size) - 1])


def _at_level(quantiles: np.ndarray, levels: Sequence[float], level: float) -> np.ndarray:
    """``[row, horizon]`` quantile at ``level``, interpolated and clamped to the level grid."""
    grid = np.asarray(levels, dtype=float)
    level = float(np.clip(level, grid[0], grid[-1]))
    j = int(np.searchsorted(grid, level))
    if j < len(grid) and abs(grid[j] - level) < 1e-12:
        return quantiles[..., j]
    lo, hi = j - 1, j
    w = (level - grid[lo]) / (grid[hi] - grid[lo])
    return (1 - w) * quantiles[..., lo] + w * quantiles[..., hi]


@dataclass
class ConformalCalibrator:
    """Split-conformal calibration of forecast quantiles, per horizon step.

    Args:
        method: One of :data:`METHODS`.

    Attributes:
        n_calibration_: Calibration rows (series times backtest windows).
        horizon_: Horizon the thresholds were computed for.
    """

    method: str = "cqr"
    n_calibration_: int = 0
    horizon_: int = 0
    _y: np.ndarray | None = field(default=None, repr=False)
    _point: np.ndarray | None = field(default=None, repr=False)
    _quantiles: np.ndarray | None = field(default=None, repr=False)
    _levels: tuple[float, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {list(METHODS)}, got {self.method!r}")

    def fit(
        self,
        y: np.ndarray,
        point: np.ndarray,
        quantiles: np.ndarray | None,
        levels: Sequence[float],
    ) -> ConformalCalibrator:
        """Store ``[row, horizon]`` actuals and the forecasts made for them."""
        y = np.asarray(y, dtype=float)
        if self.method == "cqr" and (quantiles is None or len(levels) < 2):
            raise ValueError(
                "method='cqr' needs quantile forecasts with at least two levels; "
                "use method='absolute' for point forecasts."
            )
        self._y, self._point = y, np.asarray(point, dtype=float)
        self._quantiles = None if quantiles is None else np.asarray(quantiles, dtype=float)
        self._levels = tuple(float(q) for q in levels)
        self.n_calibration_, self.horizon_ = y.shape
        if self.n_calibration_ < 20:
            warnings.warn(
                f"Only {self.n_calibration_} calibration rows: the thresholds are coarse, and "
                f"infinite for high coverage. Calibrate on more windows or series.",
                UserWarning,
                stacklevel=3,
            )
        return self

    def calibrate(
        self, point: np.ndarray, quantiles: np.ndarray | None, levels: Sequence[float]
    ) -> np.ndarray:
        """Return calibrated ``[row, horizon, level]`` quantiles for ``levels``.

        A level ``t < 0.5`` is the lower bound of the ``1 - 2t`` interval and
        ``t > 0.5`` the upper bound of the ``2t - 1`` interval; 0.5 stays the
        model's median (or its point forecast when it has no quantiles).
        """
        if self._y is None:
            raise RuntimeError("Call fit() before calibrate().")
        point = np.asarray(point, dtype=float)
        horizon = point.shape[1]
        if horizon > self.horizon_:
            raise ValueError(
                f"Calibrated for {self.horizon_} steps, cannot calibrate a {horizon}-step forecast."
            )
        center = point
        if quantiles is not None and self._quantiles is not None:
            center = _at_level(np.asarray(quantiles, dtype=float), self._levels, 0.5)
        out = np.empty(point.shape + (len(levels),))
        for j, level in enumerate(levels):
            if abs(level - 0.5) < 1e-12:
                out[..., j] = center
                continue
            alpha = 2 * min(level, 1 - level)
            lower, upper = self._interval(point, quantiles, alpha, horizon)
            out[..., j] = lower if level < 0.5 else upper
        return np.sort(out, axis=-1)

    def _interval(
        self, point: np.ndarray, quantiles: np.ndarray | None, alpha: float, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        y = self._y[:, :horizon]
        if self.method == "cqr":
            cal_lo = _at_level(self._quantiles, self._levels, alpha / 2)[:, :horizon]
            cal_hi = _at_level(self._quantiles, self._levels, 1 - alpha / 2)[:, :horizon]
            scores = np.maximum(cal_lo - y, y - cal_hi)
            threshold = np.array([conformal_quantile(scores[:, h], 1 - alpha) for h in range(horizon)])
            lo = _at_level(np.asarray(quantiles, dtype=float), self._levels, alpha / 2)
            hi = _at_level(np.asarray(quantiles, dtype=float), self._levels, 1 - alpha / 2)
            lower, upper = lo - threshold, hi + threshold
            crossed = lower > upper
            mid = (lower + upper) / 2
            return np.where(crossed, mid, lower), np.where(crossed, mid, upper)
        residual = y - self._point[:, :horizon]
        if self.method == "absolute":
            q = np.array([conformal_quantile(np.abs(residual[:, h]), 1 - alpha) for h in range(horizon)])
            return point - q, point + q
        q_lo = np.array([_lower_quantile(residual[:, h], alpha / 2) for h in range(horizon)])
        q_hi = np.array([conformal_quantile(residual[:, h], 1 - alpha / 2) for h in range(horizon)])
        return point + q_lo, point + q_hi

    def summary(self) -> dict[str, object]:
        """Method, calibration size and horizon, for forecast metadata."""
        return {
            "method": self.method,
            "n_calibration": self.n_calibration_,
            "horizon": self.horizon_,
        }

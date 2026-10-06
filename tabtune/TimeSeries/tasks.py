"""Anomaly detection, imputation and embeddings on top of a forecaster.

Every registered forecaster supports anomaly detection and imputation:

* **Anomaly detection** scores each observation against rolling forecasts
  made from the history before it, and turns the scores into conformal
  p-values against a reference period; ``alpha`` is the false-alarm rate
  when the reference is clean.
* **Imputation** fills each gap by forecasting forward from the values
  before it and, where there are enough, backward from the values after it
  (forecasting the reversed series), blending the two across the gap.

Evaluate with :meth:`AnomalyResult.evaluate` on labelled data before relying
on a detector.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .forecast import future_timestamps
from .schema import TimeSeriesPanel

__all__ = [
    "AnomalyResult",
    "ImputationResult",
    "EmbeddingResult",
    "ANOMALY_METHODS",
    "score_anomalies",
    "fill_gaps",
    "conformal_p_values",
]

ANOMALY_METHODS = ("forecast_error", "interval", "likelihood")

Forecaster = Callable[[TimeSeriesPanel, int, Sequence[float]], tuple[np.ndarray, np.ndarray | None]]


@dataclass(frozen=True, eq=False)
class AnomalyResult:
    """Per-observation anomaly scores and flags.

    Attributes:
        frame: One row per scored observation: the item column (panels
            only), the timestamp column, ``target``, ``value``, ``expected``,
            ``lower``, ``upper``, ``score``, ``p_value`` and ``is_anomaly``.
            Use :meth:`to_pandas` for a copy to modify.
        method: The scoring method.
        metadata: Settings, model and reference size.
    """

    frame: pd.DataFrame
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Own copy: the frame must not alias one the pipeline still holds.
        object.__setattr__(self, "frame", self.frame.copy())

    def to_pandas(self) -> pd.DataFrame:
        """A copy of :attr:`frame`."""
        return self.frame.copy()

    @property
    def anomalies(self) -> pd.DataFrame:
        """Only the flagged observations."""
        return self.frame[self.frame["is_anomaly"]].reset_index(drop=True)

    def evaluate(self, labels: pd.DataFrame | np.ndarray) -> dict[str, float]:
        """AUROC, AUPRC, precision, recall and F1 against boolean labels.

        Args:
            labels: A frame with the key columns of :attr:`frame` and a
                boolean ``is_anomaly`` or ``label`` column, or an array
                aligned with :attr:`frame`.
        """
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        frame = self.frame
        if isinstance(labels, pd.DataFrame):
            column = "is_anomaly" if "is_anomaly" in labels.columns else "label"
            keys = [c for c in frame.columns[:3] if c in labels.columns and c != "target"]
            renamed = labels.rename(columns={column: "__label__"})[[*keys, "__label__"]].copy()
            for key in keys:
                if pd.api.types.is_datetime64_any_dtype(frame[key]):
                    renamed[key] = pd.to_datetime(renamed[key])
            merged = frame.merge(renamed, on=keys, how="inner")
            y_true = merged["__label__"].astype(bool).to_numpy()
            scores = merged["score"].to_numpy(dtype=float)
            flags = merged["is_anomaly"].to_numpy(dtype=bool)
        else:
            y_true = np.asarray(labels, dtype=bool)
            scores = frame["score"].to_numpy(dtype=float)
            flags = frame["is_anomaly"].to_numpy(dtype=bool)
        mask = np.isfinite(scores)
        y_true, scores, flags = y_true[mask], scores[mask], flags[mask]
        out = {"n": int(len(y_true)), "n_anomalies": int(y_true.sum())}
        if y_true.any() and not y_true.all():
            out["auroc"] = float(roc_auc_score(y_true, scores))
            out["auprc"] = float(average_precision_score(y_true, scores))
        out["precision"] = float(precision_score(y_true, flags, zero_division=0))
        out["recall"] = float(recall_score(y_true, flags, zero_division=0))
        out["f1"] = float(f1_score(y_true, flags, zero_division=0))
        return out

    def __repr__(self) -> str:
        return (
            f"AnomalyResult(method={self.method!r}, scored={int(self.frame['score'].notna().sum())}, "
            f"flagged={int(self.frame['is_anomaly'].sum())})"
        )


@dataclass(frozen=True, eq=False)
class ImputationResult:
    """The history with its missing target values filled.

    Attributes:
        frame: The input columns of the schema, with filled targets, plus
            ``imputed`` (per target when there are several: ``imputed_<target>``)
            and, for a single target, the ``lower`` and ``upper`` 10-90%
            band of the fill. Use :meth:`to_pandas` for a copy to modify.
        metadata: Settings and model.
    """

    frame: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", self.frame.copy())

    def to_pandas(self) -> pd.DataFrame:
        """A copy of :attr:`frame`."""
        return self.frame.copy()

    def __repr__(self) -> str:
        filled = sum(int(self.frame[c].sum()) for c in self.frame.columns if c.startswith("imputed"))
        return f"ImputationResult(rows={len(self.frame)}, filled={filled})"


@dataclass(frozen=True, eq=False)
class EmbeddingResult:
    """One embedding vector per ``(item, target)`` series.

    Attributes:
        item_ids: One identifier per row of :attr:`embeddings`.
        targets: The target column of each row.
        embeddings: ``[row, dim]`` array.
        item_id_column: Item column name from the schema, or ``None``.
        metadata: Model and checkpoint.
    """

    item_ids: tuple[Hashable, ...]
    targets: tuple[str, ...]
    embeddings: np.ndarray
    item_id_column: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.array(self.embeddings, dtype=float)
        array.setflags(write=False)
        object.__setattr__(self, "embeddings", array)

    def __len__(self) -> int:
        return len(self.item_ids)

    def __repr__(self) -> str:
        return f"EmbeddingResult(series={len(self)}, dim={self.dim})"

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return int(self.embeddings.shape[1])

    def to_pandas(self) -> pd.DataFrame:
        """A wide frame: the item column, ``target`` and ``emb_0 ... emb_<dim-1>``."""
        data: dict[str, Any] = {}
        if self.item_id_column is not None:
            data[self.item_id_column] = pd.Index(self.item_ids)
        data["target"] = list(self.targets)
        frame = pd.DataFrame(data)
        values = pd.DataFrame(self.embeddings, columns=[f"emb_{i}" for i in range(self.dim)])
        return pd.concat([frame, values], axis=1)

    def similarity(self) -> np.ndarray:
        """``[row, row]`` cosine similarity between the series."""
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        unit = self.embeddings / np.maximum(norms, 1e-12)
        return unit @ unit.T


def check_forecasting_only(pipeline_kwargs: Mapping[str, Any], *, context: str) -> None:
    """Refuse a non-forecasting ``task_type`` up front."""
    from ..registry.errors import ConfigError

    task = pipeline_kwargs.get("task_type")
    if task is not None and task != "forecasting":
        raise ConfigError(
            f"{context} ranks models by forecast accuracy, so task_type={task!r} does "
            f"not apply. Drop it, or build a TimeSeriesPipeline directly for that task."
        )


def conformal_p_values(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``(1 + #{ref >= s}) / (n + 1)`` for every finite score; ``NaN`` elsewhere."""
    ref = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    scores = np.asarray(scores, dtype=float)
    out = np.full(scores.shape, np.nan)
    finite = np.isfinite(scores)
    if ref.size:
        greater_equal = ref.size - np.searchsorted(ref, scores[finite], side="left")
        out[finite] = (1.0 + greater_equal) / (ref.size + 1.0)
    return out


def _tail_probability(y: float, grid: np.ndarray, levels: np.ndarray) -> float:
    """Two-sided tail probability of ``y`` under the piecewise-linear CDF of a quantile grid."""
    width = max(grid[-1] - grid[0], 1e-12)
    if y <= grid[0]:
        cdf = levels[0] * np.exp(-(grid[0] - y) / width)
    elif y >= grid[-1]:
        cdf = 1.0 - (1.0 - levels[-1]) * np.exp(-(y - grid[-1]) / width)
    else:
        cdf = float(np.interp(y, grid + np.arange(len(grid)) * 1e-12, levels))
    return float(np.clip(2.0 * min(cdf, 1.0 - cdf), 1e-12, 1.0))


def _rolling_origins(
    panel: TimeSeriesPanel, stride: int, min_context: int, context: int | None
) -> tuple[TimeSeriesPanel, list[tuple[int, int]]]:
    """One panel row per (row, origin): the ``context`` observations before the origin."""
    item_ids, values, lasts, targets, pairs = [], [], [], [], []
    for row, series in enumerate(panel.values):
        stamps = pd.date_range(end=panel.last_timestamps[row], periods=len(series), freq=panel.freq)
        for origin in range(min_context, len(series), stride):
            history = series[max(0, origin - context) if context else 0 : origin]
            if not np.isfinite(history).any():
                continue
            item_ids.append((row, origin))
            values.append(history)
            lasts.append(stamps[origin - 1])
            targets.append(panel.target_names[row])
            pairs.append((row, origin))
    rolled = TimeSeriesPanel(
        item_ids=tuple(item_ids),
        values=tuple(values),
        last_timestamps=tuple(lasts),
        freq=panel.freq,
        target_names=tuple(targets),
    )
    return rolled, pairs


def score_anomalies(
    panel: TimeSeriesPanel,
    forecaster: Forecaster,
    *,
    method: str = "forecast_error",
    horizon: int = 1,
    stride: int | None = None,
    min_context: int = 32,
    context: int | None = None,
    coverage: float = 0.98,
    alpha: float = 0.01,
    threshold: str | float = "conformal",
    contamination: float = 0.01,
    reference_fraction: float = 0.3,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Score every observation of ``panel`` after ``min_context``.

    Args:
        panel: Univariate rows to score (a multivariate item's targets are
            scored independently).
        forecaster: Returns ``(point, quantiles)`` for a panel, a horizon and
            quantile levels.
        method: ``"forecast_error"`` scales ``|y - forecast|`` by the 10-90%
            forecast spread; ``"interval"`` measures the exceedance of the
            central ``coverage`` interval relative to its width;
            ``"likelihood"`` is ``-log`` of the two-sided tail probability.
        horizon: Steps forecast from each origin.
        stride: Steps between origins (default ``horizon``).
        min_context: Observations before the first scored point.
        context: Most recent observations each forecast conditions on.
        coverage: Interval coverage for ``"interval"`` and ``"likelihood"``.
        alpha: Conformal false-alarm rate for ``threshold="conformal"``.
        threshold: ``"conformal"``, ``"contamination"`` or a fixed score.
        contamination: Fraction flagged with ``threshold="contamination"``.
        reference_fraction: Leading fraction of each series' scored points
            used as the conformal reference.

    Returns:
        Per row, arrays ``expected``, ``lower``, ``upper``, ``score``,
        ``p_value`` and ``is_anomaly`` aligned with its history, and a
        metadata dict.
    """
    if method not in ANOMALY_METHODS:
        raise ValueError(f"method must be one of {list(ANOMALY_METHODS)}, got {method!r}")
    stride = stride or horizon
    rolled, pairs = _rolling_origins(panel, stride, min_context, context)
    if not len(rolled):
        raise ValueError(
            f"The series are too short to score: every series needs more than "
            f"min_context={min_context} observations. Lower task_params['min_context']."
        )
    lo, hi = round((1 - coverage) / 2, 10), round((1 + coverage) / 2, 10)
    levels = sorted({0.1, 0.5, 0.9, lo, hi})
    point, quantiles = forecaster(rolled, horizon, levels)

    per_row = [
        {k: np.full(len(v), np.nan) for k in ("score", "expected", "lower", "upper")}
        for v in panel.values
    ]
    residuals: list[list[float]] = [[] for _ in panel.values]
    level_array = np.asarray(levels)
    for r, (row, origin) in enumerate(pairs):
        out = per_row[row]
        for h in range(horizon):
            t = origin + h
            if t >= len(panel.values[row]):
                break
            y, expected = panel.values[row][t], point[r, h]
            out["expected"][t] = expected
            q = None if quantiles is None else quantiles[r, h]
            if q is not None:
                out["lower"][t], out["upper"][t] = q[levels.index(lo)], q[levels.index(hi)]
            if not np.isfinite(y):
                continue
            if method == "forecast_error":
                spread = (q[levels.index(0.9)] - q[levels.index(0.1)]) / 2.563 if q is not None else 0.0
                out["score"][t] = abs(y - expected) / spread if spread > 0 else abs(y - expected)
                residuals[row].append(y - expected)
            elif q is None:
                raise ValueError(
                    f"method={method!r} needs quantile forecasts; use method='forecast_error'."
                )
            elif method == "interval":
                width = max(out["upper"][t] - out["lower"][t], 1e-12)
                out["score"][t] = max(out["lower"][t] - y, y - out["upper"][t], 0.0) / width
            else:
                out["score"][t] = -np.log(_tail_probability(y, q, level_array))
    if method == "forecast_error" and quantiles is None:
        for row, out in enumerate(per_row):
            res = np.asarray(residuals[row])
            mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826 if res.size else 1.0
            out["score"] = out["score"] / max(mad, 1e-12)

    reference = np.concatenate(
        [
            out["score"][np.flatnonzero(np.isfinite(out["score"]))][
                : max(1, int(np.isfinite(out["score"]).sum() * reference_fraction))
            ]
            for out in per_row
        ]
    )
    all_scores = np.concatenate([out["score"] for out in per_row])
    finite = all_scores[np.isfinite(all_scores)]
    if threshold == "conformal":
        if reference.size < int(np.ceil(1 / alpha)) - 1:
            import warnings

            warnings.warn(
                f"Only {reference.size} reference scores for alpha={alpha}: the smallest "
                f"attainable p-value is {1 / (reference.size + 1):.3g}, so nothing can be "
                f"flagged. Use longer series or a larger alpha.",
                UserWarning,
                stacklevel=3,
            )
        limit = None
    elif threshold == "contamination":
        limit = float(np.quantile(finite, 1 - contamination)) if finite.size else np.inf
    else:
        limit = float(threshold)
    for out in per_row:
        scores = out["score"]
        if limit is None:
            out["p_value"] = conformal_p_values(scores, reference)
            flags = out["p_value"] <= alpha
        else:
            out["p_value"] = np.full(len(scores), np.nan)
            flags = scores >= limit
        out["is_anomaly"] = np.where(np.isfinite(scores), flags, False).astype(bool)
    info = {
        "method": method,
        "threshold": threshold,
        "alpha": alpha,
        "coverage": coverage,
        "horizon": horizon,
        "stride": stride,
        "min_context": min_context,
        "reference_points": int(reference.size),
    }
    return per_row, info


def find_gaps(values: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` index ranges of consecutive ``NaN``."""
    missing = np.isnan(np.asarray(values, dtype=float))
    edges = np.diff(np.concatenate([[0], missing.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _interpolate(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.interpolate(limit_direction="both").to_numpy(dtype=float)


def fill_gaps(
    panel: TimeSeriesPanel,
    forecaster: Forecaster,
    *,
    method: str = "bidirectional",
    context: int | None = 512,
    min_context: int = 8,
    native_missing: bool = False,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Fill the ``NaN`` values of every row of ``panel``.

    Args:
        panel: Rows with missing values.
        forecaster: Returns ``(point, quantiles)`` for a panel, a horizon and
            quantile levels.
        method: ``"bidirectional"`` blends forward and backward forecasts;
            ``"forecast"`` uses the forward pass only.
        context: Observations on each side a gap forecast conditions on.
        min_context: Observations a side needs to be forecast from. A gap
            with too little on both sides is linearly interpolated.
        native_missing: Whether the model accepts ``NaN`` in its context;
            otherwise other gaps inside a context are interpolated first.

    Every gap forecast is dated at the gap's own timestamps, in both
    directions; a backward forecast runs on the reversed series, so a model
    with calendar features sees its context dates in reverse order.

    Returns:
        Per row, the filled values and the ``(lower, upper)`` 10-90% band
        (``NaN`` outside the gaps).
    """
    if method not in ("bidirectional", "forecast"):
        raise ValueError(f"method must be 'bidirectional' or 'forecast', got {method!r}")
    levels = (0.1, 0.5, 0.9)
    requests: list[tuple[int, int, int, np.ndarray, str]] = []
    for row, values in enumerate(panel.values):
        smooth = values if native_missing else _interpolate(values)
        for start, stop in find_gaps(values):
            before, after = smooth[:start], smooth[stop:]
            if np.isfinite(values[:start]).sum() >= min_context:
                requests.append((row, start, stop, before[-context:] if context else before, "forward"))
            if method == "bidirectional" and np.isfinite(values[stop:]).sum() >= min_context:
                backward = after[:context] if context else after
                requests.append((row, start, stop, np.ascontiguousarray(backward[::-1]), "backward"))

    results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    by_length: dict[int, list[int]] = {}
    for i, (_, start, stop, _, _) in enumerate(requests):
        by_length.setdefault(stop - start, []).append(i)
    stamps = {row: rolled_timestamps(panel, row) for row in {r[0] for r in requests}}
    for length, members in by_length.items():
        batch = replace(
            panel,
            item_ids=tuple(range(len(members))),
            values=tuple(requests[i][3] for i in members),
            last_timestamps=tuple(
                stamps[requests[i][0]][requests[i][1] - 1]
                if requests[i][1] > 0
                else stamps[requests[i][0]][0] - pd.tseries.frequencies.to_offset(panel.freq)
                for i in members
            ),
            target_names=tuple(panel.target_names[requests[i][0]] for i in members),
            past_covariates=(),
            future_covariates=(),
        )
        point, quantiles = forecaster(batch, length, levels)
        for r, i in enumerate(members):
            p = np.asarray(point[r], dtype=float)
            lo = quantiles[r, :, 0] if quantiles is not None else np.full(length, np.nan)
            hi = quantiles[r, :, -1] if quantiles is not None else np.full(length, np.nan)
            if requests[i][4] == "backward":
                p, lo, hi = p[::-1], lo[::-1], hi[::-1]
            results[i] = (p, np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))

    parts: dict[tuple[int, int, int], dict[str, tuple[np.ndarray, ...]]] = {}
    for i, (row, start, stop, _, direction) in enumerate(requests):
        parts.setdefault((row, start, stop), {})[direction] = results[i]

    filled = [np.array(v, dtype=float) for v in panel.values]
    bands = [(np.full(len(v), np.nan), np.full(len(v), np.nan)) for v in panel.values]
    for (row, start, stop), sides in parts.items():
        if len(sides) == 2:
            w = (np.arange(stop - start) + 1) / (stop - start + 1)
            blend = [(1 - w) * f + w * b for f, b in zip(sides["forward"], sides["backward"], strict=True)]
        else:
            blend = list(next(iter(sides.values())))
        filled[row][start:stop] = blend[0]
        bands[row][0][start:stop] = blend[1]
        bands[row][1][start:stop] = blend[2]
    for row, values in enumerate(filled):
        if np.isnan(values).any():
            filled[row] = _interpolate(values)
    return filled, bands


def rolled_timestamps(panel: TimeSeriesPanel, row: int) -> pd.DatetimeIndex:
    """Observation timestamps of a panel row."""
    return pd.date_range(end=panel.last_timestamps[row], periods=len(panel.values[row]), freq=panel.freq)


def horizon_timestamps(panel: TimeSeriesPanel, horizon: int) -> np.ndarray:
    """``[row, horizon]`` forecast timestamps of a panel."""
    return np.stack(
        [future_timestamps(last, panel.freq, horizon).to_numpy() for last in panel.last_timestamps]
    )

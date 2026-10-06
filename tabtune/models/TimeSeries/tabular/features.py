"""Features that turn a forecasting problem into a tabular regression.

``time`` (TabPFN-TS, Hoo et al. 2025)
    One row per time step of one series: a running index, calendar sin/cos
    pairs, sin/cos pairs at the series' dominant FFT periods, and any covariate
    known over the horizon. Ports PriorLabs/tabpfn-time-series v1.3.0
    (``features/basic_features.py``, ``features/auto_features.py``) exactly.

``lags`` (direct multi-horizon, TabTune)
    One row per (series, origin ``t``, horizon ``h``), pooled over series:
    scaled lags up to ``t``, rolling statistics, the seasonal lag of the target
    step, ``h``, calendar features of ``t + h``, known covariates at ``t + h``,
    past covariates at ``t`` and static features. Training rows need
    ``t + h <= T``; test rows are the origin ``T`` with ``h = 1..H``.

Everything here is numpy/pandas only: no torch, no model code.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CALENDAR_SEASONALITIES",
    "LagDesign",
    "auto_seasonal_features",
    "calendar_features",
    "calendar_index",
    "detrend",
    "find_seasonal_periods",
    "lag_design",
    "time_design",
]

#: ``(gluonts feature, natural period)`` pairs of TabPFN-TS's ``CalendarFeature``, upstream order.
CALENDAR_SEASONALITIES: tuple[tuple[str, float], ...] = (
    ("second_of_minute", 60.0),
    ("minute_of_hour", 60.0),
    ("hour_of_day", 24.0),
    ("day_of_week", 7.0),
    ("day_of_month", 30.5),
    ("day_of_year", 365.0),
    ("week_of_year", 52.0),
    ("month_of_year", 12.0),
)

#: Defaults of TabPFN-TS's ``AutoSeasonalFeature.Config`` (v1.1+; v1.0 used max_top_k=5).
AUTO_SEASONAL_DEFAULTS: dict[str, Any] = {
    "max_top_k": 12,
    "do_detrend": True,
    "detrend_type": "linear",
    "use_peaks_only": True,
    "apply_hann_window": True,
    "zero_padding_factor": 2,
    "round_to_closest_integer": True,
    "validate_with_acf": False,
    "sampling_interval": 1.0,
    "magnitude_threshold": 0.05,
    "relative_threshold": True,
    "exclude_zero": True,
}



def calendar_index(timestamps: pd.DatetimeIndex, name: str) -> np.ndarray:
    """Zero-based calendar index, exactly as ``gluonts.time_feature.<name>_index``.

    ``day_of_month``, ``day_of_year``, ``week_of_year`` (ISO week) and
    ``month_of_year`` are shifted to start at 0; the others are raw.
    """
    ts = pd.DatetimeIndex(timestamps)
    if name == "second_of_minute":
        return np.asarray(ts.second)
    if name == "minute_of_hour":
        return np.asarray(ts.minute)
    if name == "hour_of_day":
        return np.asarray(ts.hour)
    if name == "day_of_week":
        return np.asarray(ts.dayofweek)
    if name == "day_of_month":
        return np.asarray(ts.day) - 1
    if name == "day_of_year":
        return np.asarray(ts.dayofyear) - 1
    if name == "week_of_year":
        return np.asarray(ts.isocalendar().week, dtype=np.int64) - 1
    if name == "month_of_year":
        return np.asarray(ts.month) - 1
    raise ValueError(f"Unknown calendar feature {name!r}")


def calendar_features(
    timestamps: pd.DatetimeIndex,
    *,
    components: Sequence[str] = ("year",),
    seasonalities: Sequence[tuple[str, float]] = CALENDAR_SEASONALITIES,
) -> dict[str, np.ndarray]:
    """TabPFN-TS ``CalendarFeature``: raw components, then sin/cos per calendar index.

    The angle is ``2*pi*f/(P - 1)`` with ``f`` the zero-based index and ``P``
    the natural period; the ``- 1`` is upstream's and makes the last index
    coincide with the first (hour 23 maps to the angle of hour 0).
    """
    ts = pd.DatetimeIndex(timestamps)
    out: dict[str, np.ndarray] = {}
    for component in components:
        out[component] = np.asarray(getattr(ts, component))
    for name, period in seasonalities:
        index = calendar_index(ts, name).astype(np.int32)
        angle = 2 * np.pi * index / (period - 1)
        out[f"{name}_sin"] = np.sin(angle)
        out[f"{name}_cos"] = np.cos(angle)
    return out



def detrend(x: np.ndarray, detrend_type: str) -> np.ndarray:
    """Remove a trend: ``linear`` (least squares), ``first_diff``, ``constant`` or ``loess``."""
    if detrend_type == "first_diff":
        return np.diff(x, prepend=x[0])
    if detrend_type == "loess":
        from statsmodels.api import nonparametric  # optional; upstream's only use of it

        indices = np.arange(len(x))
        return x - nonparametric.lowess(x, indices, frac=0.1)[:, 1]
    if detrend_type == "linear":
        indices = np.arange(len(x))
        if len(x) < 2:
            return x - x.mean() if len(x) else x
        coeffs = np.polyfit(indices, x, 1, rcond=None)
        return x - np.polyval(coeffs, indices)
    if detrend_type == "constant":
        return x - np.mean(x)
    raise ValueError(f"Invalid detrend method: {detrend_type}")


def find_seasonal_periods(
    values: np.ndarray,
    max_top_k: int = 10,
    do_detrend: bool = True,
    detrend_type: str = "first_diff",
    use_peaks_only: bool = True,
    apply_hann_window: bool = True,
    zero_padding_factor: int = 2,
    round_to_closest_integer: bool = True,
    validate_with_acf: bool = False,
    sampling_interval: float = 1.0,
    magnitude_threshold: float | None = 0.05,
    relative_threshold: bool = True,
    exclude_zero: bool = False,
) -> list[tuple[float, float]]:
    """Dominant periods by FFT, as ``AutoSeasonalFeature.find_seasonal_periods``.

    Steps (upstream order): drop ``NaN``; detrend; Hann window; zero-pad by
    ``zero_padding_factor``; ``rfft`` magnitudes with the DC bin zeroed; keep
    local peaks above ``magnitude_threshold`` (relative to the maximum when
    ``relative_threshold``), falling back to every bin when there is no peak;
    take the ``max_top_k`` largest; convert frequency to period; round; drop
    zero periods if asked; keep unique periods; sort by magnitude, descending.

    Returns ``(period, magnitude)`` pairs. Series shorter than 2 observations
    give no periods (upstream fails on them); a constant series follows upstream
    and yields periods from its rounding-noise spectrum.
    """
    from scipy.signal import find_peaks

    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n_original = len(x)
    if n_original < 2:
        return []
    if do_detrend:
        x = detrend(x, detrend_type)
    if apply_hann_window:
        x = x * np.hanning(n_original)
    if zero_padding_factor > 1:
        padded = np.zeros(int(n_original * zero_padding_factor))
        padded[:n_original] = x
        x = padded
    n = len(x)
    magnitudes = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=sampling_interval)
    magnitudes[0] = 0.0
    if not np.isfinite(magnitudes).all():
        return []
    if magnitude_threshold is not None and relative_threshold:
        threshold = magnitude_threshold * np.max(magnitudes)
    else:
        threshold = magnitude_threshold
    if use_peaks_only:
        if threshold is not None:
            peaks, _ = find_peaks(magnitudes, height=threshold)
        else:
            peaks, _ = find_peaks(magnitudes)
        if len(peaks) == 0:
            peaks = np.arange(len(magnitudes))
        top = peaks[np.argsort(magnitudes[peaks])[::-1]][:max_top_k]
    else:
        order = np.argsort(magnitudes)[::-1]
        if threshold is not None:
            order = np.array([i for i in order if magnitudes[i] >= threshold], dtype=int)
        top = order[:max_top_k]
    periods = np.zeros_like(freqs)
    nonzero = freqs > 0
    periods[nonzero] = 1.0 / freqs[nonzero]
    top_periods = periods[top]
    if round_to_closest_integer:
        top_periods = np.round(top_periods)
    if exclude_zero:
        keep = top_periods != 0
        top_periods, top = top_periods[keep], top[keep]
    if len(top_periods) > 0:
        unique = np.unique(top_periods, return_index=True)[1]
        top_periods, top = top_periods[unique], top[unique]
    results = [(float(top_periods[i]), float(magnitudes[top[i]])) for i in range(len(top))]
    if validate_with_acf:  # pragma: no cover - off by default, needs statsmodels
        from statsmodels.tsa.stattools import acf

        raw = np.asarray(values, dtype=float)[:n_original]
        acf_values = acf(raw, nlags=n_original, fft=True)
        acf_peaks, _ = find_peaks(acf_values, height=1.96 / np.sqrt(n_original))
        validated = [
            (p, m)
            for p, m in results
            if int(round(p)) < len(acf_values) and any(abs(int(round(p)) - k) <= 1 for k in acf_peaks)
        ]
        results = validated or results
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


def auto_seasonal_features(
    running_index: np.ndarray, periods: Sequence[float], max_top_k: int
) -> dict[str, np.ndarray]:
    """``sin_#i`` / ``cos_#i`` at ``2*pi*r/p_i`` for the first ``max_top_k`` periods; unused slots are 0."""
    r = np.asarray(running_index, dtype=float)
    out: dict[str, np.ndarray] = {}
    for i in range(max_top_k):
        if i < len(periods) and periods[i]:
            angle = 2 * np.pi * r / float(periods[i])
            out[f"sin_#{i}"] = np.sin(angle)
            out[f"cos_#{i}"] = np.cos(angle)
        else:
            out[f"sin_#{i}"] = np.zeros_like(r)
            out[f"cos_#{i}"] = np.zeros_like(r)
    return out



def _encode_covariates(
    past: Mapping[str, np.ndarray], future: Mapping[str, np.ndarray], names: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Numeric covariates as float; others as consistent integer codes (NaN stays NaN)."""
    hist: dict[str, np.ndarray] = {}
    fut: dict[str, np.ndarray] = {}
    for name in names:
        h, f = np.asarray(past[name]), np.asarray(future[name])
        if h.dtype.kind in "biuf" and f.dtype.kind in "biuf":
            hist[name], fut[name] = h.astype(float), f.astype(float)
            continue
        both = pd.Series(np.concatenate([h, f]), dtype="object")
        codes, _ = pd.factorize(both, sort=True)
        codes = codes.astype(float)
        codes[codes < 0] = np.nan
        hist[name], fut[name] = codes[: len(h)], codes[len(h) :]
    return hist, fut


def time_design(
    context_values: np.ndarray,
    context_timestamps: pd.DatetimeIndex,
    future_timestamps: pd.DatetimeIndex,
    *,
    past_covariates: Mapping[str, np.ndarray] | None = None,
    future_covariates: Mapping[str, np.ndarray] | None = None,
    max_top_k: int = 12,
    seasonal_config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """TabPFN-TS design for one series: ``(X_train, y_train, X_test)``.

    ``context_values`` must be free of ``NaN`` (TabPFN-TS drops missing rows
    before featurising; see :func:`tabtune.models.TimeSeries.tabular.handle_missing`).
    Columns, in upstream order: known covariates (present in both
    ``past_covariates`` and ``future_covariates``), ``running_index``,
    ``year``, the calendar sin/cos pairs, then ``sin_#i`` / ``cos_#i``.
    """
    y = np.asarray(context_values, dtype=float)
    if np.isnan(y).any():
        raise ValueError("time_design needs a context without missing values")
    n_train, n_test = len(y), len(future_timestamps)
    past = dict(past_covariates or {})
    future = dict(future_covariates or {})
    known = [name for name in past if name in future]
    hist_cov, fut_cov = _encode_covariates(
        {k: np.asarray(past[k])[-n_train:] for k in known},
        {k: np.asarray(future[k])[:n_test] for k in known},
        known,
    )
    running = np.arange(n_train + n_test)
    stamps = pd.DatetimeIndex(context_timestamps).append(pd.DatetimeIndex(future_timestamps))
    config = {**AUTO_SEASONAL_DEFAULTS, **dict(seasonal_config or {}), "max_top_k": max_top_k}
    periods = [p for p, _ in find_seasonal_periods(y, **config)]

    columns: dict[str, np.ndarray] = {}
    for name in known:
        columns[name] = np.concatenate([hist_cov[name], fut_cov[name]])
    columns["running_index"] = running
    columns.update(calendar_features(stamps))
    columns.update(auto_seasonal_features(running, periods, max_top_k))
    frame = pd.DataFrame(columns)
    generated = [c for c in frame.columns if c not in known and frame[c].dtype == np.float64]
    frame = frame.astype({c: np.float32 for c in generated})  # upstream's lossless downcast
    return frame.iloc[:n_train].reset_index(drop=True), y, frame.iloc[n_train:].reset_index(drop=True)



@dataclass
class LagDesign:
    """Pooled direct multi-horizon design over a panel.

    Attributes:
        X_train, y_train: Training rows (scaled targets).
        X_test: One row per (series, h), series-major, ``h = 1..H``.
        loc, scale: Per-series affine scaling; forecasts are ``z * scale + loc``.
        rows_per_series: Training rows contributed by each series.
    """

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    loc: np.ndarray
    scale: np.ndarray
    rows_per_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


def _scaling(values: np.ndarray, method: str) -> tuple[float, float]:
    observed = values[np.isfinite(values)]
    if observed.size == 0 or method == "none":
        return 0.0, 1.0
    if method == "standard":
        loc = float(observed.mean())
        scale = float(observed.std())
    elif method == "mean_abs":
        loc, scale = 0.0, float(np.abs(observed).mean())
    else:
        raise ValueError(f"scaling must be 'standard', 'mean_abs' or 'none', got {method!r}")
    if not np.isfinite(scale) or scale <= 1e-8 * max(1.0, abs(loc)):
        scale = 1.0
    return loc, scale


def _encode_block(values: np.ndarray) -> np.ndarray:
    """Numeric arrays as float; anything else as sorted integer codes (NaN for missing)."""
    arr = np.asarray(values)
    if arr.dtype.kind in "biuf":
        return arr.astype(float)
    series = pd.Series(arr, dtype="object")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        return numeric.to_numpy(dtype=float)
    codes, _ = pd.factorize(series.astype("object"), sort=True)
    out = codes.astype(float)
    out[codes < 0] = np.nan
    return out


def _encode_jointly(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Encode several arrays of one variable with a single shared mapping.

    A category must get the same code in every series, context and horizon.
    """
    blocks = [np.asarray(a) for a in arrays]
    if all(b.dtype.kind in "biuf" for b in blocks):
        return [b.astype(float) for b in blocks]
    series = pd.Series(np.concatenate([b.astype(object).ravel() for b in blocks]), dtype="object")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        flat = numeric.to_numpy(dtype=float)
    else:
        labels = series.where(series.isna(), series.astype(str))  # mixed types sort as text
        codes, _ = pd.factorize(labels, sort=True)
        flat = codes.astype(float)
        flat[codes < 0] = np.nan
    out, start = [], 0
    for b in blocks:
        out.append(flat[start:start + b.size])
        start += b.size
    return out


def lag_design(
    values: Sequence[np.ndarray],
    last_timestamps: Sequence[pd.Timestamp],
    freq: str,
    horizon: int,
    *,
    lags: int = 24,
    season: int | None = None,
    windows: Sequence[int] = (7, 28),
    seasonal_lags: int = 3,
    max_train_rows: int = 10_000,
    scaling: str = "standard",
    past_covariates: Sequence[Mapping[str, np.ndarray]] | None = None,
    future_covariates: Sequence[Mapping[str, np.ndarray]] | None = None,
    static_features: Sequence[Mapping[str, Any]] | None = None,
    item_ids: Sequence[Hashable] | None = None,
    calendar: bool = True,
) -> LagDesign:
    """Build the pooled direct multi-horizon design (see the module docstring).

    Row ``(i, t, h)`` of series ``i``, origin ``t`` (an observed step) and
    horizon ``h`` holds:

    * ``lag_l = z[t - l + 1]`` for ``l = 1..lags``;
    * ``mean_w`` and ``std_w`` over ``z[t - w + 1 .. t]``, ignoring gaps;
    * ``seasonal_lag_k = z[t + h - k * season * ceil(h / season)]`` for
      ``k = 1..seasonal_lags``: the target's own phase, one, two, ... whole
      seasons back (never later than ``t``), and ``seasonal_mean``, their
      mean ignoring gaps;
    * ``horizon = h``;
    * the calendar sin/cos pairs of the target time;
    * ``known_<c>`` at ``t + h``, ``past_<c>`` at ``t`` and ``static_<c>``.

    Here ``z`` is the scaled series. Training rows are the pairs with
    ``t + h <= T_i`` and ``z[t + h]`` observed. Origins are taken newest first,
    round-robin over series, until ``max_train_rows`` rows exist. Test rows are
    ``(i, T_i, h)`` for ``h = 1..H``, series-major. Columns never observed in
    training are dropped. Non-numeric covariates and static features become
    sorted integer codes.
    """
    n = len(values)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    lags = max(1, int(lags))
    loc = np.zeros(n)
    scale = np.ones(n)
    z_series: list[np.ndarray] = []
    for i, v in enumerate(values):
        arr = np.asarray(v, dtype=float)
        loc[i], scale[i] = _scaling(arr, scaling)
        z_series.append((arr - loc[i]) / scale[i])

    past_blocks = list(past_covariates) if past_covariates is not None else [{} for _ in range(n)]
    future_blocks = list(future_covariates) if future_covariates is not None else [{} for _ in range(n)]
    known = sorted({k for b in future_blocks for k in b} & {k for b in past_blocks for k in b})
    past_only = sorted({k for b in past_blocks for k in b} - set(known))
    lengths = [len(np.asarray(v)) for v in values]
    known_enc: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for name in known:
        pasts = [np.asarray(past_blocks[i].get(name, np.full(lengths[i], np.nan))) for i in range(n)]
        futs = [np.asarray(future_blocks[i].get(name, np.full(horizon, np.nan))) for i in range(n)]
        encoded = _encode_jointly(pasts + futs)
        known_enc[name] = (encoded[:n], encoded[n:])
    past_enc: dict[str, list[np.ndarray]] = {
        name: _encode_jointly([np.asarray(past_blocks[i].get(name, np.full(lengths[i], np.nan))) for i in range(n)])
        for name in past_only
    }
    static_blocks = list(static_features) if static_features is not None else [{} for _ in range(n)]
    static_names = sorted({k for b in static_blocks for k in b})
    static_codes = {
        name: _encode_block(np.array([b.get(name) for b in static_blocks], dtype=object))
        for name in static_names
    }

    queues = [
        [t for t in range(len(z) - 2, -1, -1) if np.isfinite(z[t])] for z in z_series
    ]
    chosen: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n)]
    total, depth = 0, 0
    steps_all = np.arange(1, horizon + 1)
    while total < max_train_rows and any(depth < len(q) for q in queues):
        for i, origins in enumerate(queues):
            if depth >= len(origins) or total >= max_train_rows:
                continue
            t = origins[depth]
            z = z_series[i]
            steps = steps_all[t + steps_all <= len(z) - 1]
            steps = steps[np.isfinite(z[t + steps])][: max_train_rows - total]
            if steps.size:
                chosen[i].append((t, steps))
                total += int(steps.size)
        depth += 1

    def block(i: int, origins: np.ndarray, steps: np.ndarray) -> dict[str, np.ndarray]:
        """Feature columns for origins x steps of series i (vectorised)."""
        z = z_series[i]
        n_i = len(z)
        stamps = pd.date_range(end=last_timestamps[i], periods=n_i, freq=freq).append(
            pd.date_range(start=last_timestamps[i], periods=horizon + 1, freq=freq)[1:]
        )
        s = pd.Series(z)
        cols: dict[str, np.ndarray] = {}

        def at(array: np.ndarray, idx: np.ndarray) -> np.ndarray:
            out = np.full(idx.shape, np.nan)
            valid = (idx >= 0) & (idx < len(array))
            out[valid] = array[idx[valid]]
            return out

        for lag in range(lags):
            cols[f"lag_{lag + 1}"] = at(z, origins - lag)
        for w in windows:
            roll = s.rolling(int(w), min_periods=1)
            cols[f"mean_{w}"] = at(roll.mean().to_numpy(), origins)
            cols[f"std_{w}"] = at(roll.std(ddof=0).to_numpy(), origins)
            # a single observation has no spread
            counts = at(s.notna().astype(float).rolling(int(w), min_periods=1).sum().to_numpy(), origins)
            cols[f"std_{w}"][counts < 2] = np.nan
        target = origins + steps
        if season and season > 1 and seasonal_lags > 0:
            span = season * np.ceil(steps / season).astype(int)
            stacked = []
            for k in range(1, seasonal_lags + 1):
                cols[f"seasonal_lag_{k}"] = at(z, target - k * span)
                stacked.append(cols[f"seasonal_lag_{k}"])
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN
                cols["seasonal_mean"] = np.nanmean(np.stack(stacked), axis=0)
        cols["horizon"] = steps.astype(float)
        if calendar:
            for name, arr in calendar_features(stamps).items():
                cols[name] = np.asarray(arr, dtype=float)[target]
        for name in known:
            past, fut = known_enc[name][0][i], known_enc[name][1][i]
            full = np.concatenate([past[:n_i], np.resize(fut, horizon) if len(fut) else np.full(horizon, np.nan)])
            cols[f"known_{name}"] = at(full, target)
        for name in past_only:
            cols[f"past_{name}"] = at(past_enc[name][i], origins)
        for name in static_names:
            cols[f"static_{name}"] = np.full(len(origins), static_codes[name][i])
        return cols

    train_parts: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    rows_per_series = np.zeros(n, dtype=int)
    for i, pairs in enumerate(chosen):
        if not pairs:
            continue
        z = z_series[i]
        o = np.concatenate([np.full(len(s), t) for t, s in pairs])
        h = np.concatenate([s for _, s in pairs])
        train_parts.append(pd.DataFrame(block(i, o, h)))
        targets.append(z[o + h])
        rows_per_series[i] = len(o)
    test_parts = [
        pd.DataFrame(block(i, np.full(horizon, len(z) - 1), np.arange(1, horizon + 1)))
        for i, z in enumerate(z_series)
    ]
    X_test = pd.concat(test_parts, ignore_index=True)
    if train_parts:
        X_train = pd.concat(train_parts, ignore_index=True)[list(X_test.columns)]
        y_train = np.concatenate(targets)
    else:
        X_train, y_train = pd.DataFrame(columns=X_test.columns, dtype=float), np.zeros(0)
    # Some learners reject all-NaN columns.
    empty = [c for c in X_train.columns if X_train[c].isna().all()]
    if empty and len(X_train):
        X_train, X_test = X_train.drop(columns=empty), X_test.drop(columns=empty)
    return LagDesign(
        X_train=X_train.reset_index(drop=True),
        y_train=np.asarray(y_train, dtype=float),
        X_test=X_test.reset_index(drop=True),
        loc=loc,
        scale=scale,
        rows_per_series=rows_per_series,
    )

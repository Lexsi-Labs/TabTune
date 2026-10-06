"""Design matrices of the tabular forecasters (``tabtune.models.TimeSeries.tabular.features``).

The TabPFN-TS time design is pinned against golden features produced by the
upstream feature generators (PriorLabs/tabpfn-time-series v1.3.0,
``tests/data/tabpfn_ts_golden.json``; generator: ``tests/data/make_tabpfn_ts_golden.py``).
The lag design is checked for leakage by building it on a series whose value
is its own index, so every feature can be traced back to a time step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabtune.models.TimeSeries.tabular.features import (
    AUTO_SEASONAL_DEFAULTS,
    CALENDAR_SEASONALITIES,
    calendar_features,
    calendar_index,
    find_seasonal_periods,
    lag_design,
    time_design,
)

pytestmark = [pytest.mark.unit, pytest.mark.time_series]

GOLDEN = Path(__file__).parent / "data" / "tabpfn_ts_golden.json"




@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


def test_golden_file_names_its_upstream_source(golden):
    assert "tabpfn-time-series@e4637598" in golden["source"]
    assert {c["freq"] for c in golden["cases"]} == {"h", "D", "15min", "W-SUN", "MS"}


@pytest.mark.parametrize("case", range(5))
def test_time_design_matches_upstream_tabpfn_ts_features(golden, case):
    c = golden["cases"][case]
    n, h = c["n"], c["horizon"]
    stamps = pd.date_range(c["start"], periods=n + h, freq=c["freq"])
    promo = np.asarray(c["promo"])
    X_train, y_train, X_test = time_design(
        np.asarray(c["y"]),
        stamps[:n],
        stamps[n:],
        past_covariates={"promo": promo[:n]},
        future_covariates={"promo": promo[n:]},
        max_top_k=12,
    )
    assert list(X_train.columns) == c["columns"] == list(X_test.columns)
    ours = pd.concat([X_train, X_test]).to_numpy(dtype=float)
    np.testing.assert_allclose(ours, np.asarray(c["X"]), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(y_train, np.asarray(c["y"]))
    assert [p for p, _ in find_seasonal_periods(np.asarray(c["y"]), **AUTO_SEASONAL_DEFAULTS)] == c["periods"]


def test_generated_features_are_float32_and_inputs_keep_their_dtype():
    stamps = pd.date_range("2024-01-01", periods=30, freq="D")
    X_train, _, _ = time_design(
        np.arange(20.0),
        stamps[:20],
        stamps[20:],
        past_covariates={"price": np.arange(20.0)},
        future_covariates={"price": np.arange(10.0)},
    )
    assert X_train["price"].dtype == np.float64
    assert X_train["hour_of_day_sin"].dtype == np.float32
    assert X_train["running_index"].tolist() == list(range(20))


def test_time_design_rejects_missing_context_and_ignores_past_only_covariates():
    stamps = pd.date_range("2024-01-01", periods=12, freq="D")
    with pytest.raises(ValueError, match="missing"):
        time_design(np.array([1.0, np.nan, 2.0]), stamps[:3], stamps[3:5])
    X_train, _, X_test = time_design(
        np.arange(8.0),
        stamps[:8],
        stamps[8:],
        past_covariates={"known": np.arange(8.0), "past_only": np.arange(8.0)},
        future_covariates={"known": np.arange(4.0)},
    )
    assert "known" in X_train.columns and "past_only" not in X_train.columns


def test_categorical_covariates_get_codes_consistent_over_context_and_horizon():
    stamps = pd.date_range("2024-01-01", periods=10, freq="D")
    past = np.array(["a", "b", "a", "c", "b", "a"], dtype=object)
    future = np.array(["c", "a", "b", "d"], dtype=object)
    X_train, _, X_test = time_design(
        np.arange(6.0), stamps[:6], stamps[6:],
        past_covariates={"store": past}, future_covariates={"store": future},
    )
    codes = dict(
        zip(
            np.concatenate([past, future]),
            np.concatenate([X_train["store"], X_test["store"]]),
            strict=True,
        )
    )
    assert codes == {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}




def test_calendar_indices_match_gluonts():
    time_feature = pytest.importorskip("gluonts.time_feature")
    stamps = pd.DatetimeIndex(
        ["2020-12-31 23:59:59", "2021-01-01", "2021-01-03 12:30:15", "2024-02-29 06:00", "2026-12-28"]
    )
    for name, _ in CALENDAR_SEASONALITIES:
        expected = np.asarray(getattr(time_feature, f"{name}_index")(stamps))
        np.testing.assert_array_equal(calendar_index(stamps, name), expected, err_msg=name)


def test_calendar_angle_keeps_upstreams_period_minus_one():
    features = calendar_features(pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 23:00"]))
    # 2*pi*23/(24-1): hour 23 lands on the same angle as hour 0 (upstream's "- 1")
    np.testing.assert_allclose(features["hour_of_day_sin"], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(features["hour_of_day_cos"], [1.0, 1.0], atol=1e-12)
    assert list(features)[:3] == ["year", "second_of_minute_sin", "second_of_minute_cos"]




def test_fft_finds_the_planted_periods():
    # 1680 = 10 weeks of hours: with 2x zero padding both periods fall on an FFT bin
    # (at 720 steps upstream's rounding turns 168 into 160).
    t = np.arange(24 * 70)
    y = np.sin(2 * np.pi * t / 24) + 0.6 * np.sin(2 * np.pi * t / 168) + 0.01 * t
    periods = [p for p, _ in find_seasonal_periods(y, **AUTO_SEASONAL_DEFAULTS)]
    assert periods[0] == 24.0 and 168.0 in periods[:3]


@pytest.mark.parametrize("values", [[], [3.0], [np.nan] * 5])
def test_series_too_short_for_an_fft_have_no_periods(values):
    assert find_seasonal_periods(np.asarray(values, dtype=float), **AUTO_SEASONAL_DEFAULTS) == []


def test_constant_series_follow_upstream_and_stay_finite():
    # The detrended spectrum is rounding noise; upstream keeps its peaks, so do we.
    periods = find_seasonal_periods(np.full(20, 5.0), **AUTO_SEASONAL_DEFAULTS)
    assert all(np.isfinite(p) and p > 0 for p, _ in periods)




def _index_series(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


def test_lag_design_never_looks_past_the_origin():
    """With y[k] = k and no scaling, every feature is a time index we can check."""
    n, horizon, season = 60, 5, 7
    design = lag_design(
        [_index_series(n)], [pd.Timestamp("2024-03-01")], "D", horizon,
        lags=10, season=season, windows=(3,), scaling="none", max_train_rows=10_000,
    )
    X, y = design.X_train, design.y_train
    origin = X["lag_1"].to_numpy()
    lag_cols = [c for c in X.columns if c.startswith("lag_")]
    assert np.nanmax(X[lag_cols].to_numpy() - origin[:, None]) <= 0  # lags <= origin
    seasonal = [c for c in X.columns if c.startswith("seasonal_lag_")]
    assert seasonal == ["seasonal_lag_1", "seasonal_lag_2", "seasonal_lag_3"]
    for column in [*seasonal, "seasonal_mean"]:
        observed = X[column].notna()
        assert (X.loc[observed, column] <= origin[observed]).all(), column
    # seasonal_lag_1 is the target's phase one whole season back
    h = X["horizon"].to_numpy()
    expected = origin + h - season * np.ceil(h / season)
    np.testing.assert_array_equal(
        X["seasonal_lag_1"].dropna(), expected[X["seasonal_lag_1"].notna()]
    )
    np.testing.assert_array_equal(y, origin + X["horizon"].to_numpy())  # target = t + h
    assert y.max() <= n - 1  # no target beyond the last observation
    # test rows: origin is the last observation, h = 1..H
    np.testing.assert_array_equal(design.X_test["lag_1"], np.full(horizon, n - 1))
    np.testing.assert_array_equal(design.X_test["horizon"], np.arange(1, horizon + 1))


def test_lag_design_is_unchanged_by_values_after_the_cut():
    rng = np.random.default_rng(0)
    y = rng.normal(size=80)
    kwargs = dict(lags=8, season=12, windows=(4,), scaling="standard", max_train_rows=500)
    a = lag_design([y[:60]], [pd.Timestamp("2024-01-01")], "h", 6, **kwargs)
    tampered = y.copy()
    tampered[60:] = 1e6  # "future" values must never reach a design cut at 60
    b = lag_design([tampered[:60]], [pd.Timestamp("2024-01-01")], "h", 6, **kwargs)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_frame_equal(a.X_test, b.X_test)


def test_row_budget_prefers_recent_origins_of_every_series():
    series = [_index_series(200), _index_series(200) + 1000, _index_series(50) + 5000]
    design = lag_design(
        series, [pd.Timestamp("2024-01-01")] * 3, "h", 4,
        lags=4, season=None, windows=(), scaling="none", max_train_rows=120,
    )
    assert len(design.y_train) == 120 and design.rows_per_series.sum() == 120
    # Round-robin over origins: each round every series adds one origin (up to H
    # rows), so the shares differ by at most one origin's rows.
    assert np.ptp(design.rows_per_series) <= 4
    first = design.X_train["lag_1"].to_numpy()
    assert first[first < 1000].min() >= 150  # only the newest origins of series 0


def test_scaling_round_trips_and_constant_series_are_safe():
    values = [np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]), np.full(6, 7.0)]
    design = lag_design(values, [pd.Timestamp("2024-01-01")] * 2, "D", 2, lags=2, season=None, windows=())
    assert design.scale[1] == 1.0 and design.loc[1] == 7.0  # constant: shift only
    z = (values[0] - design.loc[0]) / design.scale[0]
    np.testing.assert_allclose(z * design.scale[0] + design.loc[0], values[0])


def test_covariates_and_static_features_are_encoded_deterministically():
    n = 30
    values = [np.arange(n, dtype=float), np.arange(n, dtype=float) * 2]
    past = [{"promo": np.arange(n) % 2, "weather": np.arange(n, dtype=float)} for _ in range(2)]
    future = [{"promo": np.array([1, 0, 1])} for _ in range(2)]
    static = [{"region": "north", "size": 3}, {"region": "south", "size": 5}]
    kwargs = dict(lags=3, season=None, windows=(), past_covariates=past,
                  future_covariates=future, static_features=static)
    a = lag_design(values, [pd.Timestamp("2024-01-01")] * 2, "D", 3, **kwargs)
    b = lag_design(values, [pd.Timestamp("2024-01-01")] * 2, "D", 3, **kwargs)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    assert {"known_promo", "past_weather", "static_region", "static_size"} <= set(a.X_train.columns)
    assert sorted(a.X_test["static_region"].unique()) == [0.0, 1.0]
    np.testing.assert_array_equal(a.X_test["known_promo"].to_numpy()[:3], [1, 0, 1])


def test_columns_never_observed_in_training_are_dropped():
    design = lag_design(
        [np.arange(10.0)], [pd.Timestamp("2024-01-01")], "D", 2, lags=30, season=None, windows=()
    )
    assert "lag_30" not in design.X_train.columns and "lag_1" in design.X_train.columns
    assert list(design.X_train.columns) == list(design.X_test.columns)


def test_categorical_covariates_share_one_encoding_across_series_and_horizon():
    T, H = 30, 3
    values = [np.arange(T, dtype=float), np.arange(T, dtype=float)]
    past = [{"event": np.array(["none"] * (T - 1) + ["promo"], dtype=object)},
            {"event": np.array(["holiday"] + ["none"] * (T - 1), dtype=object)}]
    future = [{"event": np.array(["promo"] * H, dtype=object)},
              {"event": np.array(["none"] * H, dtype=object)}]
    design = lag_design(values, [pd.Timestamp("2024-01-30")] * 2, "D", H, lags=2, season=None,
                        windows=(), max_train_rows=1000, past_covariates=past, future_covariates=future)
    codes = {"holiday": 0.0, "none": 1.0, "promo": 2.0}  # one sorted mapping for every block
    assert design.X_test["known_event"].tolist() == [codes["promo"]] * H + [codes["none"]] * H
    assert set(design.X_train["known_event"].unique()) <= set(codes.values())
    series_1 = design.X_train.iloc[design.rows_per_series[0]:]
    assert (series_1["known_event"] == codes["none"]).all()  # 'none' is 1 in series 1 too

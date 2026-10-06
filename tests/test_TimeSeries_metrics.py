"""Forecast metrics, conformal calibration, rolling windows and the statistical baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel, split_horizon
from tabtune.TimeSeries.backtest import cut_panel, rolling_windows
from tabtune.TimeSeries.calibration import ConformalCalibrator, conformal_quantile
from tabtune.TimeSeries.metrics import score_forecast, season_length, seasonal_scale

pytestmark = [pytest.mark.unit, pytest.mark.time_series]

SCHEMA = TimeSeriesSchema(target="target", item_id="item_id")


@pytest.mark.parametrize(
    ("freq", "benchmark", "natural"),
    [("h", 24, 24), ("D", 1, 7), ("W-SUN", 1, 52), ("MS", 12, 12), ("QS", 4, 4), ("15min", 96, 4),
     ("B", 5, 5), ("YS", 1, 1), ("s", 3600, 60), ("7h", 1, 1)],
)
def test_season_length_conventions(freq, benchmark, natural):
    assert season_length(freq) == benchmark
    assert season_length(freq, natural=True) == natural


def test_seasonal_scale_falls_back_to_lag_one():
    assert seasonal_scale(np.array([1.0, 2.0, 4.0, 7.0]), 2) == pytest.approx(4.0)
    assert seasonal_scale(np.array([1.0, 3.0, 6.0]), 5) == pytest.approx(2.5)
    assert np.isnan(seasonal_scale(np.array([2.0, 2.0, 2.0]), 1))
    assert seasonal_scale(np.array([1.0, np.nan, 3.0, 4.0]), 1) == pytest.approx(1.0)


def test_score_forecast_matches_hand_computation():
    rows = pd.DataFrame(
        {
            "target": ["y"] * 4,
            "__actual__": [10.0, 12.0, 8.0, 10.0],
            "point": [11.0, 11.0, 9.0, 9.0],
            "0.1": [9.0, 9.0, 7.0, 9.5],
            "0.5": [11.0, 11.0, 9.0, 9.0],
            "0.9": [13.0, 13.0, 11.0, 9.8],
        }
    )
    metrics = score_forecast(rows, [0.1, 0.5, 0.9], {"y": 2.0})
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["mase"] == pytest.approx(0.5)
    assert metrics["smape"] == pytest.approx(np.mean([2 / 21, 2 / 23, 2 / 17, 2 / 19]))
    y = rows["__actual__"].to_numpy()
    q = rows[["0.1", "0.5", "0.9"]].to_numpy()
    levels = np.array([0.1, 0.5, 0.9])
    loss = np.maximum(levels * (y[:, None] - q), (levels - 1) * (y[:, None] - q))
    assert metrics["wql"] == pytest.approx(2 * loss.sum() / (np.abs(y).sum() * 3))
    assert metrics["coverage_80"] == pytest.approx(0.75)


def test_score_forecast_skips_series_without_a_scale():
    rows = pd.DataFrame({"target": ["a", "b"], "__actual__": [1.0, 2.0], "point": [2.0, 2.0]})
    assert score_forecast(rows, [], {"a": 0.5, "b": float("nan")})["mase"] == pytest.approx(2.0)
    assert "mase" not in score_forecast(rows, [], None)


def test_conformal_quantile_is_finite_sample_valid():
    scores = np.arange(1.0, 20.0)
    assert conformal_quantile(scores, 0.9) == 18.0
    assert conformal_quantile(scores, 0.99) == float("inf")
    assert conformal_quantile(np.array([np.nan]), 0.5) == float("inf")


@pytest.mark.parametrize("method", ["cqr", "absolute", "signed"])
def test_calibrator_reaches_nominal_coverage_on_exchangeable_errors(method):
    rng = np.random.default_rng(0)
    n, horizon = 4000, 3
    point = rng.normal(size=(n, horizon))
    y = point + rng.normal(scale=2.0, size=(n, horizon))
    levels = [0.05, 0.5, 0.95]
    narrow = point[..., None] + norm.ppf(levels) * 0.5
    half = n // 2
    calibrator = ConformalCalibrator(method).fit(y[:half], point[:half], narrow[:half], levels)
    out = calibrator.calibrate(point[half:], narrow[half:], levels)
    inside = (y[half:] >= out[..., 0]) & (y[half:] <= out[..., 2])
    assert inside.mean() == pytest.approx(0.9, abs=0.02)
    np.testing.assert_allclose(out[..., 1], point[half:])


def test_cqr_needs_quantiles():
    with pytest.raises(ValueError, match="cqr"):
        ConformalCalibrator("cqr").fit(np.zeros((30, 2)), np.zeros((30, 2)), None, [0.1, 0.9])
    with pytest.raises(ValueError, match="method"):
        ConformalCalibrator("isotonic")


def test_rolling_windows_move_the_origin_back():
    frame = make_panel(2, 50, freq="D", seed=0, covariates=("promo",))
    schema = TimeSeriesSchema(target="target", item_id="item_id", known_covariates=("promo",))
    panel = schema.to_panel(frame)
    windows = rolling_windows(panel, horizon=5, windows=3, known_covariates=("promo",))
    assert [w.index for w in windows] == [0, 1, 2]
    last = panel.last_timestamps[0]
    assert [w.context.last_timestamps[0] for w in windows] == [
        last - pd.Timedelta(days=5),
        last - pd.Timedelta(days=10),
        last - pd.Timedelta(days=15),
    ]
    np.testing.assert_array_equal(windows[1].actual[0], panel.values[0][40:45])
    assert len(windows[2].context.values[0]) == 35
    np.testing.assert_array_equal(
        windows[0].context.future_covariates[0]["promo"], panel.past_covariates[0]["promo"][45:50]
    )


def test_rolling_windows_skip_short_items_and_refuse_when_none_fit():
    frame = pd.concat(
        [
            make_panel(1, 60, freq="D", seed=0),
            make_panel(1, 12, freq="D", seed=1).assign(item_id="short"),
        ]
    )
    panel = SCHEMA.to_panel(frame)
    windows = rolling_windows(panel, horizon=5, windows=2, min_context=5)
    assert [len(w.context) for w in windows] == [2, 1]
    with pytest.raises(ValueError, match="long enough"):
        rolling_windows(panel, horizon=60, windows=1)


def test_cut_panel_keeps_the_frequency_grid():
    frame = make_panel(1, 24, freq="MS", seed=0)
    panel = SCHEMA.to_panel(frame)
    cut = cut_panel(panel, [20], horizon=4)
    assert cut.last_timestamps[0] == pd.Timestamp("2025-08-01")


def _frame():
    t = np.arange(48, dtype=float)
    return pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=48, freq="h"), "target": 10 + np.sin(2 * np.pi * t / 24) + 0.1 * t}
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("Naive", lambda y: np.full(4, y[-1])),
        ("SeasonalNaive", lambda y: y[-24:-20]),
        ("Mean", lambda y: np.full(4, y.mean())),
        ("Drift", lambda y: y[-1] + (y[-1] - y[0]) / 47 * np.arange(1, 5)),
        ("WindowAverage", lambda y: np.full(4, y[-24:].mean())),
    ],
)
def test_baseline_point_forecasts(model, expected):
    frame = _frame()
    forecast = TimeSeriesPipeline(model, forecast_params={"prediction_length": 4}).fit(
        frame, TimeSeriesSchema(target="target")
    ).predict()
    np.testing.assert_allclose(forecast.point[0], expected(frame["target"].to_numpy()), atol=1e-9)
    assert (np.diff(forecast.quantiles, axis=-1) >= 0).all()


def test_baselines_accept_missing_values_and_a_season_override():
    frame = _frame()
    frame.loc[[40, 45], "target"] = np.nan
    pipe = TimeSeriesPipeline(
        "SeasonalNaive", model_params={"season_length": 12}, forecast_params={"prediction_length": 3}
    )
    forecast = pipe.fit(frame, TimeSeriesSchema(target="target")).predict()
    assert np.isfinite(forecast.point).all()
    np.testing.assert_allclose(forecast.point[0], frame["target"].to_numpy()[36:39])


def test_seasonal_naive_is_the_mase_reference_on_its_own_history():
    frame = make_panel(2, 24 * 10, freq="h", seed=3)
    history, actual = split_horizon(frame, SCHEMA, 24)
    metrics = TimeSeriesPipeline("SeasonalNaive", forecast_params={"prediction_length": 24}).fit(
        history, SCHEMA
    ).evaluate(actual)
    assert 0.5 < metrics["mase"] < 2.0

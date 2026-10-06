from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.linear_model import Ridge

from tabtune.models.TimeSeries.tabular import (
    TABPFN_TS_VARIANTS,
    SklearnBackend,
    TabPFNBackend,
    TabPFNTSAdapter,
    handle_missing,
)
from tabtune.models.TimeSeries.tabular.backends import _probit_interpolate, gbm_backend
from tabtune.registry import (
    ConfigError,
    LicenseError,
    get_model_spec,
    get_time_series_model_spec,
    list_time_series_models,
)
from tabtune.TimeSeries import (
    TimeSeriesEnsemble,
    TimeSeriesPipeline,
    TimeSeriesSchema,
    make_panel,
    split_horizon,
)

pytestmark = [pytest.mark.unit, pytest.mark.time_series]

SCHEMA = TimeSeriesSchema(target="target", item_id="item_id")


class RecordingTabPFN:
    """TabPFN regressor stand-in: ridge + Gaussian residual quantiles; records its inputs."""

    fits: list[tuple[np.ndarray, np.ndarray, dict]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        type(self).fits.append((np.asarray(X), np.asarray(y), dict(self.kwargs)))
        self.model = Ridge(alpha=1e-3).fit(X, y)
        self.sigma = float(np.std(y - self.model.predict(X))) + 1e-9
        return self

    def predict(self, X, output_type="mean", quantiles=None):
        assert output_type == "main"
        mu = self.model.predict(X)
        return {
            "mean": mu + 1.0,
            "median": mu,
            "mode": mu - 1.0,
            "quantiles": [mu + norm.ppf(q) * self.sigma for q in quantiles],
        }


@pytest.fixture(autouse=True)
def _reset_recorder():
    RecordingTabPFN.fits = []
    yield
    RecordingTabPFN.fits = []


def _tabpfn_ts(horizon=4, checkpoint=None, quantiles=(0.1, 0.5, 0.9), **params):
    model_params = {"_factory": RecordingTabPFN, "device": "cpu", **params}
    if checkpoint:
        model_params["checkpoint"] = checkpoint
    return TimeSeriesPipeline(
        "TabPFN-TS",
        model_params=model_params,
        forecast_params={"prediction_length": horizon, "quantile_levels": list(quantiles)},
    )


def _seasonal_frame(n_items=4, length=24 * 12, noise=1.0, seed=0, freq="h"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        t = np.arange(length)
        y = 10 + 2 * np.sin(2 * np.pi * t / 24 + i) + rng.normal(scale=noise, size=length)
        stamps = pd.date_range("2024-01-01", periods=length, freq=freq)
        rows.append(pd.DataFrame({"item_id": f"s{i}", "timestamp": stamps, "target": y}))
    return pd.concat(rows, ignore_index=True)


def test_tabpfn_ts_registry_entry():
    spec = get_time_series_model_spec("tabpfn-time-series")
    assert spec.name == "TabPFN-TS" and spec.default_checkpoint == "tabpfn-ts-3.5"
    assert set(spec.checkpoints) == set(TABPFN_TS_VARIANTS)
    assert spec.native_missing and spec.supports_covariates
    assert spec.license.commercial_use_ok is False


def test_one_tabular_forecaster_per_regression_capable_tabular_model():
    from tabtune.registry.catalog import MODEL_SPECS

    expected = {f"TabularTS-{s.name}" for s in MODEL_SPECS if "inference" in s.regression_strategies}
    registered = {s.name for s in list_time_series_models() if s.name.startswith("TabularTS-")}
    assert registered == expected | {"TabularTS-GBM"}
    assert "TabularTS-TabICL" not in expected
    for name in expected:
        spec = get_time_series_model_spec(name)
        assert spec.license == get_model_spec(spec.default_checkpoint).license


def test_commercial_license_mode_refuses_tabpfn_ts():
    with pytest.raises(LicenseError):
        TimeSeriesPipeline("TabPFN-TS", license_mode="commercial", forecast_params={"prediction_length": 2})


def test_tabpfn_ts_forecasts_with_missing_values():
    frame = _seasonal_frame(n_items=2, length=80)
    frame.loc[[5, 6, 90], "target"] = np.nan
    forecast = _tabpfn_ts(horizon=6).fit(frame, SCHEMA).predict()
    assert forecast.point.shape == (2, 6) and forecast.quantiles.shape == (2, 6, 3)
    assert forecast.metadata["point_forecast"] == "median"
    assert (np.diff(forecast.quantiles, axis=-1) >= 0).all()


def test_tabpfn_receives_exactly_the_tabpfn_ts_design():
    from tabtune.models.TimeSeries.tabular.features import time_design

    frame = _seasonal_frame(n_items=1, length=100)
    _tabpfn_ts(horizon=7).fit(frame, SCHEMA).predict()
    X, y, kwargs = RecordingTabPFN.fits[-1]
    stamps = pd.date_range("2024-01-01", periods=107, freq="h")
    X_expected, y_expected, _ = time_design(frame["target"].to_numpy(), stamps[:100], stamps[100:])
    np.testing.assert_allclose(X, X_expected.to_numpy(dtype=float))
    np.testing.assert_array_equal(y, y_expected)
    assert "model_path" not in kwargs and kwargs["device"] == "cpu"


def test_tabpfn_ts_point_is_the_median_output_unless_asked_otherwise():
    frame = _seasonal_frame(n_items=1, length=60)
    median = _tabpfn_ts().fit(frame, SCHEMA).predict().point
    mean = _tabpfn_ts(output_selection="mean").fit(frame, SCHEMA).predict()
    np.testing.assert_allclose(mean.point, median + 1.0, atol=1e-9)
    assert mean.metadata["point_forecast"] == "mean"


def test_missing_rule_drops_gaps_or_fills_nearly_empty_series():
    stamps = pd.date_range("2024-01-01", periods=6, freq="D")
    values, kept_stamps, kept = handle_missing(np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0]), stamps)
    assert values.tolist() == [1.0, 3.0, 5.0, 6.0] and kept.tolist() == [0, 2, 4, 5]
    assert list(kept_stamps) == [stamps[i] for i in (0, 2, 4, 5)]
    values, _, kept = handle_missing(np.array([np.nan, 4.0, np.nan]), stamps[:3])
    assert values.tolist() == [0.0, 4.0, 0.0] and kept.tolist() == [0, 1, 2]


def test_context_is_cut_to_the_variant_limit_after_dropping_gaps():
    frame = _seasonal_frame(n_items=1, length=90)
    frame.loc[frame.index[85], "target"] = np.nan
    _tabpfn_ts(horizon=3, max_context_length=20).fit(frame, SCHEMA).predict()
    X, y, _ = RecordingTabPFN.fits[-1]
    assert len(y) == 20 and np.isfinite(y).all()
    assert X[:, 0].tolist() == list(range(20))


@pytest.mark.parametrize(
    ("checkpoint", "version", "model_path", "context", "top_k"),
    [
        ("tabpfn-ts-3.5", "v3.5", None, 32768, 12),
        ("tabpfn-ts-3", "v3", "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt", 32768, 12),
        ("tabpfn-ts-2", "v2", "tabpfn-v2-regressor-2noar4o2.ckpt", 4096, 5),
    ],
)
def test_checkpoints_select_the_upstream_release_configuration(checkpoint, version, model_path, context, top_k):
    pipe = _tabpfn_ts(horizon=3, checkpoint=checkpoint).fit(_seasonal_frame(n_items=1, length=80), SCHEMA)
    pipe.predict()
    adapter = pipe.adapter_
    assert isinstance(adapter, TabPFNTSAdapter)
    assert (adapter.version, adapter.model_path) == (version, model_path)
    assert (adapter.max_context_length, adapter.max_top_k) == (context, top_k)
    X, _, kwargs = RecordingTabPFN.fits[-1]
    assert kwargs.get("model_path") == model_path
    assert X.shape[1] == 1 + 1 + 16 + 2 * top_k


def test_known_covariates_are_used():
    frame = make_panel(2, 120, covariates=["promo"], seed=4)
    schema = TimeSeriesSchema(target="target", item_id="item_id", known_covariates=("promo",))
    history, actual = split_horizon(frame, schema, 8)
    pipe = _tabpfn_ts(horizon=8)
    forecast = pipe.fit(history, schema, future_df=actual.drop(columns="target")).predict()
    assert forecast.point.shape == (2, 8)
    X, _, _ = RecordingTabPFN.fits[-1]
    promo = history[history["item_id"] == "series_1"]["promo"].to_numpy()
    np.testing.assert_allclose(X[:, 0], promo)
    assert X.shape[1] == 1 + 1 + 1 + 16 + 24


def test_constant_series_return_the_constant_without_calling_tabpfn():
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=30, freq="D"), "target": 4.2})
    forecast = _tabpfn_ts(horizon=5, quantiles=(0.1, 0.9)).fit(frame, TimeSeriesSchema(target="target")).predict()
    np.testing.assert_allclose(forecast.point, 4.2)
    np.testing.assert_allclose(forecast.quantiles, 4.2)
    assert RecordingTabPFN.fits == []


def test_invalid_tabpfn_ts_settings_are_rejected():
    frame = _seasonal_frame(n_items=1, length=40)
    with pytest.raises(ConfigError, match="checkpoint"):
        TimeSeriesPipeline(
            "TabPFN-TS", model_params={"checkpoint": "tabpfn-ts-9"}, forecast_params={"prediction_length": 2}
        )
    with pytest.raises(ConfigError, match="point_forecast"):
        _tabpfn_ts(point_forecast="max").fit(frame, SCHEMA)
    with pytest.raises(ConfigError, match="point_forecast"):
        _tabpfn_ts(output_selection="max").fit(frame, SCHEMA)
    with pytest.raises(ConfigError, match="version"):
        TabPFNBackend("v9")


def test_tabpfn_backend_requests_quantiles_and_rearranges_them():
    backend = TabPFNBackend("v3.5", factory=RecordingTabPFN, device="cpu")
    X = pd.DataFrame({"a": np.arange(30.0), "b": np.sin(np.arange(30.0))})
    y = 2 * X["a"].to_numpy() + np.random.default_rng(0).normal(size=30)
    point, quantiles = backend.fit(X, y).predict(X.iloc[:5], [0.9, 0.1, 0.5])
    assert quantiles.shape == (5, 3) and (np.diff(quantiles, axis=1) >= 0).all()
    point_only, none = backend.predict(X.iloc[:5], [])
    assert none is None and point_only.shape == (5,)


def test_gbm_baseline_uses_the_lag_design():
    pipe = TimeSeriesPipeline("TabularTS-GBM", forecast_params={"prediction_length": 6})
    forecast = pipe.fit(_seasonal_frame(n_items=2, length=24 * 5), SCHEMA).predict()
    assert forecast.quantiles.shape == (2, 6, 3)
    assert pipe.adapter_.features == "lags"
    assert pipe.adapter_.point_forecast == "median"
    np.testing.assert_allclose(forecast.point, forecast.quantiles[..., 1])
    model_point = TimeSeriesPipeline(
        "TabularTS-GBM", model_params={"point": "model"}, forecast_params={"prediction_length": 6}
    )
    model_point.fit(_seasonal_frame(n_items=2, length=24 * 5), SCHEMA)
    assert model_point.predict().metadata["point_forecast"] == "mean"


def test_probit_interpolation_is_exact_for_gaussian_quantiles():
    mu, sigma = np.array([0.0, 5.0]), np.array([1.0, 3.0])
    anchors = [0.1, 0.5, 0.9]
    at_anchors = mu[:, None] + sigma[:, None] * norm.ppf(anchors)[None, :]
    levels = [0.02, 0.1, 0.25, 0.5, 0.7, 0.95, 0.99]
    out = _probit_interpolate(at_anchors, anchors, levels)
    np.testing.assert_allclose(out, mu[:, None] + sigma[:, None] * norm.ppf(levels)[None, :], atol=1e-12)


class _CountingQuantile:
    calls: list[float] = []

    def __init__(self, q):
        self.q = q
        type(self).calls.append(q)

    def fit(self, X, y):
        self.center = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.center + norm.ppf(self.q))


@pytest.mark.parametrize("anchors", [(0.1, 0.5, 0.9), None])
def test_anchor_levels_fit_only_the_anchors(anchors):
    _CountingQuantile.calls = []
    backend = SklearnBackend(lambda: Ridge(), _CountingQuantile, anchor_levels=anchors)
    X = pd.DataFrame({"x": np.arange(20.0)})
    backend.fit(X, np.arange(20.0))
    levels = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
    point, quantiles = backend.predict(X.iloc[:3], levels)
    assert point.shape == (3,) and quantiles.shape == (3, len(levels))
    assert sorted(_CountingQuantile.calls) == sorted(anchors or levels)
    np.testing.assert_allclose(quantiles, 9.5 + norm.ppf(levels)[None, :].repeat(3, 0), atol=1e-9)
    assert backend.clone().anchor_levels == (None if anchors is None else list(anchors))


def test_invalid_anchor_levels_are_rejected():
    for anchors in ([0.5], [0.0, 0.5], [0.5, 1.2]):
        with pytest.raises(ConfigError, match="anchor_levels"):
            SklearnBackend(lambda: Ridge(), _CountingQuantile, anchor_levels=anchors)
    assert gbm_backend(anchor_levels=None).anchor_levels is None
    assert gbm_backend().anchor_levels == [0.1, 0.5, 0.9]


def test_gbm_beats_seasonal_naive_on_noisy_seasonal_series():
    frame = _seasonal_frame(n_items=4, length=24 * 14, noise=1.0, seed=1)
    history, actual = split_horizon(frame, SCHEMA, 24)
    params = {"prediction_length": 24}
    gbm = TimeSeriesPipeline("TabularTS-GBM", forecast_params=params).fit(history, SCHEMA).evaluate(actual)
    naive = TimeSeriesPipeline("SeasonalNaive", forecast_params=params).fit(history, SCHEMA).evaluate(actual)
    # On this data the MAE ratio is 0.72-0.88 across seeds and shifts by a few points
    # between scikit-learn releases; a forecaster that misses the seasonality is above 1.
    assert gbm["mae"] < 0.95 * naive["mae"]


def test_xrfm_forecasts_through_the_real_tabular_pipeline():
    frame = _seasonal_frame(n_items=2, length=96)
    pipe = TimeSeriesPipeline("TabularTS-XRFM", forecast_params={"prediction_length": 6})
    forecast = pipe.fit(frame, SCHEMA).predict()
    assert forecast.point.shape == (2, 6) and forecast.quantiles is None
    calibrated = pipe.calibrate(method="absolute", windows=3).predict()
    assert calibrated.quantiles.shape == (2, 6, 3)


def test_custom_backend_serves_any_tabular_forecaster_entry():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, QuantileRegressor
    from sklearn.pipeline import make_pipeline

    def point():
        return make_pipeline(SimpleImputer(), LinearRegression())

    def quantile(level):
        return make_pipeline(SimpleImputer(), QuantileRegressor(quantile=level, alpha=0.0))

    frame = _seasonal_frame(n_items=2, length=60)
    probabilistic = TimeSeriesPipeline(
        "TabularTS-TabICLv2",
        model_params={"backend": SklearnBackend(point, quantile), "features": "lags"},
        forecast_params={"prediction_length": 6},
    )
    assert probabilistic.fit(frame, SCHEMA).predict().quantiles.shape == (2, 6, 3)
    point_only = TimeSeriesPipeline(
        "TabularTS-Mitra",
        model_params={"backend": SklearnBackend(point), "features": "lags"},
        forecast_params={"prediction_length": 6},
    )
    assert point_only.fit(frame, SCHEMA).predict().quantiles is None


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"features": "fourier"}, "features"),
        ({"scaling": "minmax"}, "scaling"),
        ({"max_train_rows": 0}, "max_train_rows"),
        ({"backend": object()}, "backend"),
        ({"point": "mode"}, "point"),
    ],
)
def test_tabular_forecaster_settings_are_validated(params, match):
    pipe = TimeSeriesPipeline("TabularTS-GBM", model_params=params, forecast_params={"prediction_length": 2})
    with pytest.raises(ConfigError, match=match):
        pipe.fit(_seasonal_frame(n_items=1, length=40), SCHEMA)


def test_gbm_round_trips_through_save_and_load(tmp_path):
    frame = _seasonal_frame(n_items=2, length=80)
    pipe = TimeSeriesPipeline(
        "TabularTS-GBM", tuning_params={"seed": 0}, forecast_params={"prediction_length": 5}
    ).fit(frame, SCHEMA)
    before = pipe.predict().point
    pipe.save(str(tmp_path / "gbm.joblib"))
    after = TimeSeriesPipeline.load(str(tmp_path / "gbm.joblib")).predict(frame).point
    np.testing.assert_allclose(after, before)


def test_ensemble_mixes_tabular_and_statistical_forecasters():
    frame = _seasonal_frame(n_items=3, length=24 * 6, noise=0.5, seed=2)
    ensemble = TimeSeriesEnsemble(
        [
            {"model_name": "TabPFN-TS", "model_params": {"_factory": RecordingTabPFN}},
            "TabularTS-GBM",
            "SeasonalNaive",
        ],
        forecast_params={"prediction_length": 12},
        verbose=False,
    ).fit(frame, SCHEMA)
    forecast = ensemble.predict()
    assert forecast.point.shape == (3, 12) and forecast.quantiles.shape == (3, 12, 3)
    assert set(ensemble.weights_) == {"TabPFN-TS", "TabularTS-GBM", "SeasonalNaive"}
    assert sum(ensemble.weights_.values()) == pytest.approx(1.0)


def test_selection_ranks_tabular_and_statistical_candidates():
    frame = _seasonal_frame(n_items=3, length=24 * 8, noise=0.5, seed=2)
    pipe = TimeSeriesPipeline.select(
        frame,
        SCHEMA,
        {"prediction_length": 12},
        candidates=["TabularTS-GBM", "SeasonalNaive", "Naive"],
        windows=2,
    )
    assert pipe.model_name in {"TabularTS-GBM", "SeasonalNaive"}
    assert set(pipe.selection_["Model"]) == {"TabularTS-GBM", "SeasonalNaive", "Naive"}
    assert pipe.predict().point.shape == (3, 12)


def test_pipeline_backend_clones_share_one_loaded_pipeline(monkeypatch):
    import importlib

    from tabtune.models.TimeSeries.tabular import PipelineBackend

    pipeline_module = importlib.import_module("tabtune.TabularPipeline.pipeline")
    built = []
    real = pipeline_module.TabularPipeline

    class Counting(real):
        def __init__(self, *args, **kwargs):
            built.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "TabularPipeline", Counting)
    frame = _seasonal_frame(n_items=3, length=24 * 4)
    TimeSeriesPipeline("TabularTS-XRFM", forecast_params={"prediction_length": 4}).fit(frame, SCHEMA).predict()
    assert len(built) == 1
    backend = PipelineBackend("XRFM")
    assert backend.clone()._shared is backend._shared


class _Processor:
    def __init__(self):
        self.calls = 0

    def transform(self, X):
        self.calls += 1
        return np.asarray(X, dtype=float) * 2.0


def _bare_pipeline(model, name):
    from tabtune.TabularPipeline.pipeline import TabularPipeline

    pipeline = TabularPipeline.__new__(TabularPipeline)
    pipeline.model, pipeline.model_name = model, name
    pipeline.task_type, pipeline._is_fitted = "regression", True
    pipeline.processor = _Processor()
    return pipeline


def test_predict_quantiles_covers_the_tabpfn_family_through_the_preprocessor():
    from tabtune.models.regression.tabpfnv35.regressor import TabPFNv35FastRegressorWrapper

    class Fake(TabPFNv35FastRegressorWrapper):
        def __init__(self):
            pass

        def predict(self, X, output_type="mean", quantiles=None):
            assert output_type == "quantiles"
            return [X[:, 0] + q for q in quantiles]

    pipeline = _bare_pipeline(Fake(), "TabPFNv35Fast")
    out = pipeline.predict_quantiles(pd.DataFrame({"a": [1.0, 2.0]}), [0.1, 0.9])
    assert pipeline.processor.calls == 1
    np.testing.assert_allclose(out[0.1], [2.1, 4.1])
    np.testing.assert_allclose(out[0.9], [2.9, 4.9])


def test_predict_quantiles_uses_tabiclv2s_quantile_head_on_raw_input():
    from tabtune.models.tabiclv2.sklearn.regressor import TabICLRegressor

    class Fake(TabICLRegressor):
        def __init__(self):
            pass

        def predict(self, X, output_type="mean", alphas=None):
            assert output_type == "quantiles"
            return np.asarray(X, dtype=float)[:, :1] + np.asarray(alphas)[None, :]

    pipeline = _bare_pipeline(Fake(), "TabICLv2")
    out = pipeline.predict_quantiles(pd.DataFrame({"a": [1.0, 2.0]}), [0.25, 0.75])
    assert pipeline.processor.calls == 0
    np.testing.assert_allclose(out[0.25], [1.25, 2.25])


def test_predict_quantiles_on_other_models_says_what_is_supported():
    pipeline = _bare_pipeline(object(), "Mitra")
    with pytest.raises(NotImplementedError, match="TabICLv2.*ConformalRegressor"):
        pipeline.predict_quantiles(pd.DataFrame({"a": [1.0]}), [0.5])


def _sine(n=360, horizon=24):
    t = np.arange(n + horizon)
    y = 50 + 10 * np.sin(2 * np.pi * t / 24) + 0.02 * t
    stamps = pd.date_range("2024-01-01", periods=n + horizon, freq="h")
    frame = pd.DataFrame({"timestamp": stamps, "target": y})
    return frame.iloc[:n], frame.iloc[n:], y[n - 1]


@pytest.mark.slow
@pytest.mark.weights
@pytest.mark.parametrize("model", ["TabPFN-TS", "TabularTS-TabICLv2"])
def test_released_tabular_forecasters_forecast_a_sine_wave(model):
    history, actual, last = _sine()
    model_params = {"checkpoint": "tabpfn-ts-2"} if model == "TabPFN-TS" else {}
    pipe = TimeSeriesPipeline(model, model_params=model_params, forecast_params={"prediction_length": 24})
    schema = TimeSeriesSchema(target="target")
    forecast = pipe.fit(history, schema).predict()
    naive = np.abs(actual["target"].to_numpy() - last).mean()
    assert np.abs(forecast.point[0] - actual["target"].to_numpy()).mean() < 0.5 * naive
    assert forecast.quantiles is not None


@pytest.mark.unit
def test_superseded_param_spellings_still_work_and_say_so():
    """Superseded model_params spellings map to the canonical keys with a DeprecationWarning."""
    from tabtune._internal.deprecation import reset_warning_cache

    frame = _seasonal_frame(n_items=1, length=40)
    reset_warning_cache()  # warn_once de-duplicates per process
    with pytest.warns(DeprecationWarning, match="max_context"):
        old = _tabpfn_ts(horizon=2, max_context_length=20).fit(frame, SCHEMA).predict()
    new = _tabpfn_ts(horizon=2, max_context=20).fit(frame, SCHEMA).predict()
    np.testing.assert_allclose(old.point, new.point)

    # Both spellings at once is refused; the adapter is built on fit.
    with pytest.raises(ConfigError, match="same setting"):
        _tabpfn_ts(horizon=2, max_context=20, max_context_length=20).fit(frame, SCHEMA)
    reset_warning_cache()

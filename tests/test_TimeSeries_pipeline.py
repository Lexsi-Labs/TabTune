"""Contract tests for TimeSeriesPipeline, run against fake adapters.

No weights, no network and no model library: the fake adapter forecasts the
last observed value, so every number below can be checked by hand.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from tabtune._internal.deprecation import reset_warning_cache
from tabtune.config import ForecastConfig
from tabtune.models.TimeSeries.base import AdapterOutput, TSFMAdapter
from tabtune.registry import (
    ConfigError,
    EnvelopeError,
    ModelNotFoundError,
    TimeSeriesModelSpec,
    UnsupportedStrategyError,
    UnsupportedTaskError,
    register_time_series_model,
)
from tabtune.TimeSeries import ForecastResult, TimeSeriesPipeline, TimeSeriesSchema

pytestmark = [pytest.mark.unit, pytest.mark.time_series]


class LastValueAdapter(TSFMAdapter):
    """Forecasts the last observed value; quantiles are fixed offsets around it."""

    calls = 0
    seen_lengths: list[int] = []

    def load(self) -> None:
        self._model = "loaded"

    def forecast(self, panel, config: ForecastConfig) -> AdapterOutput:
        type(self).calls += 1
        type(self).seen_lengths = [len(v) for v in panel.values]
        last = np.array([v[~np.isnan(v)][-1] for v in panel.values])
        point = np.repeat(last[:, None], config.prediction_length, axis=1)
        offsets = (np.asarray(config.quantile_levels) - 0.5) * 2.0
        quantiles = point[:, :, None] + offsets[None, None, :]
        return AdapterOutput(point=point, quantiles=quantiles)


class WrongShapeAdapter(LastValueAdapter):
    def forecast(self, panel, config):
        return AdapterOutput(point=np.zeros((len(panel), config.prediction_length + 1)))


class NonFiniteAdapter(LastValueAdapter):
    def forecast(self, panel, config):
        output = super().forecast(panel, config)
        point = output.point.copy()
        point[1, 0] = np.nan
        return AdapterOutput(point=point, quantiles=output.quantiles)


def _spec(name="LastValue", adapter=LastValueAdapter, **overrides):
    fields = {
        "name": name,
        "family": "test",
        "adapter": adapter,
        "default_checkpoint": "fake/ckpt",
        "checkpoints": ("fake/ckpt", "fake/other"),
        "max_context": 4,
        "max_horizon": 5,
    }
    fields.update(overrides)
    return TimeSeriesModelSpec(**fields)


SCHEMA = TimeSeriesSchema(target="sales", timestamp="date", item_id="store")


@pytest.fixture(autouse=True)
def fake_model(isolated_ts_registry):
    """Register the fake model for one test; the shared fixture restores the registry."""
    register_time_series_model(_spec(aliases=("last-value",)))
    LastValueAdapter.calls = 0
    LastValueAdapter.seen_lengths = []
    reset_warning_cache()
    yield
    reset_warning_cache()


def _history(n=6):
    return pd.DataFrame(
        {
            "store": ["a"] * n + ["b"] * n,
            "date": list(pd.date_range("2024-01-01", periods=n, freq="D")) * 2,
            "sales": list(np.arange(n, dtype=float)) + list(np.arange(n, dtype=float) + 10),
        }
    )


def _pipe(horizon=2, model="LastValue", **kwargs):
    kwargs.setdefault("tuning_params", {"device": "cpu"})
    return TimeSeriesPipeline(model, forecast_params={"prediction_length": horizon}, **kwargs)


def test_construction_validates_before_loading():
    pipe = _pipe()
    assert pipe.model_name == "LastValue"
    assert pipe.checkpoint == "fake/ckpt"
    assert pipe.adapter_ is None


def test_alias_resolves():
    assert _pipe(model="last_value").model_name == "LastValue"


def test_unknown_model_raises_with_registration_hint():
    with pytest.raises(ModelNotFoundError, match="register_time_series_model"):
        _pipe(model="NoSuchModel")


@pytest.mark.parametrize("validate", [True, False])
def test_unsupported_strategy_raises_even_without_validation(validate):
    with pytest.raises(UnsupportedStrategyError):
        _pipe(tuning_strategy="finetune", validate=validate)


def test_unsupported_task_raises_even_without_validation():
    with pytest.raises(UnsupportedTaskError):
        _pipe(task_type="classification", validate=False)


def test_strategy_declared_by_spec_but_not_runnable_is_rejected_before_loading():
    register_time_series_model(
        _spec(name="Tunable", strategies=frozenset({"inference", "finetune"}))
    )
    pipe = _pipe(model="Tunable", tuning_strategy="finetune", tuning_params={})
    with pytest.raises(UnsupportedStrategyError):
        pipe.fit(_history(), SCHEMA)
    assert pipe.adapter_ is None


@pytest.mark.parametrize("kwargs", [{"envelope_mode": "strict"}, {"license_mode": "open"}])
def test_modes_are_checked_at_construction(kwargs):
    with pytest.raises(ConfigError):
        _pipe(validate=False, **kwargs)


def test_forecast_params_are_validated():
    with pytest.raises(ConfigError):
        TimeSeriesPipeline("LastValue")
    with pytest.raises(ConfigError):
        TimeSeriesPipeline(
            "LastValue", forecast_params={"prediction_length": 1, "quantile_levels": [1.5]}
        )


def test_quantile_levels_are_sorted_and_deduplicated():
    config = ForecastConfig(prediction_length=1, quantile_levels=[0.9, 0.1, 0.9])
    assert config.quantile_levels == [0.1, 0.9]


def test_irrelevant_tuning_params_warn():
    with pytest.warns(UserWarning, match=r"\['epochs', 'learning_rate'\] are ignored"):
        _pipe(tuning_params={"device": "cpu", "epochs": 5, "learning_rate": 1e-3})


def test_context_length_beyond_model_maximum_warns_and_is_capped():
    with pytest.warns(UserWarning, match="exceeds LastValue's maximum context of 4"):
        pipe = TimeSeriesPipeline(
            "LastValue",
            forecast_params={"prediction_length": 1, "context_length": 50},
            tuning_params={"device": "cpu"},
        )
    pipe.fit(_history(n=10), SCHEMA).predict()
    assert LastValueAdapter.seen_lengths == [4, 4]


def test_checkpoint_must_be_validated_or_local(tmp_path):
    with pytest.raises(ConfigError, match="neither a validated checkpoint"):
        _pipe(model_params={"checkpoint": "someone/else"})
    assert _pipe(model_params={"checkpoint": str(tmp_path)}).checkpoint == str(tmp_path)
    assert _pipe(model_params={"checkpoint": "someone/else"}, validate=False).checkpoint == "someone/else"


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        _pipe().predict()


def test_fit_predict_shapes_and_timestamps():
    pipe = _pipe(horizon=3).fit(_history(), SCHEMA)
    assert pipe.training_occurred_ is False
    result = pipe.predict()

    assert isinstance(result, ForecastResult)
    assert result.item_ids == ("a", "b")
    assert result.point.shape == (2, 3)
    assert result.quantiles.shape == (2, 3, 3)
    assert result.quantile_levels == (0.1, 0.5, 0.9)
    np.testing.assert_array_equal(result.point[:, 0], [5.0, 15.0])
    assert pd.Timestamp(result.timestamps[0, 0]) == pd.Timestamp("2024-01-07")
    assert pd.Timestamp(result.timestamps[1, -1]) == pd.Timestamp("2024-01-09")
    assert result.metadata["training_occurred"] is False
    assert result.metadata["point_forecast"] == "mean"


def test_to_pandas_layout():
    frame = _pipe().fit(_history(), SCHEMA).predict().to_pandas()
    assert list(frame.columns) == ["store", "date", "target", "point", "0.1", "0.5", "0.9"]
    assert len(frame) == 4
    assert (frame["target"] == "sales").all()


def test_integer_item_ids_keep_their_dtype():
    history = _history().assign(store=lambda d: d["store"].map({"a": 1, "b": 2}))
    frame = _pipe().fit(history, SCHEMA).predict().to_pandas()
    assert pd.api.types.is_integer_dtype(frame["store"])


def test_single_series_output_has_no_item_column():
    history = _history().query("store == 'a'").drop(columns="store")
    schema = TimeSeriesSchema(target="sales", timestamp="date")
    frame = _pipe().fit(history, schema).predict().to_pandas()
    assert list(frame.columns[:2]) == ["date", "target"]


def test_predict_on_new_history():
    pipe = _pipe().fit(_history(), SCHEMA)
    new = _history(n=8).assign(store=lambda d: d["store"].map({"a": "c", "b": "d"}))
    result = pipe.predict(new)
    assert result.item_ids == ("c", "d")
    np.testing.assert_array_equal(result.point[:, 0], [7.0, 17.0])


def test_context_is_truncated_to_max_context():
    _pipe().fit(_history(n=10), SCHEMA).predict()
    assert LastValueAdapter.seen_lengths == [4, 4]


def test_context_length_below_max_context():
    TimeSeriesPipeline(
        "LastValue", forecast_params={"prediction_length": 1, "context_length": 2},
        tuning_params={"device": "cpu"},
    ).fit(_history(), SCHEMA).predict()
    assert LastValueAdapter.seen_lengths == [2, 2]


def test_all_missing_context_window_is_rejected():
    register_time_series_model(_spec(name="Gappy", native_missing=True))
    history = _history(n=10)
    history.loc[(history["store"] == "b") & (history["date"] > "2024-01-04"), "sales"] = np.nan
    pipe = _pipe(model="Gappy").fit(history, SCHEMA)  # the full history has observations
    with pytest.raises(ValueError, match="item 'b' has no observed target values in its last 4"):
        pipe.predict()


def test_horizon_beyond_envelope():
    with pytest.warns(UserWarning, match="horizons up to 5"):
        _pipe(horizon=6).fit(_history(), SCHEMA)
    with pytest.raises(EnvelopeError):
        _pipe(horizon=6, envelope_mode="error").fit(_history(), SCHEMA)


def test_adapter_output_shape_is_enforced():
    register_time_series_model(_spec(name="WrongShape", adapter=WrongShapeAdapter))
    with pytest.raises(ValueError, match="expected"):
        _pipe(model="WrongShape").fit(_history(), SCHEMA).predict()


def test_non_finite_adapter_output_is_rejected():
    register_time_series_model(_spec(name="NonFinite", adapter=NonFiniteAdapter))
    with pytest.raises(ValueError, match=r"non-finite point values for item\(s\) \['b'\]"):
        _pipe(model="NonFinite").fit(_history(), SCHEMA).predict()


def test_infinite_history_is_rejected_before_loading():
    history = _history()
    history.loc[1, "sales"] = np.inf
    pipe = _pipe()
    with pytest.raises(ValueError, match="infinite"):
        pipe.fit(history, SCHEMA)
    assert pipe.adapter_ is None


def test_missing_targets_rejected_when_model_lacks_native_support():
    history = _history()
    history.loc[2, "sales"] = np.nan
    with pytest.raises(ValueError, match="missing target"):
        _pipe().fit(history, SCHEMA)


def test_cache_hits_and_invalidation():
    pipe = _pipe(cache="memory").fit(_history(), SCHEMA)
    first = pipe.predict()
    assert pipe.predict() is first
    assert LastValueAdapter.calls == 1

    pipe.fit(_history(), SCHEMA)
    pipe.predict()
    assert LastValueAdapter.calls == 1

    pipe.fit(_history().assign(sales=lambda d: d["sales"] + 1), SCHEMA)
    pipe.predict()
    assert LastValueAdapter.calls == 2

    pipe.forecast_config = ForecastConfig(prediction_length=3)
    assert pipe.predict().prediction_length == 3
    assert LastValueAdapter.calls == 3


def test_disk_cache_is_shared_by_equal_pipelines(tmp_path):
    from tabtune.caching import PredictionCache

    cache = PredictionCache(backend="disk", cache_dir=str(tmp_path))
    first = _pipe(cache=cache).fit(_history(), SCHEMA)
    first.predict()
    _pipe(cache=cache).fit(_history(), SCHEMA).predict()
    assert LastValueAdapter.calls == 1
    first.save(str(tmp_path / "pipe.joblib"))
    TimeSeriesPipeline.load(str(tmp_path / "pipe.joblib")).predict()
    assert LastValueAdapter.calls == 1
    _pipe(cache=cache, model_params={"extra": 1}).fit(_history(), SCHEMA).predict()
    assert LastValueAdapter.calls == 2


def test_cached_results_are_immutable():
    pipe = _pipe(cache="memory").fit(_history(), SCHEMA)
    result = pipe.predict()
    for array in (result.point, result.quantiles, result.timestamps):
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = -1
    np.testing.assert_array_equal(pipe.predict().point[:, 0], [5.0, 15.0])


def test_no_cache_by_default():
    pipe = _pipe().fit(_history(), SCHEMA)
    pipe.predict()
    pipe.predict()
    assert LastValueAdapter.calls == 2


def _actuals():
    return pd.DataFrame(
        {
            "store": ["a", "a", "b", "b"],
            "date": list(pd.date_range("2024-01-07", periods=2, freq="D")) * 2,
            "sales": [6.0, 7.0, 16.0, 17.0],
        }
    )


def _pinball(y, q, alpha):
    diff = y - q
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))


def test_evaluate_matches_hand_computation():
    metrics = _pipe().fit(_history(), SCHEMA).evaluate(_actuals())
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["mse"] == pytest.approx(2.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))

    y = np.array([6.0, 7.0, 16.0, 17.0])
    point = np.array([5.0, 5.0, 15.0, 15.0])
    expected = np.mean([_pinball(y, point + (a - 0.5) * 2, a) for a in (0.1, 0.5, 0.9)])
    assert metrics["mean_pinball_loss"] == pytest.approx(expected)


def test_evaluate_scores_a_given_forecast_without_forecasting_again():
    pipe = _pipe().fit(_history(), SCHEMA)
    forecast = pipe.predict()
    assert pipe.evaluate(_actuals(), forecast=forecast) == pipe.evaluate(_actuals())
    calls = LastValueAdapter.calls
    pipe.evaluate(_actuals(), forecast=forecast)
    assert LastValueAdapter.calls == calls
    with pytest.raises(ValueError, match="either forecast or history"):
        pipe.evaluate(_actuals(), forecast=forecast, history=_history())


def test_evaluate_json_output(capsys, caplog):
    caplog.set_level(logging.WARNING, logger="tabtune")
    metrics = _pipe().fit(_history(), SCHEMA).evaluate(_actuals(), output_format="json")
    assert json.loads(capsys.readouterr().out) == pytest.approx(metrics)
    with pytest.raises(ValueError, match="output_format"):
        _pipe().fit(_history(), SCHEMA).evaluate(_actuals(), output_format="xml")


def test_evaluate_partial_overlap_warns():
    with pytest.warns(UserWarning, match="covers 2 of 4"):
        _pipe().fit(_history(), SCHEMA).evaluate(_actuals().query("store == 'a'"))


def test_evaluate_without_overlap_raises():
    late = _actuals().assign(date=lambda d: d["date"] + pd.Timedelta(days=30))
    with pytest.raises(ValueError, match="forecast window"):
        _pipe().fit(_history(), SCHEMA).evaluate(late)


def test_evaluate_timezone_mismatch_is_explained():
    history = _history().assign(date=lambda d: d["date"].dt.tz_localize("UTC"))
    with pytest.raises(ValueError, match="timezone UTC but df_actual's are in None"):
        _pipe().fit(history, SCHEMA).evaluate(_actuals())



def test_save_load_round_trip(tmp_path):
    pipe = _pipe().fit(_history(), SCHEMA)
    before = pipe.predict()
    path = tmp_path / "pipe.joblib"
    pipe.save(str(path))

    restored = TimeSeriesPipeline.load(str(path))
    assert restored.adapter_ is not None and not restored.adapter_.is_loaded
    after = restored.predict()
    assert restored.adapter_.is_loaded
    np.testing.assert_array_equal(before.point, after.point)
    np.testing.assert_array_equal(before.quantiles, after.quantiles)


def test_save_before_fit_raises(tmp_path):
    with pytest.raises(RuntimeError):
        _pipe().save(str(tmp_path / "x.joblib"))


def test_get_params_round_trips_and_tracks_config_changes():
    pipe = _pipe(cache="memory")
    clone = TimeSeriesPipeline(**pipe.get_params())
    assert clone.get_params() == pipe.get_params()
    pipe.forecast_config = pipe.forecast_config.merged(prediction_length=4)
    assert pipe.get_params()["forecast_params"]["prediction_length"] == 4



def test_adapter_registered_by_import_path(tmp_path, monkeypatch):
    """The lazy ``"module:Class"`` route real models use, not just a class object."""
    (tmp_path / "ts_fake_adapter.py").write_text(
        textwrap.dedent(
            """
            import numpy as np
            from tabtune.models.TimeSeries.base import AdapterOutput, TSFMAdapter

            class ZeroAdapter(TSFMAdapter):
                def load(self):
                    self._model = True

                def forecast(self, panel, config):
                    return AdapterOutput(point=np.zeros((len(panel), config.prediction_length)))
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    register_time_series_model(_spec(name="ByPath", adapter="ts_fake_adapter:ZeroAdapter"))
    result = _pipe(model="ByPath").fit(_history(), SCHEMA).predict()
    assert type(result).__name__ == "ForecastResult"
    assert result.quantiles is None
    np.testing.assert_array_equal(result.point, 0.0)


def test_chronos_resolves_to_its_adapter_without_loading_weights():
    pipe = TimeSeriesPipeline("Chronos", forecast_params={"prediction_length": 2})
    assert pipe._resolve_adapter_class().__name__ == "ChronosAdapter"
    assert pipe.adapter_ is None


def test_missing_adapter_module_is_actionable():
    register_time_series_model(
        _spec(name="Ghost", adapter="tabtune.models.TimeSeries.ghost:GhostAdapter")
    )
    with pytest.raises(ImportError, match="adapter for Ghost"):
        _pipe(model="Ghost").fit(_history(), SCHEMA)


def test_missing_backend_names_the_install_extra(tmp_path, monkeypatch):
    (tmp_path / "needs_backend.py").write_text("import some_backend_that_is_not_installed\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    register_time_series_model(
        _spec(name="NeedsBackend", adapter="needs_backend:Adapter", dependency_extra="ts-backend")
    )
    with pytest.raises(ImportError, match=r"pip install 'tabtune\[ts-backend\]'"):
        _pipe(model="NeedsBackend").fit(_history(), SCHEMA)


def test_pipeline_imports_no_model_backend():
    """The pipeline must stay model-agnostic: no torch and no vendored model code."""
    allowed = {"tabtune.models", "tabtune.models.TimeSeries", "tabtune.models.TimeSeries.base"}
    code = (
        "import sys, tabtune; tabtune.TimeSeriesPipeline; import tabtune.TimeSeries.pipeline; "
        f"allowed = {sorted(allowed)!r}; "
        "loaded = [m for m in sys.modules "
        "if m == 'torch' or (m.startswith('tabtune.models') and m not in allowed)]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


class EchoCovariatesAdapter(LastValueAdapter):
    """Records what actually reached the adapter, so the pipeline's job is visible."""

    seen: dict = {}

    def forecast(self, panel, config):
        type(self).seen = {
            "rows": len(panel),
            "item_groups": panel.item_groups(),
            "targets": panel.target_names,
            "past": tuple(sorted(panel.past_covariates[0])),
            "future": tuple(sorted(panel.future_covariates[0])),
            "future_length": {
                name: len(values) for name, values in panel.future_covariates[0].items()
            },
        }
        return super().forecast(panel, config)


def _rich_history(n=6):
    frame = _history(n)
    frame["traffic"] = frame["sales"] * 10
    frame["temp"] = np.linspace(15.0, 25.0, len(frame))
    frame["weather"] = ["sun", "rain"] * (len(frame) // 2)
    return frame


def _rich_future(horizon=2):
    return pd.DataFrame(
        {
            "store": ["a"] * horizon + ["b"] * horizon,
            "date": list(pd.date_range("2024-01-07", periods=horizon, freq="D")) * 2,
            "temp": np.arange(2.0 * horizon),
            "weather": ["sun"] * (2 * horizon),
        }
    )


CAPABLE = dict(supports_multivariate=True, supports_covariates=True,
               supports_categorical_covariates=True)


def _capable_pipe(horizon=2, **kwargs):
    register_time_series_model(
        _spec(name="Capable", adapter=EchoCovariatesAdapter, max_context=8, **CAPABLE)
    )
    return _pipe(horizon=horizon, model="Capable", **kwargs)


def test_multivariate_schema_becomes_one_panel_row_per_item_and_target():
    schema = TimeSeriesSchema(
        target=("sales", "traffic"), timestamp="date", item_id="store"
    )
    result = _capable_pipe().fit(_rich_history(), schema).predict()

    assert EchoCovariatesAdapter.seen["rows"] == 4
    assert EchoCovariatesAdapter.seen["item_groups"] == ((0, 1), (2, 3))
    assert result.targets == ("sales", "traffic", "sales", "traffic")
    frame = result.to_pandas()
    assert len(frame) == 4 * 2
    assert frame["target"].tolist() == ["sales"] * 2 + ["traffic"] * 2 + ["sales"] * 2 + [
        "traffic"
    ] * 2
    with pytest.raises(AttributeError, match="use .targets"):
        _ = result.target


def test_covariates_reach_the_adapter_with_the_right_lengths():
    schema = TimeSeriesSchema(
        target="sales",
        timestamp="date",
        item_id="store",
        past_covariates=("temp",),
        known_covariates=("weather",),
    )
    _capable_pipe(horizon=2).fit(
        _rich_history(), schema, future_df=_rich_future(2).drop(columns="temp")
    ).predict()

    seen = EchoCovariatesAdapter.seen
    assert seen["past"] == ("temp", "weather")
    assert seen["future"] == ("weather",)
    assert seen["future_length"] == {"weather": 2}


def test_unsupported_features_are_refused_before_any_weights_load():
    frame = _rich_history()
    with pytest.raises(ConfigError, match="one target at a time"):
        _pipe().fit(frame, TimeSeriesSchema(target=("sales", "traffic"), timestamp="date",
                                            item_id="store"))
    with pytest.raises(ConfigError, match="does not use covariates"):
        _pipe().fit(frame, TimeSeriesSchema(target="sales", timestamp="date", item_id="store",
                                            past_covariates=("temp",)))
    assert LastValueAdapter.calls == 0


def test_categorical_covariates_are_refused_by_a_numeric_only_model():
    register_time_series_model(
        _spec(name="NumericOnly", adapter=EchoCovariatesAdapter, max_context=8,
              supports_covariates=True)
    )
    schema = TimeSeriesSchema(
        target="sales", timestamp="date", item_id="store", past_covariates=("weather",)
    )
    with pytest.raises(ConfigError, match="only reads numeric covariates"):
        _pipe(model="NumericOnly").fit(_rich_history(), schema)


def test_known_covariates_without_future_values_are_refused():
    schema = TimeSeriesSchema(
        target="sales", timestamp="date", item_id="store", known_covariates=("temp",)
    )
    pipe = _capable_pipe().fit(_rich_history(), schema)
    with pytest.raises(ConfigError, match=r"predict\(future_df=\.\.\.\)"):
        pipe.predict()


def test_future_covariate_values_are_part_of_the_cache_key():
    schema = TimeSeriesSchema(
        target="sales", timestamp="date", item_id="store", known_covariates=("temp",)
    )
    future = _rich_future(2).drop(columns="weather")
    pipe = _capable_pipe(cache=True).fit(_rich_history(), schema, future_df=future)

    pipe.predict()
    calls_after_first = EchoCovariatesAdapter.calls
    pipe.predict()
    assert EchoCovariatesAdapter.calls == calls_after_first

    pipe.predict(future_df=future.assign(temp=lambda d: d["temp"] + 100))
    assert EchoCovariatesAdapter.calls == calls_after_first + 1


def test_evaluate_scores_each_target_on_its_own_scale():
    schema = TimeSeriesSchema(target=("sales", "traffic"), timestamp="date", item_id="store")
    history = _rich_history()
    pipe = _capable_pipe().fit(history, schema)
    forecast = pipe.predict()

    actual = pd.DataFrame(
        {
            "store": ["a", "a", "b", "b"],
            "date": list(pd.date_range("2024-01-07", periods=2, freq="D")) * 2,
            "sales": [5.0, 5.0, 15.0, 15.0],
            "traffic": [50.0, 50.0, 150.0, 150.0],
        }
    )
    metrics = pipe.evaluate(actual, forecast=forecast)
    assert set(metrics["per_target"]) == {"sales", "traffic"}
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["per_target"]["traffic"]["mae"] == pytest.approx(0.0)

"""Tasks, calibration, backtests and fine-tuning of TimeSeriesPipeline, on fake adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabtune._internal.deprecation import reset_warning_cache
from tabtune.models.TimeSeries.base import AdapterOutput, TSFMAdapter
from tabtune.registry import (
    ConfigError,
    TimeSeriesModelSpec,
    UnsupportedStrategyError,
    UnsupportedTaskError,
    register_time_series_model,
)
from tabtune.TimeSeries import (
    AnomalyResult,
    EmbeddingResult,
    ImputationResult,
    TimeSeriesEnsemble,
    TimeSeriesLeaderboard,
    TimeSeriesPipeline,
    TimeSeriesSchema,
    make_panel,
    split_horizon,
)
from tabtune.TimeSeries.data import make_anomalous_panel

pytestmark = [pytest.mark.unit, pytest.mark.time_series]

SCHEMA = TimeSeriesSchema(target="target", item_id="item_id")


class SeasonalAdapter(TSFMAdapter):
    """Repeats the last 24 steps; quantiles are a fixed band of +/- ``width``."""

    width = 0.2

    def load(self) -> None:
        self._model = "loaded"

    def forecast(self, panel, config):
        point = np.stack(
            [np.resize(pd.Series(v).ffill().bfill().to_numpy()[-24:], config.prediction_length) for v in panel.values]
        )
        z = (np.asarray(config.quantile_levels) - 0.5) * 2 * self.width
        return AdapterOutput(point=point, quantiles=point[..., None] + z)

    def embed(self, panel):
        return np.stack([[np.nanmean(v), np.nanstd(v), len(v)] for v in panel.values])


class PointOnlyAdapter(SeasonalAdapter):
    def forecast(self, panel, config):
        return AdapterOutput(point=super().forecast(panel, config).point)


class ContextSensitiveAdapter(SeasonalAdapter):
    """Forecasts the mean of the context: accurate only with a short context on a trend."""

    def forecast(self, panel, config):
        means = np.array([np.nanmean(v) for v in panel.values])
        point = np.repeat(means[:, None], config.prediction_length, axis=1)
        return AdapterOutput(point=point, quantiles=point[..., None] + 0 * np.asarray(config.quantile_levels))


class TrainableAdapter(SeasonalAdapter):
    """A linear map from the last 8 values to the next ``horizon`` values."""

    lora_targets = ("head",)

    def load(self) -> None:
        import torch

        torch.manual_seed(0)
        self._model = torch.nn.Sequential()
        self._model.add_module("head", torch.nn.Linear(8, 4))
        self._model.to(self.device)
        self._restore_tuned(self._model)

    def _predict(self, torch, x):
        return self._model(x)

    def forecast(self, panel, config):
        import torch

        x = np.stack([np.nan_to_num(v[-8:]) for v in panel.values]).astype(np.float32)
        with torch.no_grad():
            # import_state moves the restored module onto the adapter's device; inputs follow.
            out = (
                self._model(torch.as_tensor(x, device=self.device)).cpu().numpy().astype(float)
            )
        point = np.resize(out, (len(panel), config.prediction_length))
        return AdapterOutput(point=point)

    def _network(self):
        return self._model

    def _training_context(self, spec):
        return 8

    def _training_loss(self, torch, context, label):
        x = torch.as_tensor(np.nan_to_num(context), device=self.device)
        y = torch.as_tensor(label, device=self.device)
        pred = self._model(x)[:, : y.shape[1]]
        mask = ~torch.isnan(y)
        return ((pred - torch.nan_to_num(y)) ** 2)[mask].mean()


def _spec(name, adapter, **overrides):
    fields = dict(
        name=name,
        family="test",
        adapter=adapter,
        default_checkpoint="fake/ckpt",
        checkpoints=("fake/ckpt",),
        native_missing=True,
        tasks=frozenset({"forecasting", "anomaly_detection", "imputation", "embedding"}),
    )
    fields.update(overrides)
    return TimeSeriesModelSpec(**fields)


@pytest.fixture(autouse=True)
def fake_models(isolated_ts_registry):
    register_time_series_model(_spec("Seasonal", SeasonalAdapter))
    register_time_series_model(_spec("PointOnly", PointOnlyAdapter))
    register_time_series_model(_spec("ContextSensitive", ContextSensitiveAdapter))
    register_time_series_model(
        _spec("Trainable", TrainableAdapter, strategies=frozenset({"inference", "finetune", "peft"}))
    )
    register_time_series_model(
        _spec(
            "Untrainable",
            SeasonalAdapter,
            strategies=frozenset({"inference", "finetune"}),
        )
    )
    reset_warning_cache()
    yield
    reset_warning_cache()


def _panel(n=3, length=240, **kwargs):
    return make_panel(n, length, freq="h", seed=0, **kwargs)


def test_task_params_are_validated_per_task():
    with pytest.raises(ConfigError, match="not options of task_type='imputation'"):
        TimeSeriesPipeline("Seasonal", task_type="imputation", task_params={"alpha": 0.1})
    with pytest.raises(UnsupportedTaskError):
        TimeSeriesPipeline("SeasonalNaive", task_type="embedding")


def test_anomaly_detection_finds_injected_spikes():
    frame = make_anomalous_panel(2, 300, seed=3)
    pipe = TimeSeriesPipeline("Seasonal", task_type="anomaly_detection", task_params={"alpha": 0.02})
    result = pipe.fit(frame.drop(columns="is_anomaly"), SCHEMA).predict()
    assert isinstance(result, AnomalyResult)
    table = result.to_pandas()
    assert list(table.columns[:3]) == ["item_id", "timestamp", "target"]
    assert {"score", "p_value", "is_anomaly", "expected", "lower", "upper"} <= set(table.columns)
    assert len(table) == len(frame)
    metrics = result.evaluate(frame[["item_id", "timestamp", "is_anomaly"]])
    assert metrics["auroc"] > 0.95
    assert metrics["recall"] >= 0.75
    assert result.metadata["model"] == "Seasonal"


@pytest.mark.parametrize("method", ["interval", "likelihood"])
def test_quantile_anomaly_methods(method):
    frame = make_anomalous_panel(2, 300, seed=5)
    pipe = TimeSeriesPipeline(
        "Seasonal", task_type="anomaly_detection", task_params={"method": method, "alpha": 0.02}
    )
    metrics = pipe.fit(frame.drop(columns="is_anomaly"), SCHEMA).predict().evaluate(
        frame[["item_id", "timestamp", "is_anomaly"]]
    )
    assert metrics["auroc"] > 0.9


def test_point_only_models_score_residuals_and_refuse_quantile_methods():
    frame = make_anomalous_panel(2, 300, seed=3).drop(columns="is_anomaly")
    result = TimeSeriesPipeline("PointOnly", task_type="anomaly_detection").fit(frame, SCHEMA).predict()
    assert np.isfinite(result.frame["score"]).sum() > 400
    with pytest.raises(ValueError, match="needs quantile forecasts"):
        TimeSeriesPipeline(
            "PointOnly", task_type="anomaly_detection", task_params={"method": "interval"}
        ).fit(frame, SCHEMA).predict()


def test_contamination_threshold_flags_the_requested_fraction():
    frame = make_panel(2, 300, freq="h", seed=1)
    pipe = TimeSeriesPipeline(
        "Seasonal",
        task_type="anomaly_detection",
        task_params={"threshold": "contamination", "contamination": 0.05},
    )
    table = pipe.fit(frame, SCHEMA).predict().frame
    scored = table["score"].notna()
    assert table.loc[scored, "is_anomaly"].mean() == pytest.approx(0.05, abs=0.02)
    assert table.loc[scored, "p_value"].isna().all()


def test_imputation_fills_gaps_and_flags_them():
    full = _panel()
    gappy = full.copy()
    rows = [40, 41, 42, 43, 44, 300, 301]
    gappy.loc[rows, "target"] = np.nan
    result = TimeSeriesPipeline("Seasonal", task_type="imputation").fit(gappy, SCHEMA).predict()
    assert isinstance(result, ImputationResult)
    table = result.to_pandas()
    assert not table["target"].isna().any()
    assert table["imputed"].sum() == len(rows)
    assert table.loc[table["imputed"], "lower"].notna().all()
    error = np.abs(table.loc[rows, "target"].to_numpy() - full.loc[rows, "target"].to_numpy())
    assert error.mean() < 1.0


def test_bidirectional_imputation_meets_both_edges():
    full = make_panel(1, 200, freq="h", seed=2, kind="trend")
    gappy = full.copy()
    gappy.loc[100:111, "target"] = np.nan
    errors = {}
    for method in ("forecast", "bidirectional"):
        pipe = TimeSeriesPipeline("Seasonal", task_type="imputation", task_params={"method": method})
        filled = pipe.fit(gappy, SCHEMA).predict().frame["target"]
        errors[method] = np.abs(filled[100:112] - full["target"][100:112]).mean()
    assert errors["bidirectional"] <= errors["forecast"]


def test_multivariate_imputation_has_one_flag_per_target():
    schema = TimeSeriesSchema(target=("target_0", "target_1"), item_id="item_id")
    frame = make_panel(2, 120, freq="h", n_targets=2)
    frame.loc[10, "target_1"] = np.nan
    register_time_series_model(
        _spec("SeasonalMV", SeasonalAdapter, supports_multivariate=True), overwrite=True
    )
    table = TimeSeriesPipeline("SeasonalMV", task_type="imputation").fit(frame, schema).predict().frame
    assert {"imputed_target_0", "imputed_target_1"} <= set(table.columns)
    assert table["imputed_target_1"].sum() == 1 and table["imputed_target_0"].sum() == 0


def test_embedding_task_returns_one_vector_per_series():
    result = TimeSeriesPipeline("Seasonal", task_type="embedding").fit(_panel(), SCHEMA).predict()
    assert isinstance(result, EmbeddingResult)
    assert result.embeddings.shape == (3, 3) and result.dim == 3
    table = result.to_pandas()
    assert list(table.columns) == ["item_id", "target", "emb_0", "emb_1", "emb_2"]
    similarity = result.similarity()
    np.testing.assert_allclose(np.diag(similarity), 1.0)
    with pytest.raises(ValueError, match="read-only"):
        result.embeddings[0, 0] = 1.0


def test_calibration_widens_an_overconfident_band():
    frame = _panel(n=6, length=400)
    history, actual = split_horizon(frame, SCHEMA, 24)
    params = {"prediction_length": 24, "quantile_levels": [0.05, 0.5, 0.95]}
    pipe = TimeSeriesPipeline("Seasonal", forecast_params=params).fit(history, SCHEMA)
    raw = pipe.predict()
    before = pipe.evaluate(actual, forecast=raw)["coverage_90"]
    pipe.calibrate(windows=5)
    calibrated = pipe.predict()
    after = pipe.evaluate(actual, forecast=calibrated)["coverage_90"]
    assert before < 0.5 < after
    np.testing.assert_array_equal(raw.point, calibrated.point)
    assert calibrated.metadata["calibration"]["method"] == "cqr"
    assert calibrated.metadata["calibration"]["n_calibration"] == 30
    pipe.fit(history, SCHEMA)
    assert pipe.predict().metadata["calibration"] is None


def test_calibration_gives_point_only_models_quantiles():
    history, _ = split_horizon(_panel(n=6, length=400), SCHEMA, 24)
    pipe = TimeSeriesPipeline("PointOnly", forecast_params={"prediction_length": 24}).fit(history, SCHEMA)
    assert pipe.predict().quantiles is None
    forecast = pipe.calibrate(windows=5).predict()
    assert forecast.quantiles.shape == (6, 24, 3)
    assert forecast.metadata["calibration"]["method"] == "absolute"
    assert (np.diff(forecast.quantiles, axis=-1) >= 0).all()


def test_calibration_with_too_few_rows_is_refused():
    history, _ = split_horizon(_panel(n=1, length=200), SCHEMA, 24)
    params = {"prediction_length": 24, "quantile_levels": [0.01, 0.5, 0.99]}
    pipe = TimeSeriesPipeline("Seasonal", forecast_params=params).fit(history, SCHEMA)
    with pytest.warns(UserWarning, match="calibration rows"), pytest.raises(ValueError, match="too few"):
        pipe.calibrate(windows=3)
    assert pipe.calibrator_ is None


def test_backtest_scores_each_window():
    pipe = TimeSeriesPipeline("Seasonal", forecast_params={"prediction_length": 24}).fit(_panel(), SCHEMA)
    table = pipe.backtest(windows=3)
    assert list(table["window"]) == [0, 1, 2]
    assert table["cutoff"].is_monotonic_decreasing
    assert (table["n_series"] == 3).all()
    assert {"mase", "wql", "coverage_80"} <= set(table.columns)


def test_known_covariates_are_backtested_with_their_future_values():
    schema = TimeSeriesSchema(target="target", item_id="item_id", known_covariates=("promo",))
    seen = {}

    class Recorder(SeasonalAdapter):
        def forecast(self, panel, config):
            seen["future"] = [len(row["promo"]) for row in panel.future_covariates]
            return super().forecast(panel, config)

    register_time_series_model(_spec("Recorder", Recorder, supports_covariates=True))
    frame = _panel(covariates=("promo",))
    history, actual = split_horizon(frame, schema, 12)
    pipe = TimeSeriesPipeline("Recorder", forecast_params={"prediction_length": 12})
    pipe.fit(history, schema, future_df=actual.drop(columns="target")).backtest(windows=2)
    assert seen["future"] == [12, 12, 12]


def test_tune_context_length_keeps_the_best_candidate():
    t = np.arange(300, dtype=float)
    frame = pd.DataFrame(
        {
            "item_id": np.repeat(["a", "b"], 300),
            "timestamp": list(pd.date_range("2024-01-01", periods=300, freq="h")) * 2,
            "target": np.concatenate([0.5 * t, 10 - 0.2 * t]),
        }
    )
    pipe = TimeSeriesPipeline("ContextSensitive", forecast_params={"prediction_length": 12}).fit(frame, SCHEMA)
    table = pipe.tune_context_length([8, 64, 256], windows=2)
    assert list(table["context_length"])[0] == 8
    assert pipe.forecast_config.context_length == 8


@pytest.mark.parametrize("strategy", ["finetune", "peft"])
def test_finetuning_trains_and_survives_save_and_load(strategy, tmp_path):
    pytest.importorskip("torch")
    history, _ = split_horizon(_panel(), SCHEMA, 4)
    pipe = TimeSeriesPipeline(
        "Trainable",
        tuning_strategy=strategy,
        tuning_params={"epochs": 2, "steps_per_epoch": 20, "learning_rate": 1e-2, "batch_size": 16, "seed": 0},
        forecast_params={"prediction_length": 4, "quantile_levels": []},
    )
    before = TimeSeriesPipeline(
        "Trainable", forecast_params={"prediction_length": 4, "quantile_levels": []}
    ).fit(history, SCHEMA).predict()
    pipe.fit(history, SCHEMA)
    assert pipe.training_occurred_
    report = pipe.training_report_
    assert report["steps_run"] == 40 and report["mode"] == ("lora" if strategy == "peft" else "full")
    tuned = pipe.predict()
    assert not np.allclose(tuned.point, before.point)
    assert tuned.metadata["training_occurred"] is True
    kind = pipe.adapter_.tuned_state["__meta__"]["kind"]
    assert kind == ("lora" if strategy == "peft" else "full")

    path = tmp_path / "pipe.joblib"
    pipe.save(str(path))
    restored = TimeSeriesPipeline.load(str(path))
    np.testing.assert_allclose(restored.predict().point, tuned.point, rtol=1e-6)


def test_early_stopping_uses_held_out_windows():
    pytest.importorskip("torch")
    history, _ = split_horizon(_panel(), SCHEMA, 4)
    pipe = TimeSeriesPipeline(
        "Trainable",
        tuning_strategy="finetune",
        tuning_params={
            "epochs": 20,
            "steps_per_epoch": 5,
            "learning_rate": 1e-2,
            "early_stopping": True,
            "early_stopping_patience": 2,
            "seed": 0,
        },
        forecast_params={"prediction_length": 4, "quantile_levels": []},
    )
    report = pipe.fit(history, SCHEMA).training_report_
    assert report["best_validation_loss"] is not None
    assert report["history"] and report["history"][0]["step"] == 5


def test_training_params_that_do_not_apply_warn():
    with pytest.warns(UserWarning, match=r"\['finetune_mode'\] are ignored"):
        TimeSeriesPipeline(
            "Trainable",
            tuning_strategy="finetune",
            tuning_params={"finetune_mode": "meta-learning"},
            forecast_params={"prediction_length": 4},
        )


def test_untrainable_adapter_is_refused_before_loading():
    pipe = TimeSeriesPipeline("Untrainable", tuning_strategy="finetune", forecast_params={"prediction_length": 4})
    with pytest.raises(UnsupportedStrategyError):
        pipe.fit(_panel(), SCHEMA)
    assert pipe.adapter_ is None


@pytest.mark.parametrize("task", ["imputation", "anomaly_detection", "embedding"])
def test_non_forecasting_tasks_have_no_forecast_methods(task):
    """Every forecast-only method refuses on a non-forecasting pipeline."""
    pipe = TimeSeriesPipeline("Seasonal", task_type=task).fit(_panel(), SCHEMA)
    calls = (
        pipe.calibrate,
        pipe.backtest,
        pipe.tune_context_length,
        lambda: pipe.evaluate(_panel()),
    )
    for call in calls:
        with pytest.raises(ConfigError, match="task_type='forecasting' only"):
            call()


@pytest.mark.parametrize("task", ["anomaly_detection", "imputation"])
def test_covariates_unusable_by_a_task_are_reported_not_dropped(task):
    """Covariates these tasks cannot use are warned about, not silently dropped."""
    register_time_series_model(
        _spec("SeasonalCov", SeasonalAdapter, supports_covariates=True), overwrite=True
    )
    frame = _panel().copy()
    frame["promo"] = np.arange(len(frame)) % 7
    schema = TimeSeriesSchema(target="target", item_id="item_id", past_covariates=("promo",))
    pipe = TimeSeriesPipeline(
        "SeasonalCov", task_type=task, forecast_params={"prediction_length": 1}
    )
    with pytest.warns(UserWarning, match="not used by task_type"):
        pipe.fit(frame, schema)
    # ... and the imputation frame must not echo a covariate it never used.
    assert "promo" not in pipe.predict().to_pandas().columns


def test_imputation_forecasts_are_dated_at_the_gap():
    from tabtune.TimeSeries.schema import TimeSeriesPanel
    from tabtune.TimeSeries.tasks import fill_gaps

    values = np.arange(40, dtype=float)
    values[20:23] = np.nan
    panel = TimeSeriesPanel(
        item_ids=("a",),
        values=(values,),
        last_timestamps=(pd.Timestamp("2024-03-10"),),
        freq="D",
        target_names=("target",),
    )
    seen = []

    def forecaster(batch, horizon, levels):
        seen.extend(batch.last_timestamps)
        return np.zeros((len(batch), horizon)), None

    fill_gaps(panel, forecaster, method="forecast", min_context=4)
    stamps = pd.date_range(end="2024-03-10", periods=40, freq="D")
    assert seen == [stamps[19]]


def test_anomaly_coverage_defaults_to_what_the_model_predicts():
    from unittest import mock

    frame = make_panel(1, 80, freq="h", seed=0)
    schema = TimeSeriesSchema(target="target", item_id="item_id")
    pipe = TimeSeriesPipeline("SeasonalNaive", task_type="anomaly_detection").fit(frame, schema)
    with mock.patch.object(type(pipe.adapter_), "quantile_range", return_value=(0.1, 0.9)):
        result = pipe.predict()
    assert result.metadata["coverage"] == pytest.approx(0.8)
    explicit = TimeSeriesPipeline(
        "SeasonalNaive", task_type="anomaly_detection", task_params={"coverage": 0.9}
    ).fit(frame, schema).predict()
    assert explicit.metadata["coverage"] == pytest.approx(0.9)


PROVENANCE = frozenset({"model", "checkpoint", "device", "tuning_strategy", "training_occurred"})


@pytest.mark.parametrize("task", ["forecasting", "anomaly_detection", "imputation", "embedding"])
def test_every_task_records_the_same_provenance(task):
    """Every task's result carries the same provenance keys."""
    frame = _panel()
    frame.loc[[30, 31], "target"] = np.nan
    pipe = TimeSeriesPipeline(
        "Seasonal", task_type=task, forecast_params={"prediction_length": 4}
    ).fit(frame, SCHEMA)
    metadata = pipe.predict().metadata
    assert PROVENANCE <= set(metadata), f"{task} is missing {PROVENANCE - set(metadata)}"
    assert metadata["model"] == "Seasonal"
    assert metadata["tuning_strategy"] == "inference"
    assert metadata["training_occurred"] is False
    assert metadata["device"]


def test_a_fine_tuned_non_forecasting_result_records_the_training():
    """The case the missing keys hid: training is invisible in the result."""
    pytest.importorskip("torch")
    pipe = TimeSeriesPipeline(
        "Trainable",
        task_type="anomaly_detection",
        tuning_strategy="peft",
        tuning_params={"epochs": 1, "steps_per_epoch": 2, "batch_size": 4, "seed": 0},
    ).fit(_panel(), SCHEMA)
    metadata = pipe.predict().metadata
    assert metadata["training_occurred"] is True
    assert metadata["tuning_strategy"] == "peft"


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda frame, schema: TimeSeriesPipeline.select(
                frame, schema, {"prediction_length": 4}, task_type="imputation"
            ),
            id="select",
        ),
        pytest.param(
            lambda frame, schema: TimeSeriesEnsemble(
                [{"model_name": "Seasonal", "task_type": "imputation"}],
                "mean",
                forecast_params={"prediction_length": 4},
            ),
            id="ensemble",
        ),
    ],
)
def test_ranking_entry_points_refuse_a_non_forecasting_task(build):
    """The ranking entry points refuse a non-forecasting task_type up front."""
    frame = _panel()
    with pytest.raises(ConfigError, match="ranks models by forecast accuracy"):
        build(frame, SCHEMA)


def test_leaderboard_refuses_a_non_forecasting_task():
    board = TimeSeriesLeaderboard(_panel(), SCHEMA, forecast_params={"prediction_length": 4})
    with pytest.raises(ConfigError, match="ranks models by forecast accuracy"):
        board.add_model("Seasonal", task_type="imputation")


def test_frame_results_take_their_own_copy():
    """Frame-backed results copy the frame instead of holding a view."""
    source = pd.DataFrame({"value": [1.0, 2.0], "is_anomaly": [False, True]})
    anomaly = AnomalyResult(source, method="forecast_error")
    imputation = ImputationResult(source)
    source.loc[0, "value"] = 99.0

    assert anomaly.frame.loc[0, "value"] == 1.0
    assert imputation.frame.loc[0, "value"] == 1.0
    # to_pandas() stays the accessor for a frame to modify.
    assert anomaly.to_pandas() is not anomaly.frame

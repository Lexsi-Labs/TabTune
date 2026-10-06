"""TiRex-2 adapter on tiny random checkpoints (sLSTM and mLSTM time mixers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabtune.registry import ConfigError, get_time_series_model_spec
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel

pytestmark = [pytest.mark.unit, pytest.mark.time_series]

torch = pytest.importorskip("torch")

SCHEMA = TimeSeriesSchema(target="target", item_id="item_id")


def _pipe(checkpoint, horizon=6, **kwargs):
    model_params = {"checkpoint": checkpoint, "device": "cpu", **kwargs.pop("model_params", {})}
    return TimeSeriesPipeline(
        "TiRex2", model_params=model_params, forecast_params={"prediction_length": horizon}, **kwargs
    )


def _ts(*rows, past=None, future=None):
    from tabtune.models.tirex2 import TimeseriesType

    tensor = lambda a: None if a is None else torch.as_tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
    return TimeseriesType(tensor(np.stack(rows)), tensor(past), tensor(future))


def _covariate_frames(horizon):
    frame = make_panel(2, 48 + horizon, freq="h", seed=4, covariates=("temp", "promo"))
    cut = frame.groupby("item_id").cumcount() < 48
    return frame[cut].reset_index(drop=True), frame[~cut].drop(columns="target").reset_index(drop=True)


def test_registry_entry():
    spec = get_time_series_model_spec("TiRex-2")
    assert spec.name == "TiRex2" and spec.dependency_extra is None
    assert spec.native_missing and spec.supports_multivariate and spec.supports_covariates
    assert {"embedding", "forecasting"} <= spec.tasks
    assert spec.strategies == {"inference", "finetune", "peft"}


def test_no_third_party_model_package_is_imported():
    import sys

    import tabtune.models.tirex2  # noqa: F401

    assert not {"xlstm", "flashrnn", "mlstm_kernels", "tirex2"} & set(sys.modules)


def test_forecast_matches_the_model(tiny_tirex2_checkpoint):
    frame = make_panel(3, 60, freq="h", seed=0)
    pipe = _pipe(tiny_tirex2_checkpoint).fit(frame, SCHEMA)
    forecast = pipe.predict()
    assert forecast.point.shape == (3, 6) and forecast.quantiles.shape == (3, 6, 3)
    values = frame.loc[frame["item_id"] == "series_2", "target"].to_numpy()
    reference = pipe.adapter_._model.forecast([_ts(values)], 6, output_type="numpy")[0][0]
    np.testing.assert_allclose(forecast.point[2], reference[4], atol=1e-6)
    np.testing.assert_allclose(forecast.quantiles[2], np.sort(reference[[0, 4, 8]].T, axis=-1), atol=1e-6)


def test_long_horizons_unroll_on_the_median(tiny_tirex2_checkpoint):
    frame = make_panel(2, 60, freq="h", seed=1)
    long = _pipe(tiny_tirex2_checkpoint, horizon=20, envelope_mode="ignore").fit(frame, SCHEMA).predict()
    short = _pipe(tiny_tirex2_checkpoint, horizon=8).fit(frame, SCHEMA).predict()
    assert long.point.shape == (2, 20) and np.isfinite(long.quantiles).all()
    np.testing.assert_allclose(long.point[:, :8], short.point, atol=1e-6)
    model = _pipe(tiny_tirex2_checkpoint).fit(frame, SCHEMA).adapter_._model
    values = frame.loc[frame["item_id"] == "series_0", "target"].to_numpy()
    first = model.forecast([_ts(values)], 8, output_type="numpy")[0][0]
    second = model.forecast([_ts(np.concatenate([values, first[4]]))], 8, output_type="numpy")[0][0]
    np.testing.assert_allclose(long.point[0, 8:16], second[4], atol=1e-5)


def test_targets_of_an_item_are_forecast_jointly(tiny_tirex2_checkpoint):
    frame = make_panel(2, 50, freq="h", seed=3, n_targets=2)
    schema = TimeSeriesSchema(target=["target_0", "target_1"], item_id="item_id")
    pipe = _pipe(tiny_tirex2_checkpoint).fit(frame, schema)
    forecast = pipe.predict()
    assert forecast.point.shape == (4, 6)
    item = frame[frame["item_id"] == "series_1"]
    joint = pipe.adapter_._model.forecast(
        [_ts(item["target_0"].to_numpy(), item["target_1"].to_numpy())], 6, output_type="numpy"
    )[0]
    np.testing.assert_allclose(forecast.point[2:], joint[:, 4], atol=1e-6)


def test_covariates_are_passed_to_the_model(tiny_tirex2_checkpoint):
    history, future = _covariate_frames(6)
    schema = TimeSeriesSchema(
        target="target", item_id="item_id", past_covariates=("temp",), known_covariates=("promo",)
    )
    pipe = _pipe(tiny_tirex2_checkpoint).fit(history, schema, future_df=future)
    forecast = pipe.predict(future_df=future)
    item = history[history["item_id"] == "series_0"]
    ahead = future.loc[future["item_id"] == "series_0", "promo"].to_numpy()
    reference = pipe.adapter_._model.forecast(
        [
            _ts(
                item["target"].to_numpy(),
                past=[item["temp"].to_numpy()],
                future=[np.concatenate([item["promo"].to_numpy(), ahead])],
            )
        ],
        6,
        output_type="numpy",
    )[0][0]
    np.testing.assert_allclose(forecast.point[0], reference[4], rtol=1e-5, atol=1e-5)
    plain = _pipe(tiny_tirex2_checkpoint).fit(history[["item_id", "timestamp", "target"]], SCHEMA).predict()
    assert not np.allclose(plain.point, forecast.point)


def test_non_numeric_covariates_are_refused(tiny_tirex2_checkpoint):
    history, _ = _covariate_frames(6)
    history["temp"] = np.where(history["temp"] > 0, "hot", "cold")
    schema = TimeSeriesSchema(target="target", item_id="item_id", past_covariates=("temp",))
    with pytest.raises((ConfigError, ValueError, TypeError)):
        _pipe(tiny_tirex2_checkpoint).fit(history, schema).predict()


def test_missing_values_and_ragged_batches(tiny_tirex2_checkpoint):
    frame = make_panel(2, 60, freq="h", seed=0)
    frame.loc[[3, 4, 5], "target"] = np.nan
    frame = frame[~((frame["item_id"] == "series_1") & (frame.index % 60 < 20))]
    together = _pipe(tiny_tirex2_checkpoint).fit(frame, SCHEMA).predict()
    alone = _pipe(tiny_tirex2_checkpoint).fit(frame[frame["item_id"] == "series_1"], SCHEMA).predict()
    assert np.isfinite(together.point).all()
    np.testing.assert_allclose(alone.point[0], together.point[1], atol=1e-5)


def test_embeddings_match_upstream(tiny_tirex2_checkpoint):
    frame = make_panel(3, 50, freq="h", seed=0)
    pipe = TimeSeriesPipeline(
        "TiRex2", task_type="embedding", model_params={"checkpoint": tiny_tirex2_checkpoint, "device": "cpu"}
    ).fit(frame, SCHEMA)
    result = pipe.predict()
    assert result.embeddings.shape == (3, 16) and np.isfinite(result.embeddings).all()
    values = frame.loc[frame["item_id"] == "series_1", "target"].to_numpy()
    reference = pipe.adapter_._model.embed([_ts(values)])[0][0].numpy()
    np.testing.assert_allclose(result.embeddings[1], reference, atol=1e-5)


@pytest.mark.parametrize("strategy", ["peft", "finetune"])
def test_fine_tuning_round_trips_through_save_and_load(tiny_tirex2_checkpoint, tmp_path, strategy):
    frame = make_panel(4, 80, freq="h", seed=1)
    zero_shot = _pipe(tiny_tirex2_checkpoint).fit(frame, SCHEMA).predict().point
    pipe = _pipe(
        tiny_tirex2_checkpoint,
        horizon=12,
        envelope_mode="ignore",
        tuning_strategy=strategy,
        tuning_params={"epochs": 1, "steps_per_epoch": 3, "batch_size": 8, "learning_rate": 1e-2, "seed": 0},
    ).fit(frame, SCHEMA)
    report = pipe.training_report_
    assert report["steps_run"] == 3 and np.isfinite(report["final_train_loss"])
    assert "pinball" in report["objective"]
    tuned = pipe.predict().point
    assert tuned.shape == (4, 12) and not np.allclose(tuned[:, :6], zero_shot, atol=1e-7)
    pipe.save(str(tmp_path / "tirex2.joblib"))
    restored = TimeSeriesPipeline.load(str(tmp_path / "tirex2.joblib")).predict().point
    np.testing.assert_allclose(restored, tuned, atol=1e-6)
    again = _pipe(tiny_tirex2_checkpoint).fit(frame, SCHEMA).predict().point
    np.testing.assert_allclose(again, zero_shot, atol=1e-6)


def test_invalid_settings_are_rejected(tiny_tirex2_checkpoint):
    frame = pd.DataFrame(
        {"item_id": "a", "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"), "target": 1.0}
    )
    with pytest.raises(ConfigError, match="float32"):
        _pipe(tiny_tirex2_checkpoint, model_params={"dtype": "bfloat16"}).fit(frame, SCHEMA)
    with pytest.raises(ConfigError, match="batch_size"):
        _pipe(tiny_tirex2_checkpoint, model_params={"batch_size": 0}).fit(frame, SCHEMA)


def test_no_quantiles_requested(tiny_tirex2_checkpoint):
    """No requested levels means no quantile output."""
    pipe = TimeSeriesPipeline(
        "TiRex2",
        model_params={"checkpoint": tiny_tirex2_checkpoint, "device": "cpu"},
        forecast_params={"prediction_length": 4, "quantile_levels": []},
    )
    result = pipe.fit(make_panel(2, 60, freq="h", seed=0), SCHEMA).predict()
    assert result.quantiles is None
    assert result.quantile_levels == ()
    assert result.point.shape == (2, 4)

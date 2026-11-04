"""T0 adapter on a tiny random checkpoint (save_pretrained directory, no download)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from tabtune.registry import ConfigError, get_time_series_model_spec
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel

pytestmark = [pytest.mark.unit, pytest.mark.time_series, pytest.mark.model_t0]

torch = pytest.importorskip("torch")

SCHEMA = TimeSeriesSchema(target="target", item_id="item_id")
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RAW = {"trim_leading_missing": False, "symmetric_averaging": False, "clip_nonnegative": False, "snap_integers": False}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    from tabtune.models.t0 import T0Forecaster

    torch.manual_seed(0)
    model = T0Forecaster(
        embed_dim=16, num_layers=2, num_heads=2, mlp_hidden_dim=32, patch_size=4,
        group_every_n=2, dropout=0.0, quantile_levels=LEVELS,
    )
    path = tmp_path_factory.mktemp("t0")
    model.save_pretrained(path)
    return str(path)


def _pipe(checkpoint, horizon=6, **kwargs):
    model_params = {"checkpoint": checkpoint, "device": "cpu", **kwargs.pop("model_params", {})}
    return TimeSeriesPipeline("T0", model_params=model_params, forecast_params={"prediction_length": horizon}, **kwargs)


def test_registry_entry():
    spec = get_time_series_model_spec("t0-gamma")
    assert spec.name == "T0" and spec.dependency_extra is None
    assert spec.native_missing and spec.supports_multivariate and spec.supports_covariates
    assert not spec.supports_categorical_covariates
    assert {"embedding", "forecasting"} <= spec.tasks
    assert spec.strategies == {"inference", "finetune", "peft"}
    assert spec.license.commercial_use_ok is None


def test_no_third_party_model_package_is_imported():
    import tabtune.models.t0  # noqa: F401
    from tabtune.models.t0 import T0Forecaster  # noqa: F401  (loads the torch runtime)

    assert not {"t0", "t0.torch"} & set(sys.modules)


def test_forecast_matches_the_model(checkpoint):
    frame = make_panel(3, 60, freq="h", seed=0)
    pipe = TimeSeriesPipeline(
        "T0", model_params={"checkpoint": checkpoint, "device": "cpu", **RAW},
        forecast_params={"prediction_length": 6, "quantile_levels": [0.1, 0.5, 0.9]},
    ).fit(frame, SCHEMA)
    forecast = pipe.predict()
    assert forecast.point.shape == (3, 6) and forecast.quantiles.shape == (3, 6, 3)
    values = frame.loc[frame["item_id"] == "series_2", "target"].to_numpy()
    reference = pipe.adapter_._model.predict(torch.as_tensor(values[None], dtype=torch.float32), horizon=6,
                                             quantile_levels=[0.1, 0.5, 0.9]).quantiles[0].numpy()
    np.testing.assert_allclose(forecast.quantiles[2], reference, atol=1e-5)
    np.testing.assert_allclose(forecast.point[2], reference[:, 1], atol=1e-5)


def test_levels_beyond_the_trained_grid_are_served(checkpoint):
    pipe = TimeSeriesPipeline(
        "T0", model_params={"checkpoint": checkpoint, "device": "cpu"},
        forecast_params={"prediction_length": 4, "quantile_levels": [0.02, 0.5, 0.98]},
    ).fit(make_panel(2, 40, freq="h", seed=0), SCHEMA)
    forecast = pipe.predict()
    assert np.isfinite(forecast.quantiles).all()
    assert (forecast.quantiles[..., 0] <= forecast.quantiles[..., 2]).all()
    assert pipe.adapter_.quantile_range() is None


def test_targets_of_an_item_are_forecast_jointly(checkpoint):
    frame = make_panel(2, 50, freq="h", seed=3, n_targets=2)
    schema = TimeSeriesSchema(target=["target_0", "target_1"], item_id="item_id")
    pipe = _pipe(checkpoint, model_params=RAW).fit(frame, schema)
    forecast = pipe.predict()
    assert forecast.point.shape == (4, 6)
    item = frame[frame["item_id"] == "series_1"]
    joint = pipe.adapter_._model.predict(
        torch.as_tensor(np.stack([item["target_0"], item["target_1"]])[None], dtype=torch.float32),
        horizon=6, quantile_levels=[0.5],
    ).quantiles[0, :, :, 0].numpy()
    np.testing.assert_allclose(forecast.point[2:], joint, atol=1e-5)


def test_covariates_reach_the_model(checkpoint):
    horizon = 6
    frame = make_panel(2, 48 + horizon, freq="h", seed=4, covariates=("temp", "promo"))
    cut = frame.groupby("item_id").cumcount() < 48
    history = frame[cut].reset_index(drop=True)
    future = frame[~cut].drop(columns="target").reset_index(drop=True)
    schema = TimeSeriesSchema(target="target", item_id="item_id", past_covariates=("temp",), known_covariates=("promo",))
    pipe = _pipe(checkpoint, model_params=RAW).fit(history, schema, future_df=future)
    covariates = pipe.predict(future_df=future)
    item = history[history["item_id"] == "series_0"]
    ahead = future.loc[future["item_id"] == "series_0", "promo"].to_numpy()
    context = torch.as_tensor(np.stack([item["target"], item["temp"]])[None], dtype=torch.float32)
    known = torch.as_tensor(np.concatenate([item["promo"].to_numpy(), ahead])[None, None], dtype=torch.float32)
    reference = pipe.adapter_._model.predict(context, horizon=horizon, quantile_levels=[0.5],
                                             future_covariates=known).quantiles[0, 0, :, 0].numpy()
    np.testing.assert_allclose(covariates.point[0], reference, atol=1e-5)
    plain = _pipe(checkpoint, model_params=RAW).fit(history[["item_id", "timestamp", "target"]], SCHEMA).predict()
    assert not np.allclose(plain.point, covariates.point)


def test_non_numeric_covariates_are_refused(checkpoint):
    frame = make_panel(2, 54, freq="h", seed=4, covariates=("temp",))
    frame["temp"] = np.where(frame["temp"] > 0, "hot", "cold")
    schema = TimeSeriesSchema(target="target", item_id="item_id", past_covariates=("temp",))
    with pytest.raises((ConfigError, ValueError, TypeError)):
        _pipe(checkpoint).fit(frame, schema).predict()


def test_missing_values_and_ragged_batches(checkpoint):
    frame = make_panel(2, 60, freq="h", seed=0)
    frame.loc[[3, 4, 5], "target"] = np.nan
    frame = frame[~((frame["item_id"] == "series_1") & (frame.index % 60 < 20))]
    together = _pipe(checkpoint).fit(frame, SCHEMA).predict()
    alone = _pipe(checkpoint).fit(frame[frame["item_id"] == "series_1"], SCHEMA).predict()
    assert np.isfinite(together.point).all()
    np.testing.assert_allclose(alone.point[0], together.point[1], atol=1e-4)


def test_embeddings(checkpoint):
    frame = make_panel(3, 50, freq="h", seed=0)
    result = TimeSeriesPipeline(
        "T0", task_type="embedding", model_params={"checkpoint": checkpoint, "device": "cpu"}
    ).fit(frame, SCHEMA).predict()
    assert result.embeddings.shape == (3, 16) and np.isfinite(result.embeddings).all()
    assert not np.allclose(result.embeddings[0], result.embeddings[1])


@pytest.mark.parametrize("strategy", ["peft", "finetune"])
def test_fine_tuning_round_trips_through_save_and_load(checkpoint, tmp_path, strategy):
    frame = make_panel(4, 80, freq="h", seed=1)
    zero_shot = _pipe(checkpoint).fit(frame, SCHEMA).predict().point
    pipe = _pipe(
        checkpoint,
        horizon=12,
        tuning_strategy=strategy,
        tuning_params={"epochs": 1, "steps_per_epoch": 3, "batch_size": 8, "learning_rate": 1e-2, "seed": 0},
    ).fit(frame, SCHEMA)
    report = pipe.training_report_
    assert report["steps_run"] == 3 and np.isfinite(report["final_train_loss"])
    assert "pinball" in report["objective"]
    tuned = pipe.predict().point
    assert tuned.shape == (4, 12) and not np.allclose(tuned[:, :6], zero_shot, atol=1e-7)
    pipe.save(str(tmp_path / "t0.joblib"))
    restored = TimeSeriesPipeline.load(str(tmp_path / "t0.joblib")).predict().point
    np.testing.assert_allclose(restored, tuned, atol=1e-6)
    again = _pipe(checkpoint).fit(frame, SCHEMA).predict().point
    np.testing.assert_allclose(again, zero_shot, atol=1e-6)


def _adapter(checkpoint):
    return _pipe(checkpoint).fit(make_panel(1, 60, freq="h", seed=0), SCHEMA).adapter_


def _windows(seed=0, batch=4, context=48, horizon=12):
    rng = np.random.default_rng(seed)
    series = rng.normal(size=(batch, context + horizon)).cumsum(axis=1) * 10 + 100
    return series[:, :context], series[:, context:]


def test_training_targets_are_the_predict_step_inverted(checkpoint):
    """Mapping the model's own forecast into the loss frame recovers its normalised quantiles."""
    from tabtune.models.TimeSeries import _training

    adapter = _adapter(checkpoint)
    model = adapter._model
    context, _ = _windows()
    horizon = 12
    with torch.no_grad():
        forecast = model.predict(torch.as_tensor(context, dtype=torch.float32), horizon=horizon,
                                 quantile_levels=LEVELS).quantiles.numpy()
    seen = {}
    original = _training.pinball

    def spy(torch_, pred, target, levels, scale):
        seen["pred"], seen["target"] = pred.detach(), target
        return original(torch_, pred, target, levels, scale)

    _training.pinball = spy
    try:
        adapter._training_loss(torch, context, forecast[:, :, LEVELS.index(0.5)])
    finally:
        _training.pinball = original
    np.testing.assert_allclose(seen["target"].numpy(), seen["pred"][:, LEVELS.index(0.5)].numpy(), atol=1e-3)


def test_loss_and_gradients_stay_finite_when_the_decoder_saturates(checkpoint):
    adapter = _adapter(checkpoint)
    model = adapter._model
    with torch.no_grad():
        for parameter in model.decoder.parameters():
            parameter.mul_(1e3)
    context, label = _windows(seed=1)
    loss = adapter._training_loss(torch, context, label)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_lora_wraps_the_vendor_targets(checkpoint):
    frame = make_panel(2, 80, freq="h", seed=1)
    pipe = _pipe(
        checkpoint, tuning_strategy="peft",
        tuning_params={"epochs": 1, "steps_per_epoch": 2, "batch_size": 4, "seed": 0},
    ).fit(frame, SCHEMA)
    from tabtune._internal.lora import LoRALinear

    wrapped = [n for n, m in pipe.adapter_._model.named_modules() if isinstance(m, LoRALinear)]
    for target in ("wQKV", "wO", "mlp.0", "mlp.2"):
        assert any(target in n for n in wrapped), target
    assert not any(n.startswith(("patch_encoder", "decoder")) for n in wrapped)


def test_invalid_settings_are_rejected(checkpoint):
    frame = pd.DataFrame(
        {"item_id": "a", "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"), "target": 1.0}
    )
    with pytest.raises(ConfigError, match="float32"):
        _pipe(checkpoint, model_params={"dtype": "bfloat16"}).fit(frame, SCHEMA)
    with pytest.raises(ConfigError, match="batch_size"):
        _pipe(checkpoint, model_params={"batch_size": 0}).fit(frame, SCHEMA)

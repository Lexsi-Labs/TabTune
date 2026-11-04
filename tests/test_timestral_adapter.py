"""Timestral adapter on a tiny random checkpoint (config.json + model.safetensors, no download)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from tabtune.registry import ConfigError, get_time_series_model_spec
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel

pytestmark = [pytest.mark.unit, pytest.mark.time_series, pytest.mark.model_timestral]

torch = pytest.importorskip("torch")

SCHEMA = TimeSeriesSchema(target="target", item_id="item_id")
KNOTS = (0.01, 0.05) + tuple(round(0.1 * i, 1) for i in range(1, 10)) + (0.95, 0.99)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    from safetensors.torch import save_file

    from tabtune.models.timestral import TimestralConfig, TimestralModel

    config = TimestralConfig(
        patch_size=4, d_model=16, num_heads=2, mlp_hidden_dim=32,
        block_schedule=("time_variate", "variate_time", "time"), quantile_knots=KNOTS,
        block_size=4, context_length=64,
    )
    torch.manual_seed(0)
    model = TimestralModel(config)
    path = tmp_path_factory.mktemp("timestral")
    config.save(path / "config.json")
    save_file(model.state_dict(), path / "model.safetensors")
    return str(path)


def _pipe(checkpoint, horizon=6, **kwargs):
    model_params = {"checkpoint": checkpoint, "device": "cpu", **kwargs.pop("model_params", {})}
    return TimeSeriesPipeline(
        "Timestral", model_params=model_params, forecast_params={"prediction_length": horizon}, **kwargs
    )


def _reference(pipe, context, horizon, **covariates):
    forecaster = pipe.adapter_._model
    out = forecaster.forecast(np.asarray(context, dtype=np.float32), prediction_length=horizon, **covariates)
    return out.quantiles.float().numpy()


def test_registry_entry():
    spec = get_time_series_model_spec("Timestral-209M")
    assert spec.name == "Timestral" and spec.dependency_extra is None
    assert spec.native_missing and spec.supports_multivariate
    assert spec.supports_covariates and spec.supports_categorical_covariates
    assert {"embedding", "forecasting"} <= spec.tasks
    assert spec.strategies == {"inference", "finetune", "peft"}
    assert spec.license.commercial_use_ok is None


def test_no_third_party_model_package_is_imported():
    import tabtune.models.timestral  # noqa: F401

    assert "timestral" not in sys.modules


def test_forecast_matches_the_model(checkpoint):
    frame = make_panel(3, 60, freq="h", seed=0)
    pipe = TimeSeriesPipeline(
        "Timestral", model_params={"checkpoint": checkpoint, "device": "cpu"},
        forecast_params={"prediction_length": 6, "quantile_levels": [0.1, 0.5, 0.9]},
    ).fit(frame, SCHEMA)
    forecast = pipe.predict()
    assert forecast.point.shape == (3, 6) and forecast.quantiles.shape == (3, 6, 3)
    values = frame.loc[frame["item_id"] == "series_2", "target"].to_numpy()
    reference = _reference(pipe, values[None, None], 6)[0, 0]
    median = KNOTS.index(0.5)
    np.testing.assert_allclose(forecast.point[2], reference[:, median], atol=1e-5)
    np.testing.assert_allclose(
        forecast.quantiles[2], reference[:, [KNOTS.index(0.1), median, KNOTS.index(0.9)]], atol=1e-5
    )


def test_quantile_range_is_the_native_grid(checkpoint):
    pipe = _pipe(checkpoint).fit(make_panel(1, 40, freq="h", seed=0), SCHEMA)
    assert pipe.adapter_.quantile_range() == (0.01, 0.99)


def test_long_horizons_decode_block_by_block(checkpoint):
    frame = make_panel(2, 60, freq="h", seed=1)
    forecast = _pipe(checkpoint, horizon=40, envelope_mode="ignore").fit(frame, SCHEMA).predict()
    assert forecast.point.shape == (2, 40) and np.isfinite(forecast.point).all()


def test_targets_of_an_item_are_forecast_jointly(checkpoint):
    frame = make_panel(2, 50, freq="h", seed=3, n_targets=2)
    schema = TimeSeriesSchema(target=["target_0", "target_1"], item_id="item_id")
    pipe = _pipe(checkpoint).fit(frame, schema)
    forecast = pipe.predict()
    assert forecast.point.shape == (4, 6)
    item = frame[frame["item_id"] == "series_1"]
    joint = _reference(pipe, np.stack([item["target_0"], item["target_1"]])[None], 6)[0]
    np.testing.assert_allclose(forecast.point[2:], joint[:, :, KNOTS.index(0.5)], atol=1e-5)


def test_numeric_and_categorical_covariates_reach_the_model(checkpoint):
    horizon = 6
    frame = make_panel(2, 48 + horizon, freq="h", seed=4, covariates=("temp", "promo"))
    frame["store"] = np.where(frame.groupby("item_id").cumcount() % 3 == 0, "a", "b")
    cut = frame.groupby("item_id").cumcount() < 48
    history = frame[cut].reset_index(drop=True)
    future = frame[~cut].drop(columns="target").reset_index(drop=True)
    schema = TimeSeriesSchema(
        target="target", item_id="item_id", past_covariates=("temp",), known_covariates=("promo", "store")
    )
    covariates = _pipe(checkpoint).fit(history, schema, future_df=future).predict(future_df=future)
    plain = _pipe(checkpoint).fit(history[["item_id", "timestamp", "target"]], SCHEMA).predict()
    assert covariates.point.shape == (2, horizon)
    assert not np.allclose(plain.point, covariates.point)


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
        "Timestral", task_type="embedding", model_params={"checkpoint": checkpoint, "device": "cpu"}
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
        envelope_mode="ignore",
        tuning_strategy=strategy,
        tuning_params={"epochs": 1, "steps_per_epoch": 3, "batch_size": 8, "learning_rate": 1e-2, "seed": 0},
    ).fit(frame, SCHEMA)
    report = pipe.training_report_
    assert report["steps_run"] == 3 and np.isfinite(report["final_train_loss"])
    assert "pinball" in report["objective"]
    tuned = pipe.predict().point
    assert tuned.shape == (4, 12) and not np.allclose(tuned[:, :6], zero_shot, atol=1e-7)
    pipe.save(str(tmp_path / "timestral.joblib"))
    restored = TimeSeriesPipeline.load(str(tmp_path / "timestral.joblib")).predict().point
    np.testing.assert_allclose(restored, tuned, atol=1e-6)
    again = _pipe(checkpoint).fit(frame, SCHEMA).predict().point
    np.testing.assert_allclose(again, zero_shot, atol=1e-6)


def _adapter(checkpoint):
    return _pipe(checkpoint).fit(make_panel(1, 60, freq="h", seed=0), SCHEMA).adapter_


def _windows(seed=0, batch=4, context=48, horizon=12):
    rng = np.random.default_rng(seed)
    series = rng.normal(size=(batch, context + horizon)).cumsum(axis=1) * 10 + 100
    return series[:, :context], series[:, context:]


def test_training_targets_are_the_refine_step_inverted(checkpoint):
    """Mapping the forecaster's own refined forecast into the loss frame recovers the head output."""
    from tabtune.models.TimeSeries import _training

    adapter = _adapter(checkpoint)
    forecaster, net = adapter._model, adapter._net
    context, _ = _windows()
    horizon, ps = 12, net.config.patch_size
    x, mask = forecaster._as_context(context.astype(np.float32)[:, None, :], None)
    rows, mask, *_ = forecaster._assemble_rows(x, mask, horizon, None, None)
    rows, mask = rows[:, 0], mask[:, 0]
    ids = torch.arange(rows.shape[0])
    captured = {}
    handle = net.head.register_forward_hook(lambda _m, _i, out: captured.__setitem__("q", out))
    with torch.no_grad():
        refined = forecaster._refine_block(net.predict_block(rows, mask, ids, horizon // ps), rows, mask, None)
    handle.remove()
    median = refined[:, :horizon, forecaster.median_index].numpy()

    seen = {}
    original = _training.pinball

    def spy(torch_, pred, target, levels, scale):
        seen["pred"], seen["target"] = pred.detach(), target
        return original(torch_, pred, target, levels, scale)

    _training.pinball = spy
    try:
        adapter._training_loss(torch, context, median)
    finally:
        _training.pinball = original
    head_median = captured["q"][:, :horizon, forecaster.median_index]
    np.testing.assert_allclose(seen["target"].numpy(), head_median.numpy(), atol=1e-3)


def test_loss_and_gradients_stay_finite_when_the_head_saturates(checkpoint):
    adapter = _adapter(checkpoint)
    net = adapter._net
    with torch.no_grad():
        for parameter in net.head.parameters():
            parameter.mul_(1e3)
    context, label = _windows(seed=1)
    loss = adapter._training_loss(torch, context, label)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_lora_wraps_the_attention_and_mlp(checkpoint):
    frame = make_panel(2, 80, freq="h", seed=1)
    pipe = _pipe(
        checkpoint, tuning_strategy="peft",
        tuning_params={"epochs": 1, "steps_per_epoch": 2, "batch_size": 4, "seed": 0},
    ).fit(frame, SCHEMA)
    from tabtune._internal.lora import LoRALinear

    wrapped = [n for n, m in pipe.adapter_._net.named_modules() if isinstance(m, LoRALinear)]
    for target in ("wQKV", "wO", "mlp.w12", "mlp.w3"):
        assert any(target in n for n in wrapped), target
    assert not any("_gate" in n or "head" in n for n in wrapped)


def test_invalid_settings_are_rejected(checkpoint):
    frame = pd.DataFrame(
        {"item_id": "a", "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"), "target": 1.0}
    )
    with pytest.raises(ConfigError, match="float32"):
        _pipe(checkpoint, model_params={"dtype": "bfloat16"}).fit(frame, SCHEMA)
    with pytest.raises(ConfigError, match="batch_size"):
        _pipe(checkpoint, model_params={"batch_size": 0}).fit(frame, SCHEMA)

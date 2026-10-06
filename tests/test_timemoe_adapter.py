"""Time-MoE adapter on a tiny random checkpoint (2 layers, 4 experts, heads 1/4/8, 128 positions).

The checkpoint loads through ``TimeMoeForPrediction.from_pretrained`` as the
published weights do, never through ``trust_remote_code``. Decoding is pinned
by counting forward passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabtune.registry import ConfigError, get_time_series_model_spec
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel

pytestmark = [pytest.mark.unit, pytest.mark.time_series]

torch = pytest.importorskip("torch")

SCHEMA = TimeSeriesSchema(target="target", item_id="item_id")
SINGLE = TimeSeriesSchema(target="target")


def _pipe(checkpoint, horizon=6, **kwargs):
    model_params = {"checkpoint": checkpoint, "device": "cpu", **kwargs.pop("model_params", {})}
    return TimeSeriesPipeline(
        "TimeMoE", model_params=model_params, forecast_params={"prediction_length": horizon}, **kwargs
    )


def _series(n, fn=lambda t: np.sin(t / 4)):
    return pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=n, freq="D"), "target": fn(np.arange(n))}
    )


def _forward_lengths(pipe):
    lengths: list[int] = []
    handle = pipe.adapter_._model.register_forward_pre_hook(
        lambda _m, _args, kwargs: lengths.append(int(kwargs["input_ids"].shape[1])), with_kwargs=True
    )
    return lengths, handle


def test_registry_entry():
    spec = get_time_series_model_spec("time-moe")
    assert spec.name == "TimeMoE" and spec.dependency_extra is None
    assert not spec.native_missing and not spec.supports_multivariate
    assert {"embedding", "forecasting"} <= spec.tasks
    assert spec.strategies == {"inference", "finetune", "peft"}
    assert spec.max_context == 4096 and spec.license.commercial_use_ok is None


def test_point_forecasts_only(tiny_timemoe_checkpoint):
    frame = make_panel(3, 60, freq="h", seed=0)
    forecast = _pipe(tiny_timemoe_checkpoint).fit(frame, SCHEMA).predict()
    assert forecast.point.shape == (3, 6) and np.isfinite(forecast.point).all()
    assert forecast.quantiles is None and forecast.metadata["point_forecast"] == "mean"


def test_missing_values_are_refused(tiny_timemoe_checkpoint):
    frame = make_panel(1, 40, freq="h", seed=0)
    frame.loc[5, "target"] = np.nan
    with pytest.raises(ValueError, match="does not accept missing values"):
        _pipe(tiny_timemoe_checkpoint).fit(frame, SCHEMA)


def test_ragged_batch_matches_single_series(tiny_timemoe_checkpoint):
    frame = make_panel(2, 60, freq="h", seed=0)
    frame = frame[~((frame["item_id"] == "series_1") & (frame.index % 60 < 10))]
    together = _pipe(tiny_timemoe_checkpoint, horizon=7).fit(frame, SCHEMA).predict()
    alone = _pipe(tiny_timemoe_checkpoint, horizon=7).fit(frame[frame["item_id"] == "series_1"], SCHEMA).predict()
    np.testing.assert_allclose(alone.point[0], together.point[1], atol=1e-5)


@pytest.mark.parametrize(("horizon", "passes"), [(3, [40, 41, 42]), (8, [40]), (12, [40, 48]), (13, [40, 48, 52])])
def test_decoding_uses_the_largest_head_that_fits(tiny_timemoe_checkpoint, horizon, passes):
    pipe = _pipe(tiny_timemoe_checkpoint, horizon=horizon, envelope_mode="ignore").fit(_series(40), SINGLE)
    lengths, handle = _forward_lengths(pipe)
    try:
        forecast = pipe.predict()
    finally:
        handle.remove()
    assert lengths == passes and forecast.point.shape == (1, horizon)


def test_context_is_cut_to_fit_the_position_budget(tiny_timemoe_checkpoint):
    lengths_by_horizon = {}
    for horizon in (20, 100):
        pipe = _pipe(tiny_timemoe_checkpoint, horizon=horizon).fit(_series(300, lambda t: np.sin(t / 5)), SINGLE)
        lengths, handle = _forward_lengths(pipe)
        try:
            pipe.predict()
        finally:
            handle.remove()
        lengths_by_horizon[horizon] = lengths[0]
    assert lengths_by_horizon == {20: 128 - 20, 100: 64}


@pytest.mark.parametrize(
    "values",
    [[7.0] * 30, [0.0] * 30, [1.0, 2.0, 3.0], list(np.random.default_rng(0).normal(size=40) * 1e12)],
    ids=["constant", "zeros", "short", "huge"],
)
def test_degenerate_series_give_finite_forecasts(tiny_timemoe_checkpoint, values):
    frame = _series(len(values), lambda t: np.asarray(values))
    schema = TimeSeriesSchema(target="target", freq="D")
    assert np.isfinite(_pipe(tiny_timemoe_checkpoint, horizon=5).fit(frame, schema).predict().point).all()


def test_normalisation_never_divides_by_zero():
    from tabtune.models.TimeSeries.timemoe import normalise_context

    for values in ([5.0] * 10, [3.0], [0.1] * 7, [1e-12, 2e-12, 3e-12]):
        normed, _, scale = normalise_context(np.array(values))
        assert np.isfinite(normed).all() and scale > 0
    _, _, scale = normalise_context(np.array([1e-12, 2e-12, 3e-12]))
    assert scale == pytest.approx(1e-12)
    _, _, scale = normalise_context(np.array([0.1] * 7, dtype=np.float32))
    assert scale == 1.0


def test_embeddings(tiny_timemoe_checkpoint):
    frame = make_panel(3, 50, freq="h", seed=0)
    pipe = TimeSeriesPipeline(
        "TimeMoE", task_type="embedding", model_params={"checkpoint": tiny_timemoe_checkpoint, "device": "cpu"}
    )
    result = pipe.fit(frame, SCHEMA).predict()
    assert result.embeddings.shape == (3, 16) and np.isfinite(result.embeddings).all()


@pytest.mark.parametrize("strategy", ["peft", "finetune"])
def test_fine_tuning_round_trips_through_save_and_load(tiny_timemoe_checkpoint, tmp_path, strategy):
    frame = make_panel(4, 80, freq="h", seed=1)
    zero_shot = _pipe(tiny_timemoe_checkpoint, horizon=8).fit(frame, SCHEMA).predict().point
    pipe = _pipe(
        tiny_timemoe_checkpoint,
        horizon=8,
        tuning_strategy=strategy,
        tuning_params={"epochs": 1, "steps_per_epoch": 4, "batch_size": 8, "learning_rate": 1e-3, "seed": 0},
    ).fit(frame, SCHEMA)
    report = pipe.training_report_
    assert report["steps_run"] == 4 and np.isfinite(report["final_train_loss"])
    assert "huber" in report["objective"]
    tuned = pipe.predict().point
    assert not np.allclose(tuned, zero_shot, atol=1e-7)
    pipe.save(str(tmp_path / "timemoe.joblib"))
    restored = TimeSeriesPipeline.load(str(tmp_path / "timemoe.joblib")).predict().point
    np.testing.assert_allclose(restored, tuned, atol=1e-5)
    again = _pipe(tiny_timemoe_checkpoint, horizon=8).fit(frame, SCHEMA).predict().point
    np.testing.assert_allclose(again, zero_shot, atol=1e-6)


def test_invalid_settings_are_rejected(tiny_timemoe_checkpoint, tmp_path):
    frame = make_panel(1, 40, freq="h", seed=0)
    with pytest.raises(ConfigError, match="dtype"):
        _pipe(tiny_timemoe_checkpoint, model_params={"dtype": "float16"}).fit(frame, SCHEMA)
    with pytest.raises(ConfigError, match="attn_implementation"):
        _pipe(tiny_timemoe_checkpoint, model_params={"attn_implementation": "sdpa"}).fit(frame, SCHEMA)
    import transformers

    other = tmp_path / "t5"
    config = transformers.T5Config(d_model=8, d_ff=16, num_layers=1, num_heads=1, vocab_size=32, d_kv=8)
    transformers.T5ForConditionalGeneration(config).save_pretrained(other)
    with pytest.raises(ConfigError, match="not a Time-MoE checkpoint"):
        _pipe(str(other)).fit(frame, SCHEMA)


def test_calibration_gives_intervals(tiny_timemoe_checkpoint):
    frame = make_panel(3, 120, freq="h", seed=5)
    pipe = _pipe(tiny_timemoe_checkpoint).fit(frame, SCHEMA)
    forecast = pipe.calibrate(windows=6).predict()
    assert forecast.quantiles.shape == (3, 6, 3) and np.isfinite(forecast.quantiles).all()
    assert (np.diff(forecast.quantiles, axis=-1) >= -1e-9).all()


def test_peft_is_reproducible_with_a_seed(tiny_timemoe_checkpoint):
    frame = make_panel(3, 100, freq="h", seed=0)
    runs = []
    for offset in (0, 1):
        torch.manual_seed(100 + offset)
        pipe = _pipe(
            tiny_timemoe_checkpoint,
            horizon=8,
            tuning_strategy="peft",
            tuning_params={"epochs": 1, "steps_per_epoch": 3, "batch_size": 8, "learning_rate": 1e-2, "seed": 0},
        )
        runs.append(pipe.fit(frame, SCHEMA).predict().point)
    np.testing.assert_allclose(runs[0], runs[1], atol=1e-6)


def test_no_quantiles_requested(tiny_timemoe_checkpoint):
    """No requested levels means no quantile output."""
    pipe = TimeSeriesPipeline(
        "TimeMoE",
        model_params={"checkpoint": tiny_timemoe_checkpoint, "device": "cpu"},
        forecast_params={"prediction_length": 4, "quantile_levels": []},
    )
    result = pipe.fit(make_panel(2, 60, freq="h", seed=0), SCHEMA).predict()
    assert result.quantiles is None
    assert result.quantile_levels == ()
    assert result.point.shape == (2, 4)

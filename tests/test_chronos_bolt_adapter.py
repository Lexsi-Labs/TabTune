"""Tests for the vendored Chronos-Bolt code and its TimeSeriesPipeline adapter.

Like the Chronos v1 tests, these save a tiny randomly initialised Bolt
checkpoint to a temporary directory and drive it through the real path:
registry -> adapter imported by path -> ``ChronosBoltAdapter.load()`` ->
``ChronosBoltPipeline.from_pretrained`` -> patch -> encode -> quantile head.
Nothing is monkeypatched and nothing is downloaded. The slow test at the
bottom runs the released ``amazon/chronos-bolt-tiny`` checkpoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tabtune._internal.deprecation import reset_warning_cache  # noqa: E402
from tabtune.config import ForecastConfig  # noqa: E402
from tabtune.models.chronos import ChronosBoltModelForForecasting  # noqa: E402
from tabtune.models.TimeSeries.chronos_bolt import ChronosBoltAdapter  # noqa: E402
from tabtune.registry import ConfigError, get_time_series_model_spec  # noqa: E402
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel  # noqa: E402

pytestmark = [pytest.mark.time_series, pytest.mark.model_chronos]
@pytest.fixture(autouse=True)
def _fresh_warnings():
    """warn_once de-duplicates per process; reset so this file's assertions see the warning."""
    reset_warning_cache()
    yield
    reset_warning_cache()

SCHEMA = TimeSeriesSchema(target="y", timestamp="t", item_id="s")
TRAINED_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

_TINY_BOLT_CONFIG = {
    "context_length": 32,
    "prediction_length": 4,
    "input_patch_size": 4,
    "input_patch_stride": 4,
    "quantiles": TRAINED_QUANTILES,
    "use_reg_token": True,
}


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """A randomly initialised Chronos-Bolt checkpoint in the Hugging Face format."""
    config = transformers.T5Config(
        vocab_size=2, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    config.chronos_config = _TINY_BOLT_CONFIG
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-bolt")
    ChronosBoltModelForForecasting(config).save_pretrained(path)
    return str(path)


def _adapter(checkpoint, **model_params) -> ChronosBoltAdapter:
    adapter = ChronosBoltAdapter(
        get_time_series_model_spec("ChronosBolt"),
        checkpoint=checkpoint,
        device="cpu",
        model_params=model_params,
    )
    adapter.load()
    return adapter


def _frame():
    """Two series of different lengths, one with a missing value."""
    df = pd.DataFrame(
        {
            "s": ["a"] * 12 + ["b"] * 7,
            "t": list(pd.date_range("2024-01-01", periods=12, freq="D"))
            + list(pd.date_range("2024-01-01", periods=7, freq="D")),
            "y": np.r_[np.arange(12.0) + 1, np.arange(7.0) * 3 + 10],
        }
    )
    df.loc[4, "y"] = np.nan
    return df


def _panel():
    return SCHEMA.to_panel(_frame(), native_missing=True)



@pytest.mark.unit
def test_spec_matches_the_released_checkpoints():
    spec = get_time_series_model_spec("Chronos-Bolt")
    assert spec.name == "ChronosBolt"
    assert spec.max_context == 2048 and spec.max_horizon == 64
    assert spec.native_missing is True
    assert spec.default_checkpoint in spec.checkpoints
    assert spec.license.commercial_use_ok is True
    assert spec.tasks == frozenset(
        {"forecasting", "anomaly_detection", "imputation", "embedding"}
    )
    assert spec.strategies == frozenset({"inference", "finetune", "peft"})



@pytest.mark.unit
def test_load_reads_checkpoint_with_requested_dtype(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    assert adapter.is_loaded
    assert next(adapter._model.model.parameters()).dtype == torch.float32
    assert adapter._model.quantiles == TRAINED_QUANTILES


@pytest.mark.unit
def test_forecast_shapes_with_ragged_and_missing_history(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(_panel(), ForecastConfig(prediction_length=3))
    assert out.point.shape == (2, 3)
    assert out.quantiles.shape == (2, 3, 3)
    assert np.isfinite(out.quantiles).all()


@pytest.mark.unit
def test_forecasts_are_deterministic_without_a_seed(tiny_checkpoint):
    """Bolt samples nothing, so two independent adapters must agree exactly."""
    config = ForecastConfig(prediction_length=5)
    first = _adapter(tiny_checkpoint).forecast(_panel(), config)
    second = _adapter(tiny_checkpoint).forecast(_panel(), config)
    np.testing.assert_array_equal(first.point, second.point)
    np.testing.assert_array_equal(first.quantiles, second.quantiles)


@pytest.mark.unit
def test_horizon_beyond_builtin_length_uses_the_quantile_expansion(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(_panel(), ForecastConfig(prediction_length=11))
    assert out.point.shape == (2, 11)
    assert out.quantiles.shape == (2, 11, 3)
    assert np.isfinite(out.quantiles).all()


@pytest.mark.unit
def test_point_forecast_is_the_median(tiny_checkpoint):
    config = ForecastConfig(prediction_length=4, quantile_levels=[0.1, 0.5, 0.9])
    out = _adapter(tiny_checkpoint).forecast(_panel(), config)
    np.testing.assert_array_equal(out.point, out.quantiles[:, :, 1])


@pytest.mark.unit
def test_median_is_returned_even_when_no_quantiles_are_requested(tiny_checkpoint):
    config = ForecastConfig(prediction_length=2, quantile_levels=[])
    out = _adapter(tiny_checkpoint).forecast(_panel(), config)
    assert out.quantiles is None
    assert out.point.shape == (2, 2)


@pytest.mark.unit
def test_trained_levels_are_read_off_exactly(tiny_checkpoint):
    """A level in the trained grid must come straight from the head, not interpolation."""
    exact = _adapter(tiny_checkpoint).forecast(
        _panel(), ForecastConfig(prediction_length=3, quantile_levels=[0.2, 0.8])
    )
    full = _adapter(tiny_checkpoint).forecast(
        _panel(), ForecastConfig(prediction_length=3, quantile_levels=TRAINED_QUANTILES)
    )
    np.testing.assert_array_equal(exact.quantiles[:, :, 0], full.quantiles[:, :, 1])
    np.testing.assert_array_equal(exact.quantiles[:, :, 1], full.quantiles[:, :, 7])


@pytest.mark.unit
def test_levels_outside_the_trained_grid_warn(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    with pytest.warns(UserWarning, match="take the nearest of them"):
        adapter.forecast(_panel(), ForecastConfig(prediction_length=2, quantile_levels=[0.01, 0.5]))


@pytest.mark.unit
def test_batch_size_chunks_do_not_change_or_reorder_forecasts(tiny_checkpoint):
    """Chunking is numerically equivalent up to float32 reduction order (~1e-6)."""
    config = ForecastConfig(prediction_length=3)
    one_batch = _adapter(tiny_checkpoint).forecast(_panel(), config)
    chunked = _adapter(tiny_checkpoint, batch_size=1).forecast(_panel(), config)
    np.testing.assert_allclose(one_batch.point, chunked.point, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(one_batch.quantiles, chunked.quantiles, rtol=1e-5, atol=1e-5)
    assert chunked.point[1].mean() > chunked.point[0].mean()


@pytest.mark.unit
def test_invalid_model_params():
    spec = get_time_series_model_spec("ChronosBolt")
    with pytest.raises(ConfigError, match="point_forecast='median'"):
        ChronosBoltAdapter(spec, checkpoint="x", device="cpu", model_params={"point_forecast": "mean"})
    with pytest.raises(ConfigError, match="dtype"):
        ChronosBoltAdapter(spec, checkpoint="x", device="cpu", model_params={"dtype": "float16"})
    with pytest.raises(ConfigError, match="batch_size"):
        ChronosBoltAdapter(spec, checkpoint="x", device="cpu", model_params={"batch_size": 0})
    with pytest.warns(UserWarning, match="num_samples"):
        ChronosBoltAdapter(spec, checkpoint="x", device="cpu", model_params={"num_samples": 20})


@pytest.mark.unit
def test_dtype_defaults_to_bfloat16_on_cuda_only():
    spec = get_time_series_model_spec("ChronosBolt")
    assert ChronosBoltAdapter(spec, checkpoint="x", device="cuda:0").dtype == "bfloat16"
    assert ChronosBoltAdapter(spec, checkpoint="x", device="cpu").dtype == "float32"




@pytest.mark.unit
def test_pipeline_end_to_end_from_local_checkpoint(tiny_checkpoint, tmp_path):
    """The full architecture, offline, with no pipeline changes for this model."""
    pipe = TimeSeriesPipeline(
        "ChronosBolt",
        model_params={"checkpoint": tiny_checkpoint},
        forecast_params={"prediction_length": 3},
        tuning_params={"device": "cpu"},
    ).fit(_frame(), SCHEMA)
    assert type(pipe.adapter_) is ChronosBoltAdapter

    result = pipe.predict()
    assert result.point.shape == (2, 3)
    assert result.metadata["point_forecast"] == "median"
    assert result.metadata["checkpoint"] == tiny_checkpoint

    actual = pd.DataFrame(
        {"s": ["a", "b"], "t": [pd.Timestamp("2024-01-13"), pd.Timestamp("2024-01-08")], "y": [13.0, 31.0]}
    )
    with pytest.warns(UserWarning, match="covers 2 of 6"):
        metrics = pipe.evaluate(actual, forecast=result)
    assert {"mae", "rmse", "mse", "mean_pinball_loss"} <= set(metrics)

    path = tmp_path / "bolt.joblib"
    pipe.save(str(path))
    restored = TimeSeriesPipeline.load(str(path))
    assert not restored.adapter_.is_loaded
    np.testing.assert_array_equal(restored.predict().point, result.point)


@pytest.mark.unit
def test_context_is_cut_to_the_models_maximum(tiny_checkpoint, monkeypatch):
    """max_context is 2048 for Bolt, so a long history reaches the model trimmed."""
    seen: list[int] = []
    original = ChronosBoltAdapter.forecast

    def spy(self, panel, config):
        seen.append(panel.max_length)
        return original(self, panel, config)

    monkeypatch.setattr(ChronosBoltAdapter, "forecast", spy)
    n = 2200
    history = pd.DataFrame(
        {"s": "a", "t": pd.date_range("2020-01-01", periods=n, freq="D"), "y": np.arange(n, dtype=float)}
    )
    TimeSeriesPipeline(
        "ChronosBolt",
        model_params={"checkpoint": tiny_checkpoint},
        forecast_params={"prediction_length": 2},
        tuning_params={"device": "cpu"},
    ).fit(history, SCHEMA).predict()
    assert seen == [2048]


@pytest.mark.slow
@pytest.mark.weights
def test_released_chronos_bolt_tiny_forecasts_a_sine_wave():
    n, horizon = 240, 24
    steps = np.arange(n + horizon)
    df = pd.DataFrame(
        {
            "s": "sine",
            "t": pd.date_range("2024-01-01", periods=n + horizon, freq="h"),
            "y": 10 * np.sin(2 * np.pi * steps / 24) + 50,
        }
    )
    history, future = df.iloc[:n], df.iloc[n:]

    def run():
        return TimeSeriesPipeline(
            "ChronosBolt",
            model_params={"checkpoint": "amazon/chronos-bolt-tiny"},
            forecast_params={"prediction_length": horizon},
            tuning_params={"device": "cpu"},
        ).fit(history, SCHEMA)

    pipe = run()
    result = pipe.predict()
    assert result.point.shape == (1, horizon)
    assert result.quantiles.shape == (1, horizon, 3)
    assert (np.diff(result.quantiles, axis=-1) >= 0).all()

    naive_mae = np.abs(future["y"].to_numpy() - history["y"].iloc[-1]).mean()
    assert pipe.evaluate(future, forecast=result)["mae"] < 0.5 * naive_mae

    np.testing.assert_array_equal(run().predict().point, result.point)


@pytest.mark.unit
def test_lora_targets_match_real_linear_layers(tiny_checkpoint):
    """A wrong layer name would otherwise only surface as a confusing training error."""
    from tabtune._internal.lora import inject_lora

    adapter = _adapter(tiny_checkpoint)
    wrapped = inject_lora(adapter._network(), adapter.lora_targets, r=2, alpha=4)
    assert wrapped, f"none of {adapter.lora_targets} matched a linear layer"
    assert all("attention" in name.lower() for name in wrapped)


@pytest.mark.unit
def test_embeddings_pool_the_encoder_over_the_observed_patches(tiny_checkpoint):
    """One vector per series, and the same vector however the panel is batched."""
    pipe = TimeSeriesPipeline(
        "ChronosBolt",
        task_type="embedding",
        model_params={"checkpoint": tiny_checkpoint},
        tuning_params={"device": "cpu"},
    ).fit(_frame(), SCHEMA)
    result = pipe.predict()
    d_model = pipe.adapter_._model.model.config.d_model
    assert result.embeddings.shape == (2, d_model)
    assert np.isfinite(result.embeddings).all()

    one_at_a_time = (
        TimeSeriesPipeline(
            "ChronosBolt",
            task_type="embedding",
            model_params={"checkpoint": tiny_checkpoint, "batch_size": 1},
            tuning_params={"device": "cpu"},
        )
        .fit(_frame(), SCHEMA)
        .predict()
    )
    np.testing.assert_allclose(one_at_a_time.embeddings, result.embeddings, atol=1e-5)

    # _frame() is ragged (12 and 7 points), so this also covers batch independence.
    alone = (
        TimeSeriesPipeline(
            "ChronosBolt",
            task_type="embedding",
            model_params={"checkpoint": tiny_checkpoint},
            tuning_params={"device": "cpu"},
        )
        .fit(_frame().query("s == 'a'"), SCHEMA)
        .predict()
    )
    np.testing.assert_allclose(alone.embeddings[0], result.embeddings[0], atol=1e-5)

    # For a full series every patch is observed, so the masked mean is the plain mean ([REG] aside).
    series = _frame().query("s == 'a'")["y"].to_numpy(dtype="float32")
    upstream, _ = pipe.adapter_._model.embed(torch.as_tensor(series)[None])
    expected = upstream[0, :-1].mean(dim=0).numpy()
    np.testing.assert_allclose(result.embeddings[0], expected, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ["peft", "finetune"])
def test_fine_tuning_round_trips_through_save_and_load(tiny_checkpoint, tmp_path, strategy):
    frame = make_panel(4, 80, freq="h", seed=1)
    schema = TimeSeriesSchema(target="target", item_id="item_id")

    def build(**kwargs):
        return TimeSeriesPipeline(
            "ChronosBolt",
            model_params={"checkpoint": tiny_checkpoint},
            forecast_params={"prediction_length": 4},
            **kwargs,
        )

    zero_shot = build(tuning_params={"device": "cpu"}).fit(frame, schema).predict().point
    pipe = build(
        tuning_strategy=strategy,
        tuning_params={
            "device": "cpu",
            "epochs": 1,
            "steps_per_epoch": 4,
            "batch_size": 8,
            "learning_rate": 1e-2,
            "seed": 0,
        },
    ).fit(frame, schema)

    report = pipe.training_report_
    assert report["steps_run"] == 4 and np.isfinite(report["final_train_loss"])
    assert "quantile loss" in report["objective"]
    tuned = pipe.predict().point
    assert not np.allclose(tuned, zero_shot, atol=1e-7)

    # Training must leave the module exactly as forecasting found it.
    network = pipe.adapter_._network()
    assert not any(p.requires_grad for p in network.parameters())
    assert not network.training

    path = tmp_path / f"bolt-{strategy}.joblib"
    pipe.save(str(path))
    np.testing.assert_allclose(
        TimeSeriesPipeline.load(str(path)).predict().point, tuned, atol=1e-6
    )
    # A fresh zero-shot pipeline is untouched by any of it.
    np.testing.assert_allclose(
        build(tuning_params={"device": "cpu"}).fit(frame, schema).predict().point,
        zero_shot,
        atol=1e-6,
    )

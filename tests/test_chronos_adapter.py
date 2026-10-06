"""Tests for the vendored Chronos v1 package and its TimeSeriesPipeline adapter.

The unit tests save a tiny, randomly initialised Chronos checkpoint (a T5 with
the released checkpoints' tokenizer settings, a small vocabulary and a 4-step
built-in horizon) to a temporary directory and load it through the real path:
registry -> adapter imported by path -> ``ChronosAdapter.load()`` ->
``ChronosPipeline.from_pretrained`` -> tokenise -> sample -> decode. Nothing is
monkeypatched and nothing is downloaded. The slow test at the bottom runs the
released ``amazon/chronos-t5-tiny`` checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tabtune._internal.deprecation import reset_warning_cache  # noqa: E402
from tabtune.config import ForecastConfig  # noqa: E402
from tabtune.models.chronos import MeanScaleUniformBins  # noqa: E402
from tabtune.models.TimeSeries.chronos import ChronosAdapter  # noqa: E402
from tabtune.registry import ConfigError, get_time_series_model_spec  # noqa: E402
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema  # noqa: E402

pytestmark = [pytest.mark.time_series, pytest.mark.model_chronos]
@pytest.fixture(autouse=True)
def _fresh_warnings():
    """warn_once de-duplicates per process; reset so this file's assertions see the warning."""
    reset_warning_cache()
    yield
    reset_warning_cache()

SCHEMA = TimeSeriesSchema(target="y", timestamp="t", item_id="s")
VENDORED = Path(__file__).resolve().parents[1] / "tabtune" / "models" / "chronos"

_TINY_CHRONOS_CONFIG = {
    "tokenizer_class": "MeanScaleUniformBins",
    "tokenizer_kwargs": {"low_limit": -15.0, "high_limit": 15.0},
    "context_length": 32,
    "prediction_length": 4,
    "n_tokens": 64,
    "n_special_tokens": 2,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "use_eos_token": True,
    "model_type": "seq2seq",
    "num_samples": 5,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
}


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """A randomly initialised Chronos checkpoint in the Hugging Face format."""
    config = transformers.T5Config(
        vocab_size=64, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    config.chronos_config = _TINY_CHRONOS_CONFIG
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-chronos")
    transformers.T5ForConditionalGeneration(config).save_pretrained(path)
    return str(path)


def _adapter(checkpoint, seed=3, **model_params) -> ChronosAdapter:
    adapter = ChronosAdapter(
        get_time_series_model_spec("Chronos"),
        checkpoint=checkpoint,
        device="cpu",
        model_params=model_params,
        seed=seed,
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
def test_vendored_package_records_its_upstream_provenance():
    """The vendored copy carries no license files, so provenance lives in code."""
    import tabtune.models.chronos as vendored

    assert vendored.UPSTREAM_VERSION == "2.3.2"
    assert len(vendored.UPSTREAM_COMMIT) == 40
    files = ["chronos.py", "chronos_bolt.py", "base.py", "utils.py", "df_utils.py"]
    files += [
        f"chronos2/{name}.py"
        for name in ("config", "layers", "model", "preprocess", "dataset", "pipeline")
    ]
    for name in files:
        header = (VENDORED / name).read_text()[:400]
        assert "SPDX-License-Identifier: Apache-2.0" in header, name
        assert "Copyright Amazon.com" in header, name


@pytest.mark.unit
def test_tokenizer_class_resolves_inside_the_vendored_package(tiny_checkpoint):
    assert isinstance(_adapter(tiny_checkpoint)._model.tokenizer, MeanScaleUniformBins)



@pytest.mark.unit
def test_load_reads_checkpoint_with_requested_dtype(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    assert adapter.is_loaded
    assert next(adapter._model.model.parameters()).dtype == torch.float32
    assert str(adapter._model.model.device) == "cpu"


@pytest.mark.unit
def test_forecast_shapes_with_ragged_and_missing_history(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(_panel(), ForecastConfig(prediction_length=3))
    assert out.point.shape == (2, 3)
    assert out.quantiles.shape == (2, 3, 3)
    assert np.isfinite(out.quantiles).all()
    assert (np.diff(out.quantiles, axis=-1) >= 0).all()


@pytest.mark.unit
def test_horizon_beyond_builtin_length_is_autoregressive(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(_panel(), ForecastConfig(prediction_length=11))
    assert out.point.shape == (2, 11)
    assert out.quantiles.shape == (2, 11, 3)


@pytest.mark.unit
def test_seed_makes_forecasts_repeatable_without_touching_global_rng(tiny_checkpoint):
    config = ForecastConfig(prediction_length=5)
    state = torch.random.get_rng_state()
    first = _adapter(tiny_checkpoint, seed=11).forecast(_panel(), config)
    second = _adapter(tiny_checkpoint, seed=11).forecast(_panel(), config)
    np.testing.assert_array_equal(first.quantiles, second.quantiles)
    np.testing.assert_array_equal(first.point, second.point)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.unit
def test_median_point_forecast_is_the_half_quantile(tiny_checkpoint):
    config = ForecastConfig(prediction_length=4, quantile_levels=[0.1, 0.5, 0.9])
    out = _adapter(tiny_checkpoint, point_forecast="median").forecast(_panel(), config)
    np.testing.assert_array_equal(out.point, out.quantiles[:, :, 1])


@pytest.mark.unit
def test_median_requested_internally_is_not_returned(tiny_checkpoint):
    config = ForecastConfig(prediction_length=4, quantile_levels=[0.2, 0.8])
    out = _adapter(tiny_checkpoint, point_forecast="median").forecast(_panel(), config)
    assert out.quantiles.shape == (2, 4, 2)


@pytest.mark.unit
def test_no_quantiles_requested(tiny_checkpoint):
    config = ForecastConfig(prediction_length=2, quantile_levels=[])
    out = _adapter(tiny_checkpoint).forecast(_panel(), config)
    assert out.quantiles is None
    assert out.point.shape == (2, 2)


@pytest.mark.unit
def test_batch_size_chunks_do_not_change_or_reorder_forecasts(tiny_checkpoint):

    config = ForecastConfig(prediction_length=3)
    one_batch = _adapter(tiny_checkpoint, top_k=1).forecast(_panel(), config)
    chunked = _adapter(tiny_checkpoint, top_k=1, batch_size=1).forecast(_panel(), config)
    np.testing.assert_array_equal(one_batch.point, chunked.point)
    np.testing.assert_array_equal(one_batch.quantiles, chunked.quantiles)


@pytest.mark.unit
def test_invalid_model_params():
    spec = get_time_series_model_spec("Chronos")
    with pytest.raises(ConfigError, match="dtype"):
        ChronosAdapter(spec, checkpoint="x", device="cpu", model_params={"dtype": "float16"})
    with pytest.raises(ConfigError, match="point_forecast"):
        ChronosAdapter(spec, checkpoint="x", device="cpu", model_params={"point_forecast": "mode"})
    with pytest.raises(ConfigError, match="batch_size"):
        ChronosAdapter(spec, checkpoint="x", device="cpu", model_params={"batch_size": 0})
    with pytest.warns(UserWarning, match="num_sample"):
        ChronosAdapter(spec, checkpoint="x", device="cpu", model_params={"num_sample": 10})


@pytest.mark.unit
def test_dtype_defaults_to_bfloat16_on_cuda_only():
    spec = get_time_series_model_spec("Chronos")
    assert ChronosAdapter(spec, checkpoint="x", device="cuda:0").dtype == "bfloat16"
    assert ChronosAdapter(spec, checkpoint="x", device="cpu").dtype == "float32"
    assert ChronosAdapter(spec, checkpoint="x", device="mps").dtype == "float32"



@pytest.mark.unit
def test_pipeline_end_to_end_from_local_checkpoint(tiny_checkpoint, tmp_path):
    """The full architecture, offline: nothing is monkeypatched."""
    pipe = TimeSeriesPipeline(
        "Chronos",
        model_params={"checkpoint": tiny_checkpoint, "point_forecast": "median"},
        forecast_params={"prediction_length": 3},
        tuning_params={"device": "cpu", "seed": 0},
    ).fit(_frame(), SCHEMA)
    assert type(pipe.adapter_) is ChronosAdapter

    result = pipe.predict()
    assert result.point.shape == (2, 3)
    assert result.metadata["checkpoint"] == tiny_checkpoint
    assert result.metadata["point_forecast"] == "median"
    np.testing.assert_array_equal(result.point, result.quantiles[:, :, 1])

    actual = pd.DataFrame(
        {"s": ["a", "b"], "t": [pd.Timestamp("2024-01-13"), pd.Timestamp("2024-01-08")], "y": [13.0, 31.0]}
    )
    with pytest.warns(UserWarning, match="covers 2 of 6"):
        metrics = pipe.evaluate(actual, forecast=result)
    assert {"mae", "rmse", "mse", "mean_pinball_loss"} <= set(metrics)

    path = tmp_path / "chronos.joblib"
    pipe.save(str(path))
    restored = TimeSeriesPipeline.load(str(path))
    assert not restored.adapter_.is_loaded
    np.testing.assert_array_equal(restored.predict().point, result.point)


@pytest.mark.slow
@pytest.mark.weights
def test_released_chronos_t5_tiny_forecasts_a_sine_wave():
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
            "Chronos",
            model_params={"checkpoint": "amazon/chronos-t5-tiny"},
            forecast_params={"prediction_length": horizon},
            tuning_params={"device": "cpu", "seed": 0},
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


@pytest.mark.unit
def test_embeddings_are_one_vector_per_series_whatever_the_batching(tiny_checkpoint):
    from tabtune.TimeSeries import make_panel

    # Ragged, and not a multiple of any model's patch size.
    frame = make_panel(3, 45, freq="h", seed=0)
    frame = frame[~((frame["item_id"] == "series_2") & (frame.index % 45 < 13))]
    schema = TimeSeriesSchema(target="target", item_id="item_id")

    def embed(**model_params):
        return (
            TimeSeriesPipeline(
                "Chronos",
                task_type="embedding",
                model_params={"checkpoint": tiny_checkpoint, **model_params},
                tuning_params={"device": "cpu", "seed": 0},
            )
            .fit(frame, schema)
            .predict()
        )

    result = embed()
    assert result.embeddings.shape == (3, 16)
    assert np.isfinite(result.embeddings).all()
    np.testing.assert_allclose(
        embed(batch_size=1).embeddings, result.embeddings, atol=1e-5
    )

    # A series' vector must not depend on what it was batched with.
    alone = (
        TimeSeriesPipeline(
            "Chronos",
            task_type="embedding",
            model_params={"checkpoint": tiny_checkpoint},
            tuning_params={"device": "cpu", "seed": 0},
        )
        .fit(frame[frame["item_id"] == "series_0"], schema)
        .predict()
    )
    np.testing.assert_allclose(alone.embeddings[0], result.embeddings[0], atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ["peft", "finetune"])
def test_fine_tuning_round_trips_through_save_and_load(tiny_checkpoint, tmp_path, strategy):
    from tabtune.TimeSeries import make_panel

    frame = make_panel(4, 80, freq="h", seed=1)
    schema = TimeSeriesSchema(target="target", item_id="item_id")

    def build(**kwargs):
        return TimeSeriesPipeline(
            "Chronos",
            model_params={"checkpoint": tiny_checkpoint},
            # Seeded throughout: an unseeded sampling model consumes the global RNG.
            forecast_params={"prediction_length": 4, "context_length": 32},
            **kwargs,
        )

    def zero_shot():
        return build(tuning_params={"device": "cpu", "seed": 3}).fit(frame, schema).predict().point

    before = zero_shot()
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
    assert "token cross-entropy" in report["objective"]
    tuned = pipe.predict().point
    assert not np.allclose(tuned, before, atol=1e-7)

    # Training must leave the module exactly as forecasting found it.
    network = pipe.adapter_._network()
    assert not any(p.requires_grad for p in network.parameters())
    assert not network.training

    path = tmp_path / f"chronos-{strategy}.joblib"
    pipe.save(str(path))
    np.testing.assert_allclose(
        TimeSeriesPipeline.load(str(path)).predict().point, tuned, atol=1e-6
    )
    # Training one pipeline leaves a fresh zero-shot one untouched.
    np.testing.assert_allclose(zero_shot(), before, atol=1e-6)

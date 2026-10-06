"""Tests for the vendored Chronos-2 code and its TimeSeriesPipeline adapter.

Like the Chronos v1 and Bolt tests, these save a tiny randomly initialised
checkpoint to a temporary directory and drive it through the real path:
registry -> adapter imported by path -> ``Chronos2Adapter.load()`` ->
``Chronos2Pipeline.from_pretrained`` -> group attention -> quantile head.
Nothing is monkeypatched and nothing is downloaded.

These tests also pin the capability contract: multivariate targets are forecast
jointly, past-only and known-future covariates reach the model, and a model that
cannot do those things rejects the schema. The slow test at the bottom runs the
released ``amazon/chronos-2``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tabtune._internal.deprecation import reset_warning_cache  # noqa: E402
from tabtune.config import ForecastConfig  # noqa: E402
from tabtune.models.chronos import Chronos2Model  # noqa: E402
from tabtune.models.chronos.chronos2.config import Chronos2CoreConfig  # noqa: E402
from tabtune.models.TimeSeries.chronos2 import Chronos2Adapter  # noqa: E402
from tabtune.registry import ConfigError, get_time_series_model_spec  # noqa: E402
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema  # noqa: E402

pytestmark = [pytest.mark.time_series, pytest.mark.model_chronos]

TRAINED_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HISTORY = 24


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """warn_once de-duplicates per process; reset so this file's assertions see the warning."""
    reset_warning_cache()
    yield
    reset_warning_cache()


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """A randomly initialised Chronos-2 checkpoint in the Hugging Face format."""
    config = Chronos2CoreConfig(d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4)
    config.chronos_config = {
        "context_length": 64,
        "input_patch_size": 8,
        "input_patch_stride": 8,
        "output_patch_size": 8,
        "max_output_patches": 2,
        "quantiles": TRAINED_QUANTILES,
        "use_reg_token": True,
        "use_arcsinh": True,
    }
    config.architectures = ["Chronos2Model"]
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-chronos2")
    Chronos2Model(config).save_pretrained(path)
    return str(path)


def _adapter(checkpoint, **model_params) -> Chronos2Adapter:
    adapter = Chronos2Adapter(
        get_time_series_model_spec("Chronos2"),
        checkpoint=checkpoint,
        device="cpu",
        model_params=model_params,
    )
    adapter.load()
    return adapter


def _frame(*, missing: bool = True) -> pd.DataFrame:
    """Two items, two targets, one numeric and one categorical covariate."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=HISTORY, freq="D")
    frames = []
    for offset, item in enumerate(("a", "b")):
        sales = np.arange(HISTORY, dtype=float) + 10 * offset
        if missing and offset == 0:
            sales[3:5] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "s": item,
                    "t": dates,
                    "y": sales,
                    "z": rng.normal(100, 1, HISTORY),
                    "temp": rng.normal(20, 1, HISTORY),
                    "weather": rng.choice(["sun", "rain"], HISTORY),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _future(horizon: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=HISTORY)
    dates = pd.date_range(start, periods=horizon, freq="D")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "s": item,
                    "t": dates,
                    "temp": rng.normal(20, 1, horizon),
                    "weather": rng.choice(["sun", "rain"], horizon),
                }
            )
            for item in ("a", "b")
        ],
        ignore_index=True,
    )


UNIVARIATE = TimeSeriesSchema(target="y", timestamp="t", item_id="s")
MULTIVARIATE = TimeSeriesSchema(target=("y", "z"), timestamp="t", item_id="s")
WITH_PAST = TimeSeriesSchema(target="y", timestamp="t", item_id="s", past_covariates=("temp",))
WITH_KNOWN = TimeSeriesSchema(
    target="y", timestamp="t", item_id="s", known_covariates=("temp", "weather")
)


def _panel(schema=UNIVARIATE, *, horizon: int | None = None):
    columns = ["s", "t", *schema.target_names, *schema.covariate_names]
    frame = _frame()[columns]
    future = _future(horizon) if schema.known_covariates else None
    return schema.to_panel(frame, native_missing=True, future_df=future, horizon=horizon)


@pytest.mark.unit
def test_spec_matches_the_released_checkpoint():
    spec = get_time_series_model_spec("Chronos-2")
    assert spec.name == "Chronos2"
    assert spec.max_context == 8192 and spec.max_horizon == 1024
    assert spec.native_missing is True
    assert spec.supports_multivariate and spec.supports_covariates
    assert spec.supports_categorical_covariates
    assert spec.default_checkpoint == "amazon/chronos-2"
    assert spec.default_checkpoint in spec.checkpoints
    assert spec.license.commercial_use_ok is True


@pytest.mark.unit
def test_capability_flags_gate_the_older_chronos_models():
    """The point of the flags: v1/Bolt must refuse, not silently drop the columns."""
    from tabtune.registry import check_schema_support

    for name in ("Chronos", "ChronosBolt"):
        spec = get_time_series_model_spec(name)
        with pytest.raises(ConfigError, match="does not use covariates"):
            check_schema_support(spec, WITH_PAST)
        with pytest.raises(ConfigError, match="one target at a time"):
            check_schema_support(spec, MULTIVARIATE)
    check_schema_support(get_time_series_model_spec("Chronos2"), MULTIVARIATE)
    check_schema_support(get_time_series_model_spec("Chronos2"), WITH_KNOWN)



@pytest.mark.unit
def test_load_reads_checkpoint_with_requested_dtype(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    assert adapter.is_loaded
    assert next(adapter._model.model.parameters()).dtype == torch.float32
    assert adapter._model.quantiles == TRAINED_QUANTILES


@pytest.mark.unit
def test_forecast_shapes_with_missing_history(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(_panel(), ForecastConfig(prediction_length=3))
    assert out.point.shape == (2, 3)
    assert out.quantiles.shape == (2, 3, 3)
    assert np.isfinite(out.quantiles).all()


@pytest.mark.unit
def test_multivariate_panel_yields_one_row_per_item_and_target(tiny_checkpoint):
    panel = _panel(MULTIVARIATE)
    assert len(panel) == 4
    assert panel.target_names == ("y", "z", "y", "z")
    assert panel.item_groups() == ((0, 1), (2, 3))

    out = _adapter(tiny_checkpoint).forecast(panel, ForecastConfig(prediction_length=3))
    assert out.point.shape == (4, 3)
    assert np.isfinite(out.point).all()


@pytest.mark.unit
def test_variates_are_forecast_jointly_not_independently(tiny_checkpoint):
    """Forecasting y alongside z differs from forecasting y alone (random weights: inequality only)."""
    config = ForecastConfig(prediction_length=3)
    joint = _adapter(tiny_checkpoint).forecast(_panel(MULTIVARIATE), config)
    alone = _adapter(tiny_checkpoint).forecast(_panel(UNIVARIATE), config)
    assert not np.array_equal(joint.point[0], alone.point[0])


@pytest.mark.unit
def test_covariates_reach_the_model_and_change_the_forecast(tiny_checkpoint):
    """Each extra covariate reaches the forward pass and moves the output."""
    config = ForecastConfig(prediction_length=3)
    without = _adapter(tiny_checkpoint).forecast(_panel(UNIVARIATE), config)
    with_past = _adapter(tiny_checkpoint).forecast(_panel(WITH_PAST), config)
    with_known = _adapter(tiny_checkpoint).forecast(_panel(WITH_KNOWN, horizon=3), config)
    assert not np.array_equal(without.point, with_past.point)
    assert not np.array_equal(with_past.point, with_known.point)


@pytest.mark.unit
def test_categorical_covariates_are_passed_through(tiny_checkpoint):
    """'weather' is a string column; upstream encodes it, so it must survive as-is."""
    panel = _panel(WITH_KNOWN, horizon=3)
    assert panel.past_covariates[0]["weather"].dtype.kind in "UO"
    assert panel.future_covariates[0]["weather"].tolist() == list(
        _future(3).query("s == 'a'")["weather"]
    )
    out = _adapter(tiny_checkpoint).forecast(panel, ForecastConfig(prediction_length=3))
    assert np.isfinite(out.point).all()


@pytest.mark.unit
def test_forecasts_are_deterministic_without_a_seed(tiny_checkpoint):
    """Chronos-2 samples nothing, so two independent adapters must agree exactly."""
    config = ForecastConfig(prediction_length=5)
    first = _adapter(tiny_checkpoint).forecast(_panel(), config)
    second = _adapter(tiny_checkpoint).forecast(_panel(), config)
    np.testing.assert_array_equal(first.point, second.point)
    np.testing.assert_array_equal(first.quantiles, second.quantiles)


@pytest.mark.unit
def test_horizon_beyond_builtin_length_unrolls(tiny_checkpoint):
    """max_output_patches * output_patch_size is 16 here, so 20 steps unroll."""
    out = _adapter(tiny_checkpoint).forecast(_panel(), ForecastConfig(prediction_length=20))
    assert out.point.shape == (2, 20)
    assert np.isfinite(out.point).all()


@pytest.mark.unit
def test_point_forecast_is_the_median(tiny_checkpoint):
    config = ForecastConfig(prediction_length=4, quantile_levels=[0.1, 0.5, 0.9])
    out = _adapter(tiny_checkpoint).forecast(_panel(), config)
    np.testing.assert_array_equal(out.point, out.quantiles[:, :, 1])


@pytest.mark.unit
def test_median_is_returned_even_when_no_quantiles_are_requested(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(
        _panel(), ForecastConfig(prediction_length=2, quantile_levels=[])
    )
    assert out.quantiles is None
    assert out.point.shape == (2, 2)


@pytest.mark.unit
def test_trained_levels_are_read_off_exactly(tiny_checkpoint):
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
    """Chunking is numerically equivalent, though not bit-identical (matmul order)."""
    config = ForecastConfig(prediction_length=3)
    one_batch = _adapter(tiny_checkpoint).forecast(_panel(), config)
    chunked = _adapter(tiny_checkpoint, batch_size=1).forecast(_panel(), config)
    np.testing.assert_allclose(one_batch.point, chunked.point, rtol=1e-4, atol=1e-4)
    assert chunked.point[1].mean() > chunked.point[0].mean()


@pytest.mark.unit
def test_invalid_model_params():
    spec = get_time_series_model_spec("Chronos2")
    with pytest.raises(ConfigError, match="point_forecast='median'"):
        Chronos2Adapter(spec, checkpoint="x", device="cpu", model_params={"point_forecast": "mean"})
    with pytest.raises(ConfigError, match="dtype"):
        Chronos2Adapter(spec, checkpoint="x", device="cpu", model_params={"dtype": "float16"})
    with pytest.raises(ConfigError, match="batch_size"):
        Chronos2Adapter(spec, checkpoint="x", device="cpu", model_params={"batch_size": 0})
    # Sampling knobs belong to Chronos v1.
    with pytest.warns(UserWarning, match="num_samples"):
        Chronos2Adapter(spec, checkpoint="x", device="cpu", model_params={"num_samples": 20})


@pytest.mark.unit
def test_dtype_defaults_to_bfloat16_on_cuda_only():
    spec = get_time_series_model_spec("Chronos2")
    assert Chronos2Adapter(spec, checkpoint="x", device="cuda:0").dtype == "bfloat16"
    assert Chronos2Adapter(spec, checkpoint="x", device="cpu").dtype == "float32"



def _pipeline(checkpoint, horizon=3, cache=None, **forecast_params):
    return TimeSeriesPipeline(
        "Chronos2",
        model_params={"checkpoint": checkpoint},
        forecast_params={"prediction_length": horizon, **forecast_params},
        tuning_params={"device": "cpu"},
        cache=cache,
    )


@pytest.mark.unit
def test_pipeline_end_to_end_with_covariates(tiny_checkpoint, tmp_path):
    """The full architecture, offline: covariates in, long multi-target frame out."""
    schema = TimeSeriesSchema(
        target=("y", "z"),
        timestamp="t",
        item_id="s",
        past_covariates=("temp",),
        known_covariates=("weather",),
    )
    pipe = _pipeline(tiny_checkpoint).fit(
        _frame(), schema, future_df=_future(3).drop(columns="temp")
    )
    assert type(pipe.adapter_) is Chronos2Adapter

    result = pipe.predict()
    assert result.point.shape == (4, 3)
    assert result.targets == ("y", "z", "y", "z")
    assert result.metadata["point_forecast"] == "median"

    frame = result.to_pandas()
    assert len(frame) == 4 * 3
    assert set(frame["target"]) == {"y", "z"}

    actual = _frame().tail(6)[["s", "t", "y", "z"]].assign(
        t=pd.date_range("2024-01-25", periods=3, freq="D").tolist() * 2,
        s=["a"] * 3 + ["b"] * 3,
    )
    metrics = pipe.evaluate(actual, forecast=result)
    assert set(metrics["per_target"]) == {"y", "z"}

    path = tmp_path / "chronos2.joblib"
    pipe.save(str(path))
    restored = TimeSeriesPipeline.load(str(path))
    assert not restored.adapter_.is_loaded
    np.testing.assert_array_equal(restored.predict().point, result.point)


@pytest.mark.unit
def test_missing_future_covariates_are_refused_before_the_model_runs(tiny_checkpoint):
    pipe = _pipeline(tiny_checkpoint).fit(_frame()[["s", "t", "y", "temp", "weather"]], WITH_KNOWN)
    with pytest.raises(ConfigError, match="predict\\(future_df=...\\)"):
        pipe.predict()


@pytest.mark.unit
def test_changing_future_covariates_invalidates_the_cache(tiny_checkpoint):
    """Future covariate values are part of the cache key."""
    frame = _frame()[["s", "t", "y", "temp", "weather"]]
    pipe = _pipeline(tiny_checkpoint, cache=True).fit(frame, WITH_KNOWN, future_df=_future(3))
    first = pipe.predict()
    np.testing.assert_array_equal(pipe.predict().point, first.point)

    warmer = _future(3).assign(temp=lambda d: d["temp"] + 25)
    second = pipe.predict(future_df=warmer)
    assert not np.array_equal(first.point, second.point)
    assert pipe.history_.fingerprint() != pipe.schema_.attach_future(
        pipe.history_, warmer, horizon=3
    ).fingerprint()


@pytest.mark.unit
def test_pipeline_rejects_a_schema_the_model_cannot_read(tiny_checkpoint):
    pipe = TimeSeriesPipeline(
        "ChronosBolt",
        model_params={"checkpoint": tiny_checkpoint},
        forecast_params={"prediction_length": 3},
        validate=False,
    )
    with pytest.raises(ConfigError, match="does not use covariates"):
        pipe.fit(_frame(), WITH_PAST)


@pytest.mark.slow
@pytest.mark.weights
def test_released_checkpoint_forecasts_a_trend():
    """The released 120M model, end to end, with a known-future covariate."""
    pipe = TimeSeriesPipeline(
        "Chronos2",
        forecast_params={"prediction_length": 4, "quantile_levels": [0.1, 0.5, 0.9]},
        tuning_params={"device": "cpu"},
    )
    frame = _frame(missing=False)[["s", "t", "y", "temp", "weather"]]
    result = pipe.fit(frame, WITH_KNOWN, future_df=_future(4)).predict()

    assert result.point.shape == (2, 4)
    assert np.isfinite(result.point).all()
    assert (np.diff(result.point, axis=1) > 0).all()
    assert (np.diff(result.quantiles, axis=2) >= 0).all()


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
                "Chronos2",
                task_type="embedding",
                model_params={"checkpoint": tiny_checkpoint, **model_params},
                tuning_params={"device": "cpu", "seed": 0},
            )
            .fit(frame, schema)
            .predict()
        )

    result = embed()
    assert result.embeddings.shape == (3, 32)
    assert np.isfinite(result.embeddings).all()
    np.testing.assert_allclose(
        embed(batch_size=1).embeddings, result.embeddings, atol=1e-5
    )

    # A series' vector must not depend on what it was batched with.
    alone = (
        TimeSeriesPipeline(
            "Chronos2",
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
            "Chronos2",
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
    assert "quantile loss" in report["objective"]
    tuned = pipe.predict().point
    assert not np.allclose(tuned, before, atol=1e-7)

    # Training must leave the module exactly as forecasting found it.
    network = pipe.adapter_._network()
    assert not any(p.requires_grad for p in network.parameters())
    assert not network.training

    path = tmp_path / f"chronos2-{strategy}.joblib"
    pipe.save(str(path))
    np.testing.assert_allclose(
        TimeSeriesPipeline.load(str(path)).predict().point, tuned, atol=1e-6
    )
    # Training one pipeline leaves a fresh zero-shot one untouched.
    np.testing.assert_allclose(zero_shot(), before, atol=1e-6)

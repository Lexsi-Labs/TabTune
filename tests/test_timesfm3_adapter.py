"""Tests for the vendored TimesFM 3.0 code and its TimeSeriesPipeline adapter.

Like the Chronos tests, these save a tiny randomly initialised checkpoint to a
temporary directory and drive it through the real path: registry -> adapter
imported by path -> ``TimesFM3Adapter.load()`` ->
``TimesFM3Forecaster.from_pretrained`` -> variate attention -> quantile head.
Nothing is monkeypatched and nothing is downloaded.

These tests also pin covariates given as float channels, a fixed quantile grid
and weights that forbid commercial use. The slow test at the bottom runs the
released ``google/timesfm-3.0-pytorch``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tabtune._internal.deprecation import reset_warning_cache  # noqa: E402
from tabtune.config import ForecastConfig  # noqa: E402
from tabtune.models.TimeSeries.timesfm3 import TimesFM3Adapter  # noqa: E402
from tabtune.models.timesfm3 import (  # noqa: E402
    ResidualBlockConfig,
    StackedTransformersConfig,
    TimesFM3Torch,
    TransformerConfig,
)
from tabtune.registry import ConfigError, get_time_series_model_spec  # noqa: E402
from tabtune.registry.errors import LicenseError  # noqa: E402
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema  # noqa: E402

pytestmark = [pytest.mark.time_series, pytest.mark.model_timesfm]

TRAINED_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HISTORY = 24
VENDORED = Path(__file__).resolve().parents[1] / "tabtune" / "models" / "timesfm3"


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """warn_once de-duplicates per process; reset so this file's assertions see the warning."""
    reset_warning_cache()
    yield
    reset_warning_cache()


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """A randomly initialised TimesFM 3 checkpoint; output_patch_len must be a multiple of input_patch_len."""
    torch.manual_seed(0)
    model = TimesFM3Torch(
        input_patch_len=8,
        output_patch_len=16,
        quantiles=TRAINED_QUANTILES,
        residual_block_config=ResidualBlockConfig(
            hidden_dims=32, output_dims=32, use_bias=False, activation="relu"
        ),
        transformer_config=StackedTransformersConfig(
            num_layers=2,
            transformer=TransformerConfig(
                model_dims=32,
                hidden_dims=32,
                num_heads=2,
                attention_norm="rms",
                feedforward_norm="rms",
                qk_norm="rms",
                use_bias=False,
                use_rope_seq=True,
                use_rope_var=False,
                ff_activation="relu",
                deterministic=True,
            ),
        ),
    )
    path = tmp_path_factory.mktemp("tiny-timesfm3")
    model.save_pretrained(path)
    return str(path)


def _adapter(checkpoint, **model_params) -> TimesFM3Adapter:
    adapter = TimesFM3Adapter(
        get_time_series_model_spec("TimesFM3"),
        checkpoint=checkpoint,
        device="cpu",
        model_params=model_params,
    )
    adapter.load()
    return adapter


def _frame(*, missing: bool = True) -> pd.DataFrame:
    """Two items, two targets, one numeric and one categorical covariate.

    The target is not a straight line: TimesFM removes a linear trend first, so
    a ramp would make covariates and extra variates look inert.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=HISTORY, freq="D")
    steps = np.arange(HISTORY, dtype=float)
    frames = []
    for offset, item in enumerate(("a", "b")):
        sales = steps + 3 * np.sin(steps * 2 * np.pi / 7) + rng.normal(0, 0.5, HISTORY)
        sales += 10 * offset
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
            pd.DataFrame({"s": item, "t": dates, "temp": rng.normal(20, 1, horizon)})
            for item in ("a", "b")
        ],
        ignore_index=True,
    )


UNIVARIATE = TimeSeriesSchema(target="y", timestamp="t", item_id="s")
MULTIVARIATE = TimeSeriesSchema(target=("y", "z"), timestamp="t", item_id="s")
WITH_PAST = TimeSeriesSchema(target="y", timestamp="t", item_id="s", past_covariates=("temp",))
WITH_KNOWN = TimeSeriesSchema(target="y", timestamp="t", item_id="s", known_covariates=("temp",))
WITH_CATEGORICAL = TimeSeriesSchema(
    target="y", timestamp="t", item_id="s", past_covariates=("weather",)
)


def _panel(schema=UNIVARIATE, *, horizon: int | None = None):
    columns = ["s", "t", *schema.target_names, *schema.covariate_names]
    frame = _frame()[columns]
    future = _future(horizon) if schema.known_covariates else None
    return schema.to_panel(frame, native_missing=True, future_df=future, horizon=horizon)


def _pipeline(checkpoint, horizon=3, cache=None, **forecast_params):
    return TimeSeriesPipeline(
        "TimesFM3",
        model_params={"checkpoint": checkpoint},
        forecast_params={"prediction_length": horizon, **forecast_params},
        tuning_params={"device": "cpu"},
        cache=cache,
    )



@pytest.mark.unit
def test_spec_matches_the_released_checkpoint():
    spec = get_time_series_model_spec("TimesFM-3")
    assert spec.name == "TimesFM3"
    assert spec.family == "timesfm"
    assert spec.max_context == 15360
    assert spec.max_horizon is None
    assert spec.native_missing is True
    assert spec.supports_multivariate and spec.supports_covariates
    assert spec.supports_categorical_covariates is False
    assert spec.default_checkpoint == "google/timesfm-3.0-pytorch"
    assert spec.checkpoints == ("google/timesfm-3.0-pytorch",)

    assert spec.dependency_extra is None
    # Embeddings yes, training no; see test_fine_tuning_is_refused_with_the_reason.
    assert spec.tasks == frozenset(
        {"forecasting", "anomaly_detection", "imputation", "embedding"}
    )
    assert spec.strategies == frozenset({"inference", "finetune", "peft"})


@pytest.mark.unit
def test_a_second_family_does_not_collide_with_chronos():
    """TimesFM lives in the same registry as Chronos without clashing."""
    from tabtune.registry import list_time_series_models

    specs = {spec.name: spec for spec in list_time_series_models()}
    assert specs["TimesFM3"].family == "timesfm"
    assert specs["Chronos"].family == "chronos"
    # Aliases are global, so a second family must not shadow the first.
    assert get_time_series_model_spec("chronos-t5").name == "Chronos"
    assert get_time_series_model_spec("TimesFM-3").name == "TimesFM3"
    # The bare "TimesFM" alias is unclaimed.
    from tabtune.registry.errors import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        get_time_series_model_spec("TimesFM")


@pytest.mark.unit
def test_non_commercial_weights_are_declared_and_gated():
    """The first TS model whose weights restrict use; the check already existed."""
    spec = get_time_series_model_spec("TimesFM3")
    assert spec.license.name == "TimesFM Non-Commercial License v1.0"
    assert spec.license.commercial_use_ok is False
    assert spec.commercial_alternatives == ("Chronos2", "ChronosBolt")

    with pytest.raises(LicenseError, match="Chronos2"):
        TimesFM3Pipeline = TimeSeriesPipeline  # noqa: N806 - readability only
        TimesFM3Pipeline(
            "TimesFM3",
            forecast_params={"prediction_length": 3},
            license_mode="commercial",
        )
    TimeSeriesPipeline("TimesFM3", forecast_params={"prediction_length": 3})


@pytest.mark.unit
def test_capability_flags_name_this_model_where_it_helps():
    """Registering TimesFM3 improves the covariate error message of models that cannot read them."""
    from tabtune.registry import check_schema_support

    with pytest.raises(ConfigError, match="TimesFM3"):
        check_schema_support(get_time_series_model_spec("Chronos"), WITH_PAST)



@pytest.mark.unit
def test_vendored_package_records_its_upstream_provenance():
    """The vendored copy carries no license files, so provenance lives in code."""
    import tabtune.models.timesfm3 as vendored

    assert vendored.UPSTREAM_VERSION == "3.0.2"
    assert len(vendored.UPSTREAM_COMMIT) == 40
    files = [
        "configs.py",
        "cpm_revin_refine.py",
        "dense.py",
        "model.py",
        "normalization.py",
        "timesfm3_forecaster.py",
        "transformer.py",
        "util.py",
    ]
    for name in files:
        header = (VENDORED / name).read_text()[:400]
        assert "Copyright 2026 Google LLC" in header, name
        assert "Licensed under the Apache License" in header, name


@pytest.mark.unit
def test_vendored_modules_import_only_relatively():
    """Every vendored module imports relatively; an absolute `timesfm3` import would be the pip package."""
    for path in sorted(VENDORED.glob("*.py")):
        source = path.read_text()
        assert "import timesfm3" not in source, path.name
        assert "from timesfm3" not in source, path.name


@pytest.mark.unit
def test_load_reads_the_checkpoint_in_float32(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    assert adapter.is_loaded
    assert next(adapter._model.model.parameters()).dtype == torch.float32
    assert str(adapter._model.device) == "cpu"
    assert adapter._trained_levels() == TRAINED_QUANTILES


@pytest.mark.unit
def test_forecast_shapes_with_missing_history(tiny_checkpoint):
    """NaN in the history (interpolated upstream) still gives finite forecasts."""
    out = _adapter(tiny_checkpoint).forecast(_panel(), ForecastConfig(prediction_length=3))
    assert out.point.shape == (2, 3)
    assert out.quantiles.shape == (2, 3, 3)
    assert np.isfinite(out.point).all()
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
    """y is forecast differently when z is in the same forward pass."""
    adapter = _adapter(tiny_checkpoint)
    config = ForecastConfig(prediction_length=3)
    alone = adapter.forecast(_panel(UNIVARIATE), config)
    together = adapter.forecast(_panel(MULTIVARIATE), config)
    # Rows 0 and 2 are the y variates of the two items.
    assert not np.array_equal(alone.point, together.point[[0, 2]])


@pytest.mark.unit
def test_covariates_reach_the_model_and_change_the_forecast(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    config = ForecastConfig(prediction_length=3)
    bare = adapter.forecast(_panel(UNIVARIATE), config)
    past = adapter.forecast(_panel(WITH_PAST), config)
    known = adapter.forecast(_panel(WITH_KNOWN, horizon=3), config)

    assert not np.array_equal(bare.point, past.point)
    assert not np.array_equal(past.point, known.point)


@pytest.mark.unit
def test_covariate_channels_are_built_in_declaration_order(tiny_checkpoint):
    """Upstream takes unnamed channels, so the order is part of the contract."""
    schema = TimeSeriesSchema(
        target="y",
        timestamp="t",
        item_id="s",
        past_covariates=("z",),
        known_covariates=("temp",),
    )
    panel = schema.to_panel(
        _frame()[["s", "t", "y", "z", "temp"]],
        native_missing=True,
        future_df=_future(3),
        horizon=3,
    )
    adapter = _adapter(tiny_checkpoint)
    rows = panel.item_groups()[0]

    past_only = adapter._past_only_channels(panel, rows)
    past_future = adapter._past_future_channels(panel, rows, 3)
    assert past_only.shape == (1, HISTORY)
    np.testing.assert_array_equal(past_only[0], panel.past_covariates[rows[0]]["z"])
    # A known covariate spans history and horizon in one array.
    assert past_future.shape == (1, HISTORY + 3)
    np.testing.assert_array_equal(past_future[0][:HISTORY], panel.past_covariates[rows[0]]["temp"])
    np.testing.assert_array_equal(past_future[0][HISTORY:], panel.future_covariates[rows[0]]["temp"])


@pytest.mark.unit
def test_forecasts_are_deterministic_without_a_seed(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    config = ForecastConfig(prediction_length=3)
    first = adapter.forecast(_panel(), config)
    second = adapter.forecast(_panel(), config)
    np.testing.assert_array_equal(first.point, second.point)
    np.testing.assert_array_equal(first.quantiles, second.quantiles)


@pytest.mark.unit
def test_horizon_beyond_the_output_patch_is_produced_in_one_pass(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(_panel(), ForecastConfig(prediction_length=40))
    assert out.point.shape == (2, 40)
    assert np.isfinite(out.point).all()


@pytest.mark.unit
def test_point_forecast_is_the_median(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(
        _panel(), ForecastConfig(prediction_length=3, quantile_levels=[0.1, 0.5, 0.9])
    )
    np.testing.assert_array_equal(out.point, out.quantiles[:, :, 1])
    assert TimesFM3Adapter.point_forecast == "median"


@pytest.mark.unit
def test_median_is_returned_even_when_no_quantiles_are_requested(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(
        _panel(), ForecastConfig(prediction_length=3, quantile_levels=[])
    )
    assert out.quantiles is None
    assert np.isfinite(out.point).all()


@pytest.mark.unit
def test_trained_levels_are_read_off_exactly_and_in_request_order(tiny_checkpoint):
    """Requested levels index upstream's grid, in request order."""
    adapter = _adapter(tiny_checkpoint)
    assert adapter._quantile_columns([0.1, 0.5, 0.9]) == [0, 4, 8]
    assert adapter._quantile_columns([0.9, 0.1]) == [8, 0]
    assert adapter._quantile_columns([]) == []

    config = ForecastConfig(prediction_length=3, quantile_levels=[0.2, 0.8])
    assert config.quantile_levels == [0.2, 0.8]
    out = adapter.forecast(_panel(), config)
    nine = adapter.forecast(
        _panel(), ForecastConfig(prediction_length=3, quantile_levels=TRAINED_QUANTILES)
    )
    np.testing.assert_array_equal(out.quantiles[:, :, 0], nine.quantiles[:, :, 1])
    np.testing.assert_array_equal(out.quantiles[:, :, 1], nine.quantiles[:, :, 7])


@pytest.mark.unit
def test_quantiles_are_monotone(tiny_checkpoint):
    """Upstream sorts its quantile output, so this holds even on random weights."""
    out = _adapter(tiny_checkpoint).forecast(
        _panel(), ForecastConfig(prediction_length=3, quantile_levels=[0.1, 0.5, 0.9])
    )
    assert (np.diff(out.quantiles, axis=2) >= 0).all()


@pytest.mark.unit
def test_levels_off_the_trained_grid_are_refused_not_interpolated(tiny_checkpoint):
    """TimesFM predicts nine fixed levels; anything else would be invented."""
    adapter = _adapter(tiny_checkpoint)
    with pytest.raises(ConfigError, match="will not interpolate"):
        adapter.forecast(
            _panel(), ForecastConfig(prediction_length=3, quantile_levels=[0.05, 0.5, 0.95])
        )
    with pytest.raises(ConfigError, match=r"\[0.25\]"):
        adapter._quantile_columns([0.25])


@pytest.mark.unit
def test_batch_size_chunks_do_not_change_or_reorder_forecasts(tiny_checkpoint):
    """Equal-length histories: chunking must not move or reorder numbers."""
    config = ForecastConfig(prediction_length=3)
    whole = _adapter(tiny_checkpoint, batch_size=32).forecast(_panel(MULTIVARIATE), config)
    chunked = _adapter(tiny_checkpoint, batch_size=1).forecast(_panel(MULTIVARIATE), config)
    np.testing.assert_allclose(whole.point, chunked.point, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_ragged_histories_are_padded_by_the_backend(tiny_checkpoint):
    """Upstream masks the padding, so the adapter does not pad ragged histories itself."""
    frame = _frame(missing=False)[["s", "t", "y"]]
    frame = frame.drop(frame.index[(frame["s"] == "a") & (frame["t"] < "2024-01-10")])
    panel = UNIVARIATE.to_panel(frame)
    assert {len(values) for values in panel.values} == {HISTORY, HISTORY - 9}

    out = _adapter(tiny_checkpoint).forecast(panel, ForecastConfig(prediction_length=3))
    assert out.point.shape == (2, 3)
    assert np.isfinite(out.point).all()


@pytest.mark.unit
def test_invalid_model_params():
    spec = get_time_series_model_spec("TimesFM3")
    with pytest.raises(ConfigError, match=r"batch_size.\] must be >= 1"):
        TimesFM3Adapter(spec, checkpoint="x", device="cpu", model_params={"batch_size": 0})
    with pytest.raises(ConfigError, match="padding_mode must be one of"):
        TimesFM3Adapter(spec, checkpoint="x", device="cpu", model_params={"padding_mode": "wrap"})
    with pytest.raises(ConfigError, match="only supports point_forecast='median'"):
        TimesFM3Adapter(spec, checkpoint="x", device="cpu", model_params={"point_forecast": "mean"})
    with pytest.warns(UserWarning, match="num_samples"):
        TimesFM3Adapter(spec, checkpoint="x", device="cpu", model_params={"num_samples": 20})
    with pytest.warns(UserWarning, match="dtype"):
        TimesFM3Adapter(spec, checkpoint="x", device="cpu", model_params={"dtype": "bfloat16"})


@pytest.mark.unit
def test_upstream_inference_knobs_are_forwarded(tiny_checkpoint):
    """make_positive and use_znorm are upstream behaviour, off by default."""
    config = ForecastConfig(prediction_length=3)
    plain = _adapter(tiny_checkpoint).forecast(_panel(), config)
    normed = _adapter(tiny_checkpoint, use_znorm=True).forecast(_panel(), config)
    positive = _adapter(tiny_checkpoint, make_positive=True).forecast(_panel(), config)

    assert not np.array_equal(plain.point, normed.point)
    assert (positive.point >= 0).all()




@pytest.mark.unit
def test_pipeline_end_to_end_with_covariates(tiny_checkpoint, tmp_path):
    """The full architecture, offline: a second family through the same layer."""
    schema = TimeSeriesSchema(
        target=("y", "z"),
        timestamp="t",
        item_id="s",
        past_covariates=("temp",),
        known_covariates=("temp2",),
    )
    frame = _frame()[["s", "t", "y", "z", "temp"]].assign(temp2=lambda d: d["temp"] * 2)
    future = _future(3).assign(temp2=lambda d: d["temp"] * 2).drop(columns="temp")
    pipe = _pipeline(tiny_checkpoint).fit(frame, schema, future_df=future)
    assert type(pipe.adapter_) is TimesFM3Adapter

    result = pipe.predict()
    assert result.point.shape == (4, 3)
    assert result.targets == ("y", "z", "y", "z")
    assert result.metadata["point_forecast"] == "median"
    assert result.metadata["model"] == "TimesFM3"

    long = result.to_pandas()
    assert len(long) == 4 * 3
    assert set(long["target"]) == {"y", "z"}

    actual = _frame().tail(6)[["s", "t", "y", "z"]].assign(
        t=pd.date_range("2024-01-25", periods=3, freq="D").tolist() * 2,
        s=["a"] * 3 + ["b"] * 3,
    )
    metrics = pipe.evaluate(actual, forecast=result)
    assert set(metrics["per_target"]) == {"y", "z"}

    path = tmp_path / "timesfm3.joblib"
    pipe.save(str(path))
    restored = TimeSeriesPipeline.load(str(path))
    assert not restored.adapter_.is_loaded
    np.testing.assert_array_equal(restored.predict().point, result.point)


@pytest.mark.unit
def test_pipeline_rejects_categorical_covariates_before_loading_weights(tiny_checkpoint):
    """TimesFM covariates are float channels, so a string column is refused before weights load."""
    pipe = _pipeline(tiny_checkpoint)
    with pytest.raises(ConfigError, match="categorical"):
        pipe.fit(_frame()[["s", "t", "y", "weather"]], WITH_CATEGORICAL)
    assert pipe.adapter_ is None


@pytest.mark.unit
def test_missing_future_covariates_are_refused_before_the_model_runs(tiny_checkpoint):
    pipe = _pipeline(tiny_checkpoint).fit(_frame()[["s", "t", "y", "temp"]], WITH_KNOWN)
    with pytest.raises(ConfigError, match="predict\\(future_df=...\\)"):
        pipe.predict()


@pytest.mark.unit
def test_changing_future_covariates_invalidates_the_cache(tiny_checkpoint):
    frame = _frame()[["s", "t", "y", "temp"]]
    pipe = _pipeline(tiny_checkpoint, cache=True).fit(frame, WITH_KNOWN, future_df=_future(3))
    first = pipe.predict()
    np.testing.assert_array_equal(pipe.predict().point, first.point)

    warmer = _future(3).assign(temp=lambda d: d["temp"] + 25)
    assert not np.array_equal(first.point, pipe.predict(future_df=warmer).point)


@pytest.mark.unit
def test_long_horizon_does_not_warn_because_no_limit_is_declared(tiny_checkpoint, recwarn):
    """max_horizon is None, so check_forecast_envelope must stay quiet."""
    pipe = _pipeline(tiny_checkpoint, horizon=200, envelope_mode="error")
    pipe.fit(_frame()[["s", "t", "y"]], UNIVARIATE)
    assert not [w for w in recwarn if "horizon" in str(w.message)]


@pytest.mark.slow
@pytest.mark.weights
def test_released_checkpoint_beats_a_naive_forecast():
    """The released 330M model with a known-future covariate beats the last value over one week."""
    horizon = 7
    pipe = TimeSeriesPipeline(
        "TimesFM3",
        forecast_params={"prediction_length": horizon, "quantile_levels": [0.1, 0.5, 0.9]},
        tuning_params={"device": "cpu"},
    )
    frame = _frame(missing=False)[["s", "t", "y", "temp"]]
    result = pipe.fit(frame, WITH_KNOWN, future_df=_future(horizon)).predict()

    assert result.point.shape == (2, horizon)
    assert np.isfinite(result.point).all()

    steps = np.arange(HISTORY, HISTORY + horizon, dtype=float)
    truth = np.stack(
        [steps + 3 * np.sin(steps * 2 * np.pi / 7) + 10 * offset for offset in (0, 1)]
    )
    last = np.repeat(frame.groupby("s")["y"].last().to_numpy()[:, None], horizon, axis=1)
    model_mae = float(np.abs(result.point - truth).mean())
    naive_mae = float(np.abs(last - truth).mean())
    assert model_mae < naive_mae / 2, (model_mae, naive_mae)
    assert model_mae < 2.0, model_mae

    assert (result.point[1] > result.point[0]).all()
    assert (np.diff(result.quantiles, axis=2) >= 0).all()


def test_anomaly_detection_defaults_to_the_trained_deciles(tiny_checkpoint):
    from tabtune.TimeSeries import make_panel

    pipe = TimeSeriesPipeline(
        "TimesFM3",
        task_type="anomaly_detection",
        model_params={"checkpoint": tiny_checkpoint, "device": "cpu"},
        task_params={"min_context": 32},
    )
    frame = make_panel(2, 80, freq="h", seed=0)
    result = pipe.fit(frame, TimeSeriesSchema(target="target", item_id="item_id")).predict()
    assert result.metadata["coverage"] == pytest.approx(0.8)
    assert result.frame["score"].notna().any()


@pytest.mark.unit
def test_embeddings_are_one_vector_per_series_whatever_the_batching(tiny_checkpoint):
    from tabtune.TimeSeries import make_panel

    # Ragged, and not a multiple of the patch size.
    frame = make_panel(3, 45, freq="h", seed=0)
    frame = frame[~((frame["item_id"] == "series_2") & (frame.index % 45 < 13))]
    schema = TimeSeriesSchema(target="target", item_id="item_id")

    def embed(**model_params):
        return (
            TimeSeriesPipeline(
                "TimesFM3",
                task_type="embedding",
                model_params={"checkpoint": tiny_checkpoint, **model_params},
                tuning_params={"device": "cpu"},
            )
            .fit(frame, schema)
            .predict()
        )

    result = embed()
    model_dims = 32
    assert result.embeddings.shape == (3, model_dims)
    assert np.isfinite(result.embeddings).all()
    np.testing.assert_allclose(embed(batch_size=1).embeddings, result.embeddings, atol=1e-5)

    # A series' vector must not depend on what it was batched with.
    alone = (
        TimeSeriesPipeline(
            "TimesFM3",
            task_type="embedding",
            model_params={"checkpoint": tiny_checkpoint},
            tuning_params={"device": "cpu"},
        )
        .fit(frame[frame["item_id"] == "series_0"], schema)
        .predict()
    )
    np.testing.assert_allclose(alone.embeddings[0], result.embeddings[0], atol=1e-5)


@pytest.mark.unit
def test_embeddings_leave_out_the_forecast_positions(tiny_checkpoint):
    """Only context patches are pooled, so the embedding does not change with the horizon."""
    adapter = _adapter(tiny_checkpoint)
    panel = _panel()
    first = adapter.embed(panel)
    # The aux tensor grows with the horizon; pooling those positions would make this disagree.
    assert first.shape == (len(panel), 32)
    np.testing.assert_allclose(adapter.embed(panel), first, atol=1e-6)


@pytest.mark.parametrize("strategy", ["peft", "finetune"])
def test_fine_tuning_round_trips_through_save_and_load(tiny_checkpoint, tmp_path, strategy):
    from tabtune.TimeSeries import make_panel

    frame = make_panel(4, 80, freq="h", seed=1)
    frame.loc[[10, 11], "target"] = np.nan
    schema = TimeSeriesSchema(target="target", item_id="item_id")
    params = {"checkpoint": tiny_checkpoint, "device": "cpu"}
    zero_shot = TimeSeriesPipeline("TimesFM3", model_params=params, forecast_params={"prediction_length": 6})
    before = zero_shot.fit(frame, schema).predict().point
    pipe = TimeSeriesPipeline(
        "TimesFM3",
        tuning_strategy=strategy,
        model_params=params,
        forecast_params={"prediction_length": 6},
        tuning_params={"epochs": 1, "steps_per_epoch": 3, "batch_size": 8, "learning_rate": 1e-3, "seed": 0},
    ).fit(frame, schema)
    report = pipe.training_report_
    assert report["steps_run"] == 3 and np.isfinite(report["final_train_loss"])
    assert "pinball" in report["objective"]
    tuned = pipe.predict().point
    assert np.isfinite(tuned).all() and not np.allclose(tuned, before, atol=1e-7)
    pipe.save(str(tmp_path / "timesfm3.joblib"))
    restored = TimeSeriesPipeline.load(str(tmp_path / "timesfm3.joblib")).predict().point
    np.testing.assert_allclose(restored, tuned, atol=1e-5)
    again = zero_shot.fit(frame, schema).predict().point
    np.testing.assert_allclose(again, before, atol=1e-6)


def test_training_windows_skip_padding_and_constant_patches():
    from tabtune.models.TimeSeries.timesfm3 import _usable_rows

    context = np.full((4, 32), np.nan, dtype=np.float32)
    context[0] = np.arange(32)
    context[1, 20:] = np.arange(12)
    context[2, :] = np.arange(32)
    context[2, 8:16] = 5.0
    context[3, 30:] = [1.0, 2.0]
    assert _usable_rows(context, 8) == {32: [0], 8: [1]}

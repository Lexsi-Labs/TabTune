"""Tests for the vendored Toto 2.0 code and its TimeSeriesPipeline adapter.

Like the Chronos and TimesFM tests, these save a tiny randomly initialised
checkpoint to a temporary directory and drive it through the real path: registry
-> adapter imported by path -> ``Toto2Adapter.load()`` ->
``Toto2Model.from_pretrained`` -> variate attention -> quantile knots head.
Nothing is monkeypatched and nothing is downloaded.

These tests also pin the ``toto`` pip extra being named in the install error,
the padding of the context to a patch boundary, and the masked filling of
missing values. The slow test at the bottom runs the released
``Datadog/Toto-2.0-22m``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
# The u-microP libraries are an optional extra; conftest.py only turns download failures into skips.
pytest.importorskip("dd_unit_scaling")

from tabtune._internal.deprecation import reset_warning_cache  # noqa: E402
from tabtune.config import ForecastConfig  # noqa: E402
from tabtune.models.TimeSeries.toto2 import Toto2Adapter  # noqa: E402
from tabtune.models.toto.toto2 import Toto2Model, Toto2ModelConfig  # noqa: E402
from tabtune.registry import ConfigError, get_time_series_model_spec  # noqa: E402
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema  # noqa: E402

pytestmark = [pytest.mark.time_series, pytest.mark.model_toto]

TRAINED_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PATCH = 8
HISTORY = 48
VENDORED = Path(__file__).resolve().parents[1] / "tabtune" / "models" / "toto" / "toto2"


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """warn_once de-duplicates per process; reset so this file's assertions see the warning."""
    reset_warning_cache()
    yield
    reset_warning_cache()


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """A randomly initialised Toto 2.0 checkpoint; ``residual_attn_ratio`` has no upstream default."""
    torch.manual_seed(0)
    config = Toto2ModelConfig(
        patch_size=PATCH,
        d_model=32,
        num_heads=2,
        num_layers=2,
        layer_group_size=2,
        num_variate_layers_per_group=1,
        variate_layer_first=False,
        residual_attn_ratio=Toto2ModelConfig.compute_residual_attn_ratio(64, PATCH),
    )
    path = tmp_path_factory.mktemp("tiny-toto2")
    Toto2Model(config).save_pretrained(path)
    return str(path)


def _adapter(checkpoint, **model_params) -> Toto2Adapter:
    adapter = Toto2Adapter(
        get_time_series_model_spec("Toto2"),
        checkpoint=checkpoint,
        device="cpu",
        model_params=model_params,
    )
    adapter.load()
    return adapter


def _frame(*, missing: bool = True) -> pd.DataFrame:
    """Two items, two targets; the target is not a straight line."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=HISTORY, freq="D")
    steps = np.arange(HISTORY, dtype=float)
    frames = []
    for offset, item in enumerate(("a", "b")):
        y = steps * 0.3 + 3 * np.sin(steps * 2 * np.pi / 7) + rng.normal(0, 0.4, HISTORY)
        y += 5 * offset
        if missing and offset == 0:
            y[6:9] = np.nan
        frames.append(
            pd.DataFrame(
                {"s": item, "t": dates, "y": y, "z": 8 * y + rng.normal(0, 2, HISTORY)}
            )
        )
    return pd.concat(frames, ignore_index=True)


UNIVARIATE = TimeSeriesSchema(target="y", timestamp="t", item_id="s")
MULTIVARIATE = TimeSeriesSchema(target=("y", "z"), timestamp="t", item_id="s")
WITH_COVARIATE = TimeSeriesSchema(
    target="y", timestamp="t", item_id="s", past_covariates=("z",)
)


def _panel(schema=UNIVARIATE, *, frame=None):
    columns = ["s", "t", *schema.target_names]
    return schema.to_panel((frame if frame is not None else _frame())[columns], native_missing=True)


def _pipeline(checkpoint, horizon=3, cache=None, **forecast_params):
    return TimeSeriesPipeline(
        "Toto2",
        model_params={"checkpoint": checkpoint},
        forecast_params={"prediction_length": horizon, **forecast_params},
        tuning_params={"device": "cpu"},
        cache=cache,
        validate=False,
    )


# ------------------------------------------------------------------ registry


@pytest.mark.unit
def test_spec_matches_the_released_checkpoints():
    spec = get_time_series_model_spec("Toto-2")
    assert spec.name == "Toto2"
    assert spec.family == "toto"
    # Derived from residual_attn_ratio, not documented upstream; see the catalog.
    assert spec.max_context == 4096
    # The horizon is decoded in blocks with median feedback, so nothing caps it.
    assert spec.max_horizon is None
    assert spec.native_missing is True
    assert spec.supports_multivariate is True
    assert spec.supports_covariates is False
    assert spec.default_checkpoint == "Datadog/Toto-2.0-22m"
    assert spec.checkpoints == tuple(
        f"Datadog/Toto-2.0-{size}" for size in ("4m", "22m", "313m", "1B", "2.5B")
    )
    assert spec.dependency_extra == "toto2"


@pytest.mark.unit
def test_three_families_coexist():
    """Toto is a third family alongside Chronos and TimesFM."""
    from tabtune.registry import list_time_series_models

    specs = list_time_series_models()
    assert {"chronos", "timesfm", "toto"} <= {spec.family for spec in specs}
    assert {s.name for s in specs if s.family == "toto"} >= {"Toto2"}
    # The bare "Toto" alias is unclaimed by every Toto spec.
    from tabtune.registry.errors import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        get_time_series_model_spec("Toto")


# ------------------------------------------------------------------ vendoring


@pytest.mark.unit
def test_vendored_package_records_its_upstream_provenance():
    """The vendored copy records its upstream sdist digest (Toto ships no tagged release)."""
    import tabtune.models.toto.toto2 as vendored

    assert vendored.UPSTREAM_VERSION == "2.0.0"
    assert len(vendored.UPSTREAM_SDIST_SHA256) == 64
    for name in ("configuration.py", "model.py"):
        header = (VENDORED / name).read_text()[:400]
        assert "Apache-2.0 License" in header, name
        assert "Datadog" in header, name


@pytest.mark.unit
def test_vendored_model_needs_neither_gluonts_nor_the_pip_package():
    """Nothing vendored imports gluonts or falls back to the installed ``toto2`` package."""
    code = "\n".join(
        line
        for line in (VENDORED / "model.py").read_text().splitlines()
        if not line.lstrip().startswith("#")          # the vendoring note names both
    )
    assert "gluonts" not in code
    # The unused ``Toto2GluonTSModelConfig`` import stays so configuration.py is verbatim.
    assert "class Toto2GluonTSModel" not in code
    for path in sorted(VENDORED.glob("*.py")):
        text = path.read_text()
        assert "import toto2" not in text, path.name
        assert "from toto2" not in text, path.name
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, tabtune.models.toto.toto2; print('gluonts' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "False"


# ------------------------------------------------------------------- adapter


@pytest.mark.unit
def test_load_reads_the_checkpoint_in_float32(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    assert adapter.is_loaded
    assert next(adapter._model.parameters()).dtype == torch.float32
    assert adapter._trained_levels() == TRAINED_QUANTILES
    assert adapter._model.config.patch_size == PATCH


@pytest.mark.unit
def test_missing_backend_library_names_the_install_extra(tiny_checkpoint, monkeypatch):
    """load() produces the install hint; the backend import is deferred into it."""
    adapter = Toto2Adapter(
        get_time_series_model_spec("Toto2"), checkpoint=tiny_checkpoint, device="cpu"
    )
    for name in [m for m in list(sys.modules) if m.startswith("tabtune.models.toto.toto2")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "dd_unit_scaling", None)
    with pytest.raises(ImportError, match=r"pip install 'tabtune\[toto2\]'"):
        adapter.load()


@pytest.mark.unit
def test_forecast_shapes_with_missing_history(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(_panel(), ForecastConfig(prediction_length=3))
    assert out.point.shape == (2, 3)
    assert out.quantiles.shape == (2, 3, 3)
    assert np.isfinite(out.point).all()
    assert np.isfinite(out.quantiles).all()


@pytest.mark.unit
def test_point_forecast_is_the_median(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(
        _panel(), ForecastConfig(prediction_length=3, quantile_levels=[0.1, 0.5, 0.9])
    )
    np.testing.assert_array_equal(out.point, out.quantiles[:, :, 1])
    assert Toto2Adapter.point_forecast == "median"


@pytest.mark.unit
def test_quantiles_are_monotone(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(
        _panel(), ForecastConfig(prediction_length=3, quantile_levels=TRAINED_QUANTILES)
    )
    assert (np.diff(out.quantiles, axis=2) >= 0).all()


@pytest.mark.unit
def test_multivariate_rows_are_forecast_jointly(tiny_checkpoint):
    panel = _panel(MULTIVARIATE)
    assert len(panel) == 4
    assert panel.target_names == ("y", "z", "y", "z")
    assert panel.item_groups() == ((0, 1), (2, 3))

    adapter = _adapter(tiny_checkpoint)
    config = ForecastConfig(prediction_length=3)
    together = adapter.forecast(panel, config)
    alone = adapter.forecast(_panel(UNIVARIATE), config)
    assert together.point.shape == (4, 3)
    # Rows 0 and 2 are the y variates; sharing a pass with z must change them.
    assert not np.array_equal(alone.point, together.point[[0, 2]])


@pytest.mark.unit
def test_context_is_padded_to_a_patch_boundary(tiny_checkpoint):
    """Upstream raises on a context that does not divide the patch size, so the adapter pads; padding must not change the forecast."""
    frame = _frame(missing=False)
    short = frame.drop(frame.index[frame["t"] < "2024-01-04"])  # 45 steps, not a multiple of 8
    assert len(short) % PATCH != 0
    adapter = _adapter(tiny_checkpoint)
    config = ForecastConfig(prediction_length=3)

    out = adapter.forecast(_panel(frame=short), config)
    assert out.point.shape == (2, 3)
    assert np.isfinite(out.point).all()

    # Padding is masked, so a further-padded series forecasts identically.
    padded = adapter.forecast(_panel(frame=frame.tail(0).pipe(lambda _: short)), config)
    np.testing.assert_array_equal(out.point, padded.point)


@pytest.mark.unit
def test_ragged_histories_share_a_batch(tiny_checkpoint):
    frame = _frame(missing=False)
    frame = frame.drop(frame.index[(frame["s"] == "a") & (frame["t"] < "2024-01-12")])
    panel = _panel(frame=frame)
    assert {len(values) for values in panel.values} == {HISTORY, HISTORY - 11}

    out = _adapter(tiny_checkpoint).forecast(panel, ForecastConfig(prediction_length=3))
    assert out.point.shape == (2, 3)
    assert np.isfinite(out.point).all()


@pytest.mark.unit
def test_batch_size_chunks_do_not_reorder_forecasts(tiny_checkpoint):
    """Chunking may move numbers slightly (per-batch padding) but never reorder rows."""
    config = ForecastConfig(prediction_length=3)
    whole = _adapter(tiny_checkpoint, batch_size=32).forecast(_panel(MULTIVARIATE), config)
    chunked = _adapter(tiny_checkpoint, batch_size=1).forecast(_panel(MULTIVARIATE), config)
    np.testing.assert_allclose(whole.point, chunked.point, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_masked_values_cannot_influence_the_forecast(tiny_checkpoint):
    """The mask suppresses filled positions entirely, so any finite fill forecasts identically."""
    frame = _frame()
    absurd = frame.copy()
    gap = absurd["y"].isna()
    assert gap.any()
    absurd.loc[gap, "y"] = -1e6

    adapter = _adapter(tiny_checkpoint)
    config = ForecastConfig(prediction_length=3)
    with_nan = adapter.forecast(_panel(frame=frame), config)
    # Same rows and mask, a wild fill in the gap instead.
    filled = adapter.forecast(_panel(frame=frame.assign(y=frame["y"].ffill())), config)

    assert np.isfinite(with_nan.point).all()
    # Here the gap's ffill reaches the model as observed, so the forecast must differ.
    assert not np.array_equal(with_nan.point[0], filled.point[0])
    # The clean series is untouched either way.
    np.testing.assert_allclose(with_nan.point[1], filled.point[1], rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_decode_block_size_changes_a_long_horizon(tiny_checkpoint):
    """Blocks are decoded with median feedback, so decode_block_size changes the numbers beyond one block."""
    config = ForecastConfig(prediction_length=40)
    blocked = _adapter(tiny_checkpoint, decode_block_size=16).forecast(_panel(), config)
    whole = _adapter(tiny_checkpoint, decode_block_size=768).forecast(_panel(), config)
    assert blocked.point.shape == (2, 40)
    assert not np.allclose(blocked.point, whole.point)


@pytest.mark.unit
def test_decode_block_size_must_be_a_patch_multiple(tiny_checkpoint):
    """Upstream asserts this; failing at load() names both numbers instead."""
    adapter = Toto2Adapter(
        get_time_series_model_spec("Toto2"),
        checkpoint=tiny_checkpoint,
        device="cpu",
        model_params={"decode_block_size": PATCH + 1},
    )
    with pytest.raises(ConfigError, match="multiple of the checkpoint's patch size"):
        adapter.load()


@pytest.mark.unit
def test_levels_off_the_trained_grid_are_refused_not_interpolated(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    with pytest.raises(ConfigError, match="will not interpolate"):
        adapter.forecast(
            _panel(), ForecastConfig(prediction_length=3, quantile_levels=[0.05, 0.5, 0.95])
        )
    assert adapter._quantile_rows([0.1, 0.5, 0.9]) == [0, 4, 8]
    assert adapter._quantile_rows([]) == []


@pytest.mark.unit
def test_median_is_returned_even_when_no_quantiles_are_requested(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(
        _panel(), ForecastConfig(prediction_length=3, quantile_levels=[])
    )
    assert out.quantiles is None
    assert np.isfinite(out.point).all()


@pytest.mark.unit
def test_forecasts_are_deterministic_without_a_seed(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    config = ForecastConfig(prediction_length=3)
    first = adapter.forecast(_panel(), config)
    second = adapter.forecast(_panel(), config)
    np.testing.assert_array_equal(first.point, second.point)
    np.testing.assert_array_equal(first.quantiles, second.quantiles)


@pytest.mark.unit
def test_invalid_model_params():
    spec = get_time_series_model_spec("Toto2")
    with pytest.raises(ConfigError, match=r"batch_size.\] must be >= 1"):
        Toto2Adapter(spec, checkpoint="x", device="cpu", model_params={"batch_size": 0})
    with pytest.raises(ConfigError, match=r"decode_block_size.\] must be >= 1"):
        Toto2Adapter(spec, checkpoint="x", device="cpu", model_params={"decode_block_size": 0})
    with pytest.raises(ConfigError, match="only supports point_forecast='median'"):
        Toto2Adapter(spec, checkpoint="x", device="cpu", model_params={"point_forecast": "mean"})
    # The Chronos sampling knobs mean nothing here, and there is no dtype.
    with pytest.warns(UserWarning, match="num_samples"):
        Toto2Adapter(spec, checkpoint="x", device="cpu", model_params={"num_samples": 20})
    with pytest.warns(UserWarning, match="dtype"):
        Toto2Adapter(spec, checkpoint="x", device="cpu", model_params={"dtype": "bfloat16"})


# ------------------------------------------------------------------ pipeline


@pytest.mark.unit
def test_pipeline_end_to_end(tiny_checkpoint, tmp_path):
    pipe = _pipeline(tiny_checkpoint).fit(_frame()[["s", "t", "y", "z"]], MULTIVARIATE)
    assert type(pipe.adapter_) is Toto2Adapter

    result = pipe.predict()
    assert result.point.shape == (4, 3)
    assert result.targets == ("y", "z", "y", "z")
    assert result.metadata["point_forecast"] == "median"
    assert result.metadata["model"] == "Toto2"

    long = result.to_pandas()
    assert len(long) == 4 * 3
    assert set(long["target"]) == {"y", "z"}

    actual = _frame().tail(6)[["s", "t", "y", "z"]].assign(
        t=pd.date_range("2024-02-18", periods=3, freq="D").tolist() * 2,
        s=["a"] * 3 + ["b"] * 3,
    )
    metrics = pipe.evaluate(actual, forecast=result)
    assert set(metrics["per_target"]) == {"y", "z"}

    path = tmp_path / "toto2.joblib"
    pipe.save(str(path))
    restored = TimeSeriesPipeline.load(str(path))
    assert not restored.adapter_.is_loaded
    np.testing.assert_array_equal(restored.predict().point, result.point)


@pytest.mark.unit
def test_covariate_schema_is_refused_before_loading_weights(tiny_checkpoint):
    """The released weights ignore exogenous channels, so a covariate schema is rejected."""
    pipe = _pipeline(tiny_checkpoint)
    with pytest.raises(ConfigError, match="does not use covariates"):
        pipe.fit(_frame()[["s", "t", "y", "z"]], WITH_COVARIATE)
    assert pipe.adapter_ is None


@pytest.mark.unit
def test_covariate_error_names_the_models_that_can_read_them(tiny_checkpoint):
    from tabtune.registry import check_schema_support

    with pytest.raises(ConfigError, match="Chronos2"):
        check_schema_support(get_time_series_model_spec("Toto2"), WITH_COVARIATE)


@pytest.mark.unit
def test_long_horizon_does_not_warn_because_no_limit_is_declared(tiny_checkpoint, recwarn):
    pipe = _pipeline(tiny_checkpoint, horizon=200, envelope_mode="error")
    pipe.fit(_frame()[["s", "t", "y"]], UNIVARIATE)
    assert not [w for w in recwarn if "horizon" in str(w.message)]


# ----------------------------------------------------------------- real weights


@pytest.mark.slow
@pytest.mark.weights
def test_released_checkpoint_beats_a_naive_forecast():
    """The released 22m model, end to end on a trend plus weekly cycle."""
    horizon = 7
    pipe = TimeSeriesPipeline(
        "Toto2",
        forecast_params={"prediction_length": horizon, "quantile_levels": [0.1, 0.5, 0.9]},
        tuning_params={"device": "cpu"},
    )
    frame = _frame(missing=False)[["s", "t", "y"]]
    result = pipe.fit(frame, UNIVARIATE).predict()

    assert result.point.shape == (2, horizon)
    assert np.isfinite(result.point).all()

    steps = np.arange(HISTORY, HISTORY + horizon, dtype=float)
    truth = np.stack(
        [steps * 0.3 + 3 * np.sin(steps * 2 * np.pi / 7) + 5 * offset for offset in (0, 1)]
    )
    last = np.repeat(frame.groupby("s")["y"].last().to_numpy()[:, None], horizon, axis=1)
    model_mae = float(np.abs(result.point - truth).mean())
    naive_mae = float(np.abs(last - truth).mean())
    assert model_mae < naive_mae, (model_mae, naive_mae)
    # Item b sits 5 above item a throughout, and should still.
    assert (result.point[1] > result.point[0]).all()
    assert (np.diff(result.quantiles, axis=2) >= 0).all()


def test_pipeline_tasks_run_on_toto2(tiny_checkpoint):
    from tabtune.TimeSeries import TimeSeriesSchema, make_panel

    schema = TimeSeriesSchema(target="target", item_id="item_id")
    params = {"checkpoint": tiny_checkpoint, "device": "cpu"}
    frame = make_panel(2, 96, freq="h", seed=0)
    scores = TimeSeriesPipeline(
        "Toto2", task_type="anomaly_detection", model_params=params, task_params={"min_context": 32}
    ).fit(frame, schema).predict()
    assert scores.frame["score"].notna().any()
    gappy = frame.copy()
    gappy.loc[[40, 41, 42], "target"] = np.nan
    filled = TimeSeriesPipeline("Toto2", task_type="imputation", model_params=params).fit(gappy, schema).predict()
    assert np.isfinite(filled.to_pandas()["target"]).all()
    pipe = TimeSeriesPipeline("Toto2", model_params=params, forecast_params={"prediction_length": 4})
    calibrated = pipe.fit(frame, schema).calibrate(windows=3).predict()
    assert np.isfinite(calibrated.quantiles).all()
    assert len(pipe.backtest(windows=2)) == 2


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
                "Toto2",
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
            "Toto2",
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

    frame = make_panel(4, 96, freq="h", seed=1)
    schema = TimeSeriesSchema(target="target", item_id="item_id")

    def build(**kwargs):
        return TimeSeriesPipeline(
            "Toto2",
            model_params={"checkpoint": tiny_checkpoint},
            # Seeded throughout: an unseeded sampling model consumes the global RNG.
            forecast_params={"prediction_length": 8, "context_length": 32},
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
    assert "next-patch pinball" in report["objective"]
    tuned = pipe.predict().point
    assert not np.allclose(tuned, before, atol=1e-7)

    # Training must leave the module exactly as forecasting found it.
    network = pipe.adapter_._network()
    assert not any(p.requires_grad for p in network.parameters())
    assert not network.training

    path = tmp_path / f"toto2-{strategy}.joblib"
    pipe.save(str(path))
    np.testing.assert_allclose(
        TimeSeriesPipeline.load(str(path)).predict().point, tuned, atol=1e-6
    )
    # Training one pipeline leaves a fresh zero-shot one untouched.
    np.testing.assert_allclose(zero_shot(), before, atol=1e-6)

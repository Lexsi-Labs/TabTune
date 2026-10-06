"""Tests for the vendored Toto 1.0 code and its TimeSeriesPipeline adapter.

These save a tiny randomly initialised checkpoint to a temporary directory and
drive it through the real path: registry -> adapter imported by path ->
``Toto1Adapter.load()`` -> ``Toto.from_pretrained`` -> space-time attention ->
Student-T mixture -> sample paths. Nothing is monkeypatched and nothing is
downloaded.

They also pin what is specific to this family: arbitrary quantile levels are
served, a mean point forecast exists, the seed stays out of the global RNG, POSIX
timestamps and the step in seconds reach the model, covariates arrive in the
channel order upstream requires, the context is not padded by the adapter, and
gluonts is not a dependency. The slow test at the bottom runs the released
``Datadog/Toto-Open-Base-1.0``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rotary_embedding_torch")

import safetensors.torch as safetorch  # noqa: E402

from tabtune._internal.deprecation import reset_warning_cache  # noqa: E402
from tabtune.config import ForecastConfig  # noqa: E402
from tabtune.models.TimeSeries.toto1 import Toto1Adapter, _freq_to_seconds  # noqa: E402
from tabtune.models.toto.toto1 import Toto  # noqa: E402
from tabtune.registry import (  # noqa: E402
    ConfigError,
    ModelNotFoundError,
    get_time_series_model_spec,
)
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema  # noqa: E402

pytestmark = [pytest.mark.time_series, pytest.mark.model_toto]

PATCH = 8
# Not a multiple of PATCH: upstream pads the context itself.
HISTORY = 37
VENDORED = Path(__file__).resolve().parents[1] / "tabtune" / "models" / "toto" / "toto1"

TINY_CONFIG = {
    "patch_size": PATCH,
    "stride": PATCH,
    "embed_dim": 32,
    "num_layers": 2,
    "num_heads": 2,
    "mlp_hidden_dim": 64,
    "dropout": 0.0,
    "spacewise_every_n_layers": 2,
    "spacewise_first": False,
    "scaler_cls": "<class 'model.scaler.CausalPatchStdMeanScaler'>",
    "output_distribution_classes": [
        "<class 'model.distribution.MixtureOfStudentTsOutput'>"
    ],
    "output_distribution_kwargs": {"k_components": 4},
    "use_memory_efficient_attention": False,
}


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """warn_once de-duplicates per process; reset so this file's assertions see the warning."""
    reset_warning_cache()
    yield
    reset_warning_cache()


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """A randomly initialised Toto 1.0 checkpoint; ``Toto`` has no ``_save_pretrained``, so both files are written directly."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-toto1")
    safetorch.save_file(Toto(**TINY_CONFIG).state_dict(), str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(TINY_CONFIG))
    return str(path)


def _adapter(checkpoint, **model_params) -> Toto1Adapter:
    params = {"num_samples": 16, "samples_per_batch": 16, **model_params}
    adapter = Toto1Adapter(
        get_time_series_model_spec("Toto1"),
        checkpoint=checkpoint,
        device="cpu",
        model_params=params,
        seed=params.pop("seed", 7),
    )
    adapter.load()
    return adapter


def _frame(*, missing: bool = True, periods: int = HISTORY) -> pd.DataFrame:
    """Two items, a target, a second target and a covariate; the target is not a straight line."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    steps = np.arange(periods, dtype=float)
    frames = []
    for offset, item in enumerate(("a", "b")):
        y = steps * 0.3 + 3 * np.sin(steps * 2 * np.pi / 7) + rng.normal(0, 0.4, periods)
        y += 5 * offset
        if missing and offset == 0:
            y[6:9] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "s": item,
                    "t": dates,
                    "y": y,
                    "z": 8 * y + rng.normal(0, 2, periods),
                    "p": (rng.random(periods) < 0.3).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


UNIVARIATE = TimeSeriesSchema(target="y", timestamp="t", item_id="s")
MULTIVARIATE = TimeSeriesSchema(target=("y", "z"), timestamp="t", item_id="s")
WITH_COVARIATES = TimeSeriesSchema(
    target="y", timestamp="t", item_id="s", past_covariates=("z",), known_covariates=("p",)
)


def _panel(schema=UNIVARIATE, *, frame=None, future=False):
    frame = _frame() if frame is None else frame
    columns = ["s", "t", *schema.target_names, *schema.covariate_names]
    panel = schema.to_panel(frame[columns], native_missing=True)
    if future and schema.known_covariates:
        horizon = 3
        rows = []
        for item in ("a", "b"):
            last = frame.loc[frame["s"] == item, "t"].max()
            dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
            rows.append(pd.DataFrame({"s": item, "t": dates, "p": [0.0, 1.0, 0.0]}))
        panel = schema.attach_future(panel, pd.concat(rows, ignore_index=True), horizon=horizon)
    return panel


def _config(horizon=3, levels=(0.1, 0.5, 0.9)) -> ForecastConfig:
    return ForecastConfig(prediction_length=horizon, quantile_levels=list(levels))


def _pipeline(checkpoint, horizon=3, cache=None, **forecast_params):
    return TimeSeriesPipeline(
        "Toto1",
        model_params={"checkpoint": checkpoint, "num_samples": 16, "samples_per_batch": 16},
        forecast_params={"prediction_length": horizon, **forecast_params},
        tuning_params={"device": "cpu", "seed": 7},
        cache=cache,
        validate=False,
    )


# ------------------------------------------------------------------ registry


@pytest.mark.unit
def test_spec_matches_the_released_checkpoint():
    spec = get_time_series_model_spec("Toto1")
    assert spec.name == "Toto1"
    assert spec.family == "toto"
    assert spec.adapter == "tabtune.models.TimeSeries.toto1:Toto1Adapter"
    assert spec.checkpoints == ("Datadog/Toto-Open-Base-1.0",)
    assert spec.default_checkpoint == "Datadog/Toto-Open-Base-1.0"
    assert spec.max_context == 4096
    assert spec.max_horizon is None
    assert spec.native_missing is True
    assert spec.supports_multivariate is True
    assert spec.supports_covariates is True
    assert spec.supports_categorical_covariates is False
    assert spec.dependency_extra is None
    assert spec.license.commercial_use_ok is True


@pytest.mark.unit
def test_both_toto_generations_coexist_as_distinct_specs():
    one = get_time_series_model_spec("Toto1")
    two = get_time_series_model_spec("Toto2")
    assert one.family == two.family == "toto"
    assert one.adapter != two.adapter
    assert one.dependency_extra != two.dependency_extra
    # The bare alias is unclaimed by both.
    for alias in ("Toto-1", "Toto1.0", "TotoV1", "Toto-Open-Base-1.0"):
        assert get_time_series_model_spec(alias).name == "Toto1"
    with pytest.raises(ModelNotFoundError):
        get_time_series_model_spec("Toto")


# ------------------------------------------------------------------ provenance


@pytest.mark.unit
def test_vendored_package_records_its_upstream_provenance():
    import tabtune.models.toto.toto1 as vendored

    assert vendored.UPSTREAM_PACKAGE == "toto-ts"
    assert vendored.UPSTREAM_VERSION == "0.2.0"
    assert vendored.UPSTREAM_SDIST_SHA256 == (
        "4cb832a08abb22b307cbde2f687abd2262d424b988ce93cd20a7b48637beb178"
    )
    copied = [p for p in VENDORED.rglob("*.py") if p.name != "__init__.py"]
    assert len(copied) == 14  # 13 upstream modules plus _gluonts.py
    # Upstream's model/fusion.py has a module docstring and no license header.
    unheadered = {"fusion.py"}
    for path in copied:
        if path.name in unheadered:
            assert "Apache" not in path.read_text(), f"{path} gained a header upstream"
            continue
        assert "Apache" in path.read_text()[:600], path


@pytest.mark.unit
def test_no_vendored_file_depends_on_gluonts_or_the_toto_package():
    """No vendored file imports gluonts or the top-level ``toto`` package; ``_gluonts`` stands in."""
    for path in VENDORED.rglob("*.py"):
        source = path.read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        assert "import gluonts" not in code, path
        assert "from gluonts" not in code, path
        assert "from toto" not in code, path
        assert "import toto\n" not in code, path
        if "_gluonts import" in code:
            assert "TabTune vendoring note" in source, path
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, tabtune.models.toto.toto1; print('gluonts' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "False"


@pytest.mark.unit
def test_gluonts_shim_supplies_exactly_the_four_symbols():
    from tabtune.models.toto.toto1 import _gluonts

    for name in ("AffineTransformed", "StudentT", "Scaler", "validated"):
        assert hasattr(_gluonts, name), name
    # validated() is a passthrough: it must not alter the arguments it wraps.
    calls = []

    class Probe:
        @_gluonts.validated()
        def __init__(self, dim: int = -1, keepdim: bool = False, minimum_scale: float = 1e-10):
            calls.append((dim, keepdim, minimum_scale))
            self.dim, self.keepdim, self.minimum_scale = dim, keepdim, minimum_scale

    probe = Probe(dim=-1, keepdim=True, minimum_scale=1e-8)
    assert calls == [(-1, True, 1e-8)]
    assert (probe.dim, probe.keepdim, probe.minimum_scale) == (-1, True, 1e-8)


@pytest.mark.unit
def test_scalers_built_from_the_checkpoint_config_are_unaffected_by_the_shim(tiny_checkpoint):
    """The scaler constructs from the config, carries its arguments and scales (gluonts parity lives in the parity harness)."""
    from tabtune.models.toto.toto1.model.scaler import CausalPatchStdMeanScaler, scaler_types

    assert TINY_CONFIG["scaler_cls"] in scaler_types
    scaler = CausalPatchStdMeanScaler(
        dim=-1, patch_size=PATCH, stabilize_with_global=True, scale_factor_exponent=10.0
    )
    assert (scaler.dim, scaler.patch_size) == (-1, PATCH)
    assert scaler.stabilize_with_global is True
    assert scaler.scale_factor_exponent == 10.0
    data = torch.randn(1, 2, 16)
    observed = torch.ones_like(data, dtype=torch.bool)
    scaled, loc, scale = scaler(data, observed, torch.ones_like(data))
    assert scaled.shape == data.shape
    assert torch.isfinite(scaled).all()
    assert torch.isfinite(loc).all() and torch.isfinite(scale).all()


# ------------------------------------------------------------------ loading


@pytest.mark.unit
def test_load_builds_a_forecaster_over_the_checkpoint(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    assert adapter.is_loaded
    assert adapter._patch_stride == PATCH
    assert type(adapter._model).__name__ == "TotoForecaster"


@pytest.mark.unit
def test_missing_backend_library_names_the_package(tiny_checkpoint, monkeypatch):
    """load() produces an install hint in a broken env; the backend import is deferred into it."""
    for name in [m for m in sys.modules if m.startswith("tabtune.models.toto.toto1")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "rotary_embedding_torch", None)

    adapter = Toto1Adapter(
        get_time_series_model_spec("Toto1"), checkpoint=tiny_checkpoint, device="cpu"
    )
    with pytest.raises(ImportError, match=r"Install the 'rotary_embedding_torch' package"):
        adapter.load()


# ------------------------------------------------------------------ forecasting


@pytest.mark.unit
def test_forecast_shapes_with_missing_history(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    out = adapter.forecast(_panel(), _config())
    assert out.point.shape == (2, 3)
    assert out.quantiles.shape == (2, 3, 3)
    assert np.isfinite(out.point).all()
    assert np.isfinite(out.quantiles).all()


@pytest.mark.unit
def test_point_forecast_is_the_median_by_default(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    assert adapter.point_forecast == "median"
    out = adapter.forecast(_panel(), _config())
    np.testing.assert_allclose(out.point, out.quantiles[:, :, 1])


@pytest.mark.unit
def test_mean_point_forecast_is_available_and_differs_from_the_median(tiny_checkpoint):
    """The mean is the average of the sample paths, not the 0.5 quantile."""
    median = _adapter(tiny_checkpoint).forecast(_panel(), _config())
    mean = _adapter(tiny_checkpoint, point_forecast="mean").forecast(_panel(), _config())
    assert not np.allclose(median.point, mean.point)
    # The quantiles are unaffected by which point summary was asked for.
    np.testing.assert_allclose(median.quantiles, mean.quantiles)


@pytest.mark.unit
def test_levels_off_any_grid_are_served_not_refused(tiny_checkpoint):
    """Quantiles are order statistics of the samples, so 0.05 and 0.975 are served."""
    levels = [0.025, 0.05, 0.33, 0.5, 0.67, 0.95, 0.975]
    out = _adapter(tiny_checkpoint).forecast(_panel(), _config(levels=levels))
    assert out.quantiles.shape == (2, 3, len(levels))
    assert (np.diff(out.quantiles, axis=-1) >= -1e-9).all()
    np.testing.assert_allclose(out.point, out.quantiles[:, :, levels.index(0.5)])


@pytest.mark.unit
def test_quantiles_are_none_when_no_levels_are_requested(tiny_checkpoint):
    out = _adapter(tiny_checkpoint).forecast(_panel(), _config(levels=()))
    assert out.quantiles is None
    assert out.point.shape == (2, 3)


@pytest.mark.unit
def test_multivariate_rows_are_forecast_jointly(tiny_checkpoint):
    """An item's variates share one forward pass, so joint and separate forecasts differ."""
    adapter = _adapter(tiny_checkpoint)
    joint = adapter.forecast(_panel(MULTIVARIATE), _config())
    assert joint.point.shape == (4, 3)
    solo = adapter.forecast(_panel(UNIVARIATE), _config())
    # Rows 0 and 2 are item a's and item b's "y" in the multivariate panel.
    assert not np.allclose(joint.point[[0, 2]], solo.point)


@pytest.mark.unit
def test_a_non_patch_multiple_history_needs_no_adapter_padding(tiny_checkpoint):
    """Upstream rounds the context to the stride itself (HISTORY 37, stride 8)."""
    assert HISTORY % PATCH != 0
    adapter = _adapter(tiny_checkpoint)
    captured = {}
    original = adapter._model.forecast

    def spy(inputs, *args, **kwargs):
        captured["context"] = inputs.series.shape[-1]
        return original(inputs, *args, **kwargs)

    adapter._model.forecast = spy
    out = adapter.forecast(_panel(), _config())
    assert captured["context"] == HISTORY  # handed over unrounded
    assert np.isfinite(out.point).all()


@pytest.mark.unit
def test_ragged_histories_share_a_batch(tiny_checkpoint):
    frame = _frame(missing=False)
    frame = frame.drop(frame.index[(frame["s"] == "a")][:9])  # item a is 9 steps shorter
    out = _adapter(tiny_checkpoint).forecast(_panel(frame=frame), _config())
    assert out.point.shape == (2, 3)
    assert np.isfinite(out.point).all()


@pytest.mark.unit
def test_batch_size_chunks_do_not_reorder_forecasts(tiny_checkpoint):
    """Chunking must not permute rows; numbers move with the tensor batch, so ordering is asserted via a level difference."""
    panel = _panel(frame=_frame(missing=False))
    one = _adapter(tiny_checkpoint, batch_size=1).forecast(panel, _config())
    both = _adapter(tiny_checkpoint, batch_size=8).forecast(panel, _config())
    # Item b sits 5 above item a in both runs.
    assert (one.point[1] > one.point[0]).all()
    assert (both.point[1] > both.point[0]).all()


# ------------------------------------------------------------------ sampling


@pytest.mark.unit
def test_the_same_seed_reproduces_the_forecast(tiny_checkpoint):
    a = _adapter(tiny_checkpoint, seed=11).forecast(_panel(), _config())
    b = _adapter(tiny_checkpoint, seed=11).forecast(_panel(), _config())
    np.testing.assert_array_equal(a.point, b.point)
    c = _adapter(tiny_checkpoint, seed=12).forecast(_panel(), _config())
    assert not np.allclose(a.point, c.point)


@pytest.mark.unit
def test_seeding_leaves_the_global_rng_untouched(tiny_checkpoint):
    """The seed is applied inside torch.random.fork_rng, as for Chronos v1."""
    adapter = _adapter(tiny_checkpoint, seed=99)  # construction itself draws weights
    torch.manual_seed(1234)
    before = torch.rand(4)
    torch.manual_seed(1234)
    adapter.forecast(_panel(), _config())
    after = torch.rand(4)
    torch.testing.assert_close(before, after)


@pytest.mark.unit
def test_unseeded_forecasts_vary(tiny_checkpoint):
    adapter = Toto1Adapter(
        get_time_series_model_spec("Toto1"),
        checkpoint=tiny_checkpoint,
        device="cpu",
        model_params={"num_samples": 16, "samples_per_batch": 16},
        seed=None,
    )
    adapter.load()
    first = adapter.forecast(_panel(), _config())
    second = adapter.forecast(_panel(), _config())
    assert not np.allclose(first.point, second.point)


@pytest.mark.unit
def test_num_samples_must_be_a_multiple_of_samples_per_batch(tiny_checkpoint):
    """The error names both numbers and a divisor that works."""
    with pytest.raises(ConfigError, match="must be a multiple of"):
        _adapter(tiny_checkpoint, num_samples=128, samples_per_batch=10)


@pytest.mark.unit
def test_mean_only_mode_refuses_quantile_levels(tiny_checkpoint):
    """num_samples=None is upstream's mean-only path, so quantile levels are refused."""
    adapter = _adapter(tiny_checkpoint, num_samples=None)
    # The point forecast falls back to the mean, which is all that mode produces.
    assert adapter.point_forecast == "mean"
    with pytest.raises(ConfigError, match="num_samples=None"):
        adapter.forecast(_panel(), _config())
    out = adapter.forecast(_panel(), _config(levels=()))
    assert out.quantiles is None
    assert np.isfinite(out.point).all()
    # Explicitly asking for a median in that mode is refused up front.
    with pytest.raises(ConfigError, match="cannot produce a median"):
        _adapter(tiny_checkpoint, num_samples=None, point_forecast="median")


# ------------------------------------------------------------------ time awareness


@pytest.mark.unit
@pytest.mark.parametrize(
    ("freq", "seconds"),
    [
        ("h", 3600),
        ("D", 86400),
        ("15min", 900),
        ("5D", 432000),
        ("W", 604800),
        ("2W", 1209600),
        ("MS", 30 * 86400),
        ("ME", 30 * 86400),
        ("QS", 90 * 86400),
        ("YS", int(365.25 * 86400)),
    ],
)
def test_freq_to_seconds_follows_upstreams_recipe(freq, seconds):
    assert _freq_to_seconds(freq) == seconds


@pytest.mark.unit
def test_timestamps_and_interval_reach_the_model(tiny_checkpoint):
    """The stamps are the real POSIX seconds of the history, ending at the forecast origin."""
    adapter = _adapter(tiny_checkpoint)
    captured = {}
    original = adapter._model.forecast

    def spy(inputs, *args, **kwargs):
        captured["stamps"] = inputs.timestamp_seconds.clone()
        captured["interval"] = inputs.time_interval_seconds.clone()
        return original(inputs, *args, **kwargs)

    adapter._model.forecast = spy
    panel = _panel(frame=_frame(missing=False))
    adapter.forecast(panel, _config())

    assert (captured["interval"] == 86400).all()
    stamps = captured["stamps"][0, 0]
    origin = int(panel.last_timestamps[0].value // 10**9)
    assert int(stamps[-1]) == origin
    assert int(stamps[-2]) == origin - 86400
    # strictly increasing by one day across the whole history
    assert (np.diff(stamps.numpy()) == 86400).all()


@pytest.mark.unit
def test_timestamps_are_clamped_into_int32(tiny_checkpoint):
    """timestamp_seconds is torch.int, so a post-2038 origin is clamped as upstream does."""
    frame = _frame(missing=False)
    frame["t"] = frame["t"] + pd.DateOffset(years=100)  # ~2124, past int32 seconds
    adapter = _adapter(tiny_checkpoint)
    captured = {}
    original = adapter._model.forecast

    def spy(inputs, *args, **kwargs):
        captured["stamps"] = inputs.timestamp_seconds.clone()
        return original(inputs, *args, **kwargs)

    adapter._model.forecast = spy
    out = adapter.forecast(_panel(frame=frame), _config())
    assert int(captured["stamps"].max()) == 2**31 - 1
    assert np.isfinite(out.point).all()


# ------------------------------------------------------------------ covariates


@pytest.mark.unit
def test_covariates_arrive_as_the_last_channels_in_upstreams_order(tiny_checkpoint):
    """Channels are ``[targets..., past-only..., known-future...]``, as upstream's exogenous path requires."""
    adapter = _adapter(tiny_checkpoint)
    captured = {}
    original = adapter._model.forecast

    def spy(inputs, *args, **kwargs):
        captured["series"] = inputs.series.clone()
        captured["nev"] = inputs.num_exogenous_variables
        captured["future"] = kwargs["future_exogenous_variables"]
        return original(inputs, *args, **kwargs)

    adapter._model.forecast = spy
    frame = _frame(missing=False)
    panel = _panel(WITH_COVARIATES, frame=frame, future=True)
    adapter.forecast(panel, _config())

    # one target + one past-only ("z") + one known-future ("p")
    assert captured["series"].shape[1] == 3
    assert captured["nev"] == 1
    assert captured["future"].shape == (2, 1, 3)

    item_a = frame[frame["s"] == "a"]
    np.testing.assert_allclose(
        captured["series"][0, 0].numpy(), item_a["y"].to_numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        captured["series"][0, 1].numpy(), item_a["z"].to_numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        captured["series"][0, 2].numpy(), item_a["p"].to_numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(captured["future"][0, 0].numpy(), [0.0, 1.0, 0.0])


@pytest.mark.unit
def test_covariate_channels_are_dropped_from_the_output(tiny_checkpoint):
    """Covariates add input channels but no result rows."""
    out = _adapter(tiny_checkpoint).forecast(
        _panel(WITH_COVARIATES, frame=_frame(missing=False), future=True), _config()
    )
    assert out.point.shape == (2, 3)


@pytest.mark.unit
def test_categorical_covariates_are_refused(tiny_checkpoint):
    """Exogenous channels are float tensors, so a string covariate is refused."""
    from tabtune.registry import check_schema_support

    frame = _frame(missing=False)
    frame["cat"] = np.where(frame["p"] > 0, "yes", "no")
    schema = TimeSeriesSchema(
        target="y", timestamp="t", item_id="s", past_covariates=("cat",)
    )
    panel = schema.to_panel(frame[["s", "t", "y", "cat"]], native_missing=True)
    with pytest.raises(ConfigError, match="numeric"):
        check_schema_support(get_time_series_model_spec("Toto1"), schema, panel=panel)


# ------------------------------------------------------------------ params


@pytest.mark.unit
def test_invalid_model_params():
    spec = get_time_series_model_spec("Toto1")
    for params, match in (
        ({"batch_size": 0}, "batch_size"),
        ({"samples_per_batch": 0}, "samples_per_batch"),
        ({"num_samples": 0}, "num_samples"),
        ({"point_forecast": "mode"}, "point_forecast"),
    ):
        with pytest.raises(ConfigError, match=match):
            Toto1Adapter(spec, checkpoint="x", device="cpu", model_params=params)


@pytest.mark.unit
def test_unknown_model_params_warn(tiny_checkpoint):
    """The Chronos sampling knobs and dtype are not Toto 1.0's."""
    with pytest.warns(UserWarning, match="temperature"):
        Toto1Adapter(
            get_time_series_model_spec("Toto1"),
            checkpoint=tiny_checkpoint,
            device="cpu",
            model_params={"temperature": 1.0, "dtype": "bfloat16"},
        )


# ------------------------------------------------------------------ pipeline


@pytest.mark.unit
def test_pipeline_end_to_end(tiny_checkpoint, tmp_path):
    pipe = _pipeline(tiny_checkpoint, quantile_levels=[0.05, 0.5, 0.95])
    frame = _frame()[["s", "t", "y"]]
    result = pipe.fit(frame, UNIVARIATE).predict()

    assert result.point.shape == (2, 3)
    assert result.quantile_levels == (0.05, 0.5, 0.95)
    assert result.metadata["model"] == "Toto1"
    assert result.metadata["point_forecast"] == "median"
    assert result.metadata["training_occurred"] is False

    long = result.to_pandas()
    assert list(long.columns) == ["s", "t", "target", "point", "0.05", "0.5", "0.95"]
    assert len(long) == 6

    path = tmp_path / "toto1.joblib"
    pipe.save(path)
    reloaded = TimeSeriesPipeline.load(path)
    again = reloaded.predict()
    np.testing.assert_allclose(result.point, again.point)


@pytest.mark.unit
def test_long_horizon_does_not_warn_because_no_limit_is_declared(tiny_checkpoint, recwarn):
    pipe = _pipeline(tiny_checkpoint, horizon=40)
    out = pipe.fit(_frame()[["s", "t", "y"]], UNIVARIATE).predict()
    assert out.point.shape == (2, 40)
    assert not [w for w in recwarn if "horizon" in str(w.message).lower()]


# ------------------------------------------------------------------ real weights


@pytest.mark.slow
@pytest.mark.weights
def test_released_checkpoint_beats_a_naive_forecast():
    """The released Toto-Open-Base-1.0, end to end on a trend plus weekly cycle."""
    horizon = 7
    periods = 256
    pipe = TimeSeriesPipeline(
        "Toto1",
        model_params={"num_samples": 128, "samples_per_batch": 64},
        forecast_params={"prediction_length": horizon, "quantile_levels": [0.05, 0.5, 0.95]},
        tuning_params={"device": "cpu", "seed": 7},
    )
    frame = _frame(missing=False, periods=periods)[["s", "t", "y"]]
    result = pipe.fit(frame, UNIVARIATE).predict()

    assert result.point.shape == (2, horizon)
    assert np.isfinite(result.point).all()
    # 0.05 and 0.95 are off every other model's trained grid and served here.
    assert (np.diff(result.quantiles, axis=2) >= 0).all()

    steps = np.arange(periods, periods + horizon, dtype=float)
    truth = np.stack(
        [steps * 0.3 + 3 * np.sin(steps * 2 * np.pi / 7) + 5 * offset for offset in (0, 1)]
    )
    last = np.repeat(frame.groupby("s")["y"].last().to_numpy()[:, None], horizon, axis=1)
    model_mae = float(np.abs(result.point - truth).mean())
    naive_mae = float(np.abs(last - truth).mean())
    assert model_mae < naive_mae, (model_mae, naive_mae)
    assert (result.point[1] > result.point[0]).all()


def test_pipeline_tasks_run_on_toto1(tiny_checkpoint):
    from tabtune.TimeSeries import TimeSeriesSchema, make_panel

    schema = TimeSeriesSchema(target="target", item_id="item_id")
    params = {"checkpoint": tiny_checkpoint, "device": "cpu"}
    frame = make_panel(2, 96, freq="h", seed=0)
    scores = TimeSeriesPipeline(
        "Toto1", task_type="anomaly_detection", model_params=params, task_params={"min_context": 32}
    ).fit(frame, schema).predict()
    assert scores.frame["score"].notna().any()
    gappy = frame.copy()
    gappy.loc[[40, 41, 42], "target"] = np.nan
    filled = TimeSeriesPipeline("Toto1", task_type="imputation", model_params=params).fit(gappy, schema).predict()
    assert np.isfinite(filled.to_pandas()["target"]).all()
    pipe = TimeSeriesPipeline("Toto1", model_params=params, forecast_params={"prediction_length": 4})
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
                "Toto1",
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
            "Toto1",
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
            "Toto1",
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
    assert "negative log-likelihood" in report["objective"]
    tuned = pipe.predict().point
    assert not np.allclose(tuned, before, atol=1e-7)

    # Training must leave the module exactly as forecasting found it.
    network = pipe.adapter_._network()
    assert not any(p.requires_grad for p in network.parameters())
    assert not network.training

    path = tmp_path / f"toto1-{strategy}.joblib"
    pipe.save(str(path))
    np.testing.assert_allclose(
        TimeSeriesPipeline.load(str(path)).predict().point, tuned, atol=1e-6
    )
    # Training one pipeline leaves a fresh zero-shot one untouched.
    np.testing.assert_allclose(zero_shot(), before, atol=1e-6)

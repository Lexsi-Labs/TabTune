"""Tests for the time series model registry.

Dependency-free like ``test_registry.py``: no torch, no weights, no network.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tabtune.registry import (
    MODEL_REGISTRY,
    TS_MODEL_REGISTRY,
    ConfigError,
    EnvelopeError,
    LicenseSpec,
    ModelNotFoundError,
    TimeSeriesModelSpec,
    UnsupportedStrategyError,
    UnsupportedTaskError,
    check_forecast_envelope,
    check_license,
    get_time_series_model_spec,
    list_time_series_models,
    register_time_series_model,
    resolve_time_series_model_name,
    validate_time_series_request,
)

pytestmark = [pytest.mark.unit, pytest.mark.time_series]


def _spec(name="Dummy", **overrides):
    fields = {"name": name, "family": "test", "adapter": "some.module:Adapter"}
    fields.update(overrides)
    return TimeSeriesModelSpec(**fields)


def test_chronos_is_registered():
    spec = TS_MODEL_REGISTRY["Chronos"]
    assert spec.tasks == frozenset(
        {"forecasting", "anomaly_detection", "imputation", "embedding"}
    )
    assert spec.strategies == frozenset({"inference", "finetune", "peft"})
    assert spec.default_checkpoint in spec.checkpoints
    assert spec.license.name
    assert spec.paper


def test_tabular_registry_is_untouched():
    assert len(MODEL_REGISTRY) == 21
    assert "Chronos" not in MODEL_REGISTRY


@pytest.mark.parametrize("alias", ["Chronos", "chronos", "Chronos-T5", "chronos_v1", "ChronosT5"])
def test_aliases_resolve(alias):
    assert resolve_time_series_model_name(alias) == "Chronos"


def test_unknown_name_suggests_closest():
    with pytest.raises(ModelNotFoundError, match="Did you mean 'Chronos'"):
        get_time_series_model_spec("Chronoss")


def test_tabular_names_are_not_time_series_models():
    with pytest.raises(ModelNotFoundError):
        resolve_time_series_model_name("TabICLv2")


def test_list_filters_by_strategy():
    names = [s.name for s in list_time_series_models(task="forecasting")]
    assert {"Chronos", "Chronos2", "ChronosBolt", "TimesFM3", "TiRex", "TiRex2", "Toto1", "Toto2"} <= set(names)
    assert names == sorted(names)
    # The baselines and the tabular forecasters have no encoder.
    neural = {
        name
        for name, spec in TS_MODEL_REGISTRY.items()
        if spec.family not in ("baseline", "tabular-icl")
    }
    embedding = [s.name for s in list_time_series_models(task="embedding")]
    assert set(embedding) == neural
    assert embedding == sorted(embedding)

    trainable = [s.name for s in list_time_series_models(strategy="finetune")]
    assert trainable == [s.name for s in list_time_series_models(strategy="peft")]
    assert trainable == sorted(trainable)
    assert set(trainable) == neural
    assert "SeasonalNaive" not in trainable


@pytest.mark.parametrize(
    "alias,expected",
    [("ChronosBolt", "ChronosBolt"), ("chronos-bolt", "ChronosBolt"), ("bolt", "ChronosBolt")],
)
def test_bolt_aliases_resolve(alias, expected):
    assert resolve_time_series_model_name(alias) == expected


def test_chronos_family_members_are_distinct_models():
    """v1 and Bolt share a family but must not share a name or checkpoints."""
    v1, bolt = get_time_series_model_spec("Chronos"), get_time_series_model_spec("ChronosBolt")
    assert v1.family == bolt.family == "chronos"
    assert v1.name != bolt.name
    assert not set(v1.checkpoints) & set(bolt.checkpoints)
    assert bolt.max_context > v1.max_context



def test_register_rejects_tabular_alias_collision(isolated_ts_registry):
    with pytest.raises(ValueError, match="tabular model"):
        register_time_series_model(_spec(aliases=("TabPFN-v2",)))


def test_register_rejects_duplicate_name(isolated_ts_registry):
    register_time_series_model(_spec())
    with pytest.raises(ValueError, match="already registered"):
        register_time_series_model(_spec())
    register_time_series_model(_spec(summary="replaced"), overwrite=True)
    assert isolated_ts_registry.TS_MODEL_REGISTRY["Dummy"].summary == "replaced"


def test_register_rejects_unknown_task(isolated_ts_registry):
    with pytest.raises(ValueError, match="unknown task"):
        register_time_series_model(_spec(tasks=frozenset({"forecasting", "classification"})))


def test_spec_catalog_and_lookup_modules_agree():
    from tabtune.registry.TimeSeries import TimeSeriesModelSpec as FromRegistry
    from tabtune.registry.TimeSeriesCatalog import TS_MODEL_SPECS
    from tabtune.registry.TimeSeriesSpec import TimeSeriesModelSpec as FromSpec

    assert FromRegistry is FromSpec is TimeSeriesModelSpec
    assert {s.name for s in TS_MODEL_SPECS} <= set(TS_MODEL_REGISTRY)


def test_spec_to_dict_is_json_serialisable():
    import json

    data = get_time_series_model_spec("Chronos").to_dict()
    assert json.loads(json.dumps(data))["adapter"].endswith(":ChronosAdapter")


def test_register_rejects_non_spec():
    with pytest.raises(TypeError):
        register_time_series_model({"name": "x"})



def test_validate_accepts_zero_shot_forecasting():
    assert validate_time_series_request("chronos", "forecasting", "inference").name == "Chronos"


def test_validate_rejects_other_tasks():
    with pytest.raises(UnsupportedTaskError):
        validate_time_series_request("Chronos", "classification", "inference")


@pytest.mark.parametrize(
    "model",
    [
        "TabularTS-GBM",
        "SeasonalNaive",
    ],
)
def test_validate_rejects_finetune(model):
    with pytest.raises(UnsupportedStrategyError, match="finetune"):
        validate_time_series_request(model, "forecasting", "finetune")


def test_check_license_accepts_time_series_specs():
    spec = get_time_series_model_spec("Chronos")
    assert check_license(spec, "commercial") is spec.license


def test_check_license_blocks_restricted_weights():
    spec = _spec(license=LicenseSpec(name="NC", commercial_use_ok=False))
    with pytest.raises(Exception, match="does not permit commercial use"):
        check_license(spec, "commercial")



def test_envelope_within_limits_is_silent():
    spec = _spec(max_horizon=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_forecast_envelope(spec, prediction_length=10) == []


def test_envelope_warns_beyond_max_horizon():
    spec = _spec(max_horizon=10)
    with pytest.warns(UserWarning, match="horizons up to 10"):
        violations = check_forecast_envelope(spec, prediction_length=11)
    assert violations[0].constraint == "max_horizon"


def test_envelope_error_mode_raises():
    with pytest.raises(EnvelopeError):
        check_forecast_envelope(_spec(max_horizon=10), prediction_length=11, mode="error")


def test_envelope_ignore_mode_skips():
    assert check_forecast_envelope(_spec(max_horizon=10), prediction_length=99, mode="ignore") == []


def test_envelope_rejects_unknown_mode():
    with pytest.raises(ValueError):
        check_forecast_envelope(_spec(), prediction_length=1, mode="loud")


def test_capability_flags_are_reported_and_default_to_off():
    chronos2 = get_time_series_model_spec("Chronos2").to_dict()
    assert chronos2["supports_multivariate"] is True
    assert chronos2["supports_covariates"] is True
    assert chronos2["supports_categorical_covariates"] is True

    bolt = get_time_series_model_spec("ChronosBolt").to_dict()
    assert bolt["supports_multivariate"] is False
    assert bolt["supports_covariates"] is False
    assert bolt["supports_categorical_covariates"] is False

    timesfm = get_time_series_model_spec("TimesFM3").to_dict()
    assert timesfm["supports_multivariate"] is True
    assert timesfm["supports_covariates"] is True
    assert timesfm["supports_categorical_covariates"] is False


def test_two_families_coexist_with_distinct_adapters_and_licenses():
    """The registry holds more than one family; nothing about it is Chronos-specific."""
    specs = {spec.name: spec for spec in list_time_series_models()}
    assert {"chronos", "timesfm", "toto", "xlstm", "time-moe", "tabular-icl", "baseline"} == {
        spec.family for spec in specs.values()
    }

    foundation = [
        s for s in specs.values() if s.family in ("chronos", "timesfm", "toto", "xlstm", "time-moe")
    ]
    adapters = [spec.to_dict()["adapter"] for spec in foundation]
    assert len(set(adapters)) == len(adapters)
    assert specs["TimesFM3"].license.commercial_use_ok is False
    with pytest.raises(Exception, match="does not permit commercial use"):
        check_license(specs["TimesFM3"], "commercial")


def test_every_dependency_extra_exists_in_pyproject():
    """A spec's dependency_extra must exist in pyproject."""
    import re

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    block = pyproject.split("[project.optional-dependencies]", 1)[1]
    block = block.split("\n[", 1)[0]
    declared = set(re.findall(r"^(\w[\w.-]*)\s*=\s*\[", block, flags=re.MULTILINE))
    assert declared, "no extras parsed out of pyproject.toml"

    named = {
        spec.dependency_extra
        for spec in list_time_series_models()
        if spec.dependency_extra is not None
    }
    assert named, "no spec names a dependency_extra, so this test proves nothing"
    assert named <= declared, f"extras named by specs but missing from pyproject: {named - declared}"


def test_license_follows_the_checkpoint():
    from tabtune.registry import LicenseError
    from tabtune.TimeSeries import TimeSeriesPipeline

    spec = get_time_series_model_spec("TabPFN-TS")
    assert spec.license_for("tabpfn-ts-2").commercial_use_ok is True
    assert spec.license_for("tabpfn-ts-3").commercial_use_ok is False
    assert spec.license_for(None) is spec.license
    assert spec.to_dict()["checkpoint_licenses"]["tabpfn-ts-2"]["commercial_use_ok"] is True
    forecast = {"prediction_length": 4}
    TimeSeriesPipeline(
        "TabPFN-TS", model_params={"checkpoint": "tabpfn-ts-2"}, forecast_params=forecast,
        license_mode="commercial",
    )
    with pytest.raises(LicenseError):
        TimeSeriesPipeline("TabPFN-TS", forecast_params=forecast, license_mode="commercial")


def test_commercial_filter_can_include_unverified_licenses():
    """include_unverified_licenses makes models with an unverified license reachable."""
    strict = {s.name for s in list_time_series_models(commercial_ok=True)}
    wide = {s.name for s in list_time_series_models(
        commercial_ok=True, include_unverified_licenses=True
    )}
    assert strict < wide
    unverified = {
        name
        for name, spec in TS_MODEL_REGISTRY.items()
        if spec.license.commercial_use_ok is None
    }
    assert wide - strict == unverified
    assert unverified, "expected some specs to carry an unverified license"
    # The flag must not loosen the restricted-only direction.
    assert [s.name for s in list_time_series_models(commercial_ok=False)] == [
        s.name
        for s in list_time_series_models(commercial_ok=False, include_unverified_licenses=True)
    ]


def test_every_fixed_grid_model_reports_the_levels_it_serves():
    """Every fixed-grid model overrides quantile_range()."""
    from tabtune.models.TimeSeries.base import TSFMAdapter

    expected = {"ChronosBolt", "Chronos2", "TimesFM3", "Toto2", "TiRex", "TiRex2"}
    for name in expected:
        adapter_cls = _adapter_class(get_time_series_model_spec(name))
        assert "quantile_range" in vars(adapter_cls), (
            f"{name} predicts a fixed quantile grid but does not override quantile_range()"
        )
    for name in ("Chronos", "Toto1", "SeasonalNaive"):
        adapter_cls = _adapter_class(get_time_series_model_spec(name))
        assert "quantile_range" not in vars(adapter_cls)
        assert TSFMAdapter.quantile_range(object.__new__(adapter_cls)) is None


def _adapter_class(spec):
    from importlib import import_module

    module, _, cls = spec.adapter.partition(":")
    return getattr(import_module(module), cls)


@pytest.mark.parametrize(
    "model",
    [
        "Chronos", "ChronosBolt", "Chronos2", "TimesFM3", "Toto1", "Toto2",
        "TiRex", "TiRex2", "TimeMoE", "SeasonalNaive", "WindowAverage",
    ],
)
def test_every_adapter_validates_batch_size_the_same_way(model):
    """A junk batch_size raises ConfigError for every adapter."""
    pytest.importorskip("torch")
    spec = get_time_series_model_spec(model)
    adapter_cls = _adapter_class(spec)
    checkpoint = spec.default_checkpoint or "x"
    for bad in ("x", 0, -1):
        with pytest.raises(ConfigError, match="batch_size"):
            adapter_cls(
                spec, checkpoint=checkpoint, device="cpu", model_params={"batch_size": bad}
            )


@pytest.mark.parametrize("model", ["Chronos", "ChronosBolt", "Chronos2", "TiRex", "TimeMoE"])
def test_hub_options_are_accepted_by_every_downloading_model(model):
    """revision and local_files_only are accepted by every downloading model."""
    pytest.importorskip("torch")
    spec = get_time_series_model_spec(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an unknown-key warning would fail here
        _adapter_class(spec)(
            spec,
            checkpoint=spec.default_checkpoint,
            device="cpu",
            model_params={"revision": "main", "local_files_only": True},
        )

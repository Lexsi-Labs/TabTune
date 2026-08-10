"""Tests for the typed configuration layer.

Two properties matter most here, and both are regression guards:

1. **Backward compatibility.** Every configuration object must still accept a
   plain dict and forward unrecognised keys, because model-specific kwargs are
   a legitimate use and users have existing code passing them.
2. **Typos become visible.** The whole point of the schema is that
   ``lerning_rate`` warns instead of vanishing.
"""

from __future__ import annotations

import json
import warnings

import pytest

from tabtune.config import (
    ContextSamplingConfig,
    PeftConfig,
    PipelineConfig,
    ProcessorConfig,
    TuningConfig,
    config_from_mapping,
    dump_config,
    load_config,
    save_config,
)
from tabtune.registry.errors import ConfigError, UnsupportedStrategyError

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------- defaults


def test_defaults_match_historical_behaviour():
    cfg = TuningConfig()
    assert cfg.epochs == 5
    assert cfg.learning_rate == pytest.approx(1e-5)
    assert cfg.batch_size == 32
    assert cfg.device == "auto"
    assert cfg.finetune_mode is None


def test_peft_defaults():
    peft = PeftConfig()
    assert (peft.r, peft.lora_alpha, peft.lora_dropout) == (8, 16, 0.05)


# -------------------------------------------------------------- compatibility


def test_from_dict_accepts_plain_dicts():
    cfg = TuningConfig.from_dict({"epochs": 10, "learning_rate": 2e-5})
    assert cfg.epochs == 10 and cfg.learning_rate == pytest.approx(2e-5)


def test_unknown_keys_are_forwarded_not_dropped():
    """Model-specific kwargs must survive; they are how models get configured."""
    with pytest.warns(UserWarning):
        cfg = TuningConfig.from_dict(
            {"epochs": 3, "n_estimators": 16}, context="tuning_params"
        )
    assert cfg.extras == {"n_estimators": 16}
    assert cfg.to_dict()["n_estimators"] == 16


def test_typo_produces_a_suggestion():
    with pytest.warns(UserWarning, match="did you mean 'learning_rate'"):
        TuningConfig.from_dict({"lerning_rate": 1e-4}, context="tuning_params")


def test_strict_mode_rejects_unknown_keys():
    with pytest.raises(ConfigError, match="Unrecognised"):
        TuningConfig.from_dict({"nope": 1}, strict=True, context="tuning_params")


def test_from_dict_passes_through_existing_instances():
    cfg = TuningConfig(epochs=7)
    assert TuningConfig.from_dict(cfg) is cfg


def test_from_dict_rejects_non_mappings():
    with pytest.raises(ConfigError, match="mapping"):
        TuningConfig.from_dict([1, 2, 3])


def test_none_yields_defaults():
    assert TuningConfig.from_dict(None).epochs == 5


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"learning_rate": 0.0},
        {"learning_rate": -1.0},
        {"batch_size": 0},
        {"validation_split": 1.0},
    ],
)
def test_invalid_values_are_rejected(kwargs):
    # pydantic raises ValidationError, which is not importable in a
    # version-stable way; the contract under test is only "this is rejected".
    with pytest.raises(Exception):  # noqa: B017
        TuningConfig(**kwargs)


def test_peft_rank_bounds():
    with pytest.raises(Exception):  # noqa: B017 - see above
        PeftConfig(r=0)
    with pytest.raises(Exception):  # noqa: B017
        PeftConfig(lora_dropout=1.0)


@pytest.mark.parametrize(
    "given,expected",
    [
        ("meta_learning", "meta-learning"),
        ("META-LEARNING", "meta-learning"),
        ("metalearning", "meta-learning"),
        ("episodic", "meta-learning"),
        ("tbt", "turn_by_turn"),
        ("turn-by-turn", "turn_by_turn"),
        ("SFT", "sft"),
    ],
)
def test_finetune_mode_aliases_normalise(given, expected):
    assert TuningConfig(finetune_mode=given).finetune_mode == expected


def test_processor_treats_the_string_none_as_unset():
    cfg = ProcessorConfig(imputation_strategy="none", scaling_strategy="")
    assert cfg.imputation_strategy is None
    assert cfg.scaling_strategy is None


def test_peft_config_accepts_a_nested_dict():
    cfg = TuningConfig(peft_config={"r": 32, "lora_alpha": 64})
    assert isinstance(cfg.peft_config, PeftConfig)
    assert cfg.peft_config.r == 32


def test_peft_target_modules_accepts_a_bare_string():
    assert PeftConfig(target_modules="qkv").target_modules == ["qkv"]


# --------------------------------------------------------------- pipeline cfg


def test_pipeline_config_canonicalises_the_model_name():
    assert PipelineConfig(model_name="tabpfn-v2.6").model_name == "TabPFNv26"


def test_pipeline_config_keeps_processor_task_type_in_sync():
    cfg = PipelineConfig(model_name="TabPFNv26", task_type="regression")
    assert cfg.processor.task_type == "regression"


def test_pipeline_config_construction_does_not_validate_the_registry():
    """Construction stays cheap and total; validation is an explicit step.

    Raising inside a pydantic validator would wrap TabTune's actionable error
    in a ValidationError and bury the "here are compatible models" guidance.
    """
    cfg = PipelineConfig(model_name="Limix", tuning_strategy="finetune")
    with pytest.raises(UnsupportedStrategyError):
        cfg.validate_against_registry()


def test_pipeline_config_tolerates_unregistered_models():
    cfg = PipelineConfig(model_name="SomeFutureModel")
    assert cfg.model_name == "SomeFutureModel"


@pytest.mark.parametrize(
    "task,expected", [("classification", "meta-learning"), ("regression", "turn_by_turn")]
)
def test_resolved_finetune_mode_defaults_per_task(task, expected):
    cfg = PipelineConfig(model_name="TabPFNv26", task_type=task)
    assert cfg.resolved_finetune_mode() == expected


def test_explicit_finetune_mode_wins():
    cfg = PipelineConfig(model_name="TabPFNv26", task_type="regression", finetune_mode="sft")
    assert cfg.resolved_finetune_mode() == "sft"


# ------------------------------------------------------------------ round-trip


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_config_round_trips_through_disk(tmp_path, suffix):
    original = PipelineConfig(
        model_name="TabICLv2",
        task_type="classification",
        tuning_strategy="finetune",
        tuning=TuningConfig(epochs=12, learning_rate=3e-5, seed=7),
    )
    path = save_config(original, tmp_path / f"run{suffix}")
    assert path.exists()
    restored = load_config(path)
    assert restored.model_dump() == original.model_dump()


def test_loader_accepts_legacy_param_names(tmp_path):
    """A config written with the constructor's *_params names still loads."""
    cfg = config_from_mapping(
        {
            "model_name": "TabICLv2",
            "tuning_params": {"epochs": 4},
            "processor_params": {"imputation_strategy": "knn"},
        }
    )
    assert cfg.tuning.epochs == 4
    assert cfg.processor.imputation_strategy == "knn"


def test_loader_validates_eagerly():
    with pytest.raises(UnsupportedStrategyError):
        config_from_mapping({"model_name": "Limix", "tuning_strategy": "peft"})


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "run.ini"
    path.write_text("[x]\n")
    with pytest.raises(ConfigError, match="Unsupported config format"):
        load_config(path)


def test_dump_config_is_json_serialisable():
    cfg = PipelineConfig(model_name="TabICLv2")
    assert json.loads(json.dumps(dump_config(cfg)))["model_name"] == "TabICLv2"


def test_merged_returns_a_modified_copy():
    cfg = TuningConfig(epochs=5)
    merged = cfg.merged(epochs=50)
    assert (cfg.epochs, merged.epochs) == (5, 50)


# ------------------------------------------------------- context sampling cfg


def test_context_sampling_alias_and_enabled_flag():
    cfg = ContextSamplingConfig(context_sampling_strategy="stratified", context_size=1000)
    assert cfg.strategy == "stratified" and cfg.enabled

    legacy = cfg.to_legacy_dict()
    assert legacy["context_sampling_strategy"] == "stratified"
    assert legacy["context_size"] == 1000


def test_context_sampling_disabled_by_default():
    assert not ContextSamplingConfig().enabled


def test_to_dict_drops_none_by_default():
    cfg = TuningConfig()
    assert "finetune_mode" not in cfg.to_dict()
    assert "finetune_mode" in cfg.to_dict(drop_none=False)


def test_warnings_are_deduplicated():
    from tabtune._internal.deprecation import reset_warning_cache

    reset_warning_cache()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            TuningConfig.from_dict({"typo_key": 1}, context="tuning_params")
    assert len(caught) == 1, "repeated identical warnings should be suppressed"

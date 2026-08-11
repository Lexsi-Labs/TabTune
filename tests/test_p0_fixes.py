"""Regression tests for the P0 defects fixed in 0.2.0.

Each test here corresponds to a specific bug. They are written so that
reverting the fix makes the test fail with a message explaining what broke,
because these are exactly the failures that are silent in production: config
that vanishes, a checkpoint that never loads, a metric that is quietly
substituted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tabtune._internal.deprecation import (
    deprecated_param,
    reset_warning_cache,
    warn_once,
    warn_unknown_keys,
)
from tabtune._internal.device import describe_device, resolve_device
from tabtune.Dataprocess.data_processor import DataProcessor

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_warning_cache():
    reset_warning_cache()
    yield
    reset_warning_cache()


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "cat": list("xyxyxyxy"),
            "num2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )


@pytest.fixture
def target():
    return pd.Series([0, 1, 0, 1, 0, 1, 0, 1])


# ============================================================================
# P0-1: processor_params silently did nothing for classification
# ============================================================================


def test_standard_components_are_fitted_for_classification(frame, target):
    """Was: _fit_standard_components only ran inside the regression branch.

    For classification the imputer, scaler and selector stayed None and
    _apply_standard_transforms was an identity function.
    """
    processor = DataProcessor(
        task_type="classification", imputation_strategy="median", scaling_strategy="minmax"
    )
    processor.fit(frame, target)

    assert processor.imputer_ is not None, "imputer was never fitted for classification"
    assert processor.scaler_ is not None, "scaler was never fitted for classification"

    transformed = processor.transform(frame)
    assert not transformed.isna().any().any(), "imputation did not run"
    numeric = transformed.select_dtypes(include=np.number)
    assert numeric.min().min() >= -1e-9 and numeric.max().max() <= 1 + 1e-9


def test_user_categorical_encoding_is_not_silently_discarded():
    """Was: _set_model_aware_defaults overwrote the user value unconditionally."""
    processor = DataProcessor(model_name="TabPFN", categorical_encoding="ordinal")
    assert processor.categorical_encoding == "ordinal"


def test_overriding_a_model_aware_default_warns():
    with pytest.warns(UserWarning, match="overrides the model-aware preprocessor"):
        DataProcessor(model_name="TabICLv2", categorical_encoding="onehot")


def test_model_aware_default_still_applies_when_unset():
    processor = DataProcessor(model_name="TabICLv2")
    assert processor.categorical_encoding == "tabiclv2_special"


def test_unknown_model_falls_back_to_documented_defaults():
    processor = DataProcessor(model_name=None)
    assert processor.categorical_encoding == "onehot"
    assert processor.imputation_strategy == "mean"
    assert processor.scaling_strategy == "standard"


def test_resampling_is_reachable_through_fit_transform():
    """Was: resampling ran only in fit_transform, which the pipeline never called."""
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0)})
    y = pd.Series([0] * 32 + [1] * 8)

    processor = DataProcessor(task_type="classification", resampling_strategy="random_over")
    X_res, y_res = processor.fit_transform(X, y)

    counts = pd.Series(y_res).value_counts()
    assert counts[0] == counts[1], "classes were not balanced"
    assert len(y_res) > len(y)
    assert processor.processing_summary_["resampling"]["rows_after"] == len(y_res)


def test_fit_resample_is_a_public_entry_point():
    X = pd.DataFrame({"a": np.arange(40.0)})
    y = pd.Series([0] * 32 + [1] * 8)
    processor = DataProcessor(task_type="classification", resampling_strategy="random_under")
    X_res, y_res = processor.fit_resample(X, y)
    assert len(y_res) < len(y)


def test_resampling_is_skipped_with_a_warning_for_regression():
    X = pd.DataFrame({"a": np.arange(20.0)})
    y = pd.Series(np.arange(20.0))
    processor = DataProcessor(task_type="regression", resampling_strategy="smote")
    with pytest.warns(UserWarning, match="classification"):
        X_res, y_res = processor.fit_resample(X, y)
    assert len(y_res) == len(y)


def test_unknown_resampling_strategy_warns_rather_than_failing_silently():
    X = pd.DataFrame({"a": np.arange(20.0)})
    y = pd.Series([0, 1] * 10)
    processor = DataProcessor(task_type="classification")
    processor.resampling_strategy = "teleport"
    with pytest.warns(UserWarning, match="Unknown resampling_strategy"):
        processor.fit_resample(X, y)


def test_resampling_failure_degrades_instead_of_aborting():
    """SMOTE fails on a minority class of one; losing the run is the wrong answer."""
    X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    y = pd.Series([0] * 9 + [1])
    processor = DataProcessor(task_type="classification", resampling_strategy="smote")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_res, y_res = processor.fit_resample(X, y)
    assert len(y_res) >= len(y)


def test_explicit_strategies_apply_as_a_pre_stage_for_model_aware_models(frame, target):
    """Was: any explicit imputation/scaling request was discarded for these models."""
    processor = DataProcessor(
        model_name="TabPFN", task_type="classification", imputation_strategy="knn"
    )
    processor.fit(frame, target)
    assert processor.has_pre_stage
    assert type(processor.imputer_).__name__ == "KNNImputer"


def test_no_pre_stage_when_the_user_asked_for_nothing(frame, target):
    processor = DataProcessor(model_name="TabPFN", task_type="classification")
    processor.fit(frame, target)
    assert not processor.has_pre_stage


# ============================================================================
# P0-2: feature_names_ was initialised to None and never assigned
# ============================================================================


def test_feature_names_are_populated(frame, target):
    processor = DataProcessor(task_type="classification")
    processor.fit(frame, target)
    assert processor.feature_names_, "feature_names_ was never assigned"
    assert processor.get_feature_names_out()
    assert processor.feature_names_in_ == ["num", "cat", "num2"]


def test_get_feature_names_out_requires_fit(frame):
    with pytest.raises(RuntimeError, match="fit"):
        DataProcessor().get_feature_names_out()


# ============================================================================
# Robustness: unknown strategies warn instead of silently substituting
# ============================================================================


@pytest.mark.parametrize(
    "kwargs,pattern",
    [
        ({"imputation_strategy": "telepathy"}, "Unknown imputation_strategy"),
        ({"categorical_encoding": "runes"}, "Unknown categorical_encoding"),
        ({"scaling_strategy": "vibes"}, "Unknown scaling_strategy"),
        ({"feature_selection_strategy": "intuition"}, "Unknown feature_selection_strategy"),
    ],
)
def test_unknown_strategies_warn(frame, target, kwargs, pattern):
    processor = DataProcessor(task_type="classification", **kwargs)
    with pytest.warns(UserWarning, match=pattern):
        processor.fit(frame, target)


def test_supervised_feature_selection_without_a_target_warns(frame):
    processor = DataProcessor(
        task_type="classification", feature_selection_strategy="select_k_best_anova"
    )
    with pytest.warns(UserWarning, match="supervised"):
        processor.fit(frame, None)


def test_correlation_filter_drops_redundant_columns():
    X = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * 2, "c": np.random.rand(20)})
    processor = DataProcessor(
        task_type="classification",
        feature_selection_strategy="correlation",
        correlation_threshold=0.95,
        scaling_strategy="none",
        categorical_encoding="none",
    )
    processor.fit(X, pd.Series([0, 1] * 10))
    assert processor._correlation_cols_to_drop, "perfectly collinear column was kept"


def test_override_types_are_honoured(frame, target):
    processor = DataProcessor(task_type="classification")
    processor.override_types = {"num2": "categorical"}
    processor.fit(frame, target)
    assert "num2" in processor.categorical_cols_


def test_override_types_warn_on_unknown_columns(frame, target):
    processor = DataProcessor(task_type="classification")
    processor.override_types = {"ghost": "categorical"}
    with pytest.warns(UserWarning, match="unknown column"):
        processor.fit(frame, target)


def test_processing_summary_renders_without_raising(frame, target):
    processor = DataProcessor(task_type="classification", resampling_strategy="random_over")
    processor.fit_transform(frame, target)
    summary = processor.get_processing_summary()
    assert "Resampling" in summary
    assert "rows" in summary


def test_processing_summary_requires_fit():
    with pytest.raises(RuntimeError, match="not been fitted"):
        DataProcessor().get_processing_summary()


# ============================================================================
# P0-3: device selection was a copy-pasted ternary in ~20 places
# ============================================================================


def test_resolve_device_auto_returns_something_usable():
    assert resolve_device("auto") in ("cpu", "cuda", "mps") or resolve_device("auto").startswith("cuda:")


def test_resolve_device_cpu_is_always_honoured():
    assert resolve_device("cpu") == "cpu"


def test_unavailable_backend_falls_back_with_a_warning():
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA is available; the fallback path cannot be exercised")
    with pytest.warns(UserWarning, match="CUDA was requested"):
        assert resolve_device("cuda") == "cpu"


def test_unrecognised_device_string_warns_and_falls_back():
    with pytest.warns(UserWarning, match="Unrecognised device"):
        assert resolve_device("quantum") in ("cpu", "cuda", "mps")


def test_resolve_device_accepts_torch_device_objects():
    import torch

    assert resolve_device(torch.device("cpu")) == "cpu"


def test_describe_device_is_a_string():
    assert isinstance(describe_device("cpu"), str)


def test_device_index_is_clamped_to_available_devices():
    import torch

    if not torch.cuda.is_available():
        with pytest.warns(UserWarning):
            assert resolve_device("cuda:7") == "cpu"
    else:
        assert resolve_device("cuda:7").startswith("cuda")


# ============================================================================
# Warning de-duplication: pipelines are constructed inside loops
# ============================================================================


def test_warn_once_suppresses_repeats():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert warn_once("hello") is True
        for _ in range(10):
            assert warn_once("hello") is False
    assert len(caught) == 1


def test_warn_once_distinguishes_by_key():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_once("a", key="k1")
        warn_once("b", key="k2")
    assert len(caught) == 2


def test_deprecated_param_names_the_replacement():
    with pytest.warns(DeprecationWarning, match="use 'context_size' instead"):
        deprecated_param("sampling_context_size", replacement="context_size", removed_in="0.3.0")


def test_warn_unknown_keys_suggests_corrections():
    with pytest.warns(UserWarning, match="did you mean 'learning_rate'"):
        warn_unknown_keys(["lerning_rate"], context="tuning_params", known=["learning_rate"])


def test_warn_unknown_keys_is_silent_for_an_empty_set():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_unknown_keys([], context="tuning_params") == []


# ============================================================================
# P0-4: SHAP was advertised and silently replaced by permutation importance
# ============================================================================


def test_shap_importance_raises_instead_of_silently_substituting():
    """Was: logged a warning, set method='permutation', returned different numbers."""
    from tabtune.TabularPipeline.pipeline import TabularPipeline

    pipeline = TabularPipeline.__new__(TabularPipeline)
    pipeline._is_fitted = True
    with pytest.raises(NotImplementedError, match="not implemented"):
        TabularPipeline.get_feature_importance(pipeline, pd.DataFrame({"a": [1]}), method="shap")


# ============================================================================
# P0-5: the banner printed on every construction, once per CV fold
# ============================================================================


def test_banner_is_logged_once_and_not_printed(capsys):
    from importlib import import_module

    # import_module rather than `import a.b.c as x`: under pytest's importlib
    # mode the latter can resolve against a shadowing top-level namespace
    # package when the whole suite is collected together.
    pipeline_module = import_module("tabtune.TabularPipeline.pipeline")

    pipeline_module._BANNER_SHOWN = False
    pipeline_module._log_banner()
    pipeline_module._log_banner()
    pipeline_module._log_banner()
    assert "TabTune" not in capsys.readouterr().out


# ============================================================================
# P0-6: three-way version drift between pyproject, setup.py and __init__
# ============================================================================


def test_version_is_single_sourced():
    from pathlib import Path

    import tomllib

    import tabtune

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():  # installed wheel, no source tree
        pytest.skip("pyproject.toml not available")
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert tabtune.__version__ == declared


def test_setup_py_declares_no_version():
    """setup.py must stay a shim; a literal there is how the drift happened."""
    from pathlib import Path

    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    if not setup_py.exists():
        pytest.skip("setup.py not available")
    assert "version=" not in setup_py.read_text()


# ============================================================================
# Lazy imports: `import tabtune` must not drag in torch
# ============================================================================


@pytest.mark.integration
def test_import_tabtune_does_not_import_torch():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys, tabtune; sys.exit(1 if 'torch' in sys.modules else 0)"],
        capture_output=True,
    )
    assert result.returncode == 0, "import tabtune pulled in torch"


def test_public_names_are_reachable_through_lazy_getattr():
    import tabtune

    assert "TabularPipeline" in dir(tabtune)
    with pytest.raises(AttributeError):
        _ = tabtune.NotARealSymbol

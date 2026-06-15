"""
TabPFN v3 — Full integration into TabTune
=========================================
Complete vendored source copy from https://github.com/PriorLabs/TabPFN (v8.0.3),
which ships the v3 model (ModelVersion.V3 / architecture "tabpfn_v3").

All imports were rewritten from absolute (`tabpfn.xxx`) to the vendored package
path (`tabtune.models.tabpfnv3.xxx`). The external `tabpfn_common_utils.telemetry`
dependency is replaced with an internal no-op shim (`._compat.telemetry`), so the
tree is hermetic (no telemetry, no network callbacks).

Model name in pipeline: 'TabPFNv3'
Supports: classification + regression (inference) in this delivery. Native /
SFT / meta-learning fine-tuning is vendored under `.finetuning` and is wired into
TabTune's TuningManager in a follow-up pass (see README "What's next").

`TabPFNv3Classifier` / `TabPFNv3Regressor` are thin subclasses that pin the v3
default checkpoint, so callers get v3 without invoking `create_default_for_version`.
Upstream already defaults `settings.tabpfn.model_version = ModelVersion.V3`; we pin
the path explicitly so behavior is robust against future settings drift.
"""
from __future__ import annotations

# Re-export the base upstream classes under their original names. Several
# vendored modules (finetuning/finetuned_classifier.py, finetuning/train_util.py,
# regressor.py, model_loading.py, validation.py) do
# `from tabtune.models.tabpfnv3 import TabPFNClassifier, TabPFNRegressor`,
# so these names MUST be importable from the package root.
from .classifier import TabPFNClassifier as TabPFNClassifier
from .regressor import TabPFNRegressor as TabPFNRegressor
from .constants import ModelVersion

# Internal aliases used by the TabTune-facing pinned subclasses below.
_BaseV3Classifier = TabPFNClassifier
_BaseV3Regressor = TabPFNRegressor
from .model_loading import (
    ModelSource,
    load_fitted_tabpfn_model,
    prepend_cache_path,
    save_fitted_tabpfn_model,
)


def _v3_classifier_default_path():
    # Mirrors classifier.create_default_for_version(ModelVersion.V3).
    return prepend_cache_path(ModelSource.get_classifier_v3().default_filename)


def _v3_regressor_default_path():
    return prepend_cache_path(ModelSource.get_regressor_v3().default_filename)


class TabPFNv3Classifier(_BaseV3Classifier):
    """TabPFN v3 classifier defaulting to the v3 default checkpoint.

    Inputs:  sklearn-style X (n_samples, n_features), y (n_samples,).
    Outputs: sklearn-compatible `predict` / `predict_proba`.
    """

    def __init__(self, *args, model_path="auto", **kwargs):
        # Only override when the caller left it on "auto"; an explicit path wins.
        if model_path == "auto":
            model_path = _v3_classifier_default_path()
        super().__init__(*args, model_path=model_path, **kwargs)


class TabPFNv3Regressor(_BaseV3Regressor):
    """TabPFN v3 regressor defaulting to the v3 default checkpoint."""

    def __init__(self, *args, model_path="auto", **kwargs):
        if model_path == "auto":
            model_path = _v3_regressor_default_path()
        super().__init__(*args, model_path=model_path, **kwargs)


__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "TabPFNv3Classifier",
    "TabPFNv3Regressor",
    "ModelVersion",
    "load_fitted_tabpfn_model",
    "save_fitted_tabpfn_model",
]

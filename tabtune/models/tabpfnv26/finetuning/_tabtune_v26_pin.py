"""v2.6-pinned native fine-tuners for TabTune.

The vendored upstream wrappers hardcode ``ModelVersion.V2_5`` in
``_create_estimator``, so ``FinetunedTabPFNClassifier`` /
``FinetunedTabPFNRegressor`` from this tree would fine-tune the **v2.5**
checkpoint although TabTune registers the model as ``TabPFNv26``. These
subclasses pin ``ModelVersion.V2_6``, exactly as
``tabtune.models.tabpfnv3.finetuning._tabtune_v3_pin`` does for v3. The
override mirrors upstream except for the version, keeping
``fit_mode="batched"`` and ``differentiable_input=False`` (required by the
training loop in ``finetuned_base.py``).
"""
from __future__ import annotations

from typing import Any

from tabtune.models.tabpfnv26.classifier import TabPFNClassifier
from tabtune.models.tabpfnv26.constants import ModelVersion
from tabtune.models.tabpfnv26.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabtune.models.tabpfnv26.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
from tabtune.models.tabpfnv26.regressor import TabPFNRegressor


class V26PinnedFinetunedClassifier(FinetunedTabPFNClassifier):
    """``FinetunedTabPFNClassifier`` that fine-tunes the v2.6 checkpoint."""

    # A class attribute, not an __init__ argument: sklearn's get_params/clone
    # need the upstream __init__ signature unchanged (no varargs).
    _pinned_model_version = ModelVersion.V2_6

    def _create_estimator(self, config: dict[str, Any]) -> TabPFNClassifier:
        return TabPFNClassifier.create_default_for_version(
            version=self._pinned_model_version,
            **config,
            fit_mode="batched",
            differentiable_input=False,
        )


class V26PinnedFinetunedRegressor(FinetunedTabPFNRegressor):
    """``FinetunedTabPFNRegressor`` that fine-tunes the v2.6 checkpoint."""

    # A class attribute, not an __init__ argument: sklearn's get_params/clone
    # need the upstream __init__ signature unchanged (no varargs).
    _pinned_model_version = ModelVersion.V2_6

    def _create_estimator(self, config: dict[str, Any]) -> TabPFNRegressor:
        return TabPFNRegressor.create_default_for_version(
            version=self._pinned_model_version,
            **config,
            fit_mode="batched",
            differentiable_input=False,
        )


__all__ = ["V26PinnedFinetunedClassifier", "V26PinnedFinetunedRegressor"]

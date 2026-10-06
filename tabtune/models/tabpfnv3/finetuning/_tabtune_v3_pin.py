"""V3-pinned native fine-tuners for TabTune.

Why this module exists
----------------------
The upstream PriorLabs fine-tuning wrappers hardcode ``ModelVersion.V2_5`` inside
``_create_estimator``:

    # finetuning/finetuned_classifier.py
    def _create_estimator(self, config):
        return TabPFNClassifier.create_default_for_version(
            version=ModelVersion.V2_5, ...      # <-- NOT v3!
        )

So using ``FinetunedTabPFNClassifier`` / ``FinetunedTabPFNRegressor`` as-is would
fine-tune the **v2.5** checkpoint even though TabTune registered the model as
``TabPFNv3``. These subclasses override ``_create_estimator`` to pin
``ModelVersion.V3`` (a class attribute), so the native fine-tuning path actually
fine-tunes v3 weights.

The override mirrors the upstream method exactly except for the pinned version,
keeping ``fit_mode="batched"`` and ``differentiable_input=False`` (required by the
fine-tuning training loop — see finetuned_base.py).
"""
from __future__ import annotations

from typing import Any

from tabtune.models.tabpfnv3 import TabPFNClassifier, TabPFNRegressor
from tabtune.models.tabpfnv3.constants import ModelVersion
from tabtune.models.tabpfnv3.finetuning.finetuned_classifier import (
    FinetunedTabPFNClassifier,
)
from tabtune.models.tabpfnv3.finetuning.finetuned_regressor import (
    FinetunedTabPFNRegressor,
)


class V3PinnedFinetunedClassifier(FinetunedTabPFNClassifier):
    """``FinetunedTabPFNClassifier`` that fine-tunes the v3 checkpoint.

    The version is the class attribute ``_pinned_model_version``
    (``ModelVersion.V3``); set it on an instance to override. Constructor
    arguments are ``FinetunedTabPFNClassifier``'s, unchanged.
    """

    # A class attribute, not an __init__ argument: sklearn's get_params/clone
    # need the upstream __init__ signature unchanged (no varargs).
    _pinned_model_version = ModelVersion.V3

    def _create_estimator(self, config: dict[str, Any]) -> TabPFNClassifier:
        # Mirror upstream _create_estimator but pin the requested version.
        return TabPFNClassifier.create_default_for_version(
            version=self._pinned_model_version,
            **config,
            fit_mode="batched",
            differentiable_input=False,
        )


class V3PinnedFinetunedRegressor(FinetunedTabPFNRegressor):
    """``FinetunedTabPFNRegressor`` that fine-tunes the v3 checkpoint."""

    # A class attribute, not an __init__ argument: sklearn's get_params/clone
    # need the upstream __init__ signature unchanged (no varargs).
    _pinned_model_version = ModelVersion.V3

    def _create_estimator(self, config: dict[str, Any]) -> TabPFNRegressor:
        return TabPFNRegressor.create_default_for_version(
            version=self._pinned_model_version,
            **config,
            fit_mode="batched",
            differentiable_input=False,
        )


__all__ = ["V3PinnedFinetunedClassifier", "V3PinnedFinetunedRegressor"]

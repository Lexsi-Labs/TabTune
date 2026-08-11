"""
TabTune glue around the vendored iLTM inference engine.

The upstream engine keeps a *process-wide* backbone cache
(``_iLTMBase._model_cache`` keyed by checkpoint path + device + architecture).
That is great for repeated zero-shot inference, but it is unsafe for TabTune's
fine-tuning / LoRA flows: two pipeline instances pointing at the same
checkpoint would silently share (and mutate) the same ``torch.nn.Module``.

The subclasses below pin a **fresh, per-instance** backbone instead: the first
``_initialize_model()`` call loads the checkpoint bypassing the shared cache
and stores the module on the instance (``_pinned_model``); every later call
(including the ones issued inside ``fit``) returns that exact module, so LoRA
injection / fine-tuned weights performed through the TabTune wrapper's
``model_`` attribute are always what the engine uses for prediction.

These classes are defined at module level so fitted engines stay picklable
(``TabularPipeline.save`` uses ``joblib.dump`` of the whole pipeline).
"""
from __future__ import annotations

from .inference_interface import iLTMClassifier as EngineClassifier
from .inference_interface import iLTMRegressor as EngineRegressor


class _PinnedBackboneMixin:
    """Pin the iLTM backbone per engine instance (bypass the shared cache)."""

    def _initialize_model(self):
        pinned = getattr(self, "_pinned_model", None)
        if pinned is not None:
            return pinned
        cache = type(self)._model_cache
        snapshot = dict(cache)
        cache.clear()  # force a fresh torch.load for THIS instance
        try:
            model = super()._initialize_model()
        finally:
            cache.clear()
            cache.update(snapshot)  # never leak instance backbones into the shared cache
        self._pinned_model = model
        return model


class PinnedILTMClassifierEngine(_PinnedBackboneMixin, EngineClassifier):
    """Vendored iLTM classifier engine with a per-instance pinned backbone."""


class PinnedILTMRegressorEngine(_PinnedBackboneMixin, EngineRegressor):
    """Vendored iLTM regressor engine with a per-instance pinned backbone."""

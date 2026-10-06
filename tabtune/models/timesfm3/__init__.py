"""TimesFM 3.0 (Google Research) vendored for TabTune: the PyTorch backend.

Vendored from google-research/timesfm rather than installed as the ``timesfm``
pip package, the same way TabTune ships Chronos, TabFM and TabPFN v3. The
PyTorch backend needs only torch, numpy, ``huggingface_hub`` and
``safetensors``, all core TabTune dependencies, so there is no
``dependency_extra`` for this family.

Provenance (the constants below record it in code):

* ``configs.py``, ``cpm_revin_refine.py``, ``dense.py``, ``model.py``,
  ``normalization.py``, ``timesfm3_forecaster.py``, ``transformer.py`` and
  ``util.py`` are **verbatim** copies of upstream ``src/timesfm3/torch/``.
  Unlike Chronos, this backend needed no import fixes at all: every
  intra-package import upstream is already relative (``from . import configs``),
  so the subpackage works unchanged under a new parent.
* This ``__init__.py`` is the one file TabTune wrote rather than copied. Upstream
  has its own ``torch/__init__.py``; re-exporting from here instead keeps every
  vendored file byte-identical to the release.
* ``evaluator.py`` (a benchmarking harness, reachable only from upstream's
  ``__init__``) and ``transformations.py`` (imported by nothing) are not
  vendored, the same call as Chronos-2's ``trainer.py``.
* The MLX and Flax backends, the TimesFM 2.5 ``timesfm`` package, the
  ``timesfm3/*.py`` back-compatibility shims and ``utils/xreg_lib.py`` (which
  needs jax) are not vendored.

The *code* here is Apache-2.0, but the published **weights are not**: see the
``TimesFM3`` entry in :mod:`tabtune.registry.TimeSeriesCatalog` for the
non-commercial license TabTune records for them.

Nothing here is imported by ``import tabtune``: the adapter in
``tabtune.models.TimeSeries.timesfm3`` loads this package when a
``TimeSeriesPipeline`` is fitted.
"""

from .configs import ResidualBlockConfig, StackedTransformersConfig, TransformerConfig
from .model import TimesFM3Torch
from .timesfm3_forecaster import ForecastOutput, ModelConfig, TimesFM3Forecaster

#: Upstream release these files were taken from.
UPSTREAM_VERSION = "3.0.2"
UPSTREAM_COMMIT = "8cb0628371af142e16b8c232cc9fbf667ffb12f9"
UPSTREAM_URL = "https://github.com/google-research/timesfm"

__all__ = [
    "ForecastOutput",
    "ModelConfig",
    "ResidualBlockConfig",
    "StackedTransformersConfig",
    "TimesFM3Forecaster",
    "TimesFM3Torch",
    "TransformerConfig",
    "UPSTREAM_VERSION",
    "UPSTREAM_COMMIT",
    "UPSTREAM_URL",
]

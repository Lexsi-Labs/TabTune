"""Vendored Toto 1.0 (``Datadog/Toto-Open-Base-1.0``) inference code.

This package is TabTune's copy of the inference subset of Datadog's ``toto-ts``
release, which is Apache-2.0. It is imported only from
:mod:`tabtune.models.TimeSeries.toto1`, inside ``Toto1Adapter.load``, so neither
``import tabtune`` nor ``import tabtune.TimeSeries`` pulls in torch.

Provenance
----------
Toto 1.0 is published as a PyPI release rather than something vendored from a git
tag, so :data:`UPSTREAM_SDIST_SHA256` stands in for the commit the other families
record. ``tests/test_toto1_adapter.py`` asserts both constants.

What was copied
---------------
Thirteen modules -- the exact import closure of ``TotoForecaster``. Upstream's
subpackage layout (``model/``, ``inference/``, ``data/util/``) is preserved
because its intra-package imports are already relative, so no rewrites were
needed on that account. Not copied: ``model/lightning_module.py``,
``model/losses.py``, ``model/scheduler.py`` and ``data/datamodule/``
(fine-tuning, which would pull in ``lightning``), ``inference/gluonts_predictor.py``
and ``evaluation/`` (benchmark harnesses needing gluonts), ``scripts/``,
``test/`` and the BOOM notebooks.

Three files differ from the release, by one import statement each, and each says
so in a note at the top: ``model/distribution.py``, ``model/scaler.py`` and
``inference/forecaster.py`` import from :mod:`._gluonts` rather than from
gluonts. That module holds the four gluonts symbols Toto's model core uses; see
its own header for why gluonts is not a TabTune dependency.

Backend library
---------------
``rotary_embedding_torch`` is *not* vendored: it and ``jaxtyping`` are core
TabTune dependencies, so no extra install is needed. ``xformers`` is
genuinely optional upstream -- every use is guarded, with native PyTorch
fallbacks, and ``Toto.load_from_checkpoint`` turns
``use_memory_efficient_attention`` off by itself when it is absent.
"""

from __future__ import annotations

UPSTREAM_PACKAGE = "toto-ts"
UPSTREAM_VERSION = "0.2.0"
UPSTREAM_SDIST_SHA256 = "4cb832a08abb22b307cbde2f687abd2262d424b988ce93cd20a7b48637beb178"
UPSTREAM_URL = "https://pypi.org/project/toto-ts/0.2.0/"

from .data.util.dataset import MaskedTimeseries, pad_array, pad_id_mask  # noqa: E402
from .inference.forecaster import Forecast, TotoForecaster  # noqa: E402
from .model.toto import Toto  # noqa: E402

__all__ = [
    "UPSTREAM_PACKAGE",
    "UPSTREAM_SDIST_SHA256",
    "UPSTREAM_URL",
    "UPSTREAM_VERSION",
    "Forecast",
    "MaskedTimeseries",
    "Toto",
    "TotoForecaster",
    "pad_array",
    "pad_id_mask",
]

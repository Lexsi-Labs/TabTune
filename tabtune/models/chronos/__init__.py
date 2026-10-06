"""Chronos (Amazon) vendored for TabTune: v1 (T5), Bolt and Chronos-2.

Vendored from amazon-science/chronos-forecasting rather than installed as the
``chronos-forecasting`` pip package, the same way TabTune ships TabFM and
TabPFN v3. Both code paths need only torch, transformers, einops, numpy and
pandas, all core TabTune dependencies.

Provenance (``ATTRIBUTION.md`` used to hold this; the constants below keep it
in code):

* ``chronos.py`` (v1) is upstream apart from two documented import fixes noted
  in its header: upstream imports itself as a top-level ``chronos`` package,
  which does not exist here.
* ``chronos_bolt.py``, ``base.py``, ``utils.py`` and ``df_utils.py`` are
  verbatim. Bolt needed no edits: it only imports ``.base``.
* ``chronos2/`` (Chronos-2) is vendored with import rewrites only, noted in each
  edited file's header: upstream imports itself as a top-level ``chronos``
  package. ``chronos2/trainer.py`` is fine-tuning only and is not vendored, so
  ``Chronos2Pipeline.fit`` is unavailable.
* ``boto_utils.py`` (S3 loading) is not vendored.

Nothing here is imported by ``import tabtune``: the adapters in
``tabtune.models.TimeSeries.chronos``, ``...chronos_bolt`` and ``...chronos2``
load this package when a ``TimeSeriesPipeline`` is fitted.
"""

from .chronos import ChronosConfig, ChronosModel, ChronosPipeline, MeanScaleUniformBins
from .chronos2 import Chronos2ForecastingConfig, Chronos2Model, Chronos2Pipeline
from .chronos_bolt import (
    ChronosBoltConfig,
    ChronosBoltModelForForecasting,
    ChronosBoltPipeline,
)

UPSTREAM_VERSION = "2.3.2"
UPSTREAM_COMMIT = "0aba28e9360b6355d5a919596c90b16c25939f02"
UPSTREAM_URL = "https://github.com/amazon-science/chronos-forecasting"

__all__ = [
    "ChronosConfig",
    "ChronosModel",
    "ChronosPipeline",
    "MeanScaleUniformBins",
    "ChronosBoltConfig",
    "ChronosBoltModelForForecasting",
    "ChronosBoltPipeline",
    "Chronos2ForecastingConfig",
    "Chronos2Model",
    "Chronos2Pipeline",
    "UPSTREAM_VERSION",
    "UPSTREAM_COMMIT",
    "UPSTREAM_URL",
]

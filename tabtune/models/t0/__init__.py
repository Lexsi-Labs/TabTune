"""Open-weights t0 family of forecasting models.

``import t0`` loads no array framework. The runtimes are subpackages, each behind an
extra: ``t0.torch`` (``pip install 'tfc-t0[torch]'``) and ``t0.mlx``
(``pip install 'tfc-t0[mlx]'``, Apple silicon). ``t0.T0Forecaster`` is the PyTorch runtime.
"""

import importlib
from typing import TYPE_CHECKING

from .config import T0Config

if TYPE_CHECKING:
    from .torch import Forecast, MaskType, T0Forecaster, TimeSeries, VariateType, batch_series

__all__ = [
    "Forecast",
    "MaskType",
    "T0Config",
    "T0Forecaster",
    "TimeSeries",
    "VariateType",
    "batch_series",
]

# Served by the PyTorch runtime, imported on first access so `import t0` stays framework-free.
_TORCH_EXPORTS = frozenset({"Forecast", "MaskType", "T0Forecaster", "TimeSeries", "VariateType", "batch_series"})


def __getattr__(name: str) -> object:
    if name in _TORCH_EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.torch"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

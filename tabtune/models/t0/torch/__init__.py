"""The PyTorch runtime of t0. Requires the ``torch`` extra: ``pip install 'tfc-t0[torch]'``."""

try:
    import torch
except ImportError as error:
    raise ImportError(
        "The vendored t0 PyTorch runtime needs torch, which TabTune installs as a core dependency."
    ) from error

from .. import data
from .._ops import register
from ..data import MaskType, TimeSeries, VariateType
from ..torch._ops import TorchOps
from ..torch.model import Forecast, T0Forecaster

_OPS = TorchOps()
register(torch.Tensor, _OPS)


def batch_series(series):
    """``t0.data.batch_series`` into tensors, from tensors, NumPy arrays or lists; see there."""
    return data.batch_series(series, ops=_OPS)


__all__ = ["Forecast", "MaskType", "T0Forecaster", "TimeSeries", "VariateType", "batch_series"]

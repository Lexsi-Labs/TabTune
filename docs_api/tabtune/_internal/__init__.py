"""Internal helpers shared across TabTune modules.

Nothing here is part of the public API. These are small utilities that were
previously duplicated inline across ``pipeline.py``, ``tuning.py`` and the
preprocessors, pulled into one place so their behaviour is consistent and
testable.
"""

from __future__ import annotations

from .deprecation import (
    deprecated_param,
    reset_warning_cache,
    warn_once,
    warn_unknown_keys,
)
from .device import describe_device, resolve_device, torch_device

__all__ = [
    "resolve_device",
    "torch_device",
    "describe_device",
    "deprecated_param",
    "warn_once",
    "warn_unknown_keys",
    "reset_warning_cache",
]

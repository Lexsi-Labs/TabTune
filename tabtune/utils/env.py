"""Environment detection utilities."""

from __future__ import annotations

import functools
import os
from types import ModuleType
from typing import Optional


def _try_import(module_name: str) -> Optional[ModuleType]:
    try:
        module = __import__(module_name)  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return module


@functools.lru_cache(maxsize=1)
def is_colab() -> bool:
    """Return True when running inside a Google Colab environment."""
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True

    if os.environ.get("COLAB_GPU"):
        return True

    if _try_import("google.colab") is not None:
        return True

    return False


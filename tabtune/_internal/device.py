"""Single source of truth for device selection.

Before this module, the expression ``'cuda' if torch.cuda.is_available() else
'cpu'`` appeared roughly twenty times across ``pipeline.py`` and ``tuning.py``.
Each occurrence silently ignored Apple Silicon, could not target a specific GPU
index, and drifted as new models were added.

``resolve_device`` centralises that logic and adds:

* ``"auto"`` - prefer CUDA, then MPS, then CPU.
* explicit indices - ``"cuda:1"`` is honoured and validated.
* graceful degradation - asking for an unavailable backend warns and falls back
  rather than failing deep inside a forward pass.

The module imports torch lazily so that importing TabTune's registry, config
and evaluation layers stays torch-free.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["resolve_device", "torch_device", "describe_device"]

_VALID_PREFIXES = ("cpu", "cuda", "mps")


def _torch() -> Any | None:
    """Import torch lazily, returning ``None`` when it is unavailable."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        return None
    return torch


def resolve_device(requested: str | Any | None = None, *, warn: bool = True) -> str:
    """Resolve a device request to a concrete device string.

    Args:
        requested: ``None`` or ``"auto"`` to pick the best available backend;
            otherwise a device string such as ``"cpu"``, ``"cuda"``,
            ``"cuda:1"`` or ``"mps"``. A ``torch.device`` is also accepted.
        warn: Emit a :class:`UserWarning` when falling back from an unavailable
            backend. Disable for probing calls.

    Returns:
        A concrete device string safe to pass to ``torch.device``.

    Example:
        >>> resolve_device("auto")           # doctest: +SKIP
        'cuda'
        >>> resolve_device("cuda:3")         # on a 1-GPU box
        'cuda:0'
    """
    torch = _torch()
    if torch is None:
        return "cpu"

    if requested is not None and not isinstance(requested, str):
        requested = str(requested)

    if requested is None or requested == "auto":
        if torch.cuda.is_available():
            return "cpu" if _cuda_device_count(torch) == 0 else "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"

    requested = requested.strip().lower()
    prefix = requested.split(":", 1)[0]

    if prefix not in _VALID_PREFIXES:
        if warn:
            warnings.warn(
                f"Unrecognised device {requested!r}; falling back to automatic "
                f"selection. Valid prefixes: {_VALID_PREFIXES}.",
                UserWarning,
                stacklevel=2,
            )
        return resolve_device("auto", warn=False)

    if prefix == "cpu":
        return "cpu"

    if prefix == "cuda":
        if not torch.cuda.is_available():
            if warn:
                warnings.warn(
                    "CUDA was requested but is not available; falling back to CPU. "
                    "Inference will be substantially slower.",
                    UserWarning,
                    stacklevel=2,
                )
            return resolve_device("auto", warn=False)
        if ":" in requested:
            index = int(requested.split(":", 1)[1])
            count = _cuda_device_count(torch)
            if index >= count:
                if warn:
                    warnings.warn(
                        f"cuda:{index} was requested but only {count} CUDA "
                        f"device(s) are visible; using cuda:0.",
                        UserWarning,
                        stacklevel=2,
                    )
                return "cuda:0"
            return f"cuda:{index}"
        return "cuda"

    # prefix == "mps"
    if not _mps_available(torch):
        if warn:
            warnings.warn(
                "MPS was requested but is not available; falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
        return resolve_device("auto", warn=False)
    return "mps"


def torch_device(requested: str | Any | None = None) -> Any:
    """Return a ``torch.device`` for ``requested``.

    Thin wrapper over :func:`resolve_device` for call sites that need the
    object rather than the string. Raises if torch is not installed, since a
    caller asking for a ``torch.device`` cannot proceed without it.
    """
    torch = _torch()
    if torch is None:  # pragma: no cover - torch is a hard dep in practice
        raise ImportError(
            "torch is required for torch_device(). Install it with `pip install torch`."
        )
    return torch.device(resolve_device(requested))


def describe_device(device: str | None = None) -> str:
    """Return a human-readable description of a device, for logs and model cards.

    Example:
        >>> describe_device("cpu")
        'cpu'
    """
    resolved = resolve_device(device, warn=False)
    torch = _torch()
    if torch is None or not resolved.startswith("cuda"):
        return resolved
    try:
        index = int(resolved.split(":", 1)[1]) if ":" in resolved else 0
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        return f"{resolved} ({name}, {total:.1f} GiB)"
    except Exception:  # pragma: no cover - diagnostics must never break a run
        return resolved


def _cuda_device_count(torch: Any) -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - defensive
        return 0


def _mps_available(torch: Any) -> bool:
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:  # pragma: no cover - defensive
        return False

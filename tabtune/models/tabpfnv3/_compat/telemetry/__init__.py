"""No-op telemetry shim — replaces tabpfn_common_utils.telemetry.

Mirrors the surface used by the vendored TabPFN v3 code:
  - track_model_call (decorator)
  - set_model_config / set_init_params (no-ops)
  - interactive submodule with capture_session / ping
"""
from __future__ import annotations


def track_model_call(*args, **kwargs):
    """No-op decorator; supports both @track_model_call and @track_model_call(...)."""
    def decorator(func):
        return func
    if args and callable(args[0]):
        return args[0]
    return decorator


def set_model_config(*args, **kwargs):  # noqa: D103
    pass


def set_init_params(*args, **kwargs):  # noqa: D103
    pass

"""No-op `interactive` telemetry submodule shim."""
from __future__ import annotations


def capture_session(*args, **kwargs):  # noqa: D103
    pass


def ping(*args, **kwargs):  # noqa: D103
    pass

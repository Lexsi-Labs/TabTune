"""Exception hierarchy for TabTune.

Every error TabTune raises deliberately derives from :class:`TabTuneError`, so
callers can catch the whole family with a single ``except`` while still being
able to discriminate on the specific failure mode.

The messages here are written to be *actionable*: when a model cannot be used
for a request, the error names the concrete alternatives rather than only
stating what went wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .spec import EnvelopeViolation

__all__ = [
    "TabTuneError",
    "ModelNotFoundError",
    "UnsupportedTaskError",
    "UnsupportedStrategyError",
    "EnvelopeError",
    "LicenseError",
    "ConfigError",
]


class TabTuneError(Exception):
    """Base class for every error raised by TabTune."""


class ConfigError(TabTuneError, ValueError):
    """Raised when a configuration object fails validation.

    Subclasses :class:`ValueError` for backward compatibility: TabTune
    previously raised bare ``ValueError`` for bad configuration, and existing
    user code that catches ``ValueError`` keeps working.
    """


class ModelNotFoundError(TabTuneError, ValueError):
    """Raised when a model name does not resolve to a registered model."""

    def __init__(self, name: str, available: Sequence[str] = ()) -> None:
        self.name = name
        self.available = tuple(available)
        msg = f"Unknown model {name!r}."
        if self.available:
            suggestion = _closest(name, self.available)
            if suggestion is not None:
                msg += f" Did you mean {suggestion!r}?"
            msg += "\n  Available models: " + ", ".join(sorted(self.available))
        super().__init__(msg)


class UnsupportedTaskError(TabTuneError, ValueError):
    """Raised when a model does not implement the requested task type."""

    def __init__(self, model: str, task_type: str, supported: Sequence[str] = ()) -> None:
        self.model = model
        self.task_type = task_type
        self.supported = tuple(supported)
        msg = f"Model {model!r} does not support task_type={task_type!r}."
        if self.supported:
            msg += f" Supported: {sorted(self.supported)}."
        super().__init__(msg)


class UnsupportedStrategyError(TabTuneError, ValueError):
    """Raised when a model does not implement the requested tuning strategy."""

    def __init__(
        self,
        model: str,
        strategy: str,
        task_type: str,
        supported: Sequence[str] = (),
        alternatives: Sequence[str] = (),
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.task_type = task_type
        self.supported = tuple(supported)
        msg = (
            f"Model {model!r} does not support tuning_strategy={strategy!r} "
            f"for task_type={task_type!r}."
        )
        if self.supported:
            msg += f"\n  Supported strategies: {sorted(self.supported)}."
        if alternatives:
            msg += (
                f"\n  Models that do support {strategy!r}: "
                + ", ".join(sorted(alternatives))
            )
        super().__init__(msg)


class EnvelopeError(TabTuneError, ValueError):
    """Raised when input data exceeds a model's hard capability limits.

    These are architectural properties of the pretrained checkpoint (for
    example TabFM's ten-class output head), not configuration mistakes, so the
    message points at compatible models rather than at a parameter to change.
    """

    def __init__(self, model: str, violations: Sequence[EnvelopeViolation]) -> None:
        self.model = model
        self.violations = tuple(violations)
        lines = [f"Data exceeds the capability envelope of model {model!r}:"]
        for violation in self.violations:
            lines.append(f"  - {violation.message}")
        lines.append(
            "  These are limits of the pretrained checkpoint, not configuration "
            "issues. Pass envelope_mode='warn' to downgrade to a warning."
        )
        super().__init__("\n".join(lines))


class LicenseError(TabTuneError, ValueError):
    """Raised when a model's weight license forbids the requested usage mode.

    TabTune's license metadata is a convenience for catching obvious mismatches
    early. It is not legal advice, and the authoritative text always lives
    upstream.
    """

    def __init__(
        self,
        model: str,
        license_name: str,
        url: str = "",
        alternatives: Sequence[str] = (),
        notes: str = "",
    ) -> None:
        self.model = model
        self.license_name = license_name
        lines = [
            f"Weights for model {model!r} are distributed under "
            f"{license_name!r}, which does not permit commercial use."
        ]
        if notes:
            lines.append(f"  {notes}")
        if alternatives:
            lines.append(
                "  Commercially deployable alternatives: " + ", ".join(alternatives)
            )
            lines.append(
                "  Or distill this teacher into a model you own:\n"
                f"      TabDistiller(teachers={model!r}, student='lgbm')"
            )
        if url:
            lines.append(f"  License text: {url}")
        lines.append(
            "  To acknowledge and proceed anyway, pass license_mode='research'.\n"
            "  TabTune's license metadata is a convenience, not legal advice."
        )
        super().__init__("\n".join(lines))


def _closest(name: str, candidates: Sequence[str]) -> str | None:
    """Return the closest candidate by difflib ratio, or ``None`` if none is close."""
    import difflib

    matches = difflib.get_close_matches(name.lower(), [c.lower() for c in candidates], n=1, cutoff=0.6)
    if not matches:
        return None
    lowered = matches[0]
    for candidate in candidates:
        if candidate.lower() == lowered:
            return candidate
    return None

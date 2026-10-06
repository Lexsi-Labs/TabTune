"""Shared readers for ``model_params`` across the time series adapters.

Values are validated and coerced here (``bool("false")`` is ``False``, a bad
value raises :class:`ConfigError`); defaults stay with each adapter.
:func:`read_params` maps a superseded spelling in :data:`CANONICAL_ALIASES` onto
its canonical key with a :class:`DeprecationWarning` naming the replacement.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..._internal.deprecation import warn_once, warn_unknown_keys
from ...registry.errors import ConfigError

__all__ = [
    "CANONICAL_ALIASES",
    "as_bool",
    "as_float",
    "as_int",
    "check_median_only",
    "hub_kwargs",
    "read_params",
]

#: Superseded ``model_params`` spellings and the canonical key each maps to.
CANONICAL_ALIASES: Mapping[str, str] = {
    "max_context_length": "max_context",
    "season": "season_length",
    "point": "point_forecast",
    "output_selection": "point_forecast",
}

#: Hugging Face download options every adapter that fetches weights accepts.
HUB_KEYS: tuple[str, ...] = ("revision", "local_files_only")

_TRUTHY = {"true", "1", "yes", "on"}
_FALSEY = {"false", "0", "no", "off"}


def read_params(
    params: Mapping[str, Any],
    *,
    name: str,
    known: Iterable[str],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return ``params`` with aliases resolved, warning about unknown keys.

    Args:
        params: The caller's ``model_params``.
        name: Model name, used in the warning so it names the model.
        known: Canonical keys this adapter accepts.
        aliases: Superseded spellings to accept, mapped to their canonical key.
            Defaults to the entries of :data:`CANONICAL_ALIASES` whose target is
            in ``known``.

    Keys beginning with ``_`` are passed through untouched and never warned about.
    """
    known = tuple(known)
    if aliases is None:
        aliases = {old: new for old, new in CANONICAL_ALIASES.items() if new in known}

    resolved: dict[str, Any] = {}
    for key, value in params.items():
        target = aliases.get(key)
        if target is None:
            resolved[key] = value
            continue
        if target in params:
            raise ConfigError(
                f"{name} model_params has both {key!r} and {target!r}; they are the same "
                f"setting. Pass only {target!r}."
            )
        _warn_alias(name, key, target)
        resolved[target] = value

    unknown = [
        key for key in resolved if key not in known and not key.startswith("_")
    ]
    if unknown:
        warn_unknown_keys(sorted(unknown), context=f"{name} model_params", known=known)
    return resolved


def _warn_alias(name: str, old: str, new: str) -> None:
    warn_once(
        f"{name} model_params[{old!r}] is deprecated; use {new!r}, which every time "
        f"series adapter accepts for this setting. {old!r} still works for now.",
        DeprecationWarning,
        key=f"ts-param-alias:{name}:{old}",
    )


def as_int(
    params: Mapping[str, Any],
    key: str,
    default: int,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    """Read an integer ``model_params`` entry, raising ``ConfigError`` on junk."""
    value = params.get(key)
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ConfigError(f"{name} model_params[{key!r}] must be an integer, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{name} model_params[{key!r}] must be an integer, got {value!r}"
        ) from exc
    if minimum is not None and number < minimum:
        raise ConfigError(
            f"{name} model_params[{key!r}] must be >= {minimum}, got {number}"
        )
    return number


def as_float(
    params: Mapping[str, Any],
    key: str,
    default: float,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    """Read a float ``model_params`` entry, raising ``ConfigError`` on junk."""
    value = params.get(key)
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ConfigError(f"{name} model_params[{key!r}] must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{name} model_params[{key!r}] must be a number, got {value!r}"
        ) from exc
    if minimum is not None and number < minimum:
        raise ConfigError(
            f"{name} model_params[{key!r}] must be >= {minimum}, got {number}"
        )
    return number


def as_bool(params: Mapping[str, Any], key: str, default: bool, *, name: str) -> bool:
    """Read a boolean ``model_params`` entry; a string is read as a word, not for truthiness."""
    value = params.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUTHY:
            return True
        if word in _FALSEY:
            return False
        raise ConfigError(
            f"{name} model_params[{key!r}] must be a boolean, got {value!r}. Use True or False."
        )
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ConfigError(f"{name} model_params[{key!r}] must be a boolean, got {value!r}")


def hub_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    """The Hugging Face download options present in ``params``."""
    return {key: params[key] for key in HUB_KEYS if key in params}


def check_median_only(params: Mapping[str, Any], *, name: str) -> None:
    """Refuse a ``point_forecast`` other than the median."""
    requested = params.get("point_forecast", "median")
    if requested == "median":
        return
    raise ConfigError(
        f"{name} only supports point_forecast='median', got {requested!r}: it predicts "
        f"quantiles directly and produces no sample paths to average. For a mean "
        f"forecast use one of {', '.join(_mean_capable(name))}."
    )


def _mean_capable(exclude: str) -> list[str]:
    """Registered models whose point forecast can be a mean."""
    from ...registry.TimeSeries import list_time_series_models

    names = []
    for spec in list_time_series_models(task="forecasting"):
        if spec.name == exclude:
            continue
        # Samplers and point forecasters; quantile-native models have no mean.
        if spec.family in ("baseline", "time-moe") or spec.name in (
            "Chronos",
            "Toto1",
            "TabPFN-TS",
        ):
            names.append(spec.name)
    return names

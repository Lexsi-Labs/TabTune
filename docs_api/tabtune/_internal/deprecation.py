"""Deprecation helpers with de-duplication.

TabTune constructs pipelines inside loops (cross-validation folds, ensemble
members, leaderboard entries). A naive ``warnings.warn`` would fire once per
fold per model, drowning the useful output. These helpers warn once per unique
message per process.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

__all__ = ["warn_once", "deprecated_param", "reset_warning_cache"]

_SEEN: set[str] = set()


def reset_warning_cache() -> None:
    """Clear the de-duplication cache. Intended for tests."""
    _SEEN.clear()


def warn_once(
    message: str,
    category: type[Warning] = UserWarning,
    *,
    key: str | None = None,
    stacklevel: int = 3,
) -> bool:
    """Emit ``message`` at most once per process.

    Args:
        message: The warning text.
        category: Warning class.
        key: Optional de-duplication key; defaults to the message itself. Use a
            key when the message embeds varying detail but the underlying
            condition is the same.
        stacklevel: Forwarded to :func:`warnings.warn`.

    Returns:
        ``True`` if the warning was emitted, ``False`` if it was suppressed as
        a duplicate.
    """
    dedup_key = key if key is not None else message
    if dedup_key in _SEEN:
        return False
    _SEEN.add(dedup_key)
    warnings.warn(message, category, stacklevel=stacklevel)
    return True


def deprecated_param(
    name: str,
    *,
    replacement: str | None = None,
    removed_in: str | None = None,
    extra: str = "",
    stacklevel: int = 4,
) -> None:
    """Warn that a parameter is deprecated.

    Args:
        name: The deprecated parameter name.
        replacement: What to use instead, if anything.
        removed_in: Version in which the parameter will be removed.
        extra: Additional context appended to the message.
        stacklevel: Forwarded to :func:`warnings.warn`.
    """
    parts = [f"{name!r} is deprecated"]
    if removed_in:
        parts.append(f"and will be removed in TabTune {removed_in}")
    if replacement:
        parts.append(f"- use {replacement!r} instead")
    message = " ".join(parts) + "."
    if extra:
        message += " " + extra
    warn_once(message, DeprecationWarning, key=f"deprecated-param:{name}", stacklevel=stacklevel)


def warn_unknown_keys(
    keys: Iterable[str],
    *,
    context: str,
    known: Iterable[str] = (),
    stacklevel: int = 4,
) -> list[str]:
    """Warn about configuration keys TabTune does not recognise.

    Unknown keys are forwarded rather than dropped (models accept their own
    kwargs), but silently accepting a typo is how ``learning_rate`` becomes
    ``lerning_rate`` and a fine-tune quietly runs with the default.

    Args:
        keys: The unrecognised keys.
        context: Where they came from, for the message (e.g. ``"tuning_params"``).
        known: Recognised keys, used to suggest corrections.
        stacklevel: Forwarded to :func:`warnings.warn`.

    Returns:
        The list of unknown keys, unchanged.
    """
    unknown = sorted(keys)
    if not unknown:
        return unknown

    import difflib

    known_list = list(known)
    hints: list[str] = []
    for key in unknown:
        matches = difflib.get_close_matches(key, known_list, n=1, cutoff=0.75)
        if matches:
            hints.append(f"{key!r} (did you mean {matches[0]!r}?)")
        else:
            hints.append(repr(key))

    warn_once(
        f"Unrecognised {context} key(s): {', '.join(hints)}. "
        f"They are forwarded to the model unchanged; check for typos if a "
        f"setting appears to have no effect.",
        UserWarning,
        key=f"unknown-keys:{context}:{','.join(unknown)}",
        stacklevel=stacklevel,
    )
    return unknown

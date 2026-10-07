"""Compatibility shim for the scikit-learn 1.6 -> 1.8 validation API change.

scikit-learn deprecated ``force_all_finite`` on ``check_array`` and
``check_X_y`` in 1.6 in favour of ``ensure_all_finite`` and removed the old name
in 1.8. Vendored preprocessors still use the old keyword; importing these
wrappers keeps their validation calls compatible.

Since ``pyproject.toml`` permits ``scikit-learn>=1.6``, the validators must work
with both APIs. This module normalises the keyword so the same source works
across the whole supported range, preferring the modern spelling to avoid
deprecation warnings on 1.6 and 1.7.

Usage:
    from ..._internal.sklearn_compat import check_array

    X = check_array(X, force_all_finite="allow-nan")   # works on 1.6 and 1.8
"""

from __future__ import annotations

import inspect
from typing import Any

from sklearn.utils import check_array as _sklearn_check_array
from sklearn.utils.validation import check_X_y as _sklearn_check_X_y

__all__ = ["check_array", "check_X_y", "SUPPORTS_FORCE_ALL_FINITE"]

#: Whether the installed scikit-learn still accepts the pre-1.8 keyword.
SUPPORTS_FORCE_ALL_FINITE: bool = (
    "force_all_finite" in inspect.signature(_sklearn_check_array).parameters
)

_CHECK_ARRAY_SUPPORTS_ENSURE: bool = (
    "ensure_all_finite" in inspect.signature(_sklearn_check_array).parameters
)
_CHECK_X_Y_SUPPORTS_ENSURE: bool = (
    "ensure_all_finite" in inspect.signature(_sklearn_check_X_y).parameters
)


def _normalise_finite_kwargs(
    kwargs: dict[str, Any], *, supports_ensure: bool, validator: str
) -> dict[str, Any]:
    """Translate the finite-value option without changing validation policy."""
    has_force = "force_all_finite" in kwargs
    has_ensure = "ensure_all_finite" in kwargs

    if has_force and has_ensure:
        if kwargs["force_all_finite"] != kwargs["ensure_all_finite"]:
            raise TypeError(
                f"{validator} received conflicting values for 'force_all_finite' "
                "and 'ensure_all_finite'; pass only one."
            )
        kwargs.pop("force_all_finite")
        has_force = False

    if supports_ensure:
        if has_force:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    elif has_ensure:
        kwargs["force_all_finite"] = kwargs.pop("ensure_all_finite")
    return kwargs


def check_array(array: Any, **kwargs: Any) -> Any:
    """Call :func:`sklearn.utils.check_array`, accepting either keyword spelling.

    Args:
        array: The array to validate.
        **kwargs: Forwarded to scikit-learn. ``force_all_finite`` and
            ``ensure_all_finite`` are interchangeable; whichever the installed
            version supports is used.

    Returns:
        The validated array.

    Raises:
        TypeError: If both spellings are supplied with different values, which
            is a caller bug rather than a compatibility question.
    """
    return _sklearn_check_array(
        array,
        **_normalise_finite_kwargs(
            kwargs, supports_ensure=_CHECK_ARRAY_SUPPORTS_ENSURE, validator="check_array"
        ),
    )


def check_X_y(X: Any, y: Any, **kwargs: Any) -> tuple[Any, Any]:
    """Validate features and targets, accepting either finite-value keyword.

    Uses the same translation as :func:`check_array`, detecting support from
    ``check_X_y`` itself. All other arguments and sklearn validation errors
    are preserved; conflicting keyword values raise ``TypeError``.
    """
    return _sklearn_check_X_y(
        X,
        y,
        **_normalise_finite_kwargs(
            kwargs, supports_ensure=_CHECK_X_Y_SUPPORTS_ENSURE, validator="check_X_y"
        ),
    )

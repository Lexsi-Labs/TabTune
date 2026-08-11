"""Compatibility shim for the scikit-learn 1.6 -> 1.8 validation API change.

scikit-learn deprecated ``check_array(force_all_finite=...)`` in 1.6 in favour
of ``ensure_all_finite=...`` and **removed** the old name in 1.8. The vendored
TabICL, OrionMSP, OrionMSPv1.5 and OrionBix preprocessors all call it with the
old keyword, so on scikit-learn 1.8 they raise::

    TypeError: check_array() got an unexpected keyword argument 'force_all_finite'

Since ``pyproject.toml`` permits ``scikit-learn>=1.6``, a fresh ``pip install
tabtune`` resolves to 1.8 and four of the thirteen bundled models fail on their
first ``fit``. This module normalises the keyword so the same source works
across the whole supported range.

Usage:
    from ..._internal.sklearn_compat import check_array

    X = check_array(X, force_all_finite="allow-nan")   # works on 1.6 and 1.8
"""

from __future__ import annotations

import inspect
from typing import Any

from sklearn.utils import check_array as _sklearn_check_array

__all__ = ["check_array", "SUPPORTS_FORCE_ALL_FINITE"]

#: Whether the installed scikit-learn still accepts the pre-1.8 keyword.
SUPPORTS_FORCE_ALL_FINITE: bool = (
    "force_all_finite" in inspect.signature(_sklearn_check_array).parameters
)


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
    has_force = "force_all_finite" in kwargs
    has_ensure = "ensure_all_finite" in kwargs

    if has_force and has_ensure:
        if kwargs["force_all_finite"] != kwargs["ensure_all_finite"]:
            raise TypeError(
                "check_array received conflicting values for 'force_all_finite' "
                "and 'ensure_all_finite'; pass only one."
            )
        kwargs.pop("force_all_finite")
        has_force = False

    if SUPPORTS_FORCE_ALL_FINITE:
        if has_ensure:
            kwargs["force_all_finite"] = kwargs.pop("ensure_all_finite")
    else:
        if has_force:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")

    return _sklearn_check_array(array, **kwargs)

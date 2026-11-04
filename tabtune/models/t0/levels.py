"""Quantile-level math on plain floats, shared by every runtime.

Which levels a forecast asks for, which trained levels pin its exponential tails, and
how to assemble the output columns are all decided here, before any array exists. The
runtimes only apply the resulting plans to their arrays.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "InterpolationPlan",
    "TailPlan",
    "get_rollout_quantile_levels",
    "interpolation_plan",
    "select_tail_knots",
    "tail_plan",
    "validate_quantile_levels",
]


def validate_quantile_levels(quantile_levels: Sequence[float]) -> tuple[float, ...]:
    """Return the requested levels as floats, or raise unless they are ascending, unique and in ``(0, 1)``."""
    levels = tuple(float(level) for level in quantile_levels)
    if not levels:
        raise ValueError("quantile_levels must be non-empty")
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError(f"each quantile must be in (0, 1); got {level}")
    if levels != tuple(sorted(set(levels))):
        raise ValueError(f"quantile_levels must be sorted ascending without duplicates; got {list(levels)}")
    return levels


def select_tail_knots(
    trained_quantile_levels: Sequence[float], requested_quantile_levels: Sequence[float]
) -> tuple[float, ...] | None:
    """Return the trained levels the exponential tails pin through, or None when no tail is needed.

    A tail pins through the boundary knot at its end and that knot's inward neighbour,
    so four trained levels at most are returned, or three when the range has an odd
    middle level shared by both ends.

    Levels of exactly 0 or 1 never count as tails; the exponential tails diverge there.
    Trained levels are rounded to 6 decimals so float32 storage error
    (0.1 -> 0.10000000149...) cannot misclassify a requested level equal to a trained
    one as a tail.
    """
    ordered = sorted(round(float(level), 6) for level in trained_quantile_levels)
    lo, hi = ordered[0], ordered[-1]
    if not any((level < lo or level > hi) and 0.0 < level < 1.0 for level in requested_quantile_levels):
        return None
    if len(ordered) < 2:
        raise ValueError("need at least two trained quantile levels to pin a tail")
    return tuple(sorted({ordered[0], ordered[1], ordered[-2], ordered[-1]}))


def get_rollout_quantile_levels(
    trained_quantile_levels: Sequence[float], requested_quantile_levels: Sequence[float]
) -> tuple[float, ...]:
    """Return the quantile levels a rollout must produce in order to serve a request.

    A requested level inside the trained range is returned as asked for. A level beyond
    it is left out and replaced by the trained levels its tail pins through: the
    boundary knot at that end and its inward neighbour. Interpolation would clamp such a
    level flat onto its boundary knot, so carrying it through the rollout would cost a
    trajectory to reproduce a column the boundary knot already holds, and would reweight
    every other level through the reducer's probability mass.
    """
    knots = select_tail_knots(trained_quantile_levels, requested_quantile_levels)
    if knots is None:
        return tuple(requested_quantile_levels)
    interior = [level for level in requested_quantile_levels if knots[0] <= level <= knots[-1]]
    return tuple(sorted(set(interior).union(knots)))


@dataclass(frozen=True)
class TailPlan:
    """How to serve query levels from rollout columns plus exponential tails.

    ``knot_columns`` index the knots among the rollout columns; ``order`` indexes the
    query levels in ``left tails ‖ rollout columns ‖ right tails``.
    """

    knots: tuple[float, ...]
    knot_columns: tuple[int, ...]
    left: tuple[float, ...]
    right: tuple[float, ...]
    order: tuple[int, ...]


def tail_plan(
    query_quantile_levels: Sequence[float],
    original_quantile_levels: Sequence[float],
    trained_quantile_levels: Sequence[float],
) -> TailPlan | None:
    """Plan the columns of ``query_quantile_levels``; None when none of them needs a tail.

    ``original_quantile_levels`` are the rollout's columns, as returned by
    ``get_rollout_quantile_levels``. The tails bracket them, so the full grid is a plain
    concatenation.
    """
    return _tail_plan(tuple(query_quantile_levels), tuple(original_quantile_levels), tuple(trained_quantile_levels))


@lru_cache(maxsize=256)
def _tail_plan(query: tuple[float, ...], original: tuple[float, ...], trained: tuple[float, ...]) -> TailPlan | None:
    knots = select_tail_knots(trained, query)
    if knots is None:
        return None
    left = tuple(level for level in query if level < knots[0])
    right = tuple(level for level in query if level > knots[-1])
    full = [*left, *original, *right]
    return TailPlan(
        knots=knots,
        knot_columns=tuple(original.index(knot) for knot in knots),
        left=left,
        right=right,
        order=tuple(full.index(level) for level in query),
    )


@dataclass(frozen=True)
class InterpolationPlan:
    """Linear interpolation of query levels from columns at fixed source levels.

    Query ``i`` is ``column[lower[i]] + weight[i] * (column[upper[i]] - column[lower[i]])``.
    Beyond the source range the edge column holds flat.
    """

    lower: tuple[int, ...]
    upper: tuple[int, ...]
    weight: tuple[float, ...]


def interpolation_plan(
    query_quantile_levels: Sequence[float], source_quantile_levels: Sequence[float]
) -> InterpolationPlan:
    """Bracketing columns and weights for ``query_quantile_levels`` among sorted ``source_quantile_levels``."""
    return _interpolation_plan(
        tuple(float(level) for level in query_quantile_levels),
        tuple(float(level) for level in source_quantile_levels),
    )


@lru_cache(maxsize=256)
def _interpolation_plan(query: tuple[float, ...], source: tuple[float, ...]) -> InterpolationPlan:
    # Levels and weights in float32, as the arrays hold them, so a plan applied to float32
    # values reproduces an array-side interpolation bit for bit.
    f32 = np.float32
    levels = [f32(level) for level in source]
    pad_left, pad_right = bool(levels[0] > 0), bool(levels[-1] < 1)
    padded = np.array(([f32(0)] if pad_left else []) + levels + ([f32(1)] if pad_right else []), dtype=f32)
    lower, upper, weight = [], [], []
    for q in (f32(level) for level in query):
        up = min(int(np.searchsorted(padded, q, side="right")), len(padded) - 1)
        lo = up - 1
        span = padded[up] - padded[lo]
        # A padded endpoint repeats the edge column, so the index maps back into the source.
        lower.append(min(max(lo - int(pad_left), 0), len(levels) - 1))
        upper.append(min(max(up - int(pad_left), 0), len(levels) - 1))
        weight.append(0.0 if span == 0 else float(f32((q - padded[lo]) / span)))
    return InterpolationPlan(tuple(lower), tuple(upper), tuple(weight))

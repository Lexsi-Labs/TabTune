# Copyright 2026 The Forecasting Company
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# The weighted-quantile and probability-mass computations are adapted from
# Chronos-2 (https://github.com/amazon-science/chronos-forecasting,
# src/chronos/utils.py and src/chronos/chronos2/pipeline.py).

"""Quantile arrays: interpolation, exponential tails, and the rollout reduction.

The ``reduce_rollout_quantiles`` math (per-path empirical CDF + reduction back to the
trained quantile grid) follows Chronos-2's pipeline; see
<https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py>.

Written once for every runtime against ``ArrayOps``; the level decisions come from
``t0.levels`` as plain floats.
"""

import logging
import math
from collections.abc import Sequence

import numpy as np
from jaxtyping import Float

from ._ops import ArrayT, ops_for
from .levels import interpolation_plan, tail_plan

__all__ = [
    "extend_quantile_tails",
    "extrapolate_quantiles",
    "get_prob_mass_per_quantile_level",
    "interpolate_quantiles",
    "reduce_rollout_quantiles",
    "sanitize_predictions",
    "weighted_quantile",
]

logger = logging.getLogger(__name__)


def interpolate_quantiles(
    query_quantile_levels: Sequence[float],
    original_quantile_levels: Sequence[float],
    original_values: Float[ArrayT, "*batch orig_quantiles"],
) -> Float[ArrayT, "*batch query_quantiles"]:
    """Linearly interpolate ``original_values`` at the query levels, holding the edge quantiles flat beyond them."""
    ops = ops_for(original_values)
    plan = interpolation_plan(query_quantile_levels, original_quantile_levels)
    device = ops.device(original_values)
    lower = ops.take(original_values, ops.asarray(plan.lower, dtype=ops.int64, device=device), axis=-1)
    upper = ops.take(original_values, ops.asarray(plan.upper, dtype=ops.int64, device=device), axis=-1)
    weight = ops.asarray(plan.weight, dtype=ops.float32, device=device)
    return lower + weight * (upper - lower)


def extend_quantile_tails(
    tail_levels: Sequence[float],
    knot_levels: Sequence[float],
    knot_values: Float[ArrayT, "*batch knots"],
) -> Float[ArrayT, "*batch tails"]:
    """Evaluate IQF exponential tails (Park et al., arXiv 2111.06581) at levels outside the knot range.

    For knots k_0 < ... < k_N carrying values v_i = v(k_i), a level q below k_0 lies on
    the left tail, linear in log(q), and a level q above k_N on the right tail, linear
    in log(1 - q)::

        v(q) = v_0 + s_L * log(q / k_0)              s_L = (v_1 - v_0) / log(k_1 / k_0)
        v(q) = v_N + s_R * log((1 - k_N) / (1 - q))  s_R = (v_N - v_N-1) / log((1 - k_N-1) / (1 - k_N))

    Negative slopes (crossed knots) clamp to zero, degenerating to a flat tail, so both
    tails are continuous at the boundary knots and monotone by construction.
    ``tail_levels`` are ascending and each lies outside the knot range.
    """
    # Levels carry the float32 values the arrays hold; the log-ratios are taken in float64
    # on the host (float64 is unsupported on some accelerators) and cross to the device once.
    knots = [float(np.float32(level)) for level in knot_levels]
    levels = [float(np.float32(level)) for level in tail_levels]
    if len(knots) < 2:
        raise ValueError("need at least two knots to pin a tail")
    left = [level for level in levels if level < knots[0]]
    right = [level for level in levels if level > knots[-1]]
    if len(left) + len(right) != len(levels):
        raise ValueError("every tail level must lie outside the knot range")
    ops = ops_for(knot_values)
    device = ops.device(knot_values)
    columns = []
    if left:
        slope = ops.clip((knot_values[..., 1] - knot_values[..., 0]) / math.log(knots[1] / knots[0]), 0.0, None)
        ratio = ops.asarray([math.log(level / knots[0]) for level in left], dtype=ops.float32, device=device)
        columns.append(knot_values[..., :1] + ops.expand_dims(slope, axis=-1) * ratio)
    if right:
        slope = ops.clip(
            (knot_values[..., -1] - knot_values[..., -2]) / math.log((1.0 - knots[-2]) / (1.0 - knots[-1])), 0.0, None
        )
        ratio = ops.asarray(
            [math.log((1.0 - knots[-1]) / (1.0 - level)) for level in right], dtype=ops.float32, device=device
        )
        columns.append(knot_values[..., -1:] + ops.expand_dims(slope, axis=-1) * ratio)
    return ops.concat(columns, axis=-1)


def extrapolate_quantiles(
    query_quantile_levels: Sequence[float],
    original_quantile_levels: Sequence[float],
    original_values: Float[ArrayT, "*batch original_quantiles"],
    trained_quantile_levels: Sequence[float],
) -> Float[ArrayT, "*batch query_quantiles"]:
    """Return quantiles at the query levels, extrapolating those beyond the trained range.

    ``original_values`` holds the quantiles at ``original_quantile_levels``, as returned
    by a rollout over ``get_rollout_quantile_levels``. Query levels inside the trained
    range are taken from it unchanged; levels outside it are evaluated on exponential
    tails pinned through the outermost trained levels.
    """
    plan = tail_plan(query_quantile_levels, original_quantile_levels, trained_quantile_levels)
    if plan is None:
        return original_values
    ops = ops_for(original_values)
    device = ops.device(original_values)
    knot_values = ops.take(original_values, ops.asarray(plan.knot_columns, dtype=ops.int64, device=device), axis=-1)
    tails = extend_quantile_tails(plan.left + plan.right, plan.knots, knot_values)
    n_left = len(plan.left)
    # The tails bracket the original levels, so the full grid is a plain concatenation.
    full = ops.concat([tails[..., :n_left], original_values, tails[..., n_left:]], axis=-1)
    return ops.take(full, ops.asarray(plan.order, dtype=ops.int64, device=device), axis=-1)


def get_prob_mass_per_quantile_level(quantile_levels: Sequence[float], like: ArrayT) -> Float[ArrayT, " quantiles"]:
    """Normalized probability masses per quantile level (trapezoidal rule), as an array like ``like``.

    Adapted from Chronos-2:
    https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py#L48-L74
    """
    ops = ops_for(like)
    boundaries = ops.asarray([0.0, *quantile_levels, 1.0], dtype=ops.float32, device=ops.device(like))
    prob_mass = (boundaries[2:] - boundaries[:-2]) / 2
    return prob_mass / ops.sum(prob_mass)


def weighted_quantile(
    query_quantile_levels: Sequence[float],
    sample_weights: Float[ArrayT, " num_samples"],
    samples: Float[ArrayT, "*batch num_samples"],
) -> Float[ArrayT, "*batch query_quantiles"]:
    """Compute quantiles from weighted samples using an empirical CDF.

    Adapted from Chronos-2:
    https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/utils.py#L135-L212
    """
    ops = ops_for(samples)
    device = ops.device(samples)
    shape = samples.shape
    flat = ops.reshape(ops.astype(samples, ops.float32), (-1, shape[-1]))
    rows = flat.shape[0]
    order = ops.argsort(flat, axis=-1)
    sorted_samples = ops.take_along_axis(flat, order, axis=-1)
    weights = ops.broadcast_to(ops.expand_dims(sample_weights / ops.sum(sample_weights), axis=0), flat.shape)
    cdf = ops.clip(ops.cumulative_sum(ops.take_along_axis(weights, order, axis=-1), axis=-1), 0.0, 1.0)

    # Interpolate each row on its own CDF, holding the edge samples flat beyond it.
    levels = ops.concat(
        [
            ops.full((rows, 1), 0.0, dtype=ops.float32, device=device),
            cdf,
            ops.full((rows, 1), 1.0, dtype=ops.float32, device=device),
        ],
        axis=-1,
    )
    values = ops.concat([sorted_samples[:, :1], sorted_samples, sorted_samples[:, -1:]], axis=-1)
    query = ops.asarray(list(query_quantile_levels), dtype=ops.float32, device=device)
    # Row-wise searchsorted(side="right"); the standard's searchsorted is one-dimensional.
    upper = ops.sum(ops.astype(ops.expand_dims(levels, axis=-1) <= query, ops.int32), axis=-2)
    upper = ops.clip(upper, None, levels.shape[-1] - 1)
    lower = upper - 1
    lower_levels = ops.take_along_axis(levels, lower, axis=-1)
    upper_levels = ops.take_along_axis(levels, upper, axis=-1)
    lower_values = ops.take_along_axis(values, lower, axis=-1)
    upper_values = ops.take_along_axis(values, upper, axis=-1)
    weight = (query - lower_levels) / (upper_levels - lower_levels)
    weight = ops.where(ops.isnan(weight), ops.zeros_like(weight), weight)
    interpolated = lower_values + weight * (upper_values - lower_values)
    return ops.astype(ops.reshape(interpolated, (*shape[:-1], len(query_quantile_levels))), samples.dtype)


def reduce_rollout_quantiles(
    predictions: Float[ArrayT, "targets query_quantiles predicted_quantiles horizon"],
    predicted_quantile_levels: Sequence[float],
    query_quantile_levels: Sequence[float],
) -> Float[ArrayT, "targets horizon query_quantiles"]:
    """Reduce multi-step quantile predictions of the autoregressive rollout back to the query levels.

    Each query quantile's trajectory predicts the full trained grid; every (trajectory,
    trained level) pair is a sample weighted by the product of both levels' probability
    masses. Approach based on Chronos-2 pipeline's predict method:
    https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py#L450-L648
    """
    ops = ops_for(predictions)
    sample_weights = ops.reshape(
        ops.expand_dims(get_prob_mass_per_quantile_level(predicted_quantile_levels, predictions), axis=1)
        * ops.expand_dims(get_prob_mass_per_quantile_level(query_quantile_levels, predictions), axis=0),
        (-1,),
    )
    targets, n_query, n_predicted, horizon = predictions.shape
    samples = ops.reshape(ops.permute_dims(predictions, (0, 3, 2, 1)), (targets, horizon, n_predicted * n_query))
    return weighted_quantile(query_quantile_levels, sample_weights, samples)


def sanitize_predictions(predictions: Float[ArrayT, "*batch quantiles"]) -> Float[ArrayT, "*batch quantiles"]:
    """Cast to float32 and replace NaN/Inf with 0.0 (logged), so callers never see a poisoned array."""
    ops = ops_for(predictions)
    # Cast first: bf16 has surprising behaviour with ±inf.
    predictions = ops.astype(predictions, ops.float32)
    finite = ops.isfinite(predictions)
    non_finite_count = int(ops.sum(ops.astype(~finite, ops.int32)))
    if non_finite_count > 0:
        logger.warning("replaced %d non-finite prediction values with 0.0", non_finite_count)
        predictions = ops.where(finite, predictions, ops.zeros_like(predictions))
    return predictions

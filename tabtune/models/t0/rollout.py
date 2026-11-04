# Copyright 2026 The Forecasting Company
# The rollout logic of extending quantile predictions in different paths,
# to then reduce them through quantile projection
# follows Chronos-2's inference pipeline
# (https://github.com/amazon-science/chronos-forecasting,
# src/chronos/chronos2/pipeline.py).
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-regressive quantile rollouts, shared by every runtime.

The model can predict multiple time steps in parallel up to a maximum horizon.
Beyond this horizon, the prediction mechanism falls back to an auto-regressive rollout strategy.
"""

import dataclasses
import logging
from collections.abc import Callable, Sequence

from jaxtyping import Float

from ._ops import ArrayT, ops_for
from .data import TimeSeries
from .quantile import interpolate_quantiles, reduce_rollout_quantiles
from .types import MaskType, VariateType, round_up

__all__ = ["expand_prediction_paths", "prepare_rollout_buffer", "rollout", "update_buffer_with_predictions"]

logger = logging.getLogger(__name__)

# One forward pass over a window: ``(window, horizon) -> [rows, horizon, trained_quantiles]`` in data space.
Step = Callable[[TimeSeries, int], object]


def prepare_rollout_buffer(
    series: TimeSeries[ArrayT], prediction_length: int, context_length: int, patch_size: int
) -> TimeSeries[ArrayT]:
    """Build the rollout buffer: padded context + a forecast region.

    Over the forecast region, target and historical-covariate rows are ``WITHHELD``, as the
    model saw them in training, and known future covariates keep their values.
    """
    ops = ops_for(series.variates)
    device = ops.device(series.variates)
    rows = series.variates.shape[0]
    pad_left = (-context_length) % patch_size
    forecast_width = round_up(prediction_length, patch_size)
    known = min(series.seq_len - context_length, forecast_width)  # future cols already in `series`

    row_type = series.variate_type[:, -1:]
    row_group = series.group_ids[:, -1:]
    is_future = row_type == VariateType.FUTURE
    # A historical row is unknown over the forecast region, exactly like a target: withheld,
    # not padding (an all-PAD patch would drop the row out of attention there).
    is_withheld = (row_type == VariateType.TARGET) | (row_type == VariateType.HISTORICAL)
    forecast_values = ops.full((rows, forecast_width), 0, dtype=series.variates.dtype, device=device)
    forecast_mask = ops.where(
        ops.broadcast_to(is_withheld, (rows, forecast_width)),
        ops.full((rows, forecast_width), MaskType.WITHHELD, dtype=ops.int8, device=device),
        ops.full((rows, forecast_width), MaskType.PAD, dtype=ops.int8, device=device),
    )
    if known > 0:
        copy = is_future & (ops.expand_dims(ops.arange(forecast_width, device=device), axis=0) < known)
        unknown = forecast_width - known
        known_values = ops.concat(
            [
                series.variates[:, context_length : context_length + known],
                ops.full((rows, unknown), 0, dtype=series.variates.dtype, device=device),
            ],
            axis=1,
        )
        known_mask = ops.concat(
            [
                ops.astype(series.mask[:, context_length : context_length + known], ops.int8),
                ops.full((rows, unknown), MaskType.PAD, dtype=ops.int8, device=device),
            ],
            axis=1,
        )
        forecast_values = ops.where(copy, known_values, forecast_values)
        forecast_mask = ops.where(copy, known_mask, forecast_mask)

    def pad(value: int, dtype: object) -> object:
        return ops.full((rows, pad_left), value, dtype=dtype, device=device)

    return dataclasses.replace(
        series,
        variates=ops.concat(
            [pad(0, series.variates.dtype), series.variates[:, :context_length], forecast_values], axis=1
        ),
        mask=ops.concat(
            [pad(MaskType.PAD, ops.int8), ops.astype(series.mask[:, :context_length], ops.int8), forecast_mask], axis=1
        ),
        group_ids=ops.concat(
            [
                pad(-1, series.group_ids.dtype),
                series.group_ids[:, :context_length],
                ops.broadcast_to(row_group, (rows, forecast_width)),
            ],
            axis=1,
        ),
        variate_type=ops.concat(
            [
                pad(-1, series.variate_type.dtype),
                series.variate_type[:, :context_length],
                ops.broadcast_to(row_type, (rows, forecast_width)),
            ],
            axis=1,
        ),
    )


def expand_prediction_paths(buffer: TimeSeries[ArrayT], n_paths: int) -> TimeSeries[ArrayT]:
    """Replicate each row into ``n_paths`` trajectories with distinct group ids, one per query quantile."""
    ops = ops_for(buffer.variates)
    rows, width = buffer.variates.shape
    offsets = ops.arange(n_paths, dtype=buffer.group_ids.dtype, device=ops.device(buffer.variates))
    path_groups = ops.expand_dims(buffer.group_ids[:, -1], axis=1) * n_paths + ops.expand_dims(offsets, axis=0)
    path_types = ops.repeat(buffer.variate_type[:, -1], n_paths)
    return dataclasses.replace(
        buffer,
        variates=ops.repeat(buffer.variates, n_paths, axis=0),
        mask=ops.repeat(buffer.mask, n_paths, axis=0),
        group_ids=ops.broadcast_to(ops.reshape(path_groups, (-1, 1)), (rows * n_paths, width)),
        variate_type=ops.broadcast_to(ops.reshape(path_types, (-1, 1)), (rows * n_paths, width)),
    )


def update_buffer_with_predictions(
    buffer: TimeSeries[ArrayT],
    prediction: Float[ArrayT, "targets horizon query_quantiles"],
    at: int,
) -> TimeSeries[ArrayT]:
    """Write target predictions into the forecast region (``VALID``); other rows are left untouched."""
    ops = ops_for(buffer.variates)
    device = ops.device(buffer.variates)
    targets, horizon, n_paths = prediction.shape
    rows, width = buffer.variates.shape
    per_path = ops.reshape(ops.permute_dims(prediction, (0, 2, 1)), (targets * n_paths, horizon))  # (t q) h
    # Row i of the buffer is the k-th target path row: gather its prediction row by rank,
    # so target rows need not be contiguous (typed rows interleave targets and covariates).
    is_target = buffer.variate_type[:, -1] == VariateType.TARGET
    rank = ops.clip(ops.cumulative_sum(ops.astype(is_target, ops.int64)) - 1, 0, targets * n_paths - 1)
    placed = ops.concat(
        [
            ops.full((rows, at), 0, dtype=per_path.dtype, device=device),
            ops.take(per_path, rank, axis=0),
            ops.full((rows, width - at - horizon), 0, dtype=per_path.dtype, device=device),
        ],
        axis=1,
    )
    columns = ops.arange(width, device=device)
    window = ops.expand_dims(is_target, axis=1) & ops.expand_dims((columns >= at) & (columns < at + horizon), axis=0)
    return dataclasses.replace(
        buffer,
        variates=ops.where(window, ops.astype(placed, buffer.variates.dtype), buffer.variates),
        mask=ops.where(
            window, ops.full((), MaskType.VALID, dtype=ops.int8, device=device), ops.astype(buffer.mask, ops.int8)
        ),
    )


def rollout(
    series: TimeSeries[ArrayT],
    step: Step,
    prediction_length: int,
    context_length: int,
    patch_size: int,
    max_horizon: int,
    trained_quantile_levels: Sequence[float],
    query_quantile_levels: Sequence[float],
    target_rows: Sequence[int] | None = None,
) -> Float[ArrayT, "targets prediction_length query_quantiles"]:
    """Forecast ``prediction_length`` steps for every target row of ``series``.

    Predicts up to ``max_horizon`` steps in one pass; beyond, falls back to an
    auto-regressive rollout. Columns of ``series`` past ``context_length`` are known
    future covariates. ``target_rows`` lists the target rows when the caller knows them;
    otherwise they are read from ``series`` (one host read).
    """
    if max_horizon < patch_size or max_horizon % patch_size != 0:
        raise ValueError(f"max_horizon must be a positive multiple of patch_size ({patch_size}), got {max_horizon}")
    ops = ops_for(series.variates)
    device = ops.device(series.variates)
    if target_rows is None:
        target_rows = [
            row for row, kind in enumerate(series.variate_type[:, -1].tolist()) if kind == VariateType.TARGET
        ]
    buffer = prepare_rollout_buffer(series, prediction_length, context_length, patch_size)
    context_width = round_up(context_length, patch_size)

    horizon = min(round_up(prediction_length, patch_size), max_horizon)
    targets = ops.asarray(list(target_rows), dtype=ops.int64, device=device)
    block = ops.take(step(buffer.time_slice(0, context_width + horizon), horizon), targets, axis=0)
    prediction = interpolate_quantiles(query_quantile_levels, trained_quantile_levels, block)
    if prediction_length <= horizon:
        return prediction[:, :prediction_length]

    logger.debug(
        "prediction_length %d exceeds max_horizon %d — continuing autoregressively", prediction_length, max_horizon
    )
    n_paths = len(query_quantile_levels)
    paths = expand_prediction_paths(buffer, n_paths)
    path_targets = ops.asarray(
        [row * n_paths + path for row in target_rows for path in range(n_paths)], dtype=ops.int64, device=device
    )
    predictions = [prediction]
    decoded = horizon
    remaining = prediction_length - horizon
    while remaining > 0:
        previous_width = predictions[-1].shape[1]
        paths = update_buffer_with_predictions(paths, predictions[-1], at=context_width + decoded - previous_width)
        horizon = min(round_up(remaining, patch_size), max_horizon)
        window = paths.time_slice(decoded, context_width + decoded + horizon)
        block = ops.take(step(window, horizon), path_targets, axis=0)
        # (t q) h pq -> t q pq h
        block = ops.permute_dims(
            ops.reshape(block, (len(target_rows), n_paths, horizon, block.shape[-1])), (0, 1, 3, 2)
        )
        predictions.append(reduce_rollout_quantiles(block, trained_quantile_levels, query_quantile_levels))
        decoded += horizon
        remaining -= horizon
    return ops.concat(predictions, axis=1)[:, :prediction_length]

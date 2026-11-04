"""The model input: time series whose variates are gathered in arrays, shared by every runtime."""

import dataclasses
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Generic

from jaxtyping import Bool, Float, Int

from ._ops import ArrayOps, ArrayT, ops_for
from .types import MaskType, VariateType, round_up

__all__ = ["MaskType", "TimeSeries", "VariateType", "batch_series", "mask_nan_as_missing", "round_up"]


def mask_nan_as_missing(values: Float[ArrayT, "rows time"]) -> Int[ArrayT, "rows time"]:
    """Read every NaN in ``values`` as an absent observation, and every other cell as valid.

    This is what a context without a mask means: NaN is missing data. Padding a
    shorter series is the case this cannot express — a padded cell holds no
    observation at all rather than a missing one, and only a mask can say so.
    """
    ops = ops_for(values)
    return ops.astype(ops.where(ops.isnan(values), MaskType.MISSING, MaskType.VALID), ops.int8)


@dataclasses.dataclass
class TimeSeries(Generic[ArrayT]):
    """Model input representing one or multiple time series whose variates are gathered in an array.

    Rows are variates; ``group_ids`` says which rows belong to one series, and
    ``variate_type`` what role each row plays. Every field is ``(variates, time)``.
    """

    variates: Float[ArrayT, "variates time"]
    mask: Int[ArrayT, "variates time"]
    group_ids: Int[ArrayT, "variates time"]
    variate_type: Int[ArrayT, "variates time"]

    @cached_property
    def valid_mask(self) -> Bool[ArrayT, "variates time"]:
        return self.mask == MaskType.VALID

    @property
    def device(self) -> Any:
        """Where the arrays live; ``None`` on runtimes whose arrays have no device."""
        return ops_for(self.variates).device(self.variates)

    @property
    def seq_len(self) -> int:
        return self.variates.shape[1]

    def to(self, device: Any) -> "TimeSeries[ArrayT]":
        """The same series on ``device``; a no-op where it already lives there, or where arrays have no device."""
        ops = ops_for(self.variates)
        if device is None or device == ops.device(self.variates):
            return self
        return dataclasses.replace(
            self, **{field.name: ops.to_device(getattr(self, field.name), device) for field in dataclasses.fields(self)}
        )

    def time_slice(self, start: int, stop: int) -> "TimeSeries[ArrayT]":
        """Return the ``[start, stop)`` window along the time axis."""
        return dataclasses.replace(
            self, **{field.name: getattr(self, field.name)[:, start:stop] for field in dataclasses.fields(self)}
        )

    @classmethod
    def from_array(
        cls,
        targets: Float[ArrayT, "batch time"] | Float[ArrayT, "batch variates time"],
        future_covariates: Float[ArrayT, "batch future_variates context_plus_horizon"] | None = None,
        mask: Int[ArrayT, "batch time"] | Int[ArrayT, "batch variates time"] | None = None,
        group_ids: Int[ArrayT, " rows"] | None = None,
        horizon: int = 0,
    ) -> "TimeSeries[ArrayT]":
        """Build model input from a target context and optional future covariates.

        ``mask`` holds ``MaskType`` values shaped like ``targets``: ``MISSING``
        for an absent observation, ``PAD`` for a cell that only pads a shorter
        series out to the batch's width. Only all-``PAD`` patches leave attention.
        Without it every NaN in ``targets`` is read as an absent observation.

        ``group_ids`` holds one id per row of the flattened ``targets``; rows
        sharing an id are variates of one series and attend to one another.
        Without it every row of a ``(B, T)`` target is its own series.

        ``horizon`` extends the target rows with that many ``WITHHELD`` timesteps,
        marking the region to predict. ``future_covariates`` imply it from their width.

        All arrays belong to one runtime; the series is built where ``targets`` lives.

        Raises:
            ValueError:
                - ``targets`` is not 2-/3-D.
                - ``future_covariates`` is not ``(B, F, >= T)``.
                - ``mask`` mismatches ``targets``' shape, sets ``WITHHELD``, or
                  marks a NaN cell ``VALID``.
                - ``group_ids`` is not one non-negative id per row, or comes
                  alongside ``future_covariates``.
        """
        if targets.ndim not in (2, 3):
            raise ValueError(f"targets must be (B, T) or (B, V, T), got shape {tuple(targets.shape)}")
        ops = ops_for(targets)
        device = ops.device(targets)
        batch_size, width = targets.shape[0], targets.shape[-1]
        n_variates = targets.shape[1] if targets.ndim == 3 else 1
        n_target = batch_size * n_variates
        values = ops.reshape(targets, (n_target, width))
        is_nan = ops.isnan(values)
        if mask is None:
            target_mask = ops.astype(ops.where(is_nan, MaskType.MISSING, MaskType.VALID), ops.int8)
        else:
            if tuple(mask.shape) != tuple(targets.shape):
                raise ValueError(
                    f"mask must have the same shape as targets {tuple(targets.shape)}, got {tuple(mask.shape)}"
                )
            target_mask = ops.astype(ops.reshape(mask, (n_target, width)), ops.int8)
            # WITHHELD marks the region the model must predict; the rollout owns it.
            if bool(ops.any((target_mask < MaskType.VALID) | (target_mask > MaskType.CENSORED))):
                raise ValueError("mask values must be MaskType.VALID, PAD, MISSING or CENSORED; WITHHELD is reserved")
            if bool(ops.any(is_nan & (target_mask == MaskType.VALID))):
                raise ValueError("mask marks NaN cells VALID; mark them MaskType.MISSING or MaskType.PAD")
        values = ops.where(is_nan, ops.zeros_like(values), values)

        sample_ids = ops.arange(batch_size, dtype=ops.int64, device=device)
        if group_ids is None:
            row_groups = ops.repeat(sample_ids, n_variates)
        else:
            if future_covariates is not None:
                raise ValueError("group_ids cannot be combined with future_covariates")
            if group_ids.ndim != 1 or group_ids.shape[0] != n_target:
                raise ValueError(
                    f"group_ids must hold one id per target row ({n_target}), got {tuple(group_ids.shape)}"
                )
            # -1 is the padding sentinel the patcher and the attention masks rely on.
            if bool(ops.any(group_ids < 0)):
                raise ValueError("group_ids must be non-negative")
            row_groups = ops.astype(group_ids, ops.int64)
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")

        future_rows = None
        if future_covariates is not None and future_covariates.shape[1] != 0:
            total_width = future_covariates.shape[2]
            if future_covariates.ndim != 3 or future_covariates.shape[0] != batch_size or total_width < width:
                raise ValueError(
                    f"future_covariates must be (B={batch_size}, F, T+H>=T={width}), "
                    f"got shape {tuple(future_covariates.shape)}"
                )
            # The covariates span context + horizon, so they imply the horizon.
            implied_horizon = total_width - width
            if horizon and horizon != implied_horizon:
                raise ValueError(
                    f"horizon={horizon} contradicts future_covariates, which span "
                    f"{implied_horizon} steps past the context"
                )
            horizon = implied_horizon
            # Future rows span the full [0, T+H), VALID throughout (known context AND horizon).
            n_future = future_covariates.shape[1]
            rows = batch_size * n_future
            future_values = ops.reshape(future_covariates, (rows, total_width))
            future_nan = ops.isnan(future_values)
            future_rows = (
                ops.where(future_nan, ops.zeros_like(future_values), future_values),
                ops.astype(ops.where(future_nan, MaskType.MISSING, MaskType.VALID), ops.int8),
                ops.broadcast_to(ops.expand_dims(ops.repeat(sample_ids, n_future), axis=1), (rows, total_width)),
                ops.full((rows, total_width), VariateType.FUTURE, dtype=ops.int64, device=device),
            )

        # Target rows, extended over the horizon with WITHHELD: the region to predict.
        full_width = width + horizon
        target_rows = (
            ops.concat([values, ops.full((n_target, horizon), 0, dtype=values.dtype, device=device)], axis=1),
            ops.concat(
                [target_mask, ops.full((n_target, horizon), MaskType.WITHHELD, dtype=ops.int8, device=device)], axis=1
            ),
            ops.broadcast_to(ops.expand_dims(row_groups, axis=1), (n_target, full_width)),
            ops.full((n_target, full_width), VariateType.TARGET, dtype=ops.int64, device=device),
        )
        if future_rows is None:
            return cls(*target_rows)
        return cls(
            *(ops.concat([target, future], axis=0) for target, future in zip(target_rows, future_rows, strict=True))
        )

    @classmethod
    def batch(cls, series: Sequence["TimeSeries[ArrayT]"]) -> "TimeSeries[ArrayT]":
        """Batch T0 inputs with different widths and variate counts.

        Inputs are right-aligned so their forecast horizons remain aligned.
        Shorter inputs receive ``PAD`` cells on the left, and group ids are
        remapped so different inputs never attend to one another. This supports
        complete inputs containing target, historical, and future variates.
        Padding cells use ``-1`` for both group id and variate type metadata.

        Args:
            series: Non-empty sequence of T0 inputs of one runtime. Each row must have
                a group id and variate type at its final timestep.

        Returns:
            One ``TimeSeries`` containing all input rows, on the first input's
            device.

        Raises:
            ValueError: The sequence is empty or an input is malformed.
        """
        if not series:
            raise ValueError("series must hold at least one TimeSeries")
        ops = ops_for(series[0].variates)
        device = ops.device(series[0].variates)
        width = max(item.seq_len for item in series)
        parts, group_offset = [], 0
        for item in series:
            shape = tuple(item.variates.shape)
            if item.variates.ndim != 2 or any(
                tuple(values.shape) != shape for values in (item.mask, item.group_ids, item.variate_type)
            ):
                raise ValueError("each TimeSeries field must have the same two-dimensional shape")
            if item.seq_len == 0 or bool(ops.any(item.mask[:, -1] == MaskType.PAD)):
                raise ValueError("each TimeSeries row must end with a non-PAD timestep")
            # A handful of ids per input: remap on the host, in sorted order.
            row_groups = [int(group) for group in item.group_ids[:, -1].tolist()]
            if min(row_groups) < 0:
                raise ValueError("each TimeSeries row must end with a non-negative group id")
            remap = {group: group_offset + rank for rank, group in enumerate(sorted(set(row_groups)))}
            group_offset += len(remap)

            rows, pad = shape[0], width - item.seq_len
            item_mask = ops.to_device(ops.astype(item.mask, ops.int8), device)
            is_real = item_mask != MaskType.PAD
            remapped = ops.asarray([[remap[group]] for group in row_groups], dtype=ops.int64, device=device)
            parts.append(
                (
                    ops.concat(
                        [
                            ops.full((rows, pad), 0, dtype=ops.float32, device=device),
                            ops.to_device(ops.astype(item.variates, ops.float32), device),
                        ],
                        axis=1,
                    ),
                    ops.concat([ops.full((rows, pad), MaskType.PAD, dtype=ops.int8, device=device), item_mask], axis=1),
                    ops.concat(
                        [
                            ops.full((rows, pad), -1, dtype=ops.int64, device=device),
                            ops.where(is_real, ops.broadcast_to(remapped, (rows, item.seq_len)), -1),
                        ],
                        axis=1,
                    ),
                    ops.concat(
                        [
                            ops.full((rows, pad), -1, dtype=ops.int64, device=device),
                            ops.where(is_real, ops.to_device(ops.astype(item.variate_type, ops.int64), device), -1),
                        ],
                        axis=1,
                    ),
                )
            )
        return cls(*(ops.concat([part[i] for part in parts], axis=0) for i in range(4)))


def batch_series(
    series: Sequence[Any],
    ops: ArrayOps | None = None,
) -> tuple[Float[Any, "variates time"], Int[Any, "variates time"], Int[Any, " variates"]]:
    """Stack time series of potentially different lengths and variate counts into one model input.

    Each entry is one series, shaped ``(T,)`` or ``(V, T)``: an array of a runtime, a
    NumPy array or a list. The batch is built with ``ops`` (each runtime's
    ``batch_series`` passes its own), or else with the runtime of the first entry. Their variates are
    stacked along a single ``variates`` axis and right-aligned to the longest
    entry: cells that only widen a shorter entry are ``PAD``, NaN observations
    are ``MISSING``, and the returned group ids say which rows came from the
    same series, so its variates keep attending to one another. The batch is
    built on the first entry's device.

    Raises:
        ValueError: ``series`` is empty, or an entry is not ``(T,)`` or ``(V, T)``, or
            has no time step.
    """
    if not series:
        raise ValueError("series must hold at least one entry")
    ops = ops or ops_for(series[0])
    # Arrays keep their device: a batch of accelerator arrays never round-trips through the CPU.
    first = ops.asarray(series[0], dtype=ops.float32)
    device = ops.device(first)
    rows = [ops.to_device(ops.asarray(entry, dtype=ops.float32), device) for entry in series]
    rows = [ops.expand_dims(row, axis=0) if row.ndim == 1 else row for row in rows]
    if any(row.ndim != 2 for row in rows):
        raise ValueError("each series must be shaped (T,) or (V, T)")
    if any(row.shape[-1] < 1 for row in rows):
        raise ValueError("each series must contain at least one time step")
    width = max(row.shape[-1] for row in rows)
    contexts, masks, groups = [], [], []
    for group, row in enumerate(rows):
        n, pad = row.shape[0], width - row.shape[-1]
        # Two layers, in this order: what the data says, then what batching added. A NaN is
        # an absent observation wherever it appears; the cells that only widen a shorter
        # entry hold no observation at all.
        contexts.append(ops.concat([ops.full((n, pad), 0, dtype=ops.float32, device=device), row], axis=1))
        masks.append(
            ops.concat(
                [ops.full((n, pad), MaskType.PAD, dtype=ops.int8, device=device), mask_nan_as_missing(row)], axis=1
            )
        )
        groups.append(ops.full((n,), group, dtype=ops.int64, device=device))
    return ops.concat(contexts, axis=0), ops.concat(masks, axis=0), ops.concat(groups, axis=0)

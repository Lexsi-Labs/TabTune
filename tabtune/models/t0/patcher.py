"""Patchify time series into contiguous, non-overlapping patches, for every runtime."""

import dataclasses
from typing import Any

from jaxtyping import Shaped

from ._ops import ArrayT, ops_for
from .types import MaskType, VariateType, round_up

__all__ = ["Patcher"]


class Patcher:
    """Align a ``TimeSeries`` to the patch grid and reshape per-time-step arrays to per-patch form.

    Stateless: no parameters, no state-dict keys.
    """

    def __init__(self, patch_size: int):
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.patch_size = patch_size

    def context_end(self, series: Any) -> int:
        """Index where the forecast region begins.

        The first ``WITHHELD`` timestep across target rows, or ``seq_len`` when nothing
        is withheld. Reads the mask back to the host once.
        """
        ops = ops_for(series.variates)
        is_target = ops.expand_dims(series.variate_type[:, -1] == VariateType.TARGET, axis=1)
        withheld = ops.any(is_target & (series.mask == MaskType.WITHHELD), axis=0)
        if not bool(ops.any(withheld)):
            return series.seq_len
        return int(ops.argmax(ops.astype(withheld, ops.int32)))

    def pad(self, series: Any, context_end: int | None = None) -> Any:
        """Pad a series so the context and the forecast region both occupy whole patches.

        The model predicts one whole patch at a time, so both ends have to line up. Padding
        on the left extends the context backwards until it starts on a patch boundary, which
        holds the patch grid fixed under the forecast window. Padding on the right fills out
        the forecast region, because reading ``H`` future steps means reading
        ``ceil(H / patch_size)`` whole patches after the context.

        Left padding is inert: ``MaskType.PAD`` (1), value 0.0 and ``-1`` sentinels. Right
        padding belongs to the forecast region, so it carries each row's own group id and
        variate type, and is ``MaskType.WITHHELD`` (4) on target rows and ``PAD`` elsewhere.

        ``context_end`` defaults to ``context_end(series)``, one host read; pass it where
        the layout is already known, e.g. inside a compiled graph.

        Reading the target row's mask, where 0 is an observation, 1 is padding and 4 is a
        step to predict -- a 3-step context with nothing to predict pads on the left only::

            >>> import torch
            >>> from t0 import TimeSeries
            >>> patcher = Patcher(patch_size=2)
            >>> patcher.pad(TimeSeries.from_array(torch.zeros(1, 1, 3))).mask[0].tolist()
            [1, 0, 0, 0]

        A 4-step context is already aligned, so a 1-step horizon pads on the right only,
        rounding the forecast region up to a whole patch::

            >>> series = TimeSeries.from_array(torch.zeros(1, 1, 4), horizon=1)
            >>> patcher.pad(series).mask[0].tolist()
            [0, 0, 0, 0, 4, 4]

        A 3-step context with a 1-step horizon needs both::

            >>> series = TimeSeries.from_array(torch.zeros(1, 1, 3), horizon=1)
            >>> patcher.pad(series).mask[0].tolist()
            [1, 0, 0, 0, 4, 4]

        An aligned input is returned unchanged.
        """
        if context_end is None:
            context_end = self.context_end(series)
        horizon = series.seq_len - context_end
        pad_left = (-context_end) % self.patch_size
        pad_right = round_up(horizon, self.patch_size) - horizon
        if pad_left == 0 and pad_right == 0:
            return series

        ops = ops_for(series.variates)
        rows = series.variates.shape[0]
        device = ops.device(series.variates)
        mask = ops.astype(series.mask, ops.int8)
        row_type = series.variate_type[:, -1:]
        left = (
            ops.full((rows, pad_left), 0, dtype=series.variates.dtype, device=device),
            ops.full((rows, pad_left), MaskType.PAD, dtype=ops.int8, device=device),
            ops.full((rows, pad_left), -1, dtype=series.group_ids.dtype, device=device),
            ops.full((rows, pad_left), -1, dtype=series.variate_type.dtype, device=device),
        )
        right = (
            ops.full((rows, pad_right), 0, dtype=series.variates.dtype, device=device),
            ops.where(
                ops.broadcast_to(row_type == VariateType.TARGET, (rows, pad_right)),
                ops.full((rows, pad_right), MaskType.WITHHELD, dtype=ops.int8, device=device),
                ops.full((rows, pad_right), MaskType.PAD, dtype=ops.int8, device=device),
            ),
            ops.broadcast_to(series.group_ids[:, -1:], (rows, pad_right)),
            ops.broadcast_to(row_type, (rows, pad_right)),
        )
        fields = (series.variates, mask, series.group_ids, series.variate_type)
        variates, mask, group_ids, variate_type = (
            ops.concat([left[i], fields[i], right[i]], axis=-1) for i in range(4)
        )
        return dataclasses.replace(series, variates=variates, mask=mask, group_ids=group_ids, variate_type=variate_type)

    def patch(self, x: Shaped[ArrayT, "variates time"]) -> Shaped[ArrayT, "variates patches patch_size"]:
        """Reshape an aligned per-time-step array to per-patch form."""
        return ops_for(x).reshape(x, (*x.shape[:-1], x.shape[-1] // self.patch_size, self.patch_size))

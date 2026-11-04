# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm._memory import split_size
from tabtune.models.kumo._sdm.processing import Processor
from tabtune.models.kumo._sdm.processing.numerical._stats import _isfinite
from tabtune.models.kumo._sdm.processing.numerical.quantile import _batched_interp


class RankGaussian(Processor):
    """Map interpolated empirical mid-ranks to standard normal quantiles.

    Each fitted value receives probability ``(L + R) / (2 * N)``, where
    ``L`` and ``R`` count finite fitted values strictly below and at or below
    it, and ``N`` is the number of finite fitted values. Ties share a rank.
    Query probabilities interpolate between retained fitted value-rank pairs
    and clamp to the endpoint probabilities.

    When ``max_knots`` is set, each column retains at most that many pairs,
    selected at evenly spaced normal quantiles between the fitted extremes.

    NaN and infinite values are ignored during fitting. NaNs are preserved
    during transformation. Constant columns map to zero; columns without
    finite fitted values produce NaNs.

    Args:
        max_knots: Maximum number of knots per column, at least ``2``. If
            ``None``, retain all fitted values.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(self, *, max_knots: int | None = None) -> None:
        super().__init__()
        if max_knots is not None and max_knots < 2:
            raise ValueError("max_knots must be at least 2.")
        self.max_knots = max_knots
        self.register_buffer("_values", torch.empty(0))
        self.register_buffer("_probabilities", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        dtype = torch.promote_types(numerical.dtype, torch.float32)
        # Columns are fitted on their own, so chunks of columns bound the
        # sorting and ranking temporaries of about ten values per cell.
        size = split_size(
            num_items=numerical.size(-1),
            item_bytes=numerical[..., :1].numel() * 10 * dtype.itemsize,
            device=numerical.device,
        )
        knots = []
        for chunk in numerical.split(size, dim=-1):
            num_rows = chunk.size(-2)
            max_knots = num_rows if self.max_knots is None else self.max_knots
            columns = chunk.to(dtype).movedim(-1, -2)  # [..., C, N]
            finite = _isfinite(columns)
            count = finite.sum(dim=-1, keepdim=True)  # [..., C, 1]
            values = columns.masked_fill(~finite, torch.inf)
            values = values.sort(dim=-1).values.contiguous()
            left = torch.searchsorted(values, values, right=False)
            right = torch.searchsorted(values, values, right=True)
            probabilities = (left + right).to(values.dtype) / (
                2 * count.clamp_min(1)
            )
            # Rounding to one would give an infinite normal quantile.
            probabilities.clamp_(max=1 - torch.finfo(dtype).eps)
            last = (count - 1).clamp_min(0)
            rows = torch.arange(num_rows, device=values.device)

            if num_rows <= max_knots:
                positions = rows.minimum(last)
            else:
                # Index among distinct column values; a value starts a new
                # one at its first occurrence.
                distinct = (left == rows).cumsum(dim=-1).sub_(1)
                num_distinct = distinct.gather(-1, last) + 1
                # Knots at the first occurrence of each distinct value, padded
                # with the largest one.
                steps = torch.arange(max_knots, device=values.device)
                positions = torch.searchsorted(
                    distinct,
                    steps.minimum(num_distinct - 1),
                )
                # With more distinct values than knots, select rows whose
                # mid-ranks are closest to normal quantiles spaced evenly
                # between the extremes. Mid-ranks never decrease along rows.
                lower = torch.special.ndtri(probabilities[..., :1])
                upper = torch.special.ndtri(probabilities.gather(-1, last))
                quantiles = torch.special.ndtr(
                    lower.lerp(
                        upper,
                        steps.to(probabilities.dtype) / (max_knots - 1),
                    )
                )
                above = torch.searchsorted(probabilities, quantiles).minimum(
                    last
                )
                below = (above - 1).clamp_(min=0)
                closer = (quantiles - probabilities.gather(-1, below)) < (
                    probabilities.gather(-1, above) - quantiles
                )
                spaced = torch.where(closer, below, above)
                spaced[..., :1] = 0
                spaced[..., -1:] = last
                positions = torch.where(
                    num_distinct <= max_knots,
                    positions,
                    spaced,
                )

            missing = count == 0
            knots.append(
                (
                    values.gather(-1, positions).masked_fill_(
                        missing, torch.nan
                    ),
                    probabilities.gather(-1, positions).masked_fill_(
                        missing,
                        torch.nan,
                    ),
                )
            )

        values, probabilities = zip(*knots, strict=True)
        self._values = torch.cat(values, dim=-2)
        self._probabilities = torch.cat(probabilities, dim=-2)

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        columns = numerical.movedim(-1, -2)  # [..., C, R]
        values = self._values.flatten(end_dim=-2)
        probabilities = self._probabilities.flatten(end_dim=-2)
        output = torch.empty_like(numerical)
        # Rows transform on their own, so chunks of rows bound the
        # interpolation temporaries of about a dozen values per cell.
        size = split_size(
            num_items=columns.size(-1),
            item_bytes=columns[..., :1].numel() * 12 * values.itemsize,
            device=numerical.device,
        )
        for rows, out in zip(
            columns.split(size, dim=-1),
            output.split(size, dim=-2),
            strict=True,
        ):
            quantiles = _batched_interp(
                rows.flatten(end_dim=-2).to(values.dtype).contiguous(),
                values,
                probabilities,
            )
            normal = torch.special.ndtri(quantiles).reshape(rows.shape)
            out.copy_(normal.masked_fill_(rows.isnan(), torch.nan).mT)
        return table.replace_blocks(numerical=output)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"max_knots={self.max_knots})"
        )

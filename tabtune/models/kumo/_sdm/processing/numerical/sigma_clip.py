# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm._memory import split_size
from tabtune.models.kumo._sdm.processing import Processor
from tabtune.models.kumo._sdm.processing.numerical._stats import _isfinite


class ClipSigma(Processor):
    """Two-stage z-score outlier clipping with soft logarithmic bounds.

    The first pass masks values outside the initial z-score bounds, then the
    second pass refits bounds on the remaining values. The transform applies
    logarithmic soft clipping instead of hard truncation. NaN and infinite
    values are ignored when fitting statistics and preserved during the
    transform.

    Args:
        threshold: Positive z-score multiplier setting how many standard
            deviations from the mean mark the soft clipping bounds.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        threshold: float = 4.0,
    ) -> None:
        super().__init__()
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        self.threshold = threshold
        self.register_buffer("lower_bound", torch.empty(0))
        self.register_buffer("upper_bound", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        numerical = table.numerical
        assert not numerical.requires_grad
        finite = _isfinite(numerical)
        count_finite = finite.sum(dim=-2, keepdim=True)
        finite_or_nan = numerical.masked_fill(~finite, torch.nan)

        # Compute finite mean and standard deviation:
        mean = finite_or_nan.nansum(-2, keepdim=True).div_(count_finite)
        mean.masked_fill_(mean.isnan(), 0.0)

        var = finite_or_nan.sub_(mean).square_().nansum(-2, keepdim=True)
        del finite_or_nan
        var /= (count_finite - 1).clamp_(min=1)
        std = var.sqrt().clamp(min=1e-6)

        # Find values within range (non-finite values are never kept):
        lower = mean - self.threshold * std
        upper = mean + self.threshold * std
        keep = (numerical >= lower).logical_and_(numerical <= upper)
        keep &= finite
        del finite
        count = keep.sum(dim=-2, keepdim=True)

        # Compute mean and standard deviation of kept values:
        kept_mean = torch.where(keep, numerical, 0.0).sum(-2, keepdim=True)
        kept_mean /= count.clamp(min=1)

        centered = numerical.sub(kept_mean).masked_fill_(~keep, 0.0)
        denominator = (count - 1).clamp(min=1)
        kept_var = centered.square_().sum(-2, keepdim=True) / denominator
        del centered
        kept_std = kept_var.sqrt().clamp(min=1e-6)

        has_kept = count > 0
        mean = torch.where(has_kept, kept_mean, mean)
        std = torch.where(has_kept, kept_std, std)

        self.lower_bound = mean - self.threshold * std
        self.upper_bound = mean + self.threshold * std

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        dtype = torch.promote_types(numerical.dtype, self.lower_bound.dtype)
        if torch.is_grad_enabled():
            log_abs = numerical.abs().log1p().to(dtype)
            clipped = torch.maximum(self.lower_bound - log_abs, numerical)
            clipped = torch.minimum(log_abs + self.upper_bound, clipped)
            return table.replace_blocks(numerical=clipped)

        out = torch.empty_like(numerical, dtype=dtype)
        # Chunks of rows bound the temporary besides the output.
        size = split_size(
            num_items=numerical.size(-2),
            item_bytes=math.prod(out.size()[:-2])
            * out.size(-1)
            * out.element_size(),
            device=out.device,
        )
        for inp, clipped in zip(
            numerical.split(size, dim=-2),
            out.split(size, dim=-2),
            strict=True,
        ):
            log_abs = inp.abs().log1p_().to(dtype)
            torch.sub(self.lower_bound, log_abs, out=clipped)
            torch.maximum(clipped, inp, out=clipped)
            torch.minimum(log_abs.add_(self.upper_bound), clipped, out=clipped)
        return table.replace_blocks(numerical=out)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"threshold={self.threshold})"
        )

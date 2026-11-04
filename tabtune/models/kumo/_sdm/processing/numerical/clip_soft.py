# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.processing import Processor
from tabtune.models.kumo._sdm.processing.numerical._stats import _isfinite


class ClipSoft(Processor):
    r"""Soft-clip numerical values toward ``±max_absolute_value``.

    Values pass through

    .. math::

        \frac{x}{\sqrt{1 + (x / b)^{2}}}

    which maps monotonically onto ``±max_absolute_value`` (``b``) without a
    hard edge. NaN is preserved. ``±Inf`` maps to ``±max_absolute_value``.
    Unlike :class:`~sdm.processing.ClipSigma`, the bound is fixed rather
    than fitted from per-column statistics.

    Args:
        max_absolute_value: Bound of the soft clip.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(self, max_absolute_value: float) -> None:
        super().__init__()
        if not math.isfinite(max_absolute_value) or max_absolute_value <= 0:
            raise ValueError("max_absolute_value must be finite and positive.")
        self.max_absolute_value = max_absolute_value

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        assert not numerical.requires_grad
        bound = self.max_absolute_value
        unit = numerical.div(bound).abs_()
        root = unit.square().add_(1).sqrt_()
        unit.div_(root).masked_fill_(~_isfinite(root), 1.0)
        del root
        clipped = unit.mul_(bound).mul_(numerical.sign())
        torch.where(numerical.isnan(), numerical, clipped, out=clipped)
        return table.replace_blocks(numerical=clipped)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}"
            f"({self.max_absolute_value})"
        )

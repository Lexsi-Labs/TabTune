# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.processing import Processor


class ClipQuantiles(Processor):
    """Clamp feature columns to fitted quantile bounds.

    Values outside the fitted bounds are discarded, so this processor is not
    invertible. Quantile transform bounds are fitted independently for each
    feature column.

    Args:
        q_low: Lower quantile in ``[0, 1]`` used as the per-column lower bound.
        q_high: Upper quantile in ``[0, 1]`` used as the per-column upper
            bound. Must satisfy ``0 <= q_low <= q_high <= 1``.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        q_low: float = 0.0,
        q_high: float = 1.0,
    ) -> None:
        super().__init__()
        if q_low > q_high:
            raise ValueError("q_low and q_high must satisfy q_low <= q_high.")
        self.q_low = q_low
        self.q_high = q_high
        self.register_buffer("lower_bound", torch.empty(0))
        self.register_buffer("upper_bound", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        quantiles = numerical.new_tensor([self.q_low, self.q_high])
        q_low, q_high = torch.quantile(
            numerical,
            quantiles,
            dim=-2,
            keepdim=True,
        )
        self.lower_bound = q_low
        self.upper_bound = q_high

    def _transform(self, table: TableTensor) -> TableTensor:
        """Clamp ``table`` to the fitted lower and upper bounds."""
        numerical = table.numerical.clamp(
            min=self.lower_bound,
            max=self.upper_bound,
        )
        return table.replace_blocks(numerical=numerical)

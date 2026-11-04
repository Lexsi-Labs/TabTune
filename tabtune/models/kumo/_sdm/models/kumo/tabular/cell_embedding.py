# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from typing import Literal, cast

import torch
from torch import Tensor
from torch.nn import Linear

from tabtune.models.kumo._sdm.cache import Cache
from tabtune.models.kumo._sdm.models.tabfm.cell_embedding import CellEmbedding as TabFMCellEmbedding


class CellEmbedding(TabFMCellEmbedding):
    def __init__(
        self,
        channels: int,
        group_size: int,
        num_frequencies: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            channels=channels,
            group_size=group_size,
            num_frequencies=num_frequencies,
            device=device,
            dtype=dtype,
        )
        self.nan_lin = Linear(
            group_size,
            channels,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(  # type: ignore
        self,
        x: Tensor,  # [..., R, C]
        categorical_mask: Tensor,  # [..., C]
        *,
        train_size: int | None = None,
        cache: Cache | None = None,
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
    ) -> Tensor:  # [..., R, C, D]
        missing = x.isnan()

        if cache is not None and cache.is_replaying:
            impute_mean = cast(Tensor, cache["cell_impute_mean"])
        else:
            assert train_size is not None
            impute_mean = x[..., :train_size, :].nanmean(dim=-2, keepdim=True)
            impute_mean = impute_mean.masked_fill(impute_mean.isnan(), 0.0)

        if cache is not None and cache.is_recording:
            cache["cell_impute_mean"] = impute_mean

        x = torch.where(missing, impute_mean, x)

        return super().forward(
            x,
            categorical_mask,
            batch_size_limit=batch_size_limit,
            out=out,
            missing=missing,
        )

    def _forward(  # type: ignore
        self,
        x: Tensor,  # [..., R, C, G]
        freq: Tensor,  # [..., 1, C, G, F]
        weight: Tensor,  # [..., 1, C, G, D, 2F]
        bias: Tensor,  # [..., 1, C, D]
        *,
        missing: Tensor,  # [..., R, C, G]
        out: Tensor | None = None,
    ) -> Tensor:

        out = super()._forward(x, freq, weight, bias, out=out)
        out += self.nan_lin(missing.to(out.dtype))
        return out

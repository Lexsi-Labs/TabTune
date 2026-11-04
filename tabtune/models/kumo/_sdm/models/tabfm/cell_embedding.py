# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import Linear

from tabtune.models.kumo._sdm._memory import chunk_memory_limit


class CellEmbedding(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        group_size: int,
        num_frequencies: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.channels = channels
        self.group_size = group_size
        self.num_frequencies = num_frequencies

        self.num_freq = torch.nn.Parameter(
            torch.randn(group_size, num_frequencies, **factory_kwargs)
        )
        self.cat_freq = torch.nn.Parameter(
            torch.randn(group_size, num_frequencies, **factory_kwargs)
        )

        self.num_lin = Linear(2 * num_frequencies, channels, **factory_kwargs)
        self.cat_lin = Linear(2 * num_frequencies, channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., R, C],
        categorical_mask: Tensor,  # [..., C],
        *,
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
        **kwargs: Tensor,  # [..., R, C],
    ) -> Tensor:  # [..., R, C, D]
        if out is not None and torch.is_grad_enabled():
            raise RuntimeError(
                "'out' is only supported when gradients are disabled"
            )

        *B, R, C = x.size()

        # Feature grouping:
        index = torch.arange(C, device=x.device)
        shift = 2 ** torch.arange(self.group_size, device=x.device) - 1
        index = (index.view(C, 1) + shift.view(1, self.group_size)) % C
        x = x[..., index]  # [..., R, C, G]
        kwargs = {key: value[..., index] for key, value in kwargs.items()}

        # Compute Fourier features per semantic type (row-agnostic):
        categorical_mask = categorical_mask[..., None, index]  # [..., 1, C, G]
        freq = torch.where(
            categorical_mask.unsqueeze(-1),  # [..., 1, C, G, 1]
            self.cat_freq.to(torch.float32),  # [G, F]
            self.num_freq.to(torch.float32),  # [G, F]
        )  # [..., 1, C, G, F]

        # Gather weights and biases per semantic type (row-agnostic):
        dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            and not torch.is_grad_enabled()
            else self.num_lin.weight.dtype
        )

        weight = torch.where(
            categorical_mask[..., None, None],  # [..., G, 1, 1]
            self.cat_lin.weight.to(dtype).view(
                *(1,) * categorical_mask.dim(), *self.cat_lin.weight.size()
            ),
            self.num_lin.weight.to(dtype).view(
                *(1,) * categorical_mask.dim(), *self.num_lin.weight.size()
            ),
        )  # [..., 1, C, G, D, 2F]

        bias = torch.where(
            categorical_mask[..., None],
            self.cat_lin.bias.view(*(1,) * categorical_mask.dim(), -1),
            self.num_lin.bias.view(*(1,) * categorical_mask.dim(), -1),
        ).sum(dim=-2)  # [..., 1, C, D]
        bias = bias.to(dtype)

        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return self._forward(x, freq, weight, bias, out=out, **kwargs)

        if batch_size_limit == "auto":
            batch_size_limit = None
            if x.is_cuda:
                if torch.is_autocast_enabled(x.device.type):
                    element_size = torch.empty(
                        size=(),
                        dtype=torch.get_autocast_dtype(x.device.type),
                    ).element_size()
                else:
                    element_size = x.element_size()

                bytes_per_example = (
                    2 * self.group_size * self.num_frequencies * element_size
                    + 2 * self.channels * element_size
                )

                fixed_bytes = (
                    freq.numel() * freq.element_size()
                    + weight.numel() * weight.element_size()
                    + bias.numel() * bias.element_size()
                )

                memory_limit = chunk_memory_limit(x.device) - fixed_bytes
                batch_size_limit = memory_limit // max(bytes_per_example, 1)
                batch_size_limit = max(batch_size_limit, 1)

        if batch_size_limit is None:
            return self._forward(x, freq, weight, bias, out=out, **kwargs)

        if out is None:
            out = x.new_empty(
                (*B, R, C, self.channels),
                dtype=torch.get_autocast_dtype(x.device.type)
                if torch.is_autocast_enabled(x.device.type)
                else x.dtype,
            )

        rows_per_chunk = max(1, batch_size_limit // max(math.prod(B) * C, 1))
        for start in range(0, R, rows_per_chunk):
            self._forward(
                x=x[..., start : start + rows_per_chunk, :, :],
                freq=freq,
                weight=weight,
                bias=bias,
                out=out[..., start : start + rows_per_chunk, :, :],
                **{
                    key: value[..., start : start + rows_per_chunk, :, :]
                    for key, value in kwargs.items()
                },
            )

        return out

    def _forward(
        self,
        x: Tensor,  # [..., R, C, G]
        freq: Tensor,  # [..., 1, C, G, F]
        weight: Tensor,  # [..., 1, C, G, D, 2F]
        bias: Tensor,  # [..., 1, C, D]
        *,
        out: Tensor | None = None,
    ) -> Tensor:

        *B, R, C, G = x.size()
        F = freq.size(-1)

        if torch.is_grad_enabled():
            x = x.to(torch.float32).unsqueeze(-1) * freq  # [..., R, C, G, F]
        else:
            tmp = x.new_empty((*B, C, R, G, F), dtype=torch.float32)
            torch.mul(x.unsqueeze(-1), freq, out=tmp.transpose(-4, -3))
            x = tmp.transpose(-4, -3)  # [..., R, C, G, F]
            del tmp

        # Store rows within each column so the projection uses views.
        fourier = x.new_empty((*B, C, R, G, 2 * F), dtype=weight.dtype)
        fourier = fourier.transpose(-4, -3)  # [..., R, C, G, 2F]
        if torch.is_grad_enabled():
            fourier[..., : x.size(-1)] = x.sin()
            fourier[..., x.size(-1) :] = x.cos()
        else:
            torch.sin(x, out=fourier[..., : x.size(-1)])
            torch.cos(x, out=fourier[..., x.size(-1) :])
        del x

        # weight: [..., C, G*2F, D]
        weight = weight.transpose(-3, -2).flatten(-2).squeeze(-4).mT
        # fourier: [..., C, R, G*2F]
        fourier = fourier.transpose(-4, -3).flatten(-2)
        if out is None or out.dtype != weight.dtype:
            # projected: [..., R, C, D]
            projected = torch.matmul(fourier, weight).transpose(-3, -2)
            if out is None:
                out = projected
            else:
                out.copy_(projected)
        else:
            torch.matmul(
                fourier,
                weight,
                out=out.transpose(-3, -2),
            )

        out += bias.to(out.dtype)

        return out

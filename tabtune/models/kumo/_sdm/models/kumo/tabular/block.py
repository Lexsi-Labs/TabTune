# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import GELU, Linear, Sequential

from tabtune.models.kumo._sdm.nn import QueryScaling, RMSNorm, RotaryEmbedding, TransformerBlock


class _RoPERMSNorm(Sequential):
    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        rope, norm = self
        return norm(input, rope=rope)


class KumoTabularTransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        query_scaling: QueryScaling | None,
        rope: RotaryEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        query_transforms: list[torch.nn.Module] = []
        key_transforms: list[torch.nn.Module] = []
        if rope is not None:
            query_transforms.append(rope)
            key_transforms.append(rope)
        query_transforms.append(
            RMSNorm(
                channels // num_heads,
                eps=1e-6,
                elementwise_affine=False,
                **factory_kwargs,
            )
        )
        key_transforms.append(
            RMSNorm(
                channels // num_heads,
                eps=1e-6,
                elementwise_affine=False,
                **factory_kwargs,
            )
        )

        mlp = Sequential(
            RMSNorm(channels, **factory_kwargs),
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
            Linear(2 * channels, channels, **factory_kwargs),
        )
        torch.nn.init.zeros_(cast(Linear, mlp[-1]).weight)
        torch.nn.init.zeros_(cast(Linear, mlp[-1]).bias)

        transform = Sequential if rope is None else _RoPERMSNorm
        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            mlp=mlp,
            query_norm=RMSNorm(channels, **factory_kwargs),
            key_value_norm=RMSNorm(channels, **factory_kwargs),
            query_transform=transform(*query_transforms),
            key_transform=transform(*key_transforms),
            query_scaling=query_scaling,
            **factory_kwargs,
        )

    def peak_bytes_per_example(
        self,
        element_size: int,
        query_length: int,
        key_value_length: int | None = None,
    ) -> int:
        r""":meta private:"""  # noqa: D415
        length = max(query_length, key_value_length or 0)
        factor = 15 if element_size <= 2 else 8
        return factor * length * element_size * self.attn.q_dim


if __name__ == "__main__":
    from tabtune.models.kumo._sdm.nn import LogScale
    from tabtune.models.kumo._sdm.testing.memory import benchmark_transformer_block_memory_peak

    benchmark_transformer_block_memory_peak(
        block=lambda channels, num_heads: KumoTabularTransformerBlock(
            channels=channels,
            num_heads=num_heads,
            query_scaling=LogScale(num_heads=num_heads),
        ),
        channels_and_heads=[(256, 4), (512, 4)],
    )

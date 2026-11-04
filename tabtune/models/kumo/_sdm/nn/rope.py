# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from typing import Any, Literal

import torch
from torch import Tensor


class RotaryEmbedding(torch.nn.Module):
    r"""Rotary Positional Embedding (RoPE) from the `"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"
    <https://arxiv.org/abs/2104.09864>`_ paper.

    Args:
        channels: The number of channels per attention head.
        layout: The channel pairing layout. ``"split_half"`` pairs the first
            half of the channels with the second half. ``"interleaved"`` pairs
            adjacent even and odd channels.
        theta: The base frequency used to initialize inverse frequencies.
        requires_grad: Whether inverse frequencies are learnable.
        partial_rotary_factor: The fraction of leading channels to which RoPE
            is applied.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        layout: Literal["split_half", "interleaved"],
        theta: float = 100_000,
        requires_grad: bool = True,
        partial_rotary_factor: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.layout = layout
        self.channels = channels
        self.rotary_channels = int(channels * partial_rotary_factor)

        if self.channels % 2 != 0:
            raise ValueError(f"'channels' must be even (got {self.channels})")
        if (
            self.rotary_channels < 2
            or self.rotary_channels > channels
            or self.rotary_channels % 2 != 0
        ):
            raise ValueError(
                f"'partial_rotary_factor' must produce an even number between "
                f"2 and {self.channels} (got {self.rotary_channels})"
            )

        arange = torch.arange(0, self.rotary_channels, 2, **factory_kwargs)
        inv_freq = 1.0 / (theta ** (arange / self.rotary_channels))
        self.inv_freq = torch.nn.Parameter(
            inv_freq,
            requires_grad=requires_grad,
        )

    def forward(
        self,
        x: Tensor,  # [..., S, H, C]
    ) -> Tensor:  # [..., S, H, C]
        """The forward pass.

        Args:
            x: Tensor with shape ``[..., S, H, C]``.
                ``S`` is the query sequence length, ``H`` is the number of
                attention heads, and ``C`` is the channels per head.

        Returns:
            Tensor with shape ``[..., S, H, C]``.
        """
        if x.size(-1) != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels (got {x.size(-1)})"
            )
        seq = torch.arange(x.size(-3), device=x.device, dtype=torch.float32)
        freq = seq.view(-1, 1) * self.inv_freq.view(1, -1)  # [S, C // 2]
        sin = freq.sin()[:, None, :].to(x.dtype)  # [S, 1, C // 2]
        cos = freq.cos()[:, None, :].to(x.dtype)  # [S, 1, C // 2]

        rotary = x[..., : self.rotary_channels]
        if self.layout == "interleaved":
            x1, x2 = rotary[..., 0::2], rotary[..., 1::2]
        else:
            assert self.layout == "split_half"
            x1, x2 = rotary.split(sin.size(-1), dim=-1)

        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

        if self.layout == "interleaved":
            rotary = torch.stack((out1, out2), dim=-1).flatten(-2)
        else:
            assert self.layout == "split_half"
            rotary = torch.cat((out1, out2), dim=-1)

        if self.rotary_channels == self.channels:
            return rotary
        return torch.cat((rotary, x[..., self.rotary_channels :]), dim=-1)

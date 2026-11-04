# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from tabtune.models.kumo._sdm._kernels import rmsnorm_cast
from tabtune.models.kumo._sdm.nn.rope import RotaryEmbedding


class RMSNorm(torch.nn.RMSNorm):
    """A :class:`torch.nn.RMSNorm` that dispatches to an efficient kernel."""

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding | None = None,
    ) -> Tensor:
        """Apply optional rotary embeddings before normalization.

        Args:
            x: Input with shape ``[..., C]``, or ``[..., S, H, C]`` with
                rotary embeddings, where ``S`` is the sequence length,
                ``H`` is the number of heads, and ``C`` is channels per head.
            rope: Optional rotary embedding applied to the input.
        """
        if (
            x.is_cuda
            and torch.is_autocast_enabled("cuda")
            and not torch.is_grad_enabled()
            and x.dtype in {torch.float16, torch.bfloat16, torch.float32}
            and x.shape[-1:] == self.normalized_shape
            and (
                self.weight is None
                or self.weight.dtype
                in {torch.float16, torch.bfloat16, torch.float32}
            )
        ):
            cos_sin: tuple[Tensor, Tensor] | None = None
            if rope is not None:
                if (
                    not torch.compiler.is_compiling()
                    and rope.layout == "split_half"
                    and rope.rotary_channels == rope.channels == x.size(-1)
                ):
                    seq = torch.arange(
                        x.size(-3), device=x.device, dtype=torch.float32
                    )
                    freq = seq.view(-1, 1) * rope.inv_freq.view(1, -1)
                    cos_sin = (freq.cos().to(x.dtype), freq.sin().to(x.dtype))
                else:
                    x = rope(x)
            eps = (
                self.eps
                if self.eps is not None
                else torch.finfo(torch.float32).eps
            )
            return rmsnorm_cast(x, self.weight, eps, rope=cos_sin)
        return super().forward(x if rope is None else rope(x))

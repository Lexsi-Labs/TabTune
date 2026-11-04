# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

_RMSNormCast: TypeAlias = Callable[
    [Tensor, Tensor | None, float, tuple[Tensor, Tensor] | None], Tensor
]

_triton_rmsnorm_cast: _RMSNormCast | None = None
try:
    from tabtune.models.kumo._sdm._kernels.triton.rmsnorm_cast import (
        rmsnorm_cast as _triton_rmsnorm_cast_impl,
    )
except ImportError:
    pass
else:
    _triton_rmsnorm_cast = _triton_rmsnorm_cast_impl


def rmsnorm_cast(
    x: Tensor,
    weight: Tensor | None,
    eps: float,
    rope: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Apply RMSNorm over the last dimension and cast to the autocast dtype.

    The output uses the configured autocast dtype for the input device,
    even when autocast is disabled.

    Args:
        x: Input tensor with shape ``[..., C]``, where ``C`` is the number
            of channels.
        weight: Scale tensor with shape ``[C]``, or ``None`` for no scaling.
        eps: Constant added to the mean square before normalization.
        rope: Optional cosine and sine tables with shape ``[S, C // 2]``
            for split-half rotary embeddings. The input then has shape
            ``[..., S, H, C]``, where ``S`` is the sequence length and ``H``
            is the number of heads. Rotation rounds in the input dtype.
    """
    if (
        _triton_rmsnorm_cast is not None
        and x.is_cuda
        and x.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and x.size(-1) in {32, 64, 128, 256, 512}
        and (
            weight is None
            or (
                weight.dtype in {torch.float16, torch.bfloat16, torch.float32}
                and weight.device == x.device
                and weight.shape == x.shape[-1:]
            )
        )
        and (
            rope is None
            or (
                x.dim() >= 3
                and all(
                    table.shape == (x.size(-3), x.size(-1) // 2)
                    and table.dtype == x.dtype
                    and table.device == x.device
                    for table in rope
                )
            )
        )
        and x.numel() > 0
        and not (
            (
                x.requires_grad
                or (weight is not None and weight.requires_grad)
                or (
                    rope is not None
                    and any(table.requires_grad for table in rope)
                )
            )
            and torch.is_grad_enabled()
        )
    ):
        return _triton_rmsnorm_cast(x, weight, eps, rope)
    if rope is not None:
        cos, sin = (table.unsqueeze(-2) for table in rope)
        x1, x2 = x.chunk(2, dim=-1)
        x = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return F.rms_norm(x, (x.size(-1),), weight, eps).to(
        torch.get_autocast_dtype(x.device.type)
    )

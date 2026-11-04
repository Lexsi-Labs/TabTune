# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _load_input(
    x_ptr,
    cos_ptr,
    sin_ptr,
    row,
    c,
    mask,
    channels: tl.constexpr,
    seq_len,
    num_heads,
) -> tl.tensor:
    offsets = row[:, None] * channels
    value = tl.load(x_ptr + offsets + c, mask=mask, other=0).to(tl.float32)
    if cos_ptr is not None:
        first = c < channels // 2
        partner = tl.where(first, c + channels // 2, c - channels // 2)
        other = tl.load(x_ptr + offsets + partner, mask=mask, other=0).to(
            tl.float32
        )
        table = (row // num_heads % seq_len)[:, None] * (channels // 2)
        table += c % (channels // 2)
        cos = tl.load(cos_ptr + table, mask=mask, other=0).to(tl.float32)
        sin = tl.load(sin_ptr + table, mask=mask, other=0).to(tl.float32)
        # Eager RoPE rounds both products and their sum in the input dtype.
        dtype = x_ptr.dtype.element_ty
        a = (value * cos).to(dtype).to(tl.float32)
        b = (other * sin).to(dtype).to(tl.float32)
        value = tl.where(first, a - b, a + b).to(dtype).to(tl.float32)
    return value


@triton.jit(do_not_specialize=["rows", "seq_len", "num_heads"])
def _rmsnorm_cast_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    rows,
    seq_len,
    num_heads,
    channels: tl.constexpr,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    offsets = row[:, None] * channels
    mask = row[:, None] < rows
    groups = tl.arange(0, channels // 4)[None, :]
    total = tl.full((block_rows, channels // 4), 0, tl.float32)
    for i in tl.static_range(4):
        v = _load_input(  # ty: ignore[invalid-argument-type]
            x_ptr,
            cos_ptr,
            sin_ptr,
            row,
            groups * 4 + i,
            mask,
            channels,
            seq_len,
            num_heads,
        )
        total = tl.fma(v, v, total)
    if channels == 32:
        # Preserve CUDA's shuffle order when Triton packs adjacent groups.
        for shift in tl.static_range(2, -1, -1):
            indices = tl.broadcast_to(groups ^ (1 << shift), (block_rows, 8))
            total += tl.gather(total, indices, axis=1)
        total = tl.where(groups == 0, total, 0)
    variance = tl.sum(total, 1)  # ty: ignore[invalid-argument-type]
    c = tl.arange(0, channels)[None, :]
    value = _load_input(  # ty: ignore[invalid-argument-type]
        x_ptr,
        cos_ptr,
        sin_ptr,
        row,
        c,
        mask,
        channels,
        seq_len,
        num_heads,
    )
    value *= tl.rsqrt(variance[:, None] / channels + eps)
    if weight_ptr is not None:
        value *= tl.load(weight_ptr + c).to(tl.float32)
    tl.store(out_ptr + offsets + c, value, mask=mask)


def rmsnorm_cast(
    x: Tensor,
    weight: Tensor | None,
    eps: float,
    rope: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    x = x.contiguous()
    if weight is not None:
        weight = weight.contiguous()
    cos, sin = (
        (None, None)
        if rope is None
        else (table.contiguous() for table in rope)
    )
    channels = x.size(-1)
    rows = x.numel() // channels
    block_rows = 8
    grid = ((rows + block_rows - 1) // block_rows,)
    out = torch.empty_like(x, dtype=torch.get_autocast_dtype("cuda"))
    with torch.cuda.device(x.device):
        cast(Any, _rmsnorm_cast_kernel)[grid](
            x,
            weight,
            cos,
            sin,
            out,
            rows,
            x.size(-3) if rope is not None else 1,
            x.size(-2) if rope is not None else 1,
            channels,
            eps,
            block_rows,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out

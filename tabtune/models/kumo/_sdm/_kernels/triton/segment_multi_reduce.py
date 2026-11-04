# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor

_NUM_STATISTICS = 5


@triton.jit
def _segment_multi_reduce_kernel(
    src_ptr,
    index_ptr,
    edge_attr_ptr,
    edge_type_ptr,
    offsets_ptr,
    output_ptr,
    num_channels: tl.constexpr,
    num_statistics: tl.constexpr,
    block_channels: tl.constexpr,
    has_edge_type: tl.constexpr,
) -> None:
    segment = tl.program_id(0)
    channels = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    channel_mask = channels < num_channels
    start = tl.load(offsets_ptr + segment)
    end = tl.load(offsets_ptr + segment + 1)

    total = tl.zeros((block_channels,), tl.float32)  # ty: ignore[invalid-argument-type]
    square_total = tl.zeros((block_channels,), tl.float32)  # ty: ignore[invalid-argument-type]
    minimum = tl.full((block_channels,), float("inf"), tl.float32)
    maximum = tl.full((block_channels,), -float("inf"), tl.float32)
    has_nan = tl.zeros((block_channels,), tl.int1)  # ty: ignore[invalid-argument-type]

    edge = start
    while edge < end:
        row = tl.load(index_ptr + edge)
        value = tl.load(
            src_ptr + row.to(tl.int64) * num_channels + channels,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if has_edge_type:
            edge_attr_row = tl.load(edge_type_ptr + edge)
        else:
            edge_attr_row = edge
        value += tl.load(
            edge_attr_ptr
            + edge_attr_row.to(tl.int64) * num_channels
            + channels,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        total += value
        square_total += value * value
        minimum = tl.minimum(minimum, value)
        maximum = tl.maximum(maximum, value)
        has_nan |= value != value
        edge += 1

    count = end - start
    divisor = tl.maximum(count, 1)
    mean = total / divisor
    variance = square_total / divisor - mean * mean
    clamped_variance = tl.where(variance < 1e-5, 1e-5, variance)
    std = tl.where(
        variance <= 1e-5,
        0.0,
        tl.sqrt(clamped_variance),
    )
    minimum_is_inf = (minimum == float("inf")) | (minimum == -float("inf"))
    maximum_is_inf = (maximum == float("inf")) | (maximum == -float("inf"))
    minimum = tl.where(minimum_is_inf, 0.0, minimum)
    maximum = tl.where(maximum_is_inf, 0.0, maximum)
    minimum = tl.where(has_nan, float("nan"), minimum)
    maximum = tl.where(has_nan, float("nan"), maximum)

    output_offsets = (
        segment.to(tl.int64) * num_statistics * num_channels + channels
    )
    tl.store(output_ptr + output_offsets, total, mask=channel_mask)
    tl.store(
        output_ptr + output_offsets + num_channels,
        mean,
        mask=channel_mask,
    )
    tl.store(
        output_ptr + output_offsets + 2 * num_channels,
        std,
        mask=channel_mask,
    )
    tl.store(
        output_ptr + output_offsets + 3 * num_channels,
        minimum,
        mask=channel_mask,
    )
    tl.store(
        output_ptr + output_offsets + 4 * num_channels,
        maximum,
        mask=channel_mask,
    )


def segment_multi_reduce(
    src: Tensor,
    index: Tensor,
    edge_attr: Tensor,
    offsets: Tensor,
    edge_type: Tensor | None = None,
) -> Tensor:
    src = src.contiguous()
    index = index.contiguous()
    edge_attr = edge_attr.contiguous()
    offsets = offsets.contiguous()
    if edge_type is not None:
        edge_type = edge_type.contiguous()
    if edge_type is None and edge_attr.shape != (
        index.numel(),
        src.size(1),
    ):
        raise ValueError(
            "edge_attr must have shape (index.numel(), src.size(1))"
        )
    if edge_type is not None and (
        edge_attr.dim() != 2
        or edge_attr.size(1) != src.size(1)
        or edge_type.numel() != index.numel()
    ):
        raise ValueError(
            "edge_type must match index and edge_attr must have "
            "src.size(1) columns"
        )

    shape = (offsets.numel() - 1, _NUM_STATISTICS, src.size(1))
    output = src.new_empty(shape)
    if shape[0] == 0 or shape[2] == 0:
        return output

    block_channels = min(
        triton.next_power_of_2(shape[2]),  # ty: ignore[invalid-argument-type]
        512,
    )
    if block_channels >= 256:
        num_warps = 8
    elif block_channels >= 128:
        num_warps = 4
    else:
        num_warps = 1
    grid = (
        shape[0],
        triton.cdiv(  # ty: ignore[invalid-argument-type]
            shape[2], block_channels
        ),
    )
    with torch.cuda.device(src.device):
        cast(Any, _segment_multi_reduce_kernel)[grid](
            src,
            index,
            edge_attr,
            edge_type if edge_type is not None else index,
            offsets,
            output,
            num_channels=shape[2],
            num_statistics=shape[1],
            block_channels=block_channels,
            num_warps=num_warps,
            has_edge_type=edge_type is not None,
        )
    return output

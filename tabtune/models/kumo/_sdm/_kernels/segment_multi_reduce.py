# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TypeAlias

import torch
from torch import Tensor

_SegmentMultiReduce: TypeAlias = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor | None],
    Tensor,
]

_triton_segment_multi_reduce: _SegmentMultiReduce | None = None
try:
    from tabtune.models.kumo._sdm._kernels.triton.segment_multi_reduce import (
        segment_multi_reduce as _triton_segment_multi_reduce_impl,
    )
except ImportError:
    pass
else:
    _triton_segment_multi_reduce = _triton_segment_multi_reduce_impl


def _eager_segment_multi_reduce(
    src: Tensor,
    index: Tensor,
    edge_attr: Tensor,
    offsets: Tensor,
    edge_type: Tensor | None = None,
) -> Tensor:
    if edge_type is not None:
        edge_attr = edge_attr[edge_type]
    values = src[index]
    if src.dtype in {torch.float16, torch.bfloat16}:
        values = values.float() + edge_attr.float()
    else:
        values = values + edge_attr
    total = torch.segment_reduce(
        values,
        offsets=offsets,
        reduce="sum",
        unsafe=True,
        initial=0,
    )
    mean = total / offsets.diff().clamp(min=1).unsqueeze(1)
    variance = (
        torch.segment_reduce(
            values.square(),
            offsets=offsets,
            reduce="mean",
            unsafe=True,
            initial=0,
        )
        - mean.square()
    )
    std = torch.where(
        variance <= 1e-5,
        0.0,
        variance.clamp(min=1e-5).sqrt(),
    )
    minimum = torch.segment_reduce(
        values,
        offsets=offsets,
        reduce="min",
        unsafe=True,
    )
    maximum = torch.segment_reduce(
        values,
        offsets=offsets,
        reduce="max",
        unsafe=True,
    )
    minimum = torch.where(minimum.isinf(), 0.0, minimum)
    maximum = torch.where(maximum.isinf(), 0.0, maximum)
    return torch.stack(
        (
            total,
            mean,
            std,
            minimum,
            maximum,
        ),
        dim=1,
    ).to(src.dtype)


def segment_multi_reduce(
    src: Tensor,
    index: Tensor,
    edge_attr: Tensor,
    offsets: Tensor,
    edge_type: Tensor | None = None,
) -> Tensor:
    edge_attr_matches = (
        edge_attr.shape == (index.numel(), src.size(1))
        if edge_type is None
        else (
            edge_attr.dim() == 2
            and edge_attr.size(1) == src.size(1)
            and edge_type.numel() == index.numel()
        )
    )
    if (
        _triton_segment_multi_reduce is not None
        and not (
            (src.requires_grad or edge_attr.requires_grad)
            and torch.is_grad_enabled()
        )
        and src.is_cuda
        and src.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and edge_attr.dtype == src.dtype
        and (
            edge_type is None
            or (
                edge_type.device == src.device
                and edge_type.dtype in {torch.int32, torch.int64}
            )
        )
        and index.device == src.device
        and edge_attr.device == src.device
        and offsets.device == src.device
        and index.dtype in {torch.int32, torch.int64}
        and offsets.dtype in {torch.int32, torch.int64}
        and edge_attr_matches
    ):
        return _triton_segment_multi_reduce(
            src,
            index,
            edge_attr,
            offsets,
            edge_type,
        )
    return _eager_segment_multi_reduce(
        src,
        index,
        edge_attr,
        offsets,
        edge_type,
    )

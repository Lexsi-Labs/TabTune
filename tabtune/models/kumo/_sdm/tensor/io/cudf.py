# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    import cudf


def to_cudf(tensor: Tensor, valid_mask: Tensor | None = None) -> cudf.Series:
    r"""Convert a CUDA tensor to a flat :class:`cudf.Series`.

    Args:
        tensor: The CUDA tensor.
        valid_mask: Boolean mask indicating valid, non-null tensor elements.
    """
    import cudf

    if not tensor.is_cuda:
        raise ValueError(
            f"Expected tensor to be on a CUDA device (got '{tensor.device}')"
        )

    tensor = tensor.detach().contiguous().view(-1)
    with torch.cuda.device(tensor.device):
        ser = cudf.Series(tensor, copy=False, nan_as_null=valid_mask is None)

    if valid_mask is None:
        return ser

    if valid_mask.device != tensor.device:
        raise ValueError(
            f"Expected 'valid_mask' to be on device '{tensor.device}' "
            f"(got '{valid_mask.device}')"
        )

    with torch.cuda.device(tensor.device):
        mask = to_cudf(valid_mask.contiguous().view(-1))._column.as_mask()
        if not isinstance(mask, tuple):
            mask = (mask,)

        return cudf.Series._from_column(ser._column.set_mask(*mask))

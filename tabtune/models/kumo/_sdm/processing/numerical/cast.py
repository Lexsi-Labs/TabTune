# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.processing import Processor


class Cast(Processor):
    """Cast numerical columns to a floating-point dtype.

    Args:
        dtype: The floating-point dtype of the numerical columns.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        if not dtype.is_floating_point:
            raise ValueError(f"Expected a floating-point dtype (got {dtype})")
        self.dtype = dtype

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical.to(self.dtype))

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}({self.dtype})"

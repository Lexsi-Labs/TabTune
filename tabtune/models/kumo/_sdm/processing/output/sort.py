# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.processing import Processor


class SortQuantiles(Processor):
    r"""Sort quantile predictions in ascending order."""

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical.sort(dim=-1).values
        return table.replace_blocks(numerical=numerical)

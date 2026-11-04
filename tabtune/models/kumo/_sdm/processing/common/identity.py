# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.processing import InvertibleMixin, Processor


class Identity(Processor, InvertibleMixin):
    r"""Return inputs unchanged."""

    handles_stypes = frozenset(Stype)
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table

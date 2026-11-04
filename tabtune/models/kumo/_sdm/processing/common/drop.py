# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tabtune.models.kumo._sdm import Stype, StypeLike, TableTensor
from tabtune.models.kumo._sdm.processing.base import Processor


class DropStypes(Processor):
    r"""Remove all columns for specific semantic types.

    Args:
        stypes: Semantic column types to remove. All other semantic types are
            preserved unchanged.
    """

    requires_fit = False

    def __init__(self, *stypes: StypeLike) -> None:
        super().__init__()
        self._stypes = stypes
        self.handles_stypes = frozenset(Stype(stype) for stype in stypes)

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.drop_stypes(self.handles_stypes)

    def __repr__(self, *, indent: int = 0) -> str:
        stypes = ", ".join(map(repr, self._stypes))
        return f"{' ' * indent}{self.__class__.__name__}({stypes})"

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.processing import Processor


class Callable(Processor):
    """Adapt a stateless callable to the :class:`Processor` interface."""

    handles_stypes = frozenset(Stype)
    requires_fit = False

    def __init__(self, function: Callable[[TableTensor], TableTensor]) -> None:
        super().__init__()
        self.function = function

    def _transform(self, table: TableTensor) -> TableTensor:
        return self.function(table)

    def __repr__(self, *, indent: int = 0) -> str:
        name = getattr(
            self.function,
            "__name__",
            self.function.__class__.__name__,
        )
        return f"{' ' * indent}{self.__class__.__name__}({name})"

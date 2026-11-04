# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.processing import Processor


class Clip(Processor):
    """Clamp numerical values to a fixed interval.

    Values below ``min_value`` are set to ``min_value``, and values above
    ``max_value`` are set to ``max_value``. Values within the interval are
    unchanged.

    Args:
        min_value: Inclusive lower bound for every numerical value.
        max_value: Inclusive upper bound for every numerical value.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(
        self,
        min_value: float,
        max_value: float,
    ) -> None:
        super().__init__()
        if min_value > max_value:
            raise ValueError(
                "min_value must be less than or equal to max_value"
            )
        self.min_value = min_value
        self.max_value = max_value

    def _transform(self, table: TableTensor) -> TableTensor:
        """Clamp ``table`` to the configured interval."""
        numerical = table.numerical.clamp(
            min=self.min_value,
            max=self.max_value,
        )
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"{self.min_value}, {self.max_value})"
        )

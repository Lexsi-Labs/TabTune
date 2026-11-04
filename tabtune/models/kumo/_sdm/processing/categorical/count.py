# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch import Tensor

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.nn._buffer import BufferList
from tabtune.models.kumo._sdm.processing import Processor
from tabtune.models.kumo._sdm.processing.categorical._categorical import (
    _check_categorical_codes,
    _check_categories,
)


class AddCategoryCounts(Processor):
    """Add the log row count of each categorical value.

    Each eligible column receives one numerical column ``<column>__count``
    holding ``log1p`` of the number of fitted rows sharing the row's code.
    Negative codes (missing or unknown values) share one count.

    Transform inputs must use the fitted per-column category vocabularies.
    Use :class:`~sdm.processing.AlignCategories` before this processor when
    training and transform inputs were tensorized independently.

    Args:
        min_cardinality: A count column is added when its fitted vocabulary
            size exceeds this value. Missing values do not contribute to
            vocabulary size. Defaults to 0.
    """

    handles_stypes = frozenset({Stype.categorical})
    requires_fit = True

    def __init__(self, min_cardinality: int = 0) -> None:
        super().__init__()
        self.min_cardinality = min_cardinality
        self._columns: tuple[str, ...] = ()
        self._categories: BufferList[Tensor] = BufferList()
        self.register_buffer("_log_counts", torch.empty(0))
        self.register_buffer(
            "_num_categories",
            torch.empty(0, dtype=torch.long),
        )

    def get_extra_state(self) -> tuple[str, ...]:
        r""":meta private:"""  # noqa: D415
        return self._columns

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._columns = cast(tuple[str, ...], state)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        _check_categorical_codes(table)
        self._columns = tuple(
            column
            for column, categories in zip(
                table.columns[Stype.categorical], table.categorical.categories
            )
            if categories.numel() > self.min_cardinality
        )
        categorical = table.select_columns(self._columns).categorical
        codes = categorical.code  # [*batch, num_rows, num_columns]
        sizes = [categories.numel() for categories in categorical.categories]
        num_categories = codes.new_tensor(sizes, dtype=torch.long)
        # Negative codes accumulate in the slot behind their column's
        # vocabulary: [*batch, num_columns, max(sizes) + 1].
        slots = torch.where(codes >= 0, codes.long(), num_categories)
        slots = slots.transpose(-2, -1)
        counts = torch.zeros(
            (*codes.shape[:-2], len(sizes), max(sizes, default=0) + 1),
            dtype=torch.long,
            device=codes.device,
        )
        counts.scatter_add_(-1, slots, torch.ones_like(slots))

        dtype = table.numerical.dtype
        count_dtype = (
            torch.float32
            if dtype in {torch.float16, torch.bfloat16}
            else dtype
        )
        self._log_counts = counts.to(count_dtype).log1p().to(dtype)
        self._num_categories = num_categories
        self._categories = BufferList(table.categorical.categories)

    def _transform(self, table: TableTensor) -> TableTensor:
        _check_categories(table, self._categories)
        _check_categorical_codes(table)
        if not self._columns:
            return table
        codes = table.select_columns(self._columns).categorical.code
        slots = torch.where(codes >= 0, codes.long(), self._num_categories)
        counts = self._log_counts.gather(-1, slots.transpose(-2, -1))

        out_table = TableTensor(
            columns={
                Stype.numerical: tuple(
                    f"{column}__count" for column in self._columns
                ),
            },
            numerical=counts.transpose(-2, -1).to(table.numerical.dtype),
        )
        return cast(TableTensor, torch.cat([table, out_table], dim=-1))

    def __repr__(self, *, indent: int = 0) -> str:
        if self.min_cardinality == 0:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}"
            f"(min_cardinality={self.min_cardinality!r})"
        )

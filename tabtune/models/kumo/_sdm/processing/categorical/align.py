# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from tabtune.models.kumo._sdm import (
    CategoricalTensor,
    ColumnarTensor,
    EnsembleTable,
    StringTensor,
    Stype,
    TableTensor,
)
from tabtune.models.kumo._sdm.nn._buffer import BufferList
from tabtune.models.kumo._sdm.processing import EnsembleProcessor
from tabtune.models.kumo._sdm.relational.join import join_index

_UNSIGNED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class AlignCategories(EnsembleProcessor):
    """Align categorical columns to categories observed during fitting.

    A categorical column stores each value as an integer code indexing an
    ordered list of categories. Separately created tables can use different
    codes for the same value. Fitting learns the category list for each column,
    and transforming remaps another table to use it. Missing values and
    categories not retained during fitting receive code ``-1``.

    Args:
        sort_by: How to order fitted categories.
            ``"code"`` keeps observed categories in original order.
            ``"frequency"`` orders observed categories by descending frequency.
            ``"value"`` orders observed categories by ascending value.
        min_frequency: Minimum number of observations required to retain a
            category. Values of rarer categories receive code ``-1``.
            Must be positive.

    >>> import pandas as pd
    >>> import sdm
    >>> from sdm.processing import AlignCategories
    >>> table1 = sdm.TableTensor.from_pandas(
    ...     pd.DataFrame({"color": ["red", "blue", "red"]}),
    ...     stypes={"color": "categorical"},
    ... )
    >>> table2 = sdm.TableTensor.from_pandas(
    ...     pd.DataFrame({"color": ["blue", "green", None]}),
    ...     stypes={"color": "categorical"},
    ... )
    >>> table1.categorical.categories[0].tolist()
    ['red', 'blue']
    >>> table2.categorical.categories[0].tolist()
    ['blue', 'green']
    >>> table2.categorical.code[:, 0].tolist()
    [0, 1, -1]
    >>> processor = AlignCategories().fit(table1)
    >>> table2 = processor.transform(table2)
    >>> table2.categorical.categories[0].tolist()
    ['red', 'blue']
    >>> table2.categorical.code[:, 0].tolist()
    [1, -1, -1]

    Here, ``"blue"`` changes from code ``0`` to ``1``, while unseen
    ``"green"`` and the missing value use ``-1``.
    """

    handles_stypes = frozenset({Stype.categorical})
    requires_fit = True

    def __init__(
        self,
        sort_by: Literal["code", "frequency", "value"] = "code",
        *,
        min_frequency: int = 1,
    ) -> None:
        super().__init__()
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        self.sort_by = sort_by
        self.min_frequency = min_frequency
        self._categories: BufferList[BufferList[Tensor]] = BufferList()

    def _fit_column(
        self,
        input_categories: Tensor,
        codes: Tensor,
        *,
        align_codes: bool,
    ) -> tuple[tuple[Tensor, ...], Tensor | None]:
        batch_size = codes.size(0)
        if input_categories.numel() == 0:
            aligned_codes = torch.full_like(codes, -1) if align_codes else None
            return (input_categories,) * batch_size, aligned_codes

        mask = codes >= 0
        indices = codes.clamp_min(0).long()
        # Count categories independently per batch: [B, N] -> [B, K].
        counts = codes.new_zeros((batch_size, input_categories.numel()))
        counts.scatter_add_(1, indices, mask.to(codes.dtype))

        if self.sort_by == "frequency":
            order = counts.argsort(dim=1, descending=True, stable=True)
            ordered_categories = input_categories
        elif self.sort_by == "value":
            if (
                input_categories.is_cuda
                and input_categories.dtype in _UNSIGNED_DTYPES
            ):
                key = input_categories.to(torch.int64)
                if input_categories.dtype == torch.uint64:
                    # Map unsigned integer order onto signed integer order.
                    key = key.bitwise_xor(torch.iinfo(torch.int64).min)
                perm = key.argsort()
                ordered_categories = input_categories.index_select(0, perm)
            else:
                ordered_categories, perm = input_categories.sort()
            order = perm.expand(batch_size, -1)
        else:
            assert self.sort_by == "code"
            order = torch.arange(
                input_categories.numel(),
                device=codes.device,
            ).expand(batch_size, -1)
            ordered_categories = input_categories

        observed = counts.gather(1, order) >= self.min_frequency
        # Only ragged vocabularies require per-batch materialization.
        fitted_categories = []
        for batch_index in range(batch_size):
            selected_indices = order[batch_index, observed[batch_index]]
            if (
                selected_indices.numel() == input_categories.numel()
                and self.sort_by != "frequency"
            ):
                fitted_categories.append(ordered_categories)
            elif (
                input_categories.dtype in _UNSIGNED_DTYPES
                and input_categories.is_cpu
            ):
                # PyTorch CPU index_select is not implemented for these dtypes.
                fitted_categories.append(input_categories[selected_indices])
            else:
                fitted_categories.append(
                    input_categories.index_select(0, selected_indices)
                )

        if not align_codes:
            return tuple(fitted_categories), None

        rank = observed.cumsum(dim=1, dtype=codes.dtype) - 1
        lookup = torch.full_like(counts, -1)
        lookup.scatter_(1, order, torch.where(observed, rank, -1))
        aligned_codes = torch.where(mask, lookup.gather(1, indices), -1)
        return tuple(fitted_categories), aligned_codes

    def _fit_columns(
        self,
        table: TableTensor,
        *,
        align_codes: bool,
    ) -> tuple[tuple[tuple[Tensor, ...], ...], Tensor | None]:
        codes = table.categorical.code
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        aligned_codes = torch.full_like(codes, -1) if align_codes else None

        # Accumulate ragged vocabularies in batch-major order: [batch][column].
        categories_by_batch: list[list[Tensor]] = [
            [] for _ in range(codes.size(0))
        ]
        for column_index, input_categories in enumerate(
            table.categorical.categories
        ):
            fitted_categories, aligned_column_codes = self._fit_column(
                input_categories=input_categories,
                codes=codes[..., column_index],
                align_codes=align_codes,
            )
            if aligned_codes is not None:
                assert aligned_column_codes is not None
                aligned_codes[..., column_index] = aligned_column_codes
            for batch_categories, column_categories in zip(
                categories_by_batch,
                fitted_categories,
                strict=True,
            ):
                batch_categories.append(column_categories)

        fitted_categories = tuple(
            tuple(batch_categories) for batch_categories in categories_by_batch
        )
        return fitted_categories, aligned_codes

    def _fit_and_align(
        self,
        table: TableTensor,
    ) -> tuple[tuple[tuple[Tensor, ...], ...], tuple[TableTensor, ...]]:
        single_table = table.categorical.code.dim() == 2
        fitted_categories, aligned_codes = self._fit_columns(
            table,
            align_codes=True,
        )
        assert aligned_codes is not None
        aligned_tables = tuple(
            (table if single_table else table[batch_index]).replace_blocks(
                categorical=CategoricalTensor(
                    aligned_codes[batch_index],
                    categories=batch_categories,
                ),
            )
            for batch_index, batch_categories in enumerate(fitted_categories)
        )
        return fitted_categories, aligned_tables

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        fitted_categories, _ = self._fit_columns(
            table,
            align_codes=False,
        )
        self._categories = BufferList(
            BufferList(categories) for categories in fitted_categories
        )

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        fitted_categories, aligned_tables = self._fit_and_align(table)
        self._categories = BufferList(
            BufferList(categories) for categories in fitted_categories
        )
        return aligned_tables[0]

    def _transform(self, table: TableTensor) -> TableTensor:
        return self._align_to_categories(
            table,
            cast(Sequence[Sequence[Tensor]], self._categories),
        )[0]

    @staticmethod
    def _member_table_ids(
        ensemble_table: EnsembleTable,
    ) -> tuple[int, ...]:
        group_offsets = []
        offset = 0
        for group in ensemble_table._iter_groups():
            group_offsets.append(offset)
            offset += group.size(0)

        # TODO: Replace this private access with a public EnsembleTable
        # operation returning flattened tables and member table IDs.
        return tuple(
            group_offsets[group_index] + position
            for group_index, position in ensemble_table._locations
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        fitted_categories = []
        for group in ensemble_table._iter_groups():
            group_categories, _ = self._fit_columns(
                group,
                align_codes=False,
            )
            fitted_categories.extend(group_categories)
        self._categories = BufferList(
            BufferList(categories) for categories in fitted_categories
        )

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        fitted_categories = []
        aligned_tables = []
        for group in ensemble_table._iter_groups():
            group_categories, group_tables = self._fit_and_align(group)
            fitted_categories.extend(group_categories)
            aligned_tables.extend(group_tables)

        member_table_ids = self._member_table_ids(ensemble_table)
        output = ensemble_table.replace_tables(
            tables=aligned_tables,
            member_table_ids=member_table_ids,
        )
        self._categories = BufferList(
            BufferList(categories) for categories in fitted_categories
        )
        return output

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        table_ids = self._member_table_ids(ensemble_table)
        aligned_tables = []
        offset = 0
        for group in ensemble_table._iter_groups():
            end = offset + group.size(0)
            aligned_tables.extend(
                self._align_to_categories(
                    group,
                    tuple(
                        cast(Sequence[Tensor], self._categories[index])
                        for index in range(offset, end)
                    ),
                )
            )
            offset = end

        return ensemble_table.replace_tables(
            tables=aligned_tables,
            member_table_ids=table_ids,
        )

    @staticmethod
    def _string_category_lookups(
        input_categories: tuple[StringTensor, ...],
        fitted_categories: tuple[Tensor, ...],
        codes: Tensor,
    ) -> tuple[Tensor, ...]:
        if not input_categories:
            return ()

        # Pair IDs for each input category: (total_input_categories,).
        left_pair = torch.cat(
            [
                codes.new_full((categories.numel(),), pair_index)
                for pair_index, categories in enumerate(input_categories)
            ]
        )
        # Pair IDs for each fitted category: (total_fitted_categories,).
        right_pair = torch.cat(
            [
                codes.new_full((categories.numel(),), pair_index)
                for pair_index, categories in enumerate(fitted_categories)
            ]
        )
        left_index, right_index = join_index(
            left_table=TableTensor(
                columns={"id": ("pair", "value")},
                id=ColumnarTensor(
                    (left_pair, torch.cat(input_categories)),
                ),
            ),
            right_table=TableTensor(
                columns={"id": ("pair", "value")},
                id=ColumnarTensor(
                    (right_pair, torch.cat(fitted_categories)),
                ),
            ),
            left_keys=["pair", "value"],
            right_keys=["pair", "value"],
            dtype=codes.dtype,
        )
        fitted_codes = torch.cat(
            [
                torch.arange(
                    categories.numel(),
                    dtype=codes.dtype,
                    device=codes.device,
                )
                for categories in fitted_categories
            ]
        )
        # Fitted code for each input category: (total_input_categories,).
        lookup = codes.new_full(
            (sum(categories.numel() for categories in input_categories),),
            -1,
        )
        lookup[left_index] = fitted_codes[right_index]
        return lookup.split(
            [categories.numel() for categories in input_categories]
        )

    @staticmethod
    def _category_lookup(
        input_categories: Tensor,
        fitted_categories: Tensor,
        codes: Tensor,
    ) -> Tensor:
        lookup = codes.new_full((input_categories.numel(),), -1)
        if fitted_categories.numel() == 0:
            return lookup

        comparable_categories = input_categories
        sorted_categories = fitted_categories
        if (
            comparable_categories.dtype == torch.bool
            or comparable_categories.dtype in _UNSIGNED_DTYPES
        ):
            comparable_categories = comparable_categories.to(torch.int64)
            sorted_categories = sorted_categories.to(torch.int64)

        sorted_categories, perm = sorted_categories.sort()
        position = torch.searchsorted(
            sorted_categories,
            comparable_categories,
        )
        position = position.clamp(max=sorted_categories.numel() - 1)
        match = sorted_categories[position] == comparable_categories
        left_index = match.nonzero().view(-1)
        right_index = perm[position[left_index]]
        lookup[left_index] = right_index.to(codes.dtype)
        return lookup

    def _align_to_categories(
        self,
        table: TableTensor,
        fitted_categories: Sequence[Sequence[Tensor]],
    ) -> tuple[TableTensor, ...]:
        codes = table.categorical.code
        single_table = codes.dim() == 2
        if single_table:
            codes = codes.unsqueeze(0)
        aligned_codes = torch.full_like(codes, -1)
        lookups_by_column: list[dict[int, Tensor]] = [
            {} for _ in table.categorical.categories
        ]
        string_requests: list[tuple[int, int, StringTensor, Tensor]] = []
        for column_index, input_categories in enumerate(
            table.categorical.categories
        ):
            if input_categories.numel() == 0:
                continue

            lookup_by_categories = lookups_by_column[column_index]
            pending_categories: set[int] = set()
            for batch_categories in fitted_categories:
                column_categories = batch_categories[column_index]
                identity = id(column_categories)
                lookup = lookup_by_categories.get(identity)
                if lookup is not None or identity in pending_categories:
                    continue
                if column_categories.numel() == 0:
                    lookup_by_categories[identity] = codes.new_full(
                        (input_categories.numel(),),
                        -1,
                    )
                    continue
                if isinstance(input_categories, StringTensor) != isinstance(
                    column_categories, StringTensor
                ):
                    raise NotImplementedError(
                        "Cannot align string categories with non-string "
                        "categories"
                    )
                if isinstance(input_categories, StringTensor):
                    pending_categories.add(identity)
                    string_requests.append(
                        (
                            column_index,
                            identity,
                            input_categories,
                            column_categories,
                        )
                    )
                    continue
                lookup_by_categories[identity] = self._category_lookup(
                    input_categories,
                    column_categories,
                    codes,
                )

        string_lookups = self._string_category_lookups(
            input_categories=tuple(
                input_categories
                for _, _, input_categories, _ in string_requests
            ),
            fitted_categories=tuple(
                fitted_categories
                for _, _, _, fitted_categories in string_requests
            ),
            codes=codes,
        )
        for (column_index, identity, _, _), lookup in zip(
            string_requests,
            string_lookups,
            strict=True,
        ):
            lookups_by_column[column_index][identity] = lookup

        for column_index, input_categories in enumerate(
            table.categorical.categories
        ):
            if input_categories.numel() == 0:
                continue
            lookup_by_categories = lookups_by_column[column_index]
            lookups = [
                lookup_by_categories[id(batch_categories[column_index])]
                for batch_categories in fitted_categories
            ]
            column_codes = codes[..., column_index]
            mask = column_codes >= 0
            lookup = torch.stack(lookups)
            indices = column_codes.clamp_min(0).long()
            aligned_codes[..., column_index] = torch.where(
                mask,
                lookup.gather(1, indices),
                -1,
            )

        return tuple(
            (table if single_table else table[batch_index]).replace_blocks(
                categorical=CategoricalTensor(
                    aligned_codes[batch_index],
                    categories=batch_categories,
                ),
            )
            for batch_index, batch_categories in enumerate(fitted_categories)
        )

    def __repr__(self, *, indent: int = 0) -> str:
        arguments = []
        if self.sort_by != "code":
            arguments.append(f"sort_by={self.sort_by!r}")
        if self.min_frequency != 1:
            arguments.append(f"min_frequency={self.min_frequency!r}")
        if not arguments:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}({', '.join(arguments)})"
        )

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast

import torch
from torch import Tensor

from tabtune.models.kumo._sdm import CategoricalTensor, EnsembleTable, Stype, TableTensor
from tabtune.models.kumo._sdm.nn._buffer import BufferList
from tabtune.models.kumo._sdm.processing import EnsembleProcessor


class ShuffleCategories(EnsembleProcessor):
    """Permute the integer codes of categorical columns.

    One permutation per categorical column is drawn when the processor is
    fitted. Codes and their corresponding category vectors are permuted
    together so decoded values remain unchanged. Negative codes represent
    missing values and are preserved unchanged. Only categorical columns
    are supported; use :class:`~sdm.processing.StypeDispatch` to apply this
    processor to the categorical block of a mixed feature table.

    Args:
        method: Permutation strategy. ``"random"`` remaps codes
            with an independent permutation. ``"shift"`` uses a cyclic
            offset, cycling through every offset in random order.
    """

    handles_stypes = frozenset({Stype.categorical})
    requires_fit = True

    def __init__(
        self,
        method: Literal["shift", "random"] = "random",
    ) -> None:
        super().__init__()
        self.method = method
        self._permutations: BufferList[BufferList[Tensor]] = BufferList()
        self._permutation_ids: tuple[int, ...] = ()

    def get_extra_state(self) -> tuple[int, ...]:
        r""":meta private:"""  # noqa: D415
        return self._permutation_ids

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._permutation_ids = cast(tuple[int, ...], state)

    def _draw_permutations(
        self,
        table: TableTensor,
        *,
        unused_offsets: dict[tuple[str, int], Tensor],
        generator: torch.Generator | None = None,
    ) -> list[Tensor]:
        device = table.categorical.device
        permutations: list[Tensor] = []
        for column_index, category in enumerate(table.categorical.categories):
            n_classes = category.numel()
            if n_classes <= 1:
                permutation = torch.arange(n_classes, device=device)
            elif self.method == "shift":
                column_name = table.columns[Stype.categorical][column_index]
                key = (column_name, n_classes)
                offsets = unused_offsets.get(key)
                if offsets is None or offsets.numel() == 0:
                    offsets = torch.randperm(
                        n_classes,
                        generator=generator,
                        device=device,
                    )
                offset = offsets[:1]
                unused_offsets[key] = offsets[1:]
                permutation = (
                    torch.arange(n_classes, device=device) - offset
                ) % n_classes
            else:
                assert self.method == "random"
                permutation = torch.randperm(
                    n_classes,
                    generator=generator,
                    device=device,
                )
            permutations.append(permutation)
        return permutations

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        permutations_by_id = []
        permutation_ids = []
        permutation_id_by_key: dict[tuple[tuple[int, ...], ...], int] = {}

        # (column name, n_classes) -> unused cyclic offsets for that column.
        unused_offsets: dict[tuple[str, int], Tensor] = {}
        for member_id in range(len(ensemble_table)):
            permutations = self._draw_permutations(
                ensemble_table[member_id],
                unused_offsets=unused_offsets,
                generator=generator,
            )
            key = tuple(
                tuple(permutation.tolist()) for permutation in permutations
            )
            permutation_id = permutation_id_by_key.get(key)
            if permutation_id is None:
                permutation_id = len(permutations_by_id)
                permutation_id_by_key[key] = permutation_id
                permutations_by_id.append(permutations)
            permutation_ids.append(permutation_id)

        self._permutations = BufferList(
            BufferList(permutations) for permutations in permutations_by_id
        )
        self._permutation_ids = tuple(permutation_ids)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._permutation_ids) != len(ensemble_table):
            raise RuntimeError(
                "ShuffleCategories must be fitted with the same number of "
                "ensemble members before transform."
            )

        member_ids_by_permutation: dict[int, list[int]] = {}
        for member_id, permutation_id in enumerate(self._permutation_ids):
            member_ids_by_permutation.setdefault(permutation_id, []).append(
                member_id
            )

        if len(member_ids_by_permutation) == len(ensemble_table):
            member_tables: list[TableTensor] = []
            for member_id, permutation_id in enumerate(self._permutation_ids):
                permutations = self._permutations[permutation_id]
                member_tables.append(
                    self._permute(
                        ensemble_table[member_id],
                        permutations,
                    )
                )
            return EnsembleTable.from_tables(
                tables=member_tables,
                member_table_ids=range(len(ensemble_table)),
            )

        outputs: dict[int, EnsembleTable] = {}
        for permutation_id, member_ids in member_ids_by_permutation.items():
            selected = ensemble_table[member_ids]
            permutations = self._permutations[permutation_id]
            outputs[permutation_id] = selected.replace_groups(
                [
                    self._permute(group, permutations)
                    for group in selected._iter_groups()
                ]
            )

        output_tables = []
        member_ids = []
        next_member_id_by_permutation: dict[int, int] = {}
        for permutation_id in self._permutation_ids:
            output_tables.append(outputs[permutation_id])
            member_id = next_member_id_by_permutation.get(permutation_id, 0)
            member_ids.append(member_id)
            next_member_id_by_permutation[permutation_id] = member_id + 1

        return ensemble_table.gather_members(
            tables=output_tables,
            member_ids=member_ids,
        )

    @staticmethod
    def _permute(
        table: TableTensor,
        permutations: BufferList[Tensor],
    ) -> TableTensor:
        code = table.categorical.code.clone()
        valid_mask = table.categorical.isfinite()
        categories: list[Tensor] = []
        for index, (category, permutation) in enumerate(
            zip(table.categorical.categories, permutations, strict=True)
        ):
            codes = code[..., index]
            valid = valid_mask[..., index]
            valid_codes = codes[valid].to(torch.long)
            codes[valid] = permutation[valid_codes].to(codes.dtype)
            categories.append(category[permutation.argsort()])

        categorical = CategoricalTensor(
            code=code,
            categories=categories,
        )
        return table.replace_blocks(categorical=categorical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )

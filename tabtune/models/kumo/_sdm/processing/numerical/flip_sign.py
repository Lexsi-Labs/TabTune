# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from tabtune.models.kumo._sdm import EnsembleTable, Stype
from tabtune.models.kumo._sdm.nn._buffer import BufferList
from tabtune.models.kumo._sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor


class FlipSign(EnsembleProcessor, EnsembleInvertibleMixin):
    """Randomly negate numerical columns.

    Args:
        probability: Probability of negating each numerical column.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.probability = probability
        self._signs: BufferList[Tensor] = BufferList()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        signs = []
        for member_id in range(len(ensemble_table)):
            numerical = ensemble_table[member_id].numerical
            sign = numerical.new_empty(
                (*numerical.size()[:-2], 1, numerical.size(-1))
            )
            sign.bernoulli_(self.probability, generator=generator)
            sign.mul_(-2).add_(1)
            signs.append(sign)
        self._signs = BufferList(signs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:

        member_ids_by_group = [[] for _ in range(ensemble_table.num_groups)]
        for member_id, (group_id, _) in enumerate(ensemble_table._locations):
            member_ids_by_group[group_id].append(member_id)

        groups = []
        locations = [(-1, -1)] * len(ensemble_table)
        for group_id, member_ids in enumerate(member_ids_by_group):
            group = ensemble_table.expanded_group(group_id)
            sign = torch.stack([self._signs[i] for i in member_ids], dim=0)
            group = group.replace_blocks(numerical=group.numerical * sign)
            groups.append(group)
            for position, member_id in enumerate(member_ids):
                locations[member_id] = (group_id, position)

        return EnsembleTable(groups, locations)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return self._transform_ensemble(ensemble_table)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}"
            f"(probability={self.probability})"
        )

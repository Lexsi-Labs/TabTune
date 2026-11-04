# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast

import torch

from tabtune.models.kumo._sdm import EnsembleTable, Stype, TableTensor
from tabtune.models.kumo._sdm.nn._buffer import BufferList
from tabtune.models.kumo._sdm.processing import EnsembleProcessor


class RandomProjection(EnsembleProcessor):
    r"""Randomly project numerical columns.

    Args:
        channels: Number of output numerical columns.
        init: The projection weight initialization.
            ``"normal"`` samples Gaussian weights scaled by
            ``1 / sqrt(channels)``.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        channels: int,
        *,
        init: Literal["normal"] = "normal",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.init = init
        self._weights: BufferList[torch.Tensor] = BufferList()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        weights = []
        for i, group in enumerate(ensemble_table._iter_groups()):
            num_members = ensemble_table.num_members_in_group(i)
            weight = group.numerical.new_empty(
                (
                    num_members,
                    *group.size()[1:-2],
                    self.channels,
                    group.numerical.size(-1),
                )
            )
            assert self.init == "normal"
            weight.normal_(std=self.channels**-0.5, generator=generator)
            weights.append(weight)
        self._weights = BufferList(weights)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        groups = []
        for i in range(ensemble_table.num_groups):
            group = ensemble_table.expanded_group(i)
            projected = TableTensor(
                columns={
                    Stype.numerical: [f"rp_{i}" for i in range(self.channels)]
                },
                numerical=group.numerical @ self._weights[i].transpose(-1, -2),
            )
            group = cast(
                TableTensor,
                torch.cat(
                    [group.drop_stypes(Stype.numerical), projected], dim=-1
                ),
            )
            groups.append(group)

        locations = []
        group_position = [0] * ensemble_table.num_groups
        for group_id, _ in ensemble_table._locations:
            locations.append((group_id, group_position[group_id]))
            group_position[group_id] += 1

        return EnsembleTable(groups=groups, locations=locations)

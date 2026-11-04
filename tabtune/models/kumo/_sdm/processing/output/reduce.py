# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch import Tensor

from tabtune.models.kumo._sdm import EnsembleTable, Stype, TableTensor
from tabtune.models.kumo._sdm.processing import EnsembleProcessor


class AverageEstimators(EnsembleProcessor):
    """Reduce the leading ensemble dimension of model outputs.

    Inputs must be a numerical output table with shape ``[E, ...]``, where
    ``E`` is the non-empty leading ensemble dimension.

    Args:
        trim_fraction: Fraction of estimators discarded from each tail after
            sorting each output coordinate, in ``[0, 0.5)``. This can make the
            aggregation less sensitive to outlying estimator predictions.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(
        self,
        *,
        trim_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("'trim_fraction' must be in [0, 0.5)")
        self.trim_fraction = trim_fraction

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:

        if len(ensemble_table) == 1:
            return ensemble_table

        groups: list[Tensor] = [
            ensemble_table.expanded_group(group_id)
            for group_id in range(ensemble_table.num_groups)
        ]
        table = torch.cat(groups, dim=0) if len(groups) > 1 else groups[0]
        table = cast(TableTensor, table)

        out = table.numerical
        cut = int(self.trim_fraction * table.size(0))
        if cut > 0:
            out = out.sort(dim=0).values
            out = out[cut : out.size(0) - cut]
        out = out.mean(dim=0)

        return EnsembleTable.from_table(
            table=TableTensor(
                columns={Stype.numerical: table.columns[Stype.numerical]},
                numerical=out,
            ),
            num_members=1,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        if table.dim() < 3:
            raise ValueError(
                f"Expected a leading ensemble dimension (got {table.dim()}D)"
            )

        ensemble_table = EnsembleTable(
            groups=(table,),
            locations=tuple((0, i) for i in range(table.size(0))),
        )
        return self._transform_ensemble(ensemble_table)[0]

    def __repr__(self, *, indent: int = 0) -> str:
        if self.trim_fraction == 0:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"trim_fraction={self.trim_fraction})"
        )

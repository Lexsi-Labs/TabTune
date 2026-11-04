# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tabtune.models.kumo._sdm import CategoricalTensor, Stype, TableTensor
from tabtune.models.kumo._sdm.nn._buffer import BufferList
from tabtune.models.kumo._sdm.processing import Processor
from tabtune.models.kumo._sdm.processing.categorical._categorical import (
    _check_categorical_codes,
    _check_categories,
)


class ImputeMode(Processor):
    """Replace missing categorical values with fitted per-column modes.

    Negative category codes are missing values. The fitted fill value is
    the most frequent observed category code learned independently for every
    categorical column and applied without changing its category vocabulary.
    Ties select the lowest category code.

    Transform inputs must use the fitted per-column category vocabularies.
    Use :class:`~sdm.processing.AlignCategories` before this processor when
    training and transform inputs were tensorized independently.
    """

    handles_stypes = frozenset({Stype.categorical})
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        # TODO: Add a separate processor that encodes missing values as their
        # own category instead of imputing an observed one.
        self._categories: BufferList[torch.Tensor] = BufferList()
        self.register_buffer(
            "_fill_values",
            torch.empty(0, dtype=torch.long),
        )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        data = table.categorical
        _check_categorical_codes(table)
        fill_values: list[torch.Tensor] = []
        columns = table.columns[Stype.categorical]
        observed_mask = data.isfinite()
        for index, category in enumerate(table.categorical.categories):
            codes = data[..., index]  # [*batch, n_samples]
            observed = observed_mask[..., index]
            if not bool(observed.any(dim=-1).all()):
                raise ValueError(
                    "Cannot fit 'ImputeMode' because categorical "
                    f"column {columns[index]!r} has no observed values."
                )

            # Accumulate category counts per batch.
            counts = torch.zeros(
                (*codes.shape[:-1], category.numel()),
                dtype=torch.long,
                device=codes.device,
            )
            counts.scatter_add_(
                -1,
                codes.clamp_min(0).to(torch.long),
                observed.to(torch.long),
            )
            fill_values.append(counts.argmax(dim=-1, keepdim=True))

        self._fill_values = (
            torch.stack(fill_values, dim=-1)
            if len(fill_values) > 0
            else torch.empty(0, dtype=torch.long, device=data.device)
        )
        self._categories = BufferList(table.categorical.categories)

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        self._fit(table, generator=generator)
        return self._replace_missing(table)

    def _transform(self, table: TableTensor) -> TableTensor:
        _check_categories(table, self._categories)
        _check_categorical_codes(table)
        return self._replace_missing(table)

    def _replace_missing(self, table: TableTensor) -> TableTensor:
        code = table.categorical.where(
            table.categorical.isfinite(),
            self._fill_values.to(
                dtype=table.categorical.dtype,
                device=table.categorical.device,
            ),
        )
        categorical = CategoricalTensor(
            code=code,
            categories=table.categorical.categories,
        )
        return table.replace_blocks(categorical=categorical)

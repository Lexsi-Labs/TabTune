# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from itertools import repeat
from typing import cast

import torch
from torch.nn import ModuleList

from tabtune.models.kumo._sdm import EnsembleTable, Stype
from tabtune.models.kumo._sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    InvertibleMixin,
    Processor,
)


class EnsembleProcessorAdapter(EnsembleProcessor, EnsembleInvertibleMixin):
    """Adapt an ordinary processor to ensemble-aware processing.

    The adapter turns a :class:`~sdm.processing.base.Processor` into an
    :class:`~sdm.processing.ensemble.EnsembleProcessor`. It copies and fits the
    processor separately for each group of compatible tables in an
    :class:`~sdm.EnsembleTable`.

    The wrapped processor must preserve row and leading dimensions as required
    by the :class:`~sdm.processing.base.Processor` contract. A processor that
    changes the ensemble structure must implement
    :class:`~sdm.processing.ensemble.EnsembleProcessor` directly. Inverse
    transformation requires the wrapped processor to implement
    :class:`~sdm.processing.base.InvertibleMixin`.

    Args:
        processor: Processor to fit separately for each ensemble table group.
    """

    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.processor = processor
        self.requires_fit = processor.requires_fit

        self._processors: ModuleList[Processor] = ModuleList()
        self._fitted_locations: tuple[tuple[int, int], ...] = ()

    def get_extra_state(
        self,
    ) -> tuple[int, tuple[tuple[int, int], ...]]:
        r""":meta private:"""  # noqa: D415
        return len(self._processors), self._fitted_locations

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        num_processors, self._fitted_locations = cast(
            tuple[int, tuple[tuple[int, int], ...]], state
        )
        self._processors = ModuleList(
            [copy.deepcopy(self.processor) for _ in range(num_processors)]
        )

    @property
    def handles_stypes(self) -> frozenset[Stype]:
        r""":meta private:"""  # noqa: D415
        return self.processor.handles_stypes

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fitted_locations = ensemble_table._locations
        self._processors = ModuleList()
        for group_id, group in enumerate(ensemble_table._iter_groups()):
            if group_id == 0:
                processor = self.processor
            else:
                processor = copy.deepcopy(self.processor)
            processor.fit(group, generator=generator)
            self._processors.append(processor)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if not self.requires_fit:
            return self._transform_ensemble(ensemble_table)

        self._fitted_locations = ensemble_table._locations
        outputs = []
        self._processors = ModuleList()
        for group_id, group in enumerate(ensemble_table._iter_groups()):
            if group_id == 0:
                processor = self.processor
            else:
                processor = copy.deepcopy(self.processor)
            outputs.append(processor.fit_transform(group, generator=generator))
            self._processors.append(processor)
        return ensemble_table.replace_groups(outputs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.requires_fit:
            processors = self._aligned_processors(ensemble_table)
        else:
            processors = repeat(self.processor, ensemble_table.num_groups)
        outputs = [
            processor.transform(group)
            for group, processor in zip(
                ensemble_table._iter_groups(), processors, strict=True
            )
        ]
        return ensemble_table.replace_groups(outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.requires_fit:
            processors = self._aligned_processors(ensemble_table)
        else:
            processors = repeat(self.processor, ensemble_table.num_groups)

        outputs = []
        for group, processor in zip(
            ensemble_table._iter_groups(), processors, strict=True
        ):
            if not isinstance(processor, InvertibleMixin):
                raise AttributeError(
                    f"{self.processor.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform'"
                )
            outputs.append(processor.inverse_transform(group))
        return ensemble_table.replace_groups(outputs)

    def _aligned_processors(
        self,
        ensemble_table: EnsembleTable,
    ) -> tuple[Processor, ...]:
        if ensemble_table._locations == self._fitted_locations:
            return tuple(self._processors)

        # One fitted processor per transform ensemble group. Multiple
        # transform groups can share the same fitted processor.
        processors: list[Processor | None] = [None] * ensemble_table.num_groups
        # Current and fitted batch of every member sharing a fitted processor.
        batches_by_processor: dict[int, list[tuple[int, int, int]]] = {}
        for member_id, (group_id, batch_id) in enumerate(
            ensemble_table._locations
        ):
            fitted_group_id, fitted_batch_id = self._fitted_locations[
                member_id
            ]
            processor = self._processors[fitted_group_id]
            if processors[group_id] is None:
                processors[group_id] = processor
            elif processors[group_id] is not processor:
                raise RuntimeError(
                    "Cannot apply fitted processor state to an ensemble "
                    "group containing members from different fitted groups"
                )
            batches_by_processor.setdefault(fitted_group_id, []).append(
                (group_id, batch_id, fitted_batch_id)
            )

        # A processor fitted on a stacked tensor table stores per-batch state.
        # It can only operate on stacked tensor tables with the same layout,
        # but not on splits or reordered batches.
        for batches in batches_by_processor.values():
            if len({fitted for _, _, fitted in batches}) > 1 and (
                len({group for group, _, _ in batches}) > 1
                or any(current != fitted for _, current, fitted in batches)
            ):
                raise RuntimeError(
                    "Cannot apply position-dependent fitted processor state "
                    "after its ensemble group was split or reordered"
                )

        return tuple(cast(Processor, processor) for processor in processors)

    def __repr__(self, *, indent: int = 0) -> str:
        return self.processor.__repr__(indent=indent)

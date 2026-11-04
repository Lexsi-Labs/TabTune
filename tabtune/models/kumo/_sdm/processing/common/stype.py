# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch.nn import ModuleDict

from tabtune.models.kumo._sdm import EnsembleTable, Stype
from tabtune.models.kumo._sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    Processor,
)


class StypeDispatch(EnsembleProcessor, EnsembleInvertibleMixin):
    r"""Apply separate processor pipelines to columns grouped by semantic type.

    For each configured route, matching columns from a
    :class:`~sdm.tensor.TableTensor` or
    :class:`~sdm.EnsembleTable` are passed to that processor. Ordinary
    processors learn separate state for each compatible ensemble group, while
    ensemble-aware processors operate on all groups directly. Compatible
    members are processed together and route outputs are concatenated in
    semantic type order while preserving logical member order. A route may
    change column values, names, count, or order. With the default passthrough
    behavior, unconfigured semantic types follow in input order.

    Inverse transform supports routes that preserve their semantic type. Every
    active route must be invertible, and routes must not share an output
    semantic type. Passthrough columns are preserved.

    Args:
        numerical: Processor or stateless callable route for numerical
            columns.
        categorical: Processor or stateless callable route for categorical
            columns.
        datetime: Processor or stateless callable route for datetime columns.
        text: Processor or stateless callable route for text columns.
        id: Processor or stateless callable route for identifier columns.
    """

    def __init__(
        self,
        *,
        numerical: object = None,
        categorical: object = None,
        datetime: object = None,
        id: object = None,
        text: object = None,
    ) -> None:
        super().__init__()
        self.processors: ModuleDict[EnsembleProcessor] = ModuleDict()
        for stype, processor in (
            (Stype.numerical, numerical),
            (Stype.categorical, categorical),
            (Stype.datetime, datetime),
            (Stype.text, text),
            (Stype.id, id),
        ):
            if processor is None:
                continue
            processor = EnsembleProcessor.as_processor(processor)
            self.processors[str(stype)] = processor

        self.handles_stypes = frozenset(
            Stype(stype) for stype in self.processors
        )
        self.requires_fit = any(
            processor.requires_fit for processor in self.processors.values()
        )
        self._active_routes: tuple[str, ...] | None = None

    def get_extra_state(
        self,
    ) -> tuple[str, ...] | None:
        r""":meta private:"""  # noqa: D415
        return self._active_routes

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._active_routes = cast(tuple[str, ...] | None, state)

    def _find_active_routes(
        self,
        ensemble_table: EnsembleTable,
    ) -> tuple[str, ...]:
        return tuple(
            stype
            for stype in self.processors
            if any(
                len(group.columns[Stype(stype)]) > 0
                for group in ensemble_table._iter_groups()
            )
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._active_routes = self._find_active_routes(ensemble_table)
        for stype in self._active_routes:
            self.processors[stype].fit_ensemble(
                ensemble_table.select_stypes(stype),
                generator=generator,
            )

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self._active_routes = self._find_active_routes(ensemble_table)
        outputs = [
            self.processors[stype].fit_transform_ensemble(
                ensemble_table.select_stypes(stype),
                generator=generator,
            )
            for stype in self._active_routes
        ]
        outputs.append(self._remainder_ensemble(ensemble_table))
        if len(outputs) == 0:
            return ensemble_table.select_stypes(())
        return EnsembleTable.concatenate_columns(outputs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        active_routes = self._active_routes
        if not self.requires_fit or active_routes is None:
            active_routes = self._find_active_routes(ensemble_table)
            self._active_routes = active_routes
        outputs = [
            self.processors[stype].transform_ensemble(
                ensemble_table.select_stypes(stype)
            )
            for stype in active_routes
        ]
        outputs.append(self._remainder_ensemble(ensemble_table))
        if len(outputs) == 0:
            return ensemble_table.select_stypes(())
        return EnsembleTable.concatenate_columns(outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        active_routes = self._active_routes
        if not self.requires_fit or active_routes is None:
            active_routes = self._find_active_routes(ensemble_table)
            self._active_routes = active_routes

        outputs = []
        for stype in active_routes:
            processor = self.processors[stype]
            if not isinstance(processor, EnsembleInvertibleMixin):
                raise TypeError(
                    f"{processor.__class__.__name__!r} is not invertible."
                )
            # TODO This is wrong. A processor may not map to same stype back!
            outputs.append(
                processor.inverse_transform_ensemble(
                    ensemble_table.select_stypes(stype)
                )
            )

        outputs.append(self._remainder_ensemble(ensemble_table))
        return EnsembleTable.concatenate_columns(outputs)

    def _remainder_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return ensemble_table.select_stypes(
            tuple(
                stype for stype in Stype if str(stype) not in self.processors
            )
        )

    def __repr__(self, *, indent: int = 0) -> str:
        if len(self.processors) == 0:
            return super().__repr__(indent=indent)

        reprs = []
        for stype, processor in self.processors.items():
            processor = cast(Processor, processor)
            processor_repr = processor.__repr__(indent=indent + 2)
            processor_repr = processor_repr[indent + 2 :]
            reprs.append(f"{' ' * (indent + 2)}{stype}={processor_repr}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )

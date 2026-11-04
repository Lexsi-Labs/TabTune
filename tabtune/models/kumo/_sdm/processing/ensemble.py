# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from typing_extensions import Self

import torch

from tabtune.models.kumo._sdm import EnsembleTable, TableTensor
from tabtune.models.kumo._sdm.processing import InvertibleMixin, Processor


class EnsembleProcessor(Processor):
    r"""Base processor for ensemble-aware table transformations.

    An :class:`EnsembleProcessor` defines a reusable transformation on
    :class:`~sdm.EnsembleTable` for feature, target and output
    pre/post-processing across ensemble members.
    An :class:`EnsembleProcessor` learns any required state via
    :meth:`fit_ensemble` and applies the transformation via
    :meth:`transform_ensemble`. :meth:`fit_ensemble`,
    :meth:`transform_ensemble`, and :meth:`fit_transform_ensemble`
    are no-ops for stypes outside of
    :attr:`~sdm.processing.base.Processor.handles_stypes`.

    As a :class:`~sdm.processing.base.Processor`, it also accepts a
    :class:`~sdm.tensor.TableTensor` and processes it as an ensemble
    with one member.
    """

    @staticmethod
    def as_processor(processor: object) -> EnsembleProcessor:
        r"""Normalize a processor-like object to a :class:`EnsembleProcessor`.

        Args:
            processor: A processor-like object. An :class:`EnsembleProcessor`
                is returned as-is, and any ordinary processor-like object is
                adapted to ensemble processing.
        """
        from tabtune.models.kumo._sdm.processing import EnsembleProcessorAdapter  # noqa: PLC0415

        processor = Processor.as_processor(processor)
        if isinstance(processor, EnsembleProcessor):
            return processor
        return EnsembleProcessorAdapter(processor)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_ensemble(
            EnsembleTable.from_table(table, num_members=1),
            generator=generator,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        output = self._transform_ensemble(
            EnsembleTable.from_table(table, num_members=1)
        )
        return output[0]

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        if not self.requires_fit:
            return self._transform(table)
        output = self._fit_transform_ensemble(
            EnsembleTable.from_table(table, num_members=1),
            generator=generator,
        )
        return output[0]

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        pass

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if self.requires_fit:
            self._fit_ensemble(ensemble_table, generator=generator)
        return self._transform_ensemble(ensemble_table)

    def fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        """Fit the processor on an ensemble table.

        Args:
            ensemble_table: Ensemble table used to compute the processor state.
            generator: Pseudorandom number generator used for sampling.
        """
        if not any(
            group.active_stypes & self.handles_stypes
            for group in ensemble_table._iter_groups()
        ):
            return self
        if self.requires_fit:
            self._fit_ensemble(ensemble_table, generator=generator)
            self._set_fitted(ensemble_table[0].device)
        return self

    def transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        """Transform an ensemble table with fitted state.

        Args:
            ensemble_table: Ensemble table to transform.

        Returns:
            The transformed ensemble table.
        """
        if not any(
            group.active_stypes & self.handles_stypes
            for group in ensemble_table._iter_groups()
        ):
            return ensemble_table
        self._check_is_fitted()
        return self._transform_ensemble(ensemble_table)

    def fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        """Fit the processor and transform an ensemble table.

        Args:
            ensemble_table: Ensemble table to fit on and transform.
            generator: Pseudorandom number generator used for sampling.

        Returns:
            The transformed ensemble table.
        """
        if not any(
            group.active_stypes & self.handles_stypes
            for group in ensemble_table._iter_groups()
        ):
            return ensemble_table
        output = self._fit_transform_ensemble(
            ensemble_table,
            generator=generator,
        )
        if self.requires_fit:
            self._set_fitted(ensemble_table[0].device)
        return output


class EnsembleInvertibleMixin(InvertibleMixin):
    r"""Extend a :class:`EnsembleProcessor` by an inverse transformation."""

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        output = self._inverse_transform_ensemble(
            EnsembleTable.from_table(table, num_members=1)
        )
        return output[0]

    @abc.abstractmethod
    def _inverse_transform_ensemble(
        self, ensemble_table: EnsembleTable
    ) -> EnsembleTable: ...

    def inverse_transform_ensemble(
        self, ensemble_table: EnsembleTable
    ) -> EnsembleTable:
        r"""Apply the inverse transformation to ``ensemble_table``.

        Args:
            ensemble_table: The ensemble table in transformed representation.

        Returns:
            The table restored to the representation before
            :meth:`~EnsembleProcessor.transform_ensemble`.
        """
        self._check_is_fitted()
        return self._inverse_transform_ensemble(ensemble_table)

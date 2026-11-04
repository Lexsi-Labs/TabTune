# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Iterator
from typing import cast
from typing_extensions import Self

import torch

from tabtune.models.kumo._sdm import EnsembleTable, Stype, TableTensor
from tabtune.models.kumo._sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
)


class Sequential(EnsembleProcessor, EnsembleInvertibleMixin):
    r"""Apply processors and callables to a table in sequence.

    Args:
        args: Sequence of :class:`Processor` instances or callables.
    """

    def __init__(self, *args: object) -> None:
        super().__init__()
        self.requires_fit = False
        self.extend(args)

    @property
    def handles_stypes(self) -> frozenset[Stype]:
        r""":meta private:"""  # noqa: D415
        if len(self) == 0:
            return frozenset(Stype)
        return frozenset(
            stype for child in self for stype in child.handles_stypes
        )

    def append(self, processor: object) -> Self:
        r"""Append a processor or callable to this sequence.

        Args:
            processor: The processor to append.
        """
        processor = EnsembleProcessor.as_processor(processor)
        if isinstance(processor, Sequential):
            for child in processor.children():
                self.add_module(str(len(self)), child)
        else:
            self.add_module(str(len(self)), processor)

        self.requires_fit = any(child.requires_fit for child in self)
        self._fitted = False
        return self

    def prepend(self, processor: object) -> Self:
        r"""Prepend a processor or callable to this sequence.

        Args:
            processor: The processor to prepend.
        """
        processor = EnsembleProcessor.as_processor(processor)

        if isinstance(processor, Sequential):
            processors = tuple(processor.children())
        else:
            processors = (processor,)

        processors = (*processors, *self.children())

        self._modules.clear()
        for child in processors:
            self.add_module(str(len(self)), child)

        self.requires_fit = any(child.requires_fit for child in self)
        self._fitted = False
        return self

    def extend(self, processors: Iterable[object]) -> Self:
        r"""Append multiple processors or callables to this sequence.

        Args:
            processors: The processors to append.
        """
        for processor in processors:
            self.append(processor)
        return self

    def _transform(self, table: TableTensor) -> TableTensor:
        out = table
        for child in self:
            out = child.transform(out)
        return out

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        out = ensemble_table
        for i, child in enumerate(self):
            if i < len(self) - 1:
                out = child.fit_transform_ensemble(out, generator=generator)
            else:
                child.fit_ensemble(out, generator=generator)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        out = ensemble_table
        for child in self:
            out = child.fit_transform_ensemble(out, generator=generator)
        return out

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        out = ensemble_table
        for child in self:
            out = child.transform_ensemble(out)
        return out

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        out = ensemble_table
        for child in reversed(list(self)):
            if not isinstance(child, EnsembleInvertibleMixin):
                raise AttributeError(
                    f"{child.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform_ensemble'"
                )
            out = child.inverse_transform_ensemble(out)
        return out

    def __iter__(self) -> Iterator[EnsembleProcessor]:
        return cast(Iterator[EnsembleProcessor], self.children())

    def __len__(self) -> int:
        return len(self._modules)

    def __iadd__(self, other: object) -> Self:
        self.append(EnsembleProcessor.as_processor(other))
        return self

    def __repr__(self, *, indent: int = 0) -> str:
        if len(self) == 0:
            return super().__repr__(indent=indent)
        reprs = "".join(
            [f"{child.__repr__(indent=indent + 2)},\n" for child in self]
        )
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n{reprs}{' ' * indent})"
        )

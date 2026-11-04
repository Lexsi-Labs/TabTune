# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from typing_extensions import Self

import torch

from tabtune.models.kumo._sdm import Stype, TableTensor

if TYPE_CHECKING:
    from tabtune.models.kumo._sdm.processing import Sequential


class Processor(torch.nn.Module, abc.ABC):
    r"""Base processor for tensor-aware table transformations.

    A :class:`Processor` defines a reusable transformation on
    :class:`~sdm.tensor.TableTensor` for feature, target and output
    preprocessing.
    A :class:`Processor` learns any required state via :meth:`fit`, and applies
    the transformation via :meth:`transform`. Implementations preserve batch
    dimensions, and preserve rows unless documented otherwise. Batch
    dimensions are processed independently.

    :meth:`fit`, :meth:`transform`, and :meth:`fit_transform` are no-ops for
    stypes outside of :attr:`handles_stypes`.
    """

    #: Semantic types this processor operates on.
    handles_stypes: frozenset[Stype]

    #: Whether this processor requires fitting.
    requires_fit: bool

    _fitted_state: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_fitted_state", torch.tensor(False))
        self.__fitted = False

    @property
    def _fitted(self) -> bool:
        return self.__fitted

    @_fitted.setter
    def _fitted(self, value: bool) -> None:
        self.__fitted = value
        self._fitted_state.fill_(value)

    @property
    def is_fitted(self) -> bool:
        """Whether the processor has all state required for transformation."""
        return not self.requires_fit or self._fitted

    @staticmethod
    def as_processor(processor: object) -> Processor:
        r"""Normalize a processor-like object to a :class:`Processor`.

        Args:
            processor: A processor-like object. A :class:`Processor` is
                returned as-is, a callable is wrapped as a stateless processor,
                and a sequence of processor-like objects is normalized to
                :class:`~sdm.processing.common.Sequential`.
        """
        from tabtune.models.kumo._sdm.processing import Callable, Sequential  # noqa: PLC0415

        if isinstance(processor, Processor):
            return processor
        if callable(processor):
            return Callable(processor)  # type: ignore
        if isinstance(processor, Sequence) and not isinstance(processor, str):
            return Sequential(*processor)
        raise TypeError(
            f"Input must be a 'Processor', callable, or sequence of them "
            f"(got '{type(processor).__name__}')"
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Resize dynamically shaped buffers before PyTorch copies saved values.
        for name, buffer in self._buffers.items():
            state = state_dict.get(f"{prefix}{name}")
            if (
                buffer is not None
                and isinstance(state, torch.Tensor)
                and buffer.shape != state.shape
            ):
                buffer.resize_(state.shape)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.__fitted = bool(self._fitted_state)

    def _set_fitted(self, device: torch.device) -> None:
        self._fitted_state = self._fitted_state.to(device=device)
        self._fitted = True

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__!r} is not fitted; "
                "call 'fit()' before."
            )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _transform(self, table: TableTensor) -> TableTensor:
        pass

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        if self.requires_fit:
            self._fit(table, generator=generator)
        return self._transform(table)

    def fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        r"""Fit the processor.

        Args:
            table: The table used to compute the processor state.
            generator: Pseudorandom number generator used for sampling.
        """
        if not table.active_stypes & self.handles_stypes:
            return self
        if self.requires_fit:
            self._fit(table, generator=generator)
            self._set_fitted(table.device)
        return self

    def transform(self, table: TableTensor) -> TableTensor:
        r"""Transform ``table``.

        Args:
            table: The table to transform.

        Returns:
            The transformed table.
        """
        if not table.active_stypes & self.handles_stypes:
            return table
        self._check_is_fitted()
        return self._transform(table)

    def forward(self, table: TableTensor) -> TableTensor:
        r"""Alias of :meth:`transform`."""
        return self.transform(table)

    def fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        r"""Fit the processor and transform ``table``.

        Args:
            table: The table to fit on and transform.
            generator: Pseudorandom number generator used for sampling.

        Returns:
            The transformed table.
        """
        if not table.active_stypes & self.handles_stypes:
            return table
        out = self._fit_transform(table, generator=generator)
        if self.requires_fit:
            self._set_fitted(table.device)
        return out

    def __add__(self, other: object) -> Sequential:
        from tabtune.models.kumo._sdm.processing import Sequential  # noqa: PLC0415

        try:
            other = Processor.as_processor(other)
        except TypeError:
            return NotImplemented
        return Sequential(self, other)

    def __radd__(self, other: object) -> Sequential:
        from tabtune.models.kumo._sdm.processing import Sequential  # noqa: PLC0415

        try:
            other = Processor.as_processor(other)
        except TypeError:
            return NotImplemented
        return Sequential(other, self)

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}()"


class InvertibleMixin(abc.ABC):
    r"""Extend a :class:`Processor` by an inverse transformation."""

    @abc.abstractmethod
    def _inverse_transform(self, table: TableTensor) -> TableTensor: ...

    def inverse_transform(self, table: TableTensor) -> TableTensor:
        r"""Apply the inverse transformation to ``table``.

        Args:
            table: The table in transformed representation.

        Returns:
            The table restored to the representation before
            :meth:`~Processor.transform`.
        """
        self._check_is_fitted()
        return self._inverse_transform(table)

    if TYPE_CHECKING:
        # Provided at runtime by `Processor` via the MRO.
        def _check_is_fitted(self) -> None: ...

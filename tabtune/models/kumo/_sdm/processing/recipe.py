# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing_extensions import Self

import tabtune.models.kumo._sdm.processing as sp
from tabtune.models.kumo._sdm.processing import EnsembleProcessor, Processor


@dataclass(init=False, repr=False)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe bundles three processing pipelines, one per role the data plays
    relative to the model:

    - ``features``: model inputs, transformed before the model.
    - ``target``: labels transformed forward before the model. Regression
      predictions are inverted through this pipeline; classification outputs
      are reconstructed from the fitted target categories instead.
    - ``output``: transforms member outputs after they have been mapped to a
      common class or target space and stacked as ``[E, ..., R, O]``. An
      explicit dimension-changing step such as
      :class:`~sdm.processing.AverageEstimators` removes ``E``; without one,
      the output remains stacked. Steps before the reducer must support
      stacked outputs, while steps after it receive already-reduced outputs.

    Each pipeline exposes ``fit``/``transform``/``fit_transform`` and, when its
    steps are invertible, ``inverse_transform``. Call them directly, e.g.
    ``recipe.features.transform(table)`` or
    ``recipe.target.inverse_transform(prediction)``. Recipes do not infer each
    step's non-finite input contract; order steps so values are imputed before
    processors that do not explicitly document non-finite support.

    Args:
        features: Steps applied to model inputs before model execution.
        target: Steps applied to labels before model execution.
        output: Steps applied to stacked model outputs.
    """

    _features: EnsembleProcessor
    _target: EnsembleProcessor
    _output: EnsembleProcessor

    def __init__(
        self,
        features: Processor | Iterable[Processor] | None = None,
        target: Processor | Iterable[Processor] | None = None,
        output: Processor | Iterable[Processor] | None = None,
    ) -> None:
        self.features = features
        self.target = target
        self.output = output

    @property
    def features(self) -> EnsembleProcessor:
        r"""The steps applied to model inputs."""
        return self._features

    @features.setter
    def features(
        self,
        processor: Processor | Iterable[Processor] | None,
    ) -> None:
        if processor is None:
            processor = sp.Identity()
        self._features = EnsembleProcessor.as_processor(processor)

    @property
    def target(self) -> EnsembleProcessor:
        r"""The steps applied to labels."""
        return self._target

    @target.setter
    def target(
        self,
        processor: Processor | Iterable[Processor] | None,
    ) -> None:
        if processor is None:
            processor = sp.Identity()
        self._target = EnsembleProcessor.as_processor(processor)

        if any(isinstance(m, sp.TaskDispatch) for m in self.target.modules()):
            raise ValueError(
                "'TaskDispatch' is not supported in 'Recipe.target'"
            )
        if any(isinstance(m, sp.TableDispatch) for m in self.target.modules()):
            raise ValueError(
                "'TableDispatch' is not supported in 'Recipe.target'"
            )

    @property
    def output(self) -> EnsembleProcessor:
        r"""The steps applied to model outputs."""
        return self._output

    @output.setter
    def output(
        self,
        processor: Processor | Iterable[Processor] | None,
    ) -> None:
        if processor is None:
            processor = sp.Identity()
        self._output = EnsembleProcessor.as_processor(processor)

        if any(isinstance(m, sp.TableDispatch) for m in self.output.modules()):
            raise ValueError(
                "'TableDispatch' is not supported in 'Recipe.output'"
            )
        if self.output.requires_fit:
            raise ValueError("'Recipe.output' should not require fitting")

    def prepend_features(self, processor: object) -> Self:
        """Prepend a processor to the feature pipeline."""
        self.features = processor + self.features
        return self

    def append_features(self, processor: object) -> Self:
        """Append a processor to the feature pipeline."""
        self.features = self.features + processor
        return self

    def prepend_target(self, processor: object) -> Self:
        """Prepend a processor to the target pipeline."""
        self.target = processor + self.target
        return self

    def append_target(self, processor: object) -> Self:
        """Append a processor to the target pipeline."""
        self.target = self.target + processor
        return self

    def prepend_output(self, processor: object) -> Self:
        """Prepend a processor to the output pipeline."""
        self.output = processor + self.output
        return self

    def append_output(self, processor: object) -> Self:
        """Append a processor to the output pipeline."""
        self.output = self.output + processor
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  features={self.features.__repr__(indent=2)[2:]},\n"
            f"  target={self.target.__repr__(indent=2)[2:]},\n"
            f"  output={self.output.__repr__(indent=2)[2:]},\n"
            ")"
        )

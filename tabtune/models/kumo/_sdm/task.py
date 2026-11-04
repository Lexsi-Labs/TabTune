# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tabtune.models.kumo._sdm._compat import StrEnum
from typing import TypeAlias

from tabtune.models.kumo._sdm import Stype, StypeLike


class Task(StrEnum):
    r"""The prediction task type.

    A task type describes the kind of target a model predicts.
    Possible values are:

    Attributes:
        classification: Binary, multi-class, or multi-label classification.
        regression: Continuous numerical prediction.
    """

    classification = "classification"
    regression = "regression"

    @classmethod
    def from_stype(cls, stype: StypeLike) -> Task:
        r"""The prediction task associated with a target semantic type.

        Args:
            stype: The target semantic type.
        """
        if stype == Stype.categorical:
            return cls.classification
        if stype == Stype.numerical:
            return cls.regression
        raise ValueError(
            f"Cannot map semantic type {str(stype)!r} to prediction task"
        )

    @property
    def stype(self) -> Stype:
        r"""The semantic type associated with a prediction task."""
        if self == Task.classification:
            return Stype.categorical
        assert self == Task.regression
        return Stype.numerical

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


TaskLike: TypeAlias = Task | str

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Processors for structured data tables."""

from tabtune.models.kumo._sdm.processing.base import Processor, InvertibleMixin
from tabtune.models.kumo._sdm.processing.ensemble import (
    EnsembleProcessor,
    EnsembleInvertibleMixin,
)
from tabtune.models.kumo._sdm.processing.common import (
    Identity,
    Callable,
    Sequential,
    DropStypes,
    StypeDispatch,
    TaskDispatch,
    TableDispatch,
    EnsembleProcessorAdapter,
    Choice,
    ToNumerical,
    ShuffleColumns,
    SelectColumns,
)
from tabtune.models.kumo._sdm.processing.numerical import (
    Cast,
    Clip,
    ClipQuantiles,
    ClipSigma,
    ClipSoft,
    ImputeMean,
    PowerTransform,
    QuantileTransform,
    RankGaussian,
    Standardize,
    RobustScale,
    FlipSign,
    DropConstantColumns,
    PCA,
    RandomProjection,
)
from tabtune.models.kumo._sdm.processing.categorical import (
    AlignCategories,
    ShuffleCategories,
    ImputeMode,
    AddCategoryCounts,
)
from tabtune.models.kumo._sdm.processing.datetime import AddCalendarFields
from tabtune.models.kumo._sdm.processing.output import AverageEstimators, Softmax, SortQuantiles
from tabtune.models.kumo._sdm.processing.recipe import Recipe

__all__ = [
    "Processor",
    "InvertibleMixin",
    "EnsembleInvertibleMixin",
    "EnsembleProcessorAdapter",
    "Identity",
    "Callable",
    "Sequential",
    "DropStypes",
    "StypeDispatch",
    "TaskDispatch",
    "TableDispatch",
    "EnsembleProcessor",
    "Choice",
    "ToNumerical",
    "ShuffleColumns",
    "SelectColumns",
    "Cast",
    "Clip",
    "ClipQuantiles",
    "ClipSigma",
    "ClipSoft",
    "ImputeMean",
    "PowerTransform",
    "QuantileTransform",
    "RankGaussian",
    "Standardize",
    "RobustScale",
    "FlipSign",
    "DropConstantColumns",
    "PCA",
    "RandomProjection",
    "AlignCategories",
    "ShuffleCategories",
    "ImputeMode",
    "AddCategoryCounts",
    "AddCalendarFields",
    "AverageEstimators",
    "Softmax",
    "SortQuantiles",
    "Recipe",
]

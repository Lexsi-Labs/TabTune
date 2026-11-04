# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured data modeling primitives."""

from importlib.metadata import PackageNotFoundError, version

from tabtune.models.kumo._sdm._constants import NaT
from tabtune.models.kumo._sdm.stype import Stype, StypeLike, infer_stypes
from tabtune.models.kumo._sdm.task import Task, TaskLike
from tabtune.models.kumo._sdm.tensor import (
    VarLenTensor,
    StringTensor,
    NullableTensor,
    CategoricalTensor,
    ColumnarTensor,
    TableTensor,
)
from tabtune.models.kumo._sdm.ensemble import EnsembleData, EnsembleTable
from tabtune.models.kumo._sdm.relational import (
    Relationship,
    RelationalData,
    TaskLink,
    RelatedTables,
)
from tabtune.models.kumo._sdm.processing import Recipe

try:
    __version__ = version("structured-data-models")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "NaT",
    "Stype",
    "StypeLike",
    "Task",
    "TaskLike",
    "infer_stypes",
    "VarLenTensor",
    "StringTensor",
    "NullableTensor",
    "CategoricalTensor",
    "ColumnarTensor",
    "TableTensor",
    "EnsembleData",
    "EnsembleTable",
    "Relationship",
    "RelationalData",
    "TaskLink",
    "RelatedTables",
    "Recipe",
    "__version__",
]

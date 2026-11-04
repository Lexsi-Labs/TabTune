# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from torch import Tensor

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.nn._buffer import BufferList


def _check_categorical_codes(table: TableTensor) -> None:
    """Raise if a categorical code exceeds its column's category vocabulary.

    Negative codes encode missing values and are allowed.

    Args:
        table: The table whose categorical codes are validated.
    """
    if os.getenv("SDM_ENABLE_CATEGORY_CHECKS", "1") == "0":
        return
    columns = table.columns[Stype.categorical]
    for index, category in enumerate(table.categorical.categories):
        codes = table.categorical[..., index]
        if (codes >= category.numel()).any():
            raise ValueError(
                f"Categorical column {columns[index]!r} contains a code "
                "outside its category vocabulary."
            )


def _check_categories(
    table: TableTensor,
    categories: BufferList[Tensor],
) -> None:
    """Raise if a category vocabulary differs from the fitted one.

    Args:
        table: The table whose category vocabularies are validated.
        categories: The fitted category vocabulary of every column.
    """
    if os.getenv("SDM_ENABLE_CATEGORY_CHECKS", "1") == "0":
        return
    columns = table.columns[Stype.categorical]
    if len(table.categorical.categories) != len(categories):
        raise ValueError(
            f"Expected {len(categories)} fitted categorical "
            f"columns (got {len(columns)})."
        )
    for index, (actual, expected) in enumerate(
        zip(table.categorical.categories, categories)
    ):
        expected = expected.to(device=actual.device)
        if not actual.equal(expected):
            raise ValueError(
                "Expected the category vocabulary for categorical column "
                f"{columns[index]!r} to match the fitted values and order. "
                "Use 'AlignCategories' before this processor for "
                "independently tensorized inputs."
            )

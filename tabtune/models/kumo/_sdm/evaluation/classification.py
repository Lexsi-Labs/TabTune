# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch import Tensor

from tabtune.models.kumo._sdm import CategoricalTensor, StringTensor, Stype, TableTensor


def to_class_indices(
    pred: TableTensor,
    target: TableTensor | CategoricalTensor,
    *,
    missing_score: float | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Convert classification predictions and targets to class-index form.

    Args:
        pred: Prediction table whose numerical columns contain class scores.
            Column names define the class order.
        target: Single-column categorical target.
        missing_score: Score assigned to target classes that are absent from
            the prediction columns. If ``None``, will raise an error when a
            ``target`` value refers to an absent class in ``pred``.

    >>> import torch
    >>> import sdm
    >>> from sdm.evaluation import to_class_indices
    >>> pred = sdm.TableTensor.from_tensor(
    ...     torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]),
    ...     columns=["cat", "dog", "bird"],
    ... )
    >>> target = sdm.CategoricalTensor(
    ...     code=torch.tensor([[0], [2]]),
    ...     categories=(sdm.StringTensor.from_list(["cat", "dog", "bird"]),),
    ... )
    >>> class_scores, class_indices = to_class_indices(pred, target)
    >>> class_scores
    tensor([[0.8000, 0.1000, 0.1000],
            [0.1000, 0.2000, 0.7000]])
    >>> class_indices
    tensor([0, 2])
    >>> pred = sdm.TableTensor.from_tensor(
    ...     torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.1, 0.2]]),
    ...     columns=["bird", "cat", "dog"],
    ... )
    >>> class_scores, class_indices = to_class_indices(pred, target)
    >>> class_scores
    tensor([[0.8000, 0.1000, 0.1000],
            [0.1000, 0.2000, 0.7000]])
    >>> class_indices
    tensor([0, 2])
    """
    if isinstance(target, TableTensor):
        if target.size(-1) != 1:
            raise ValueError(
                f"Expected a single categorical target column "
                f"(got {target.size(-1)} columns)"
            )
        target = target.categorical

    if target.size(-1) != 1:
        raise ValueError(
            f"Expected a single categorical target column "
            f"(got {target.size(-1)} columns)"
        )

    if target.isnan().any():
        raise ValueError("Expected target to not contain missing values")

    if pred.size()[:-1] != target.size()[:-1]:
        raise ValueError(
            f"Expected prediction and target row dimensions to match "
            f"(got {tuple(pred.size()[:-1])} and "
            f"{tuple(target.size())[:-1]})"
        )

    columns = pred.columns[Stype.numerical]
    pred: Tensor = pred.numerical

    category = target.categories[0]
    code = target.code.squeeze(-1).to(torch.int64)

    if isinstance(category, StringTensor):
        classes = StringTensor.from_list(
            [_parse_column(column, category) for column in columns],
            device=category.device,
        )
    else:
        classes = torch.tensor(
            [_parse_column(column, category) for column in columns],
            dtype=category.dtype,
            device=category.device,
        )

    match = category.unsqueeze(-1) == classes.unsqueeze(0)  # [C_t, C_p]
    if missing_score is None and not match.any(dim=-1)[code].all():
        raise ValueError("Target contains classes missing from prediction")

    matched_target, matched_pred = match.nonzero(as_tuple=True)
    aligned = pred.new_full(
        (*pred.size()[:-1], category.size(0)),
        fill_value=missing_score or 0,
    )
    aligned[..., matched_target] = pred[..., matched_pred]
    return aligned, code


def to_binary_class(
    pred: TableTensor,
    target: TableTensor | CategoricalTensor,
    positive_class: bool | int | float | str,
) -> tuple[Tensor, Tensor]:
    r"""Convert binary predictions and targets to positive-class form.

    Args:
        pred: Prediction table whose numerical columns contain class scores.
            Column names define the class order.
        target: Single-column categorical target.
        positive_class: Class value treated as the positive class.

    >>> import torch
    >>> import sdm
    >>> from sdm.evaluation import to_binary_class
    >>> pred = sdm.TableTensor.from_tensor(
    ...     torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]]),
    ...     columns=["false", "true"],
    ... )
    >>> target = sdm.CategoricalTensor(
    ...     code=torch.tensor([[0], [1], [1]]),
    ...     categories=(sdm.StringTensor.from_list(["false", "true"]),),
    ... )
    >>> positive_scores, binary_target = to_binary_class(
    ...     pred,
    ...     target,
    ...     positive_class="true",
    ... )
    >>> positive_scores
    tensor([0.2000, 0.9000, 0.6000])
    >>> binary_target
    tensor([False,  True,  True])
    >>> pred = sdm.TableTensor.from_tensor(
    ...     torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.6, 0.4]]),
    ...     columns=["true", "false"],
    ... )
    >>> positive_scores, binary_target = to_binary_class(
    ...     pred,
    ...     target,
    ...     positive_class="true",
    ... )
    >>> positive_scores
    tensor([0.2000, 0.9000, 0.6000])
    >>> binary_target
    tensor([False,  True,  True])
    """
    if isinstance(target, TableTensor):
        if target.size(-1) != 1:
            raise ValueError(
                f"Expected a single categorical target column "
                f"(got {target.size(-1)} columns)"
            )
        target = target.categorical

    if target.size(-1) != 1:
        raise ValueError(
            f"Expected a single categorical target column "
            f"(got {target.size(-1)} columns)"
        )

    if pred.size()[:-1] != target.size()[:-1]:
        raise ValueError(
            f"Expected prediction and target row dimensions to match "
            f"(got {tuple(pred.size()[:-1])} and "
            f"{tuple(target.size())[:-1]})"
        )

    score = pred[str(positive_class)].numerical.squeeze(-1)
    match = cast(Tensor, target.categories[0] == positive_class)
    index = match.nonzero().view(-1)
    if index.numel() != 1:
        raise ValueError(
            f"Expected positive class {positive_class!r} to appear exactly "
            f"once in target categories (found {index.numel()} matches)"
        )

    return score, target.code.squeeze(-1) == index


def _parse_column(
    column: str,
    category: Tensor,
) -> bool | int | float | str:
    if isinstance(category, StringTensor):
        return column
    if category.dtype == torch.bool:
        if column == "True":
            return True
        if column == "False":
            return False
        raise ValueError(f"Cannot parse prediction column {column!r} as bool")
    if category.is_floating_point():
        return float(column)
    return int(column)

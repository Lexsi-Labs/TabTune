# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
from collections.abc import Sequence
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor

from tabtune.models.kumo._sdm import TableTensor
from tabtune.models.kumo._sdm._warnings import warn_once
from tabtune.models.kumo._sdm.tensor.io import arrow_as_tensor, to_cudf

PREFIX = "sdm_internal"
LEFT_ROW_ID = f"__{PREFIX}_left_row_id__"
RIGHT_ROW_ID = f"__{PREFIX}_right_row_id__"


def join_index(
    left_table: TableTensor,
    right_table: TableTensor,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    how: Literal["inner"] = "inner",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Return row-index pairs for matching rows in two tables.

    Args:
        left_table: The left table.
        right_table: The right table.
        left_keys: Column names from ``left_table`` used as join keys.
        right_keys: Column names from ``right_table`` used as join keys.
        how: The join type.
        dtype: The dtype.
        device: The device.

    Returns:
        ``(left_index, right_index)`` pair with one entry per matched row.
    """
    assert how == "inner"

    if left_table.device != right_table.device:
        raise RuntimeError(
            "Expected 'left_table' and 'right_table' to be on the same device "
            f"(got '{left_table.device}' and '{right_table.device}')"
        )

    dtype = dtype or torch.long
    device = device or left_table.device

    left_table = left_table[list(left_keys)]
    right_table = right_table[list(right_keys)]
    left_rows = math.prod(left_table.size()[:-1])
    right_rows = math.prod(right_table.size()[:-1])

    if max(left_rows, right_rows) - 1 > torch.iinfo(dtype).max:
        raise ValueError(
            f"Creating row indices up to {max(left_rows, right_rows) - 1:,} "
            f"in '{dtype}' would overflow"
        )

    backend: Literal["arrow", "cudf"] = "arrow"
    if left_table.is_cuda:
        if importlib.util.find_spec("cudf") is not None:
            backend = "cudf"
        else:
            warn_once(
                key="missing-cudf-join",
                message=(
                    "Falling back to a CPU-based join because cuDF is not "
                    "installed. Install cuDF to enable faster CUDA-based "
                    "joins without device synchronization."
                ),
            )

    if backend == "arrow":
        left = left_table.to_arrow().append_column(
            LEFT_ROW_ID,
            pa.array(torch.arange(left_rows, dtype=dtype).numpy()),
        )
        right = right_table.to_arrow().append_column(
            RIGHT_ROW_ID,
            pa.array(torch.arange(right_rows, dtype=dtype).numpy()),
        )

        left = dropna(left, subset=left_keys)
        right = dropna(right, subset=right_keys)

        right_schema = right.schema
        for left_key, right_key in zip(left_keys, right_keys):
            left_type = left[left_key].type
            if left_type != right[right_key].type:
                index = right.schema.get_field_index(right_key)
                field = right.schema.field(index).with_type(left_type)
                right_schema = right_schema.set(index, field)
        if right_schema != right.schema:
            right = right.cast(right_schema, safe=True)

        joined = left.join(
            right,
            keys=left_keys,
            right_keys=right_keys,
            join_type=how,
        )

        return (
            arrow_as_tensor(joined[LEFT_ROW_ID], device=device),
            arrow_as_tensor(joined[RIGHT_ROW_ID], device=device),
        )

    assert backend == "cudf"
    with torch.cuda.device(left_table.device):
        left = left_table.to_cudf()
        left[LEFT_ROW_ID] = to_cudf(
            torch.arange(left_rows, dtype=dtype, device=left_table.device)
        )
        right = right_table.to_cudf()
        right[RIGHT_ROW_ID] = to_cudf(
            torch.arange(right_rows, dtype=dtype, device=right_table.device)
        )

        joined = left.dropna(subset=list(left_keys)).merge(
            right.dropna(subset=list(right_keys)),
            left_on=left_keys,
            right_on=right_keys,
            how=how,
        )

        return (
            torch.as_tensor(joined[LEFT_ROW_ID]).to(device),
            torch.as_tensor(joined[RIGHT_ROW_ID]).to(device),
        )


def dropna(
    table: pa.Table,
    subset: Sequence[str] | None = None,
) -> pa.Table:
    r"""Drop rows with null or NaN values in the given columns."""
    columns = table.column_names if subset is None else subset

    mask = None
    for name in columns:
        column = table[name]

        _mask = None
        if column.null_count > 0:
            _mask = column.is_valid()

        if pa.types.is_floating(column.type):
            __mask = pc.call_function(
                "invert", [pc.call_function("is_nan", [column])]
            )
            if _mask is not None:
                _mask = pc.call_function("and_kleene", [_mask, __mask])
            else:
                _mask = __mask

        if _mask is None:
            continue

        if mask is not None:
            mask = pc.call_function("and_kleene", [mask, _mask])
        else:
            mask = _mask

    return table if mask is None else table.filter(mask)

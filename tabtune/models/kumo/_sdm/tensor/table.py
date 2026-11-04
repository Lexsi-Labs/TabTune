# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, SupportsIndex, cast
from typing_extensions import Self

import numpy as np
import pyarrow as pa
import torch
from torch import Tensor
from typing_extensions import override

from tabtune.models.kumo._sdm import NaT, Stype, StypeLike
from tabtune.models.kumo._sdm.tensor import CategoricalTensor, ColumnarTensor, StringTensor
from tabtune.models.kumo._sdm.tensor.io import arrow_as_tensor, to_arrow, to_cudf
from tabtune.models.kumo._sdm.tensor.mixin import _resolve_device

if TYPE_CHECKING:
    import cudf
    import pandas as pd

aten = torch.ops.aten

_NUMPY_TORCH_DTYPES = {
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
}
_TORCH_NUMPY_DTYPES = {
    torch_dtype: numpy_dtype
    for numpy_dtype, torch_dtype in _NUMPY_TORCH_DTYPES.items()
}


def preserve_view_inference_mode(fn: Callable) -> Callable:
    r"""Preserve input inference state for tensor view operations."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.inference_mode(args[0].is_inference()):
            return fn(*args, **kwargs)

    return wrapper


def preserve_autograd_state(fn: Callable) -> Callable:
    r"""Re-apply the caller's autograd/inference-mode state for non-view ops.

    ``__torch_dispatch__`` redispatch does not automatically retain the
    caller's inference-mode and grad-enabled state for non-view ops, so a
    real (non-view) copy performed here would otherwise silently lose
    ``requires_grad``/``grad_fn`` even though gradients are enabled in the
    caller's scope.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        inference_mode_enabled = torch.is_inference_mode_enabled()
        grad_enabled = torch.is_grad_enabled()
        with (
            torch.inference_mode(inference_mode_enabled),
            torch.set_grad_enabled(grad_enabled),
        ):
            return fn(*args, **kwargs)

    return wrapper


def _move_categories(
    categories: Sequence[Tensor],
    device: torch.device | str | None,
) -> tuple[Tensor, ...]:
    device = _resolve_device(device)
    if device is None or device.type == "cpu":
        return tuple(categories)

    groups: dict[
        tuple[type[Tensor], torch.dtype, torch.dtype | None], list[int]
    ] = defaultdict(list)
    moved: list[Tensor | None] = [None] * len(categories)
    for i, category in enumerate(categories):
        if type(category) is Tensor or isinstance(category, StringTensor):
            offset_dtype = (
                category.data_offset[1].dtype
                if isinstance(category, StringTensor)
                else None
            )
            groups[(type(category), category.dtype, offset_dtype)].append(i)
        else:
            moved[i] = category.to(device)

    for indices in groups.values():
        tensors = [categories[i] for i in indices]
        if len(tensors) == 1:
            parts = (tensors[0].to(device),)
        else:
            sizes = [tensor.numel() for tensor in tensors]
            parts = torch.cat(tensors).to(device).split(sizes)
        for i, part in zip(indices, parts):
            moved[i] = part

    assert all(category is not None for category in moved)
    return cast(tuple[Tensor, ...], tuple(moved))


@dataclass(frozen=True)
class TableSchema:
    r"""The schema of a :class:`TableTensor`.

    Args:
        columns: Column names grouped by semantic type.
    """

    columns: Mapping[Stype, tuple[str, ...]]

    def __hash__(self) -> int:
        return hash(frozenset(self.columns.items()))


class TableTensor(Tensor):
    r"""A :class:`torch.Tensor` for tensorized, lossless table data.

    A :class:`TableTensor` stores column blocks separately per semantic type,
    while exposing a single tensor-shaped table interface.
    The last dimension represents named columns.

    .. testcode:: drop_stypes, select_columns, drop_columns

        import torch
        from tabtune.models.kumo._sdm import CategoricalTensor, StringTensor, Stype, TableTensor

        table = TableTensor(
            columns={
                "numerical": ["age", "income"],
                "categorical": ["country", "segment"],
            },
            numerical=torch.randn(10, 2),
            categorical=CategoricalTensor(
                code=torch.randint(0, 2, size=(10, 2)),
                categories=(
                    StringTensor.from_list(["USA", "Germany"]),
                    StringTensor.from_list(["enterprise", "startup"]),
                ),
            ),
        )

        print(table)
        # TableTensor(
        #   size=(10, 4),
        #   blocks={
        #     numerical (2): ['age', 'income'],
        #     categorical (2): ['country', 'segment'],
        #   },
        # )

        # DataFrame-like column selection, but still tensor-native:
        features = table[["age", "country"]]
        assert features.size() == (10, 2)

        # Normal PyTorch indexing still works on row/batch dimensions:
        batch = table[[1, 0, 2], ["income", "segment"]]
        assert batch.size() == (3, 2)

        # Semantic blocks stay separate for model input:
        x_num = table.numerical
        x_cat = table.categorical

        # Tensor ops preserve the table container:
        stacked = torch.stack([table, table], dim=0)
        assert stacked.size() == (2, 10, 4)

        # Column-wise cat extends the schema:
        wide = torch.cat([table[["age"]], table[["country"]]], dim=-1)

    .. testoutput:: drop_stypes, select_columns, drop_columns
        :hide:
        :options: +ELLIPSIS

        ...

    Args:
        size: The shape of the tensor ``[..., C]``.
        columns: Column names grouped by semantic type.
        numerical: The numerical column block of shape ``[..., C_num]``.
        categorical: The categorical column block of shape ``[..., C_cat]``.
        datetime: The ``datetime64[us]`` column block of shape ``[..., C_dt]``.
        text: The text column block of shape ``[..., C_text]``.
        id: The identifier column block of shape ``[..., C_id]``.
        device: The device.
    """

    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _numerical: Tensor
    _categorical: CategoricalTensor
    _datetime: Tensor
    _text: StringTensor
    _id: ColumnarTensor
    _columns: dict[Stype, tuple[str, ...]]
    _column_to_loc: dict[str, tuple[Stype, int]]

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    # Constructors ############################################################

    def __init__(
        self,
        size: Sequence[int] | None = None,
        columns: Mapping[StypeLike, Sequence[str]] | None = None,
        numerical: Tensor | None = None,
        categorical: CategoricalTensor | None = None,
        datetime: Tensor | None = None,
        text: StringTensor | None = None,
        id: ColumnarTensor | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        pass

    def __new__(
        cls,
        size: Sequence[int] | None = None,
        columns: Mapping[StypeLike, Sequence[str]] | None = None,
        numerical: Tensor | None = None,
        categorical: CategoricalTensor | None = None,
        datetime: Tensor | None = None,
        text: StringTensor | None = None,
        id: ColumnarTensor | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create a tensor wrapper."""
        if size is not None and len(size) == 0:
            raise ValueError("Expected 'size' to be non-empty")

        size = tuple(size) if size is not None else size
        device = _resolve_device(device)

        for stype, block in (
            (Stype.numerical, numerical),
            (Stype.categorical, categorical),
            (Stype.datetime, datetime),
            (Stype.text, text),
            (Stype.id, id),
        ):
            if block is None:
                continue

            if stype == Stype.datetime and block.dtype != torch.int64:
                raise ValueError(
                    f"Expected {str(stype)!r} block to have dtype "
                    f"'{torch.int64}' (got '{block.dtype}')"
                )

            size = tuple(block.size()[:-1]) if size is None else size
            device = block.device if device is None else device

            if block.dim() < 2:
                raise ValueError(
                    f"Expected {str(stype)!r} block to be at least 2D "
                    f"(got {block.dim()}D)"
                )
            if size != block.size()[:-1]:
                raise ValueError(
                    f"Expected {str(stype)!r} block size of "
                    f"{_block_size_repr(size)} (got {tuple(block.size())})"
                )
            if device != block.device:
                raise ValueError(
                    f"Expected {str(stype)!r} block to be on device "
                    f"'{device}' (got '{block.device}')"
                )

        if size is None:
            raise ValueError(
                "Expected 'size' to be given when all blocks are 'None'"
            )

        if numerical is None:
            numerical = torch.empty((*size, 0), device=device)
        if categorical is None:
            categorical = CategoricalTensor(
                code=torch.empty((*size, 0), dtype=torch.int32, device=device),
                categories=(),
            )
        if datetime is None:
            datetime = torch.empty((*size, 0), dtype=torch.long, device=device)
        if text is None:
            text = StringTensor(
                data=torch.empty(0, dtype=torch.uint8, device=device),
                offset=torch.zeros(1, dtype=torch.int32, device=device),
                valid=None,
                size=(*size, 0),
            )
        if id is None:
            id = ColumnarTensor((), size=size, device=device)

        columns = {
            Stype(stype): tuple(names)
            for stype, names in (columns or {}).items()
        }

        for stype, block, name in (
            (Stype.numerical, numerical, "num"),
            (Stype.categorical, categorical, "cat"),
            (Stype.datetime, datetime, "dt"),
            (Stype.text, text, "text"),
            (Stype.id, id, "id"),
        ):
            if stype not in columns:
                columns[stype] = tuple(
                    f"{name}_{i}" for i in range(block.size(-1))
                )

            if block.size(-1) != len(columns[stype]):
                _columns = "column" if len(columns[stype]) == 1 else "columns"
                raise ValueError(
                    f"Expected {str(stype)!r} block to hold "
                    f"{len(columns[stype])} {_columns} (got {block.size(-1)})"
                )

        column_names = list(chain.from_iterable(columns.values()))
        column_to_loc: dict[str, tuple[Stype, int]] = {}
        for stype_like, names in columns.items():
            stype = Stype(stype_like)
            for i, name in enumerate(names):
                column_to_loc[name] = (stype, i)
        if len(column_names) != len(column_to_loc):
            raise ValueError("Expected column names to be unique")

        out = Tensor._make_wrapper_subclass(
            cls,
            size=(*size, len(column_names)),
            dtype=numerical.dtype,
            device=numerical.device,
            requires_grad=False,
        )

        out._numerical = numerical
        out._categorical = categorical
        out._datetime = datetime
        out._text = text
        out._id = id
        out._columns = columns
        out._column_to_loc = column_to_loc

        return out

    @classmethod
    def from_arrow(
        cls,
        table: pa.Table,
        stypes: Mapping[str, StypeLike],
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create a tensor from a :class:`pyarrow.Table`.

        .. testcode::

            import pyarrow as pa
            from tabtune.models.kumo._sdm import TableTensor

            arrow_table = pa.table({
                "age": pa.array([25, 31, 42], type=pa.int64()),
                "city": pa.array(["SF", "NYC", "SF"], type=pa.string()),
            })
            tensor = TableTensor.from_arrow(
                table=arrow_table,
                stypes={"age": "numerical", "city": "categorical"},
            )

        Args:
            table: The table.
            stypes: The semantic type for each column. Columns that are present
                in ``table`` but not included in ``stypes`` will be ignored.
            device: The device.
        """
        columns: dict[Stype, list[str]] = defaultdict(list)
        for column, stype in stypes.items():
            columns[Stype(stype)].append(column)

        blocks: dict[Stype, Tensor] = {}
        for stype in columns:
            tensors: list[Tensor] = []
            for column in columns[stype]:
                array = table.column(column)
                if stype == Stype.numerical:
                    tensor = arrow_as_tensor(
                        array,
                        dtype=torch.get_default_dtype(),
                    ).unsqueeze(-1)
                elif stype == Stype.categorical:
                    tensor = CategoricalTensor.from_arrow(array)
                elif stype == Stype.datetime:
                    array = array.cast(pa.timestamp("us"))
                    values = array.to_numpy(zero_copy_only=False)
                    values = values.astype("int64")
                    tensor = torch.from_numpy(values).unsqueeze(-1)
                elif stype == Stype.text:
                    tensor = StringTensor.from_arrow(array).unsqueeze(-1)
                elif stype == Stype.id:
                    tensor = ColumnarTensor.from_arrow(array)
                else:
                    raise NotImplementedError
                tensors.append(tensor)

            blocks[stype] = torch.cat(tensors, dim=-1).to(device)

        return cls(
            columns=cast(Mapping[StypeLike, Sequence[str]], columns),
            **blocks,
        )

    def to_arrow(self) -> pa.Table:
        r"""Convert this tensor to a :class:`pyarrow.Table`."""
        arrays: list[pa.Array] = []
        columns: list[str] = []
        for stype, tensor in self.items():
            if tensor.size(-1) == 0:
                continue

            columns.extend(self._columns[stype])

            if isinstance(tensor, CategoricalTensor | ColumnarTensor):
                arrays.extend(tensor.to_arrow().itercolumns())
            elif isinstance(tensor, StringTensor):
                tensor = cast(
                    StringTensor,
                    tensor.movedim(-1, 0).contiguous().cpu(),
                )
                arrays.extend(cast(StringTensor, t).to_arrow() for t in tensor)
            elif stype == Stype.datetime:
                tensor = tensor.movedim(-1, 0).contiguous().cpu()
                array = tensor.numpy().reshape(tensor.size(0), -1)
                array = array.view("datetime64[us]")
                arrays.extend(pa.array(a) for a in array)
            else:
                tensor = tensor.detach().movedim(-1, 0).contiguous().cpu()
                arrays.extend(to_arrow(t) for t in tensor)

        return pa.Table.from_arrays(arrays, names=columns)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        stypes: Mapping[str, StypeLike],
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create a tensor from a :class:`pandas.DataFrame`.

        Args:
            df: The dataframe.
            stypes: The semantic type for each column. Columns that are present
                in ``df`` but not included in ``stypes`` will be ignored.
            device: The device.
        """
        import pandas as pd

        columns: dict[Stype, list[str]] = defaultdict(list)
        for name, stype in stypes.items():
            columns[Stype(stype)].append(name)

        if columns and set(columns).issubset(
            {Stype.numerical, Stype.categorical}
        ):
            target_device = (
                torch.get_default_device() if device is None else device
            )
            blocks: dict[Stype, Tensor] = {}

            if numerical_columns := columns.get(Stype.numerical):
                dtype = torch.get_default_dtype()
                numpy_dtype = _TORCH_NUMPY_DTYPES.get(
                    dtype, np.dtype(np.float32)
                )
                numerical_df = (
                    df
                    if len(numerical_columns) == len(df.columns)
                    and numerical_columns == df.columns.tolist()
                    else df[numerical_columns]
                )
                if len(df) == 0:
                    if any(
                        pd.api.types.is_complex_dtype(dtype)
                        for dtype in numerical_df.dtypes
                    ):
                        raise TypeError(
                            "Expected numerical columns to contain real values"
                        )
                    blocks[Stype.numerical] = torch.empty(
                        (0, len(numerical_columns)),
                        device=target_device,
                        dtype=dtype,
                    )
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter(
                                "error", np.exceptions.ComplexWarning
                            )
                            values = numerical_df.to_numpy(
                                dtype=numpy_dtype,
                                na_value=np.nan,
                            )
                    except np.exceptions.ComplexWarning as error:
                        raise TypeError(
                            "Expected numerical columns to contain real values"
                        ) from error
                    values = values.copy(order="C")
                    blocks[Stype.numerical] = torch.as_tensor(
                        values,
                        device=target_device,
                        dtype=dtype,
                    )

            if categorical_columns := columns.get(Stype.categorical):
                categorical_series = [df[name] for name in categorical_columns]
                all_categorical = all(
                    isinstance(ser.dtype, pd.CategoricalDtype)
                    for ser in categorical_series
                )
                use_int64_code = not all_categorical and any(
                    isinstance(ser.dtype, pd.ArrowDtype)
                    and pa.types.is_dictionary(ser.dtype.pyarrow_dtype)
                    and pa.types.is_int64(ser.dtype.pyarrow_dtype.index_type)
                    for ser in categorical_series
                )
                numpy_code_dtype = np.int64 if use_int64_code else np.int32
                torch_code_dtype = (
                    torch.int64 if use_int64_code else torch.int32
                )
                if all_categorical:
                    code = np.stack(
                        [ser.array.codes for ser in categorical_series], axis=1
                    ).astype(
                        numpy_code_dtype,
                        copy=False,
                    )
                else:
                    code = np.empty(
                        (len(df), len(categorical_columns)),
                        dtype=numpy_code_dtype,
                    )
                categories: list[Tensor] = []
                for i, ser in enumerate(categorical_series):
                    if all_categorical:
                        categorical = ser.array
                        values = categorical.categories
                    elif isinstance(ser.dtype, pd.CategoricalDtype):
                        categorical = ser.array
                        code[:, i] = categorical.codes
                        values = categorical.categories
                    elif ser.dtype.kind == "f" or (
                        isinstance(ser.dtype, pd.ArrowDtype)
                        and pa.types.is_dictionary(ser.dtype.pyarrow_dtype)
                    ):
                        categorical = CategoricalTensor.from_arrow(
                            pa.array(ser, from_pandas=True)
                        )
                        code[:, i] = categorical.code[:, 0].numpy()
                        categories.append(categorical.categories[0])
                        continue
                    else:
                        column_code, values = pd.factorize(
                            ser,
                            sort=False,
                            use_na_sentinel=True,
                        )
                        code[:, i] = column_code

                    category_dtype = values.dtype
                    if category_dtype.kind in "biuf":
                        category = torch.from_numpy(np.asarray(values).copy())
                    else:
                        array = pa.array(values, from_pandas=True)
                        is_string = pa.types.is_string(array.type)
                        is_large_string = pa.types.is_large_string(array.type)
                        if is_string or is_large_string:
                            category = StringTensor.from_arrow(
                                array,
                                device="cpu",
                            )
                        elif pa.types.is_null(array.type):
                            category = torch.empty(
                                0,
                                dtype=torch.int64,
                                device="cpu",
                            )
                        elif ser.dtype.kind == "O" and (
                            pa.types.is_boolean(array.type)
                            or pa.types.is_integer(array.type)
                            or pa.types.is_floating(array.type)
                        ):
                            categorical = CategoricalTensor.from_arrow(
                                pa.array(ser, from_pandas=True)
                            )
                            code[:, i] = categorical.code[:, 0].numpy()
                            category = categorical.categories[0]
                        else:
                            category = arrow_as_tensor(array, device="cpu")
                    categories.append(category)

                code_tensor = (
                    torch.from_numpy(code).to(target_device)
                    if len(df) > 0
                    else torch.empty(
                        (0, len(categorical_columns)),
                        dtype=torch_code_dtype,
                        device=target_device,
                    )
                )
                blocks[Stype.categorical] = CategoricalTensor(
                    code=code_tensor,
                    categories=_move_categories(categories, target_device),
                )

            return cls(
                size=(len(df),),
                columns=cast(Mapping[StypeLike, Sequence[str]], columns),
                **blocks,
            )

        df = df[stypes.keys()]
        # Resolve period ordinals before Arrow casts them as timestamps, which
        # would interpret the ordinals as microseconds since the Unix epoch.
        period_timestamps = {}
        for name, stype in stypes.items():
            if Stype(stype) != Stype.datetime:
                continue
            if not isinstance(df[name].dtype, pd.PeriodDtype):
                continue
            period_timestamps[name] = df[name].dt.to_timestamp(how="start")
        if period_timestamps:
            df = df.assign(**period_timestamps)

        return cls.from_arrow(
            table=pa.Table.from_pandas(df, preserve_index=False),
            stypes=stypes,
            device=device,
        )

    @classmethod
    def from_columns(
        cls,
        data: Mapping[str, Sequence[Any]],
        stypes: Mapping[str, StypeLike],
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create a tensor from column data.

        Args:
            data: Column data keyed by column name.
            stypes: The semantic type for each column. Columns that are present
                in ``data`` but not included in ``stypes`` will be ignored.
            device: The device.
        """
        import pandas as pd

        return cls.from_pandas(
            df=pd.DataFrame(data),
            stypes=stypes,
            device=device,
        )

    def to_pandas(self) -> pd.DataFrame:
        r"""Convert this tensor to a :class:`pandas.DataFrame`."""
        return self.to_arrow().to_pandas()

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        columns: Sequence[str] | None = None,
    ) -> Self:
        r"""Create a table from a :class:`torch.Tensor`.

        Args:
            tensor: The input tensor with shape ``[..., C]``, interpreted as:

                * A floating-point :class:`torch.Tensor` becomes numerical
                  columns.
                * An integer :class:`torch.Tensor` becomes categorical columns.
                * A :class:`StringTensor` becomes text columns.
            columns: The ``C`` column names.
        """
        if columns is None:
            columns = [str(i) for i in range(tensor.size(-1))]

        if isinstance(tensor, StringTensor):
            return cls(
                columns={Stype.text: columns},
                text=tensor,
            )

        if not tensor.is_floating_point():
            return cls(
                columns={Stype.categorical: columns},
                categorical=CategoricalTensor.from_tensor(tensor),
            )

        return cls(
            columns={Stype.numerical: columns},
            numerical=tensor,
        )

    @classmethod
    def from_cudf(
        cls,
        df: cudf.DataFrame,
        stypes: Mapping[str, StypeLike],
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create a tensor from a :class:`cudf.DataFrame`.

        Args:
            df: The dataframe.
            stypes: The semantic type for each column. Columns that are present
                in ``df`` but not included in ``stypes`` will be ignored.
            device: The device.
        """
        columns: dict[Stype, list[str]] = defaultdict(list)
        for column, stype in stypes.items():
            columns[Stype(stype)].append(column)

        blocks: dict[Stype, Tensor] = {}
        for stype in columns:
            tensors: list[Tensor] = []
            for column in columns[stype]:
                ser = df[column]
                if stype == Stype.numerical:
                    ser = ser.astype("float32", copy=False)
                    if ser.null_count > 0:
                        ser = ser.fillna(float("nan"))
                    tensor = torch.from_dlpack(ser.to_dlpack()).unsqueeze(-1)
                    tensor = tensor.to(device)
                elif stype == Stype.categorical:
                    tensor = CategoricalTensor.from_cudf(ser, device=device)
                elif stype == Stype.datetime:
                    ser = ser.astype("datetime64[us]", copy=False)
                    ser = ser.astype("int64", copy=False)
                    if ser.null_count > 0:
                        ser = ser.fillna(NaT)
                    tensor = torch.from_dlpack(ser.to_dlpack()).unsqueeze(-1)
                    tensor = tensor.to(device)
                elif stype == Stype.text:
                    tensor = StringTensor.from_cudf(ser, device=device)
                    tensor = tensor.unsqueeze(-1)
                elif stype == Stype.id:
                    tensor = ColumnarTensor.from_cudf(ser, device=device)
                else:
                    raise NotImplementedError
                tensors.append(tensor)

            blocks[stype] = torch.cat(tensors, dim=-1)

        return cls(
            columns=cast(Mapping[StypeLike, Sequence[str]], columns),
            **blocks,
        )

    def to_cudf(self) -> cudf.DataFrame:
        r"""Convert this tensor to a :class:`cudf.DataFrame`."""
        import cudf

        dfs: list[cudf.DataFrame] = []
        for stype, tensor in self.items():
            if tensor.size(-1) == 0:
                continue

            if isinstance(tensor, CategoricalTensor | ColumnarTensor):
                dfs.append(tensor.to_cudf(self._columns[stype]))
            elif isinstance(tensor, StringTensor):
                tensor = cast(StringTensor, tensor.movedim(-1, 0).contiguous())
                df = cudf.DataFrame(
                    {
                        name: cast(StringTensor, column).to_cudf()
                        for name, column in zip(self._columns[stype], tensor)
                    }
                )
                dfs.append(df)
            elif stype == Stype.datetime:
                tensor = tensor.movedim(-1, 0).contiguous()
                df = cudf.DataFrame(
                    {
                        name: to_cudf(column, mask).astype(
                            "datetime64[us]", copy=False
                        )
                        for name, column, mask in zip(
                            self._columns[stype],
                            tensor,
                            tensor != NaT,
                        )
                    }
                )
                dfs.append(df)
            else:
                tensor = tensor.detach().movedim(-1, 0).contiguous()
                valid = (
                    tensor.isnan().logical_not_()
                    if tensor.is_floating_point()
                    else None
                )
                col_dict: dict[str, cudf.Series] = {}
                for idx, (name, column) in enumerate(
                    zip(self._columns[stype], tensor)
                ):
                    col_dict[name] = to_cudf(
                        tensor=column,
                        valid_mask=None if valid is None else valid[idx],
                    )

                df = cudf.DataFrame(col_dict)
                dfs.append(df)

        if len(dfs) == 1:
            return dfs[0]

        return cudf.concat(dfs, axis=1)

    # Properties ##############################################################

    @property
    def columns(self) -> Mapping[Stype, tuple[str, ...]]:
        r"""Return column names grouped by semantic type."""
        return self._columns.copy()

    @property
    def column_names(self) -> frozenset[str]:
        r"""The column names of this tensor."""
        return frozenset(self._column_to_loc)

    @property
    def stypes(self) -> Mapping[str, Stype]:
        r"""Return the semantic type for each column."""
        return {key: stype for key, (stype, _) in self._column_to_loc.items()}

    def stype(self, column: str) -> Stype:
        r"""Return the semantic type for a column.

        Args:
            column: The column name.
        """
        return self._column_to_loc[column][0]

    @property
    def active_stypes(self) -> frozenset[Stype]:
        r"""Semantic types with at least one column."""
        return frozenset(
            stype for stype, tensor in self.items() if tensor.size(-1) > 0
        )

    @property
    def numerical(self) -> Tensor:
        r"""Return the numerical column block."""
        return self._numerical

    @property
    def categorical(self) -> CategoricalTensor:
        r"""Return the categorical column block."""
        return self._categorical

    @property
    def datetime(self) -> Tensor:
        r"""Return the datetime column block."""
        return self._datetime

    @property
    def text(self) -> StringTensor:
        r"""Return the text column block."""
        return self._text

    @property
    def id(self) -> ColumnarTensor:
        r"""Return the identifier column block."""
        return self._id

    def items(self) -> Iterator[tuple[Stype, Tensor]]:
        r"""Yield ``(stype, block)`` pairs for typed column blocks."""
        yield Stype.numerical, self._numerical
        yield Stype.categorical, self._categorical
        yield Stype.datetime, self._datetime
        yield Stype.text, self._text
        yield Stype.id, self._id

    @property
    def blocks(self) -> Mapping[Stype, Tensor]:
        r"""Return typed column blocks per semantic type."""
        return dict(self.items())

    @property
    def schema(self) -> TableSchema:
        r"""The schema of this table."""
        return TableSchema(columns=self._columns)

    def is_same_schema(self, other: TableTensor) -> bool:
        r"""Whether ``other`` has the same schema layout.

        Args:
            other: The object to compare against.
        """
        return self.schema == other.schema

    def replace_blocks(
        self,
        *,
        numerical: Tensor | None = None,
        categorical: CategoricalTensor | None = None,
        datetime: Tensor | None = None,
        text: StringTensor | None = None,
        id: ColumnarTensor | None = None,
    ) -> Self:
        r"""Return a table with one or more semantic blocks replaced.

        Provided blocks replace the corresponding semantic type while omitted
        blocks are reused from this table. The returned table preserves the
        current column schema and is validated by the ``TableTensor``
        constructor.

        Args:
            numerical: Replacement numerical block with shape
                ``[..., C_num]``.
            categorical: Replacement categorical block with shape
                ``[..., C_cat]``.
            datetime: Replacement datetime block with shape ``[..., C_dt]``.
            text: Replacement text block with shape ``[..., C_text]``.
            id: Replacement identifier block with shape ``[..., C_id]``.
        """
        return self.__class__(
            columns=cast(Mapping[StypeLike, Sequence[str]], self.columns),
            numerical=self.numerical if numerical is None else numerical,
            categorical=(
                self.categorical if categorical is None else categorical
            ),
            datetime=self.datetime if datetime is None else datetime,
            text=self.text if text is None else text,
            id=self.id if id is None else id,
        )

    def select_stypes(
        self,
        stypes: StypeLike | Iterable[StypeLike],
    ) -> Self:
        r"""Return a table containing only ``stypes`` columns.

        Args:
            stypes: The semantic type or semantic types to select.
        """
        if isinstance(stypes, (str, Stype)):
            stypes = (stypes,)
        stypes = tuple(Stype(stype) for stype in stypes)

        return self.__class__(
            columns={stype: self._columns[stype] for stype in stypes},
            **{stype: getattr(self, stype) for stype in stypes},
        )

    def drop_stypes(
        self,
        stypes: StypeLike | Iterable[StypeLike],
    ) -> Self:
        r"""Return a table with ``stypes`` columns removed.

        .. testcode:: drop_stypes

            assert table.columns[Stype.categorical] == ("country", "segment")
            table = table.drop_stypes("categorical")
            assert table.columns[Stype.categorical] == ()
            assert table.columns[Stype.numerical] == ("age", "income")

        Args:
            stypes: The semantic type or semantic types to drop.
        """
        if isinstance(stypes, (str, Stype)):
            stypes = (stypes,)
        stypes = {Stype(stype) for stype in stypes}

        keep = tuple(stype for stype in self._columns if stype not in stypes)
        return self.__class__(
            size=self.size()[:-1],
            columns={stype: self._columns[stype] for stype in keep},
            device=self.device,
            **{stype: getattr(self, stype) for stype in keep},
        )

    def select_columns(self, columns: str | Iterable[str]) -> Self:
        r"""Return a table containing only ``columns``.

        .. testcode:: select_columns

            assert table.size() == (10, 4)
            table = table.select_columns(["age", "country"])
            assert table.size() == (10, 2)

        Args:
            columns: The columns to select.
        """
        columns = {columns} if isinstance(columns, str) else set(columns)

        for column in columns:
            if column not in self._column_to_loc:
                raise KeyError(column)

        index_dict: dict[Stype, list[int]] = defaultdict(list)
        columns_dict: dict[StypeLike, list[str]] = defaultdict(list)
        for stype, stype_columns in self._columns.items():
            for i, column in enumerate(stype_columns):
                if column in columns:
                    index_dict[stype].append(i)
                    columns_dict[stype].append(column)

        blocks: dict[Stype, Tensor] = {}
        for stype, tensor in self.items():
            indices = index_dict[stype]
            if len(indices) == 0:
                blocks[stype] = tensor.narrow(-1, 0, 0)
            elif len(indices) == len(self._columns[stype]):
                blocks[stype] = tensor
            elif len(indices) == 1:
                blocks[stype] = tensor.narrow(-1, indices[0], 1)
            else:
                blocks[stype] = torch.cat(
                    [tensor.narrow(-1, index, 1) for index in indices],
                    dim=-1,
                )

        return self.__class__(columns=columns_dict, **blocks)

    def drop_columns(self, columns: str | Iterable[str]) -> Self:
        r"""Return a table with ``columns`` removed.

        .. testcode:: drop_columns

            assert table.size() == (10, 4)
            table = table.drop_columns(["age", "country"])
            assert table.size() == (10, 2)

        Args:
            columns: The columns to drop.
        """
        columns = {columns} if isinstance(columns, str) else set(columns)

        for column in columns:
            if column not in self._column_to_loc:
                raise KeyError(column)

        columns = set(chain.from_iterable(self._columns.values())) - columns
        return self.select_columns(columns)

    # Decorators ##############################################################

    @classmethod
    def implements(
        cls,
        torch_function: Callable[..., Any],
    ) -> Callable[..., Any]:
        r"""Register a ``__torch_dispatch__`` implementation.

        See PyTorch's
        :ref:`calling convention <torch-dispatch-calling-convention>`.
        """
        if "HANDLED_FUNCTIONS" not in cls.__dict__:
            cls.HANDLED_FUNCTIONS = cls.HANDLED_FUNCTIONS.copy()

        def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
            cls.HANDLED_FUNCTIONS[torch_function] = my_function
            return my_function

        return decorator

    # PyTorch/Python builtins #################################################

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        args = (
            tuple(self.size()[:-1]),
            self._columns,
            self._numerical,
            self._categorical,
            self._datetime,
            self._text,
            self._id,
        )
        return (self.__class__, args)

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if (handler := cls.HANDLED_FUNCTIONS.get(func)) is not None:
            return handler(*args, **(kwargs or {}))

        raise NotImplementedError(
            f"'{func}' is not supported for {cls.__name__!r}"
        )

    def __getitem__(self, indices: Any) -> TableTensor:
        def is_column_index(index: Any) -> bool:
            return isinstance(index, str) or (
                isinstance(index, list)
                and all(isinstance(value, str) for value in index)
            )

        if is_column_index(indices):
            return self.select_columns(indices)
        if not isinstance(indices, tuple):
            return cast(TableTensor, Tensor.__getitem__(self, indices))
        if not any(is_column_index(index) for index in indices):
            return cast(TableTensor, Tensor.__getitem__(self, indices))
        if any(is_column_index(index) for index in indices[:-1]):
            raise IndexError(
                "Column names can only index the column dimension"
            )

        out = Tensor.__getitem__(self, (*indices[:-1], slice(None)))
        return cast(TableTensor, out).select_columns(indices[-1])

    @override
    def is_shared(self) -> bool:
        return all(tensor.is_shared() for _, tensor in self.items())

    @override
    def share_memory_(self) -> Self:
        for _, tensor in self.items():
            tensor.share_memory_()
        return self

    @override
    def is_contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> bool:
        return all(
            tensor.is_contiguous(memory_format=memory_format)
            for _, tensor in self.items()
        )

    @override
    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Self:
        if self.is_contiguous(memory_format=memory_format):
            return self
        return _contiguous(self, memory_format=memory_format)

    @override
    def tolist() -> Any:
        raise NotImplementedError("'tolist() is not yet implemented")  # TODO

    def __repr__(self, *, indent: int = 0) -> str:  # type: ignore
        def _columns_repr(
            columns: Sequence[str],
            max_cols: int = 3,
            max_item_len: int = 24,
        ) -> str:
            columns = [
                column
                if len(column) <= max_item_len
                else column[: max_item_len - 1] + "…"
                for column in columns
            ]
            if len(columns) > max_cols:
                columns = [*columns[: max_cols - 1], "...", columns[-1]]
            return "[" + ", ".join(column for column in columns) + "]"

        stype_repr = [
            (
                f"{' ' * (indent + 4)}{stype} ({tensor.size(-1):,}): "
                f"{_columns_repr(self._columns[stype])},"
            )
            for stype, tensor in self.items()
            if tensor.size(-1) > 0
        ]

        out = f"{' ' * indent}{self.__class__.__name__}(\n"
        out += f"{' ' * (indent + 2)}size={tuple(self.size())},\n"
        if len(stype_repr) > 0:
            out += f"{' ' * (indent + 2)}blocks={{\n"
            out += "\n".join(stype_repr) + "\n"
            out += f"{' ' * (indent + 2)}}},\n"
        if not self.is_cpu:
            out += f"{' ' * (indent + 2)}device={self.device},\n"
        out += f"{' ' * indent})"
        return out

    def _repr_html_(self) -> str:
        import pandas as pd

        max_columns = 10
        rows = [
            [column, str(stype)]
            for stype, columns in self._columns.items()
            for column in columns
        ]
        if len(rows) > max_columns + 1:
            rows = [
                *rows[: max_columns // 2],
                ["...", "..."],
                *rows[-max_columns // 2 :],
            ]
        df = pd.DataFrame(
            data=rows,
            columns=pd.Index(["Column", "Stype"]),
        )

        size = f"{self.size(-2)} rows x {self.size(-1)} columns"
        if self.dim() > 2:
            examples = " x ".join(str(dim) for dim in self.size()[:-2])
            size = f"{examples} examples x {size}"
        if not self.is_cpu:
            size += f", device={self.device}"

        return df.to_html(index=False, escape=True) + f"<p>{size}</p>"


@TableTensor.implements(aten.alias.default)
@preserve_view_inference_mode
def _alias(inp: TableTensor) -> TableTensor:
    blocks = {
        stype: aten.alias.default(tensor) for stype, tensor in inp.items()
    }
    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.record_stream.default)
def _record_stream(inp: TableTensor, stream: torch.Stream) -> None:
    for _, tensor in inp.items():
        tensor.record_stream(stream)


@TableTensor.implements(aten.to.dtype_layout)
@preserve_autograd_state
def _to_dtype_layout(
    inp: TableTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool | None = None,  # Ignored by PyTorch.
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> TableTensor:

    if (
        not copy
        and (dtype is None or dtype == inp.dtype)
        and (device is None or torch.device(device) == inp.device)
        and (layout is None or layout == inp.layout)
        and (
            memory_format is None
            or memory_format == torch.preserve_format
            or (
                memory_format == torch.contiguous_format
                and inp.is_contiguous()
            )
        )
    ):
        return inp

    blocks = {
        stype: aten.to.dtype_layout(
            tensor,
            dtype=dtype
            if (
                stype not in (Stype.categorical,)
                or dtype in (torch.int32, torch.int64)
            )
            and stype not in (Stype.datetime, Stype.text, Stype.id)
            else None,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            copy=copy,
            memory_format=memory_format,
        )
        for stype, tensor in inp.items()
    }
    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.to.dtype)
def _to_dtype(
    inp: TableTensor,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> TableTensor:
    return _to_dtype_layout(
        inp,
        dtype=dtype,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@TableTensor.implements(aten.to.device)
def _to_device(
    inp: TableTensor,
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> TableTensor:
    return _to_dtype_layout(
        inp,
        dtype=dtype,
        device=device,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@TableTensor.implements(aten.to.other)
def _to_other(
    inp: TableTensor,
    other: Tensor,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> TableTensor:
    return _to_dtype_layout(
        inp,
        dtype=other.dtype,
        layout=other.layout,
        device=other.device,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@TableTensor.implements(aten._to_copy.default)
def _to_copy(
    inp: TableTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,  # Ignored by PyTorch.
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> TableTensor:
    return _to_dtype_layout(
        inp,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        copy=True,
        memory_format=memory_format,
    )


@TableTensor.implements(aten.clone.default)
def _clone(
    inp: TableTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> TableTensor:
    return _to_dtype_layout(inp, copy=True, memory_format=memory_format)


@TableTensor.implements(aten.contiguous.default)
def _contiguous(
    inp: TableTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> TableTensor:
    blocks = {
        stype: tensor.contiguous(memory_format=memory_format)
        for stype, tensor in inp.items()
    }
    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.is_pinned.default)
def _is_pinned(inp: TableTensor) -> bool:
    return all(
        tensor.is_pinned() for _, tensor in inp.items() if tensor.numel() > 0
    )


@TableTensor.implements(aten._pin_memory.default)
def _pin_memory(inp: TableTensor) -> TableTensor:
    blocks = {
        stype: tensor.pin_memory() if tensor.numel() > 0 else tensor
        for stype, tensor in inp.items()
    }
    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.pin_memory.default)
def _pin_memory_composite(
    inp: TableTensor,
    device: torch.device | None = None,
) -> TableTensor:
    if _is_pinned(inp):
        return inp
    return _pin_memory(inp)


@TableTensor.implements(aten.equal.default)
def _equal(inp: TableTensor, other: Tensor) -> bool:
    if inp.__class__ is not other.__class__:
        return False
    if inp.size() != other.size():
        return False
    if inp.stypes != other.stypes:
        return False

    for stype, block in _align_like(inp, other).items():
        if not block.equal(other.blocks[stype]):
            return False

    return True


@TableTensor.implements(aten.allclose.default)
def _allclose(
    inp: TableTensor,
    other: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    if inp.__class__ is not other.__class__:
        return False
    if inp.size() != other.size():
        return False
    if inp.stypes != other.stypes:
        return False

    for stype, block in _align_like(inp, other).items():
        if stype != Stype.numerical:  # Tolerance only applies for numerical:
            if not block.equal(other.blocks[stype]):
                return False
        elif not block.allclose(other.blocks[stype], rtol, atol, equal_nan):
            return False

    return True


@TableTensor.implements(aten.view.default)
@preserve_view_inference_mode
def _view(inp: TableTensor, size: Sequence[int]) -> TableTensor:
    size = tuple(size)
    for i, dim_size in enumerate(size):
        if dim_size < -1:
            raise RuntimeError(
                f"Invalid shape dimension {dim_size} at index {i} of shape "
                f"{size}"
            )

    if size.count(-1) > 1:
        raise RuntimeError("Only one dimension can be inferred")

    if -1 in size:
        known = math.prod(dim_size for dim_size in size if dim_size != -1)
        if known == 0:
            raise RuntimeError(
                f"Cannot reshape tensor of {inp.numel()} elements into "
                f"shape {size} because the unspecified dimension size -1 can "
                f"be any value and is ambiguous"
            )
        if inp.numel() % known != 0:
            raise RuntimeError(
                f"Shape {size} is invalid for input of size {inp.numel()}"
            )
        dim = size.index(-1)
        size = (*size[:dim], inp.numel() // known, *size[dim + 1 :])

    if len(size) == 0 or size[-1] != inp.size(-1):
        _columns = "column" if inp.size(-1) == 1 else "columns"
        raise RuntimeError(
            f"Can't reshape {inp.__class__.__name__!r} with "
            f"{inp.size(-1)} {_columns} into shape {size}"
        )

    blocks = {
        stype: tensor.view((*size[:-1], tensor.size(-1)))
        for stype, tensor in inp.items()
    }
    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten._unsafe_view.default)
@preserve_view_inference_mode
def _unsafe_view(inp: TableTensor, size: Sequence[int]) -> TableTensor:
    return _view(inp, size)


@TableTensor.implements(aten.reshape.default)
def _reshape(inp: TableTensor, size: Sequence[int]) -> TableTensor:
    return cast(
        TableTensor,
        aten.reshape.default.decompose(inp, size),
    )


@TableTensor.implements(aten.flatten.using_ints)
def _flatten(
    inp: TableTensor,
    start_dim: int = 0,
    end_dim: int = -1,
) -> TableTensor:
    return cast(
        TableTensor,
        aten.flatten.using_ints.decompose(inp, start_dim, end_dim),
    )


@TableTensor.implements(aten.squeeze.default)
@preserve_view_inference_mode
def _squeeze(inp: TableTensor) -> TableTensor:
    return _squeeze_dims(inp, range(inp.dim() - 1))


@TableTensor.implements(aten.squeeze.dim)
@preserve_view_inference_mode
def _squeeze_dim(inp: TableTensor, dim: int) -> TableTensor:
    return _squeeze_dims(inp, (dim,))


@TableTensor.implements(aten.squeeze.dims)
@preserve_view_inference_mode
def _squeeze_dims(inp: TableTensor, dim: Sequence[int]) -> TableTensor:
    dims = tuple(dim)
    blocks = {stype: tensor.squeeze(dims) for stype, tensor in inp.items()}

    if inp.dim() - 1 in tuple(dim % inp.dim() for dim in dims):
        raise RuntimeError(
            f"Can't squeeze the column dimension of {inp.__class__.__name__!r}"
        )

    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.unsqueeze.default)
@preserve_view_inference_mode
def _unsqueeze(inp: TableTensor, dim: int) -> TableTensor:
    blocks = {stype: tensor.unsqueeze(dim) for stype, tensor in inp.items()}

    if dim % (inp.dim() + 1) == inp.dim():
        raise RuntimeError(
            f"Can't unsqueeze after the column dimension of "
            f"{inp.__class__.__name__!r}"
        )

    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.expand.default)
@preserve_view_inference_mode
def _expand(
    inp: TableTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> TableTensor:
    size = tuple(size)
    if len(size) == 0 or size[-1] not in (-1, inp.size(-1)):
        _columns = "column" if inp.size(-1) == 1 else "columns"
        raise RuntimeError(
            f"Can't expand {inp.__class__.__name__!r} with "
            f"{inp.size(-1)} {_columns} to shape {size}"
        )

    blocks = {
        stype: aten.expand.default(
            tensor,
            (*size[:-1], tensor.size(-1)),
            implicit=implicit,
        )
        for stype, tensor in inp.items()
    }
    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.transpose.int)
@preserve_view_inference_mode
def _transpose(inp: TableTensor, dim0: int, dim1: int) -> TableTensor:
    blocks = {
        stype: tensor.transpose(dim0, dim1) for stype, tensor in inp.items()
    }

    dim0 %= inp.dim()
    dim1 %= inp.dim()
    if dim0 != dim1 and inp.dim() - 1 in (dim0, dim1):
        raise RuntimeError(
            f"Can't transpose the column dimension of "
            f"{inp.__class__.__name__!r}"
        )

    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.permute.default)
@preserve_view_inference_mode
def _permute(inp: TableTensor, dims: Sequence[int]) -> TableTensor:
    dims = tuple(dims)
    blocks = {stype: tensor.permute(dims) for stype, tensor in inp.items()}

    dims = tuple(dim % inp.dim() for dim in dims)
    if dims[-1] != inp.dim() - 1:
        raise RuntimeError(
            f"Can't permute the column dimension of {inp.__class__.__name__!r}"
        )

    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.movedim.int)
def _movedim_int(
    inp: TableTensor,
    source: int,
    destination: int,
) -> TableTensor:
    return cast(
        TableTensor,
        aten.movedim.int.decompose(inp, source, destination),
    )


@TableTensor.implements(aten.movedim.intlist)
def _movedim_intlist(
    inp: TableTensor,
    source: Sequence[int],
    destination: Sequence[int],
) -> TableTensor:
    return cast(
        TableTensor,
        aten.movedim.intlist.decompose(inp, source, destination),
    )


@TableTensor.implements(aten.select.int)
@preserve_view_inference_mode
def _select(inp: TableTensor, dim: int, index: int) -> TableTensor:
    if _is_column_dim(inp, dim):
        raise RuntimeError(
            f"Can't select the column dimension of {inp.__class__.__name__!r}"
        )

    blocks = {
        stype: tensor.select(dim, index) for stype, tensor in inp.items()
    }

    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.slice.Tensor)
@preserve_view_inference_mode
def _slice(
    inp: TableTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> TableTensor:
    if _is_column_dim(inp, dim):
        if (
            (start is None or start == 0 or start <= -inp.size(-1))
            and (end is None or end >= inp.size(dim))
            and step == 1
        ):
            return _alias(inp)
        raise RuntimeError(
            f"Can't slice the column dimension of '{inp.__class__.__name__}'"
        )

    blocks = {
        stype: aten.slice.Tensor(tensor, dim, start, end, step)
        for stype, tensor in inp.items()
    }

    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.narrow.default)
@preserve_view_inference_mode
def _narrow(
    inp: TableTensor,
    dim: int,
    start: int,
    length: int,
) -> TableTensor:
    if _is_column_dim(inp, dim):
        if (start == 0 or start == -inp.size(-1)) and length == inp.size(-1):
            return _alias(inp)
        raise RuntimeError(
            f"Can't narrow the column dimension of '{inp.__class__.__name__}'"
        )

    blocks = {
        stype: tensor.narrow(dim, start, length)
        for stype, tensor in inp.items()
    }

    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.unbind.int)
@preserve_view_inference_mode
def _unbind(inp: TableTensor, dim: int = 0) -> tuple[TableTensor, ...]:
    if _is_column_dim(inp, dim):
        return _split(inp, split_size=1, dim=dim)

    tensors_dict: dict[Stype, tuple[Tensor, ...]] = {
        stype: tensor.unbind(dim) for stype, tensor in inp.items()
    }

    stypes = tuple(tensors_dict.keys())
    return tuple(
        inp.__class__(
            columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
            **dict(zip(stypes, blocks)),
        )
        for blocks in zip(*tensors_dict.values())
    )


@TableTensor.implements(aten.split.Tensor)
@preserve_view_inference_mode
def _split(
    inp: TableTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[TableTensor, ...]:
    if _is_column_dim(inp, dim):
        if split_size != 1:
            raise RuntimeError(
                f"Can only split the column dimension of "
                f"{inp.__class__.__name__!r} with split size 1"
            )
        return tuple(
            inp.__class__(
                columns={stype: (name,)},
                **{stype: tensor.narrow(-1, i, 1)},
            )
            for stype, tensor in inp.items()
            for i, name in enumerate(inp._columns[stype])
        )

    tensors_dict: dict[Stype, tuple[Tensor, ...]] = {
        stype: tensor.split(split_size, dim) for stype, tensor in inp.items()
    }

    stypes = tuple(tensors_dict.keys())
    return tuple(
        inp.__class__(
            columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
            **dict(zip(stypes, blocks)),
        )
        for blocks in zip(*tensors_dict.values())
    )


@TableTensor.implements(aten.split.sizes)
@TableTensor.implements(aten.split.default)
@TableTensor.implements(aten.split_with_sizes.default)
@preserve_view_inference_mode
def _split_with_sizes(
    inp: TableTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[TableTensor, ...]:
    if _is_column_dim(inp, dim):
        raise RuntimeError(
            f"Can't split the column dimension of {inp.__class__.__name__!r}"
        )

    split_sizes = tuple(split_sizes)
    blocks_dict: dict[Stype, tuple[Tensor, ...]] = {
        stype: tensor.split(split_sizes, dim) for stype, tensor in inp.items()
    }

    stypes = tuple(blocks_dict.keys())
    return tuple(
        inp.__class__(
            columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
            **dict(zip(stypes, blocks)),
        )
        for blocks in zip(*blocks_dict.values())
    )


@TableTensor.implements(aten.index_select.default)
def _index_select(
    inp: TableTensor,
    dim: int,
    index: Tensor,
) -> TableTensor:
    if _is_column_dim(inp, dim):
        raise RuntimeError(
            f"Can't index the column dimension of {inp.__class__.__name__!r}"
        )

    blocks = {
        stype: tensor.index_select(dim, index) for stype, tensor in inp.items()
    }

    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.index.Tensor)
def _index(
    inp: TableTensor,
    indices: Sequence[Tensor | None],
) -> TableTensor:

    current_dim = 0
    for index in indices:
        if index is None:
            current_dim += 1
            continue

        # Check whether we index the column dimension:
        num_indexed_dims = index.dim() if index.dtype == torch.bool else 1
        if current_dim <= inp.dim() - 1 < current_dim + num_indexed_dims:
            raise RuntimeError(
                f"Can't index the column dimension of "
                f"{inp.__class__.__name__!r}"
            )
        current_dim += num_indexed_dims

    blocks = {
        stype: aten.index.Tensor(tensor, indices)
        for stype, tensor in inp.items()
    }

    return inp.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], inp._columns),
        **blocks,
    )


@TableTensor.implements(aten.cat.default)
@preserve_autograd_state
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> TableTensor:
    if len(tensors) == 0:
        raise ValueError("torch.cat(): expected a non-empty list of Tensors")

    if not all(isinstance(tensor, TableTensor) for tensor in tensors):
        raise TypeError(
            f"Expected all tensors to be {TableTensor.__name__!r} instances"
        )
    tensors = cast(Sequence[TableTensor], tensors)

    ref = tensors[0]
    if not _is_column_dim(ref, dim):
        tensors = (ref, *(_align_like(tensor, ref) for tensor in tensors[1:]))

    blocks: dict[Stype, Tensor] = {}
    for stype, _ in ref.items():
        block_list = [tensor.blocks[stype] for tensor in tensors]
        block_list = [block for block in block_list if block.size(-1) > 0]
        if len(block_list) == 1:
            blocks[stype] = block_list[0]
        elif len(block_list) > 1:
            blocks[stype] = torch.cat(block_list, dim=dim)

    size: Sequence[int] | None = None
    if not _is_column_dim(ref, dim):
        columns = ref._columns
        if len(blocks) == 0:
            size = list(tensors[0].size())
            for tensor in tensors[1:]:
                size[dim] += tensor.size(dim)
    else:
        columns = {
            stype: tuple(
                chain.from_iterable(t._columns[stype] for t in tensors)
            )
            for stype, _ in ref.items()
        }
        size = ref.size()

    return ref.__class__(
        size=size[:-1] if size is not None else None,
        columns=cast(dict[StypeLike, tuple[str, ...]], columns),
        device=ref.device if len(blocks) == 0 else None,
        **blocks,
    )


@TableTensor.implements(aten.stack.default)
@preserve_autograd_state
def _stack(tensors: Sequence[Tensor], dim: int = 0) -> TableTensor:
    if len(tensors) == 0:
        raise RuntimeError("stack expects a non-empty TensorList")

    if not all(isinstance(tensor, TableTensor) for tensor in tensors):
        raise TypeError(
            f"Expected all tensors to be {TableTensor.__name__!r} instances"
        )

    dim %= tensors[0].dim() + 1
    if dim >= tensors[0].dim():
        raise RuntimeError(
            f"Can't stack after the column dimension of "
            f"{tensors[0].__class__.__name__!r}"
        )

    tensors = cast(Sequence[TableTensor], tensors)

    ref = tensors[0]
    tensors = (ref, *(_align_like(tensor, ref) for tensor in tensors[1:]))

    blocks = {
        stype: torch.stack([tensor.blocks[stype] for tensor in tensors], dim)
        for stype in ref._columns
    }

    return ref.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], ref._columns),
        **blocks,
    )


# Helpers #####################################################################


def _block_size_repr(size: Sequence[int]) -> str:
    if len(size) == 0:
        return "(*,)"
    if len(size) == 1:
        return f"({size[0]}, *)"
    return f"{str(tuple(size))[:-1]}, *)"


def _is_column_dim(inp: Tensor, dim: int) -> bool:
    if dim < -inp.dim() or dim >= inp.dim():
        return False
    return dim % inp.dim() == inp.dim() - 1


def _align_like(inp: TableTensor, ref: TableTensor) -> TableTensor:
    if inp._columns == ref._columns:
        return inp

    if inp.stypes != ref.stypes:
        raise ValueError(
            "Expected tensors to have the same column names and stypes"
        )

    blocks: dict[Stype, Tensor] = {}
    for stype, ref_columns in ref._columns.items():
        if len(ref_columns) > 0:
            column_to_index = {
                column: i for i, column in enumerate(inp._columns[stype])
            }
            indices = [column_to_index[column] for column in ref_columns]
            block = inp.blocks[stype]
            if indices == list(range(len(inp._columns[stype]))):
                blocks[stype] = block
            elif len(indices) == 1:
                blocks[stype] = block.narrow(-1, indices[0], 1)
            else:
                blocks[stype] = torch.cat(
                    [block.narrow(-1, index, 1) for index in indices],
                    dim=-1,
                )
        else:
            blocks[stype] = inp.blocks[stype]

    return inp.__class__(
        columns=cast(Mapping[StypeLike, Sequence[str]], ref._columns),
        **blocks,
    )

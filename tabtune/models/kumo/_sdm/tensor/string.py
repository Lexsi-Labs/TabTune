# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from typing_extensions import Self

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor
from typing_extensions import override

from tabtune.models.kumo._sdm._warnings import warn_once
from tabtune.models.kumo._sdm.tensor import VarLenTensor
from tabtune.models.kumo._sdm.tensor.io import arrow_as_tensor, combine_arrow_chunks

if TYPE_CHECKING:
    import cudf

aten = torch.ops.aten


class StringTensor(VarLenTensor):
    r"""A :class:`torch.Tensor` for UTF-8 encoded string values.

    Args:
        data: Flat contiguous ``uint8`` tensor containing all string values.
        offset: One-dimensional offsets into ``data``.
        valid: One-dimensional mask indicating valid, non-null string values.
        size: The shape of the tensor.
        stride: The stride of the tensor.
        storage_offset: The offset into the logical ``offset`` storage.
    """

    # NOTE Assume that `data` stores valid UTF-8 bytes and do not validate it.
    ALLOWED_DTYPES: ClassVar[tuple[torch.dtype, ...] | None] = (torch.uint8,)
    __hash__ = Tensor.__hash__

    @override
    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a string :class:`pyarrow.Array`.

        .. testcode::

            import pyarrow as pa
            from tabtune.models.kumo._sdm import StringTensor

            array = pa.array(["foo", "bar", "hello world", ""])
            tensor = StringTensor.from_arrow(array, size=(2, 2))

        Args:
            array: The string :class:`pyarrow.Array` or
                :class:`pyarrow.ChunkedArray`.
            size: The shape of the tensor.
            device: The device.
        """
        if isinstance(array, pa.ChunkedArray):
            array = combine_arrow_chunks(array)

        if size is None:
            size = (len(array),)
        elif math.prod(size) != len(array):
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}.from_arrow' to contain "
                f"{len(array)} elements (got {math.prod(size)})"
            )

        is_string = pa.types.is_string(array.type)
        is_large_string = pa.types.is_large_string(array.type)
        if not is_string and not is_large_string:
            raise TypeError(
                f"Expected 'array' in '{cls.__name__}.from_arrow' to have "
                f"'string' or 'large_string' type (got '{array.type}')"
            )

        buffers = array.buffers()
        offset = torch.frombuffer(
            buffers[1],
            dtype=torch.int32 if is_string else torch.int64,
        )
        valid: Tensor | None = None
        storage_offset = array.offset
        if array.null_count > 0:
            offset = offset[array.offset : array.offset + len(array) + 1]
            valid = arrow_as_tensor(
                array.is_valid(),
                dtype=torch.bool,
                device=device,
            )
            storage_offset = 0

        return cls(
            data=torch.frombuffer(buffers[2], dtype=torch.uint8).to(device)
            if buffers[2] is not None and buffers[2].size > 0
            else torch.empty(0, dtype=torch.uint8, device=device),
            offset=offset.to(device),
            valid=valid,
            size=size,
            storage_offset=storage_offset,
        )

    @override
    def to_arrow(self) -> pa.Array:
        r"""Convert this tensor to a flat :class:`pyarrow.Array`."""
        tensor = cast(StringTensor, self.contiguous().cpu())

        return pa.Array.from_buffers(
            pa.string()
            if tensor._offset.dtype == torch.int32
            else pa.large_string(),
            length=tensor.numel(),
            buffers=[
                pa.array(tensor._valid.numpy(), type=pa.bool_()).buffers()[1]
                if tensor._valid is not None
                else None,
                pa.py_buffer(tensor._offset.numpy()),
                pa.py_buffer(tensor._data.numpy()),
            ],
            null_count=-1,
            offset=int(tensor.storage_offset()),
        )

    @classmethod
    def from_cudf(
        cls,
        ser: cudf.Series | cudf.Index,
        *,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a string :class:`cudf.Series`.

        Args:
            ser: The string :class:`cudf.Series` or :class:`cudf.Index`.
            size: The shape of the tensor.
            device: The device.
        """
        import cupy as cp
        import pylibcudf as plc
        from cudf.api.types import is_string_dtype

        if size is None:
            size = (len(ser),)
        elif math.prod(size) != len(ser):
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}.from_cudf' to contain "
                f"{len(ser)} elements (got {math.prod(size)})"
            )

        if not is_string_dtype(ser.dtype):
            raise TypeError(
                f"Expected 'values' in '{cls.__name__}.from_cudf' to have "
                f"string type (got '{ser.dtype}')"
            )

        # `Series.to_pylibcudf` returns a zero-copy Arrow-style view: base
        # character/offset buffers plus a row offset into the offsets.
        column, _ = ser.to_pylibcudf()

        # `None` or zero-length when the column holds no characters:
        chars = column.data()
        data = torch.from_dlpack(
            cp.asarray(chars) if chars is not None else cp.empty(0, cp.uint8)
        ).to(device)

        if len(ser) == 0:
            return cls(
                data=data,
                offset=torch.zeros(1, dtype=torch.int32, device=data.device),
                valid=None,
                size=size,
            )

        offsets = column.children()[0]  # (base_size + 1,) INT32/INT64 values
        offset_dtype = (
            cp.int32
            if offsets.type().id() == plc.types.TypeId.INT32
            else cp.int64
        )
        offset = torch.from_dlpack(
            cp.asarray(offsets.data()).view(offset_dtype)
        )

        valid: Tensor | None = None
        storage_offset = column.offset()
        if ser.hasnans:
            offset = offset[column.offset() : column.offset() + len(ser) + 1]
            valid = torch.from_dlpack(ser.notnull().to_cupy()).to(device)
            storage_offset = 0

        return cls(
            data=data,
            offset=offset.to(device),
            valid=valid,
            size=size,
            storage_offset=storage_offset,
        )

    def to_cudf(self) -> cudf.Series:
        r"""Convert this CUDA tensor to a flat :class:`cudf.Series`."""
        if not self.is_cuda:
            raise RuntimeError(
                f"Expected tensor to be on a CUDA device (got '{self.device}')"
            )

        tensor = cast(StringTensor, self.contiguous())

        with torch.cuda.device(self.device):
            import cudf
            import pylibcudf as plc

            mask = None
            null_count = 0
            start = int(tensor.storage_offset())
            if tensor._valid is not None:
                mask = cudf.Series(tensor._valid, copy=False)._column.as_mask()
                if isinstance(mask, tuple):
                    mask = mask[0]
                null_count = plc.null_mask.null_count(
                    mask, start, start + tensor.numel()
                )

            plc_column = plc.Column(
                data_type=plc.DataType(plc.TypeId.STRING),
                size=tensor.numel(),
                data=plc.gpumemoryview(tensor._data),
                mask=mask,
                null_count=null_count,
                offset=start,
                children=[plc.Column.from_array(obj=tensor._offset)],
            )
            return cudf.Series.from_pylibcudf(plc_column)

    @classmethod
    @override
    def from_list(
        cls,
        values: str | Sequence[Any] | None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        offset_dtype: torch.dtype = torch.int64,
    ) -> Self:
        r"""Create tensor from a rectangular Python list of strings.

        .. testcode::

            from tabtune.models.kumo._sdm import StringTensor

            tensor = StringTensor.from_list([
                ["foo", "bar"],
                ["hello world", ""],
                [None, "xzy"],
            ])

        Args:
            values: The rectangular Python list of strings.
            dtype: The dtype of the ``value`` tensor.
            device: The device.
            offset_dtype: The dtype of the ``offset`` tensor.
        """
        if dtype is not None and dtype != torch.uint8:
            raise ValueError(
                f"Expected 'dtype' in '{cls.__name__}.from_list' to be "
                f"'torch.uint8' (got '{dtype}')"
            )
        if offset_dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"Expected 'offset_dtype' in '{cls.__name__}.from_list' "
                f"to be 'torch.int32' or 'torch.int64' "
                f"(got '{offset_dtype}')"
            )

        data: list[str | None] = []

        def flatten(value: Any) -> tuple[int, ...]:
            if value is None or isinstance(value, str):
                data.append(value)
                return ()

            if not isinstance(value, Sequence):
                raise ValueError(f"{cls.__name__!r} data must be rectangular")

            if len(value) == 0:
                return (0,)

            if not isinstance(value[0], Sequence) or isinstance(value[0], str):
                data.extend(value)
                return (len(value),)

            child_size: tuple[int, ...] | None = None
            for item in value:
                item_size = flatten(item)
                if child_size is None:
                    child_size = item_size
                elif item_size != child_size:
                    raise ValueError(
                        f"{cls.__name__!r} data must be rectangular"
                    )

            assert child_size is not None
            return (len(value), *child_size)

        size = flatten(values)

        pa_type = pa.large_string()
        if offset_dtype == torch.int32:
            pa_type = pa.string()

        return cls.from_arrow(
            array=pa.array(data, type=pa_type),
            device=device,
            size=size,
        )

    @override
    def item(self) -> str | None:  # type: ignore
        return cast(str | None, super().item())

    def __eq__(self, other: object) -> Tensor:  # type: ignore
        if isinstance(other, str):
            return _eq(self, other)
        return cast(Tensor, super().__eq__(other))

    def __ne__(self, other: object) -> Tensor:  # type: ignore
        if isinstance(other, str):
            return _ne(self, other)
        return cast(Tensor, super().__ne__(other))

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        # TODO Support tensor content printing.
        out = f"{self.__class__.__name__}("
        out += f"size={tuple(self.size())}"
        if self.valid is not None:
            out += f", null_count={int((~self.valid).sum())}"
        if not self.is_cpu:
            out += f", device={self.device}"
        out += ")"
        return out


@StringTensor.implements(aten.eq.Tensor)
@StringTensor.implements(aten.eq.str)
def _eq(inp: StringTensor, other: Tensor | str) -> Tensor:
    if isinstance(other, Tensor) and inp.device != other.device:
        raise RuntimeError(
            f"Expected both tensors to be on the same device "
            f"(got '{inp.device}' and '{other.device}')"
        )

    if not isinstance(other, StringTensor | str):
        return torch.zeros(
            torch.broadcast_shapes(inp.size(), other.size()),
            dtype=torch.bool,
            device=inp.device,
        )

    if isinstance(other, Tensor):
        size = torch.broadcast_shapes(inp.size(), other.size())
        inp = cast(StringTensor, inp.expand(size))
        other = cast(StringTensor, other.expand(size))
    else:
        size = inp.size()

    backend: Literal["arrow", "cudf"] = "arrow"
    if inp.is_cuda:
        if importlib.util.find_spec("cudf") is not None:
            backend = "cudf"
        else:
            warn_once(
                key="missing-cudf-eq",
                message=(
                    "Falling back to a CPU-based string comparison because "
                    "cuDF is not installed. Install cuDF to enable faster "
                    "CUDA-based string comparisons without device "
                    "synchronization."
                ),
            )

    if backend == "arrow":
        out = pc.call_function(
            "equal",
            [
                inp.to_arrow(),
                other.to_arrow() if isinstance(other, StringTensor) else other,
            ],
        )
        if out.null_count > 0:
            out = out.fill_null(False)
        mask = arrow_as_tensor(out, dtype=torch.bool, device=inp.device)
    else:
        assert backend == "cudf"
        with torch.cuda.device(inp.device):
            if isinstance(other, StringTensor):
                out = inp.to_cudf() == other.to_cudf()
            else:
                out = inp.to_cudf() == other
            if out.hasnans:
                out = out.fillna(False)
            mask = torch.from_dlpack(out.to_cupy()).view(size)

    return mask.view(size)


@StringTensor.implements(aten.ne.Tensor)
@StringTensor.implements(aten.ne.str)
def _ne(inp: StringTensor, other: Tensor | str) -> Tensor:
    if isinstance(other, Tensor) and inp.device != other.device:
        raise RuntimeError(
            f"Expected both tensors to be on the same device "
            f"(got '{inp.device}' and '{other.device}')"
        )

    if not isinstance(other, StringTensor | str):
        return torch.ones(
            torch.broadcast_shapes(inp.size(), other.size()),
            dtype=torch.bool,
            device=inp.device,
        )

    if isinstance(other, Tensor):
        size = torch.broadcast_shapes(inp.size(), other.size())
        inp = cast(StringTensor, inp.expand(size))
        other = cast(StringTensor, other.expand(size))
    else:
        size = inp.size()

    backend: Literal["arrow", "cudf"] = "arrow"
    if inp.is_cuda:
        if importlib.util.find_spec("cudf") is not None:
            backend = "cudf"
        else:
            warn_once(
                key="missing-cudf-eq",
                message=(
                    "Falling back to a CPU-based string comparison because "
                    "cuDF is not installed. Install cuDF to enable faster "
                    "CUDA-based string comparisons without device "
                    "synchronization."
                ),
            )

    if backend == "arrow":
        out = pc.call_function(
            "not_equal",
            [
                inp.to_arrow(),
                other.to_arrow() if isinstance(other, StringTensor) else other,
            ],
        )
        if out.null_count > 0:
            out = out.fill_null(False)
        mask = arrow_as_tensor(out, dtype=torch.bool, device=inp.device)
    else:
        assert backend == "cudf"
        with torch.cuda.device(inp.device):
            if isinstance(other, StringTensor):
                out = inp.to_cudf() != other.to_cudf()
            else:
                out = inp.to_cudf() != other
            if out.hasnans:
                out = out.fillna(False)
            mask = torch.from_dlpack(out.to_cupy()).view(size)

    return mask.view(size)


@StringTensor.implements(aten.sort.default)
@StringTensor.implements(aten.sort.stable)
def _sort(
    inp: StringTensor,
    dim: int = -1,
    descending: bool = False,
    *,
    stable: bool | None = None,
) -> tuple[StringTensor, Tensor]:
    if dim < -inp.dim() or dim >= inp.dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.dim()}, {inp.dim() - 1}], but got {dim})"
        )

    if inp.dim() != 1:
        raise NotImplementedError("'sort' only supports one-dimensional input")

    backend: Literal["arrow", "cudf"] = "arrow"
    if inp.is_cuda:
        if importlib.util.find_spec("cudf") is not None:
            backend = "cudf"
        else:
            warn_once(
                key="missing-cudf-sort",
                message=(
                    "Falling back to a CPU-based string sort because cuDF is "
                    "not installed. Install cuDF to enable faster CUDA-based "
                    "string sorting without device synchronization."
                ),
            )

    if backend == "arrow":
        out = pc.call_function(
            "array_sort_indices",
            [inp.to_arrow()],
            options=pc.ArraySortOptions(
                order="descending" if descending else "ascending",
            ),
        )
        perm = arrow_as_tensor(out, dtype=torch.int64, device=inp.device)
    else:
        assert backend == "cudf"
        with torch.cuda.device(inp.device):
            perm_ser = inp.to_cudf().argsort(ascending=not descending)
            perm = torch.from_dlpack(perm_ser.astype("int64").to_cupy())

    return cast(StringTensor, inp[perm]), perm

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, SupportsIndex, cast
from typing_extensions import Self

import pyarrow as pa
import torch
from torch import Tensor
from typing_extensions import override

from tabtune.models.kumo._sdm.tensor.io import (
    ARROW_TORCH_DTYPES,
    arrow_as_tensor,
    combine_arrow_chunks,
    to_arrow,
    to_cudf,
)

if TYPE_CHECKING:
    import cudf

aten = torch.ops.aten


def preserve_view_inference_mode(fn: Callable) -> Callable:
    r"""Preserve input inference state for tensor view operations."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.inference_mode(args[0].is_inference()):
            return fn(*args, **kwargs)

    return wrapper


class NullableTensor(Tensor):
    r"""A :class:`torch.Tensor` for nullable values.

    Args:
        data: Tensor containing the values.
        valid: Boolean mask indicating valid, non-null values.
    """

    ALLOWED_DTYPES: ClassVar[tuple[torch.dtype, ...]] = (
        torch.bool,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _data: Tensor
    _valid: Tensor

    # Constructors ############################################################

    def __init__(self, data: Tensor, valid: Tensor) -> None:
        pass

    def __new__(cls, data: Tensor, valid: Tensor) -> Self:
        r"""Create a tensor wrapper."""
        if data.dtype not in cls.ALLOWED_DTYPES:
            raise ValueError(
                f"Expected 'data' in {cls.__name__!r} to have integer or "
                f"boolean dtype (got '{data.dtype}')"
            )
        if valid.dtype != torch.bool:
            raise ValueError(
                f"Expected 'valid' in {cls.__name__!r} to have 'torch.bool' "
                f"dtype (got '{valid.dtype}')"
            )
        if valid.size() != data.size():
            raise ValueError(
                f"Expected 'data' and 'valid' in {cls.__name__!r} to have the "
                f"same size (got {tuple(data.size())} and "
                f"{tuple(valid.size())})"
            )
        if valid.device != data.device:
            raise ValueError(
                f"Expected 'data' and 'valid' in {cls.__name__!r} to be on "
                f"the same device (got '{data.device}' and '{valid.device}')"
            )

        out = Tensor._make_wrapper_subclass(
            cls,
            size=data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            device=data.device,
            requires_grad=False,
        )

        out._data = data
        out._valid = valid

        return out

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> Self:
        r"""Wrap a tensor into a nullable tensor.

        Args:
            tensor: The tensor to wrap.
        """
        return cls(tensor, valid=torch.ones_like(tensor, dtype=torch.bool))

    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        dtype: torch.dtype | None = None,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a :class:`pyarrow.Array`.

        Args:
            array: The :class:`pyarrow.Array` or :class:`pyarrow.ChunkedArray`.
            dtype: The dtype.
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

        arrow_dtype = ARROW_TORCH_DTYPES.get(array.type)
        if arrow_dtype is None:
            raise TypeError(f"Unsupported value type '{array.type}'")

        buffer = array.buffers()[1]
        storage_offset = array.offset
        if buffer is not None and buffer.size > 0:
            if arrow_dtype == torch.bool:
                data = arrow_as_tensor(
                    array.fill_null(False) if array.null_count > 0 else array,
                    dtype=arrow_dtype,
                )
                storage_offset = 0
            else:
                data = torch.frombuffer(buffer, dtype=arrow_dtype)
        else:
            data = torch.empty(0, dtype=arrow_dtype, device=device)
        data = torch.as_strided(
            data,
            size=size,
            stride=_contiguous_stride(size),
            storage_offset=storage_offset,
        ).to(device, dtype)

        if array.null_count > 0:
            valid = arrow_as_tensor(
                array.is_valid(),
                dtype=torch.bool,
                device=device,
            ).view(size)
        else:
            valid = torch.ones_like(data, dtype=torch.bool)

        return cls(data, valid)

    def to_arrow(self) -> pa.Array:
        r"""Convert this tensor to a flat :class:`pyarrow.Array`."""
        return to_arrow(self._data, self._valid)

    @classmethod
    def from_cudf(
        cls,
        ser: cudf.Series | cudf.Index,
        *,
        dtype: torch.dtype | None = None,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a :class:`cudf.Series`.

        Args:
            ser: The :class:`cudf.Series` or :class:`cudf.Index`.
            dtype: The dtype.
            size: The shape of the tensor.
            device: The device.
        """
        import cupy as cp
        import pylibcudf as plc

        if size is None:
            size = (len(ser),)
        elif math.prod(size) != len(ser):
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}.from_cudf' to contain "
                f"{len(ser)} elements (got {math.prod(size)})"
            )

        column, _ = ser.to_pylibcudf()
        cp_dtype = {
            plc.TypeId.BOOL8: cp.bool_,
            plc.TypeId.UINT8: cp.uint8,
            plc.TypeId.UINT16: cp.uint16,
            plc.TypeId.UINT32: cp.uint32,
            plc.TypeId.UINT64: cp.uint64,
            plc.TypeId.INT8: cp.int8,
            plc.TypeId.INT16: cp.int16,
            plc.TypeId.INT32: cp.int32,
            plc.TypeId.INT64: cp.int64,
        }[column.type().id()]
        data = torch.from_dlpack(cp.asarray(column.data()).view(cp_dtype))
        start = column.offset()
        data = data[start : start + len(ser)].to(device=device, dtype=dtype)

        if ser.hasnans:
            valid = torch.from_dlpack(ser.notnull().to_cupy()).to(device)
        else:
            valid = torch.ones_like(data, dtype=torch.bool)

        return cls(data.view(size), valid.view(size))

    def to_cudf(self) -> cudf.Series:
        r"""Convert this CUDA tensor to a flat :class:`cudf.Series`."""
        return to_cudf(self._data, self._valid)

    @classmethod
    def from_list(
        cls,
        values: int | bool | Sequence[Any] | None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a rectangular Python list.

        Args:
            values: The rectangular Python list.
            dtype: The dtype.
            device: The device.
        """
        data: list[Any] = []
        valid: list[bool] = []

        def is_sequence(value: Any) -> bool:
            return isinstance(value, Sequence) and not isinstance(
                value, str | bytes | bytearray
            )

        def flatten(value: Any) -> tuple[int, ...]:
            if value is None:
                data.append(False)
                valid.append(False)
                return ()

            if not is_sequence(value):
                data.append(value)
                valid.append(True)
                return ()

            if len(value) == 0:
                return (0,)

            if not is_sequence(value[0]):
                data.extend(False if item is None else item for item in value)
                valid.extend(item is not None for item in value)
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
        if dtype is None and not any(valid):
            dtype = torch.int64

        return cls(
            torch.tensor(data, dtype=dtype, device=device).view(size),
            torch.tensor(valid, dtype=torch.bool, device=device).view(size),
        )

    # Properties ##############################################################

    @property
    def data(self) -> Tensor:
        r"""Return the physical data.

        Values at positions where :attr:`valid` is ``False`` are unspecified.
        """
        return self._data

    @property
    def valid(self) -> Tensor:
        r"""Return the logical validity mask."""
        return self._valid

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

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        attrs = ["_data", "_valid"]
        ctx = (self.__class__,)
        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        ctx: tuple[Any, ...],
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> NullableTensor:
        (cls,) = ctx
        return cls(inner_tensors["_data"], inner_tensors["_valid"])

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        return (self.__class__, (self._data, self._valid))

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func is torch.isfinite or func is Tensor.isfinite:
            assert isinstance(args[0], NullableTensor)
            return _isfinite(args[0])

        with torch._C.DisableTorchFunction():
            return func(*args, **(kwargs or {}))

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if (handler := cls.HANDLED_FUNCTIONS.get(func)) is not None:
            return handler(*args, **(kwargs or {}))

        raise NotImplementedError(
            f"'{func}' is not supported for {cls.__name__!r}"
        )

    @override
    def is_shared(self) -> bool:
        return self._data.is_shared() and self._valid.is_shared()

    @override
    def share_memory_(self) -> Self:
        self._data.share_memory_()
        self._valid.share_memory_()
        return self

    @override
    def is_contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> bool:
        return self._data.is_contiguous(
            memory_format=memory_format
        ) and self._valid.is_contiguous(memory_format=memory_format)

    @override
    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Self:
        if self.is_contiguous(memory_format=memory_format):
            return self
        return self.__class__(
            data=self._data.contiguous(memory_format=memory_format),
            valid=self._valid.contiguous(memory_format=memory_format),
        )

    @override
    def tolist(self) -> Any:
        def apply_valid(data: Any, valid: Any) -> Any:
            if isinstance(valid, bool):
                return data if valid else None

            return [
                apply_valid(value, is_valid)
                for value, is_valid in zip(data, valid)
            ]

        return apply_valid(
            data=self._data.cpu().tolist(),
            valid=self._valid.cpu().tolist(),
        )

    @override
    def item(self) -> int | bool | None:  # type: ignore
        if self._data.numel() != 1:
            raise RuntimeError(
                f"{self.__class__.__name__!r} with {self._data.numel()} "
                "elements cannot be converted to a single item"
            )
        return self.view(-1).tolist()[0]

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        # TODO Support tensor content printing.
        out = f"{self.__class__.__name__}("
        out += f"size={tuple(self.size())}"
        if self.dtype != torch.int64:
            out += f", dtype={self.dtype}"
        out += f", null_count={int((~self.valid).sum())}"
        if not self.is_cpu:
            out += f", device={self.device}"
        out += ")"
        return out


@NullableTensor.implements(aten.alias.default)
@preserve_view_inference_mode
def _alias(inp: NullableTensor) -> NullableTensor:
    return inp.__class__(
        data=aten.alias.default(inp._data),
        valid=aten.alias.default(inp._valid),
    )


@NullableTensor.implements(aten.record_stream.default)
def _record_stream(inp: NullableTensor, stream: torch.Stream) -> None:
    inp._data.record_stream(stream)
    inp._valid.record_stream(stream)


@NullableTensor.implements(aten.to.dtype_layout)
def _to_dtype_layout(
    inp: NullableTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool | None = None,  # Ignored by PyTorch.
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> NullableTensor:

    if dtype is not None and dtype not in inp.ALLOWED_DTYPES:
        raise TypeError(
            f"Can't convert {inp.__class__.__name__!r} to dtype '{dtype}'"
        )

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

    return inp.__class__(
        data=aten.to.dtype_layout(
            inp._data,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            copy=copy,
            memory_format=memory_format,
        ),
        valid=aten.to.dtype_layout(
            inp._valid,
            dtype=None,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            copy=copy,
            memory_format=memory_format,
        ),
    )


@NullableTensor.implements(aten.to.dtype)
def _to_dtype(
    inp: NullableTensor,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> NullableTensor:
    return _to_dtype_layout(
        inp,
        dtype=dtype,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@NullableTensor.implements(aten.to.device)
def _to_device(
    inp: NullableTensor,
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> NullableTensor:
    return _to_dtype_layout(
        inp,
        dtype=dtype,
        device=device,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@NullableTensor.implements(aten.to.other)
def _to_other(
    inp: NullableTensor,
    other: Tensor,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> NullableTensor:
    return _to_dtype_layout(
        inp,
        dtype=other.dtype,
        layout=other.layout,
        device=other.device,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@NullableTensor.implements(aten._to_copy.default)
def _to_copy(
    inp: NullableTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,  # Ignored by PyTorch.
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> NullableTensor:
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


@NullableTensor.implements(aten.clone.default)
def _clone(
    inp: NullableTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> NullableTensor:
    return _to_dtype_layout(inp, copy=True, memory_format=memory_format)


@NullableTensor.implements(aten.contiguous.default)
def _contiguous(
    inp: NullableTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> NullableTensor:
    return inp.__class__(
        data=inp._data.contiguous(memory_format=memory_format),
        valid=inp._valid.contiguous(memory_format=memory_format),
    )


@NullableTensor.implements(aten.is_pinned.default)
def _is_pinned(inp: NullableTensor) -> bool:
    return inp._data.is_pinned() and inp._valid.is_pinned()


@NullableTensor.implements(aten._pin_memory.default)
def _pin_memory(inp: NullableTensor) -> NullableTensor:
    return inp.__class__(
        data=inp._data.pin_memory(),
        valid=inp._valid.pin_memory(),
    )


@NullableTensor.implements(aten.pin_memory.default)
def _pin_memory_composite(
    inp: NullableTensor,
    device: torch.device | None = None,
) -> NullableTensor:
    if _is_pinned(inp):
        return inp
    return _pin_memory(inp)


@NullableTensor.implements(aten.isnan.default)
def _isnan(inp: NullableTensor) -> Tensor:
    return ~inp._valid


@NullableTensor.implements(aten.isfinite.default)
def _isfinite(inp: NullableTensor) -> Tensor:
    return inp._valid.clone()


@NullableTensor.implements(aten.nan_to_num.default)
def _nan_to_num(
    inp: NullableTensor,
    nan: int = 0,
    posinf: Any = None,
    neginf: Any = None,
) -> Tensor:
    return torch.where(inp._valid, inp._data, nan)


@NullableTensor.implements(aten.equal.default)
def _equal(inp: NullableTensor, other: Tensor) -> bool:
    if inp.__class__ is not other.__class__:
        return False
    if inp.size() != other.size():
        return False

    if not inp._valid.equal(other._valid):
        return False

    return inp._data[inp._valid].equal(other._data[other._valid])


@NullableTensor.implements(aten.allclose.default)
def _allclose(
    inp: NullableTensor,
    other: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    return _equal(inp, other)


@NullableTensor.implements(aten.view.default)
@preserve_view_inference_mode
def _view(inp: NullableTensor, size: Sequence[int]) -> NullableTensor:
    return _apply(inp, lambda x: aten.view.default(x, size))


@NullableTensor.implements(aten._unsafe_view.default)
@preserve_view_inference_mode
def _unsafe_view(
    inp: NullableTensor,
    size: Sequence[int],
) -> NullableTensor:
    return _apply(inp, lambda x: aten._unsafe_view.default(x, size))


@NullableTensor.implements(aten.reshape.default)
def _reshape(inp: NullableTensor, size: Sequence[int]) -> NullableTensor:
    return cast(
        NullableTensor,
        aten.reshape.default.decompose(inp, size),
    )


@NullableTensor.implements(aten.flatten.using_ints)
def _flatten(
    inp: NullableTensor,
    start_dim: int = 0,
    end_dim: int = -1,
) -> NullableTensor:
    return cast(
        NullableTensor,
        aten.flatten.using_ints.decompose(inp, start_dim, end_dim),
    )


@NullableTensor.implements(aten.squeeze.default)
@preserve_view_inference_mode
def _squeeze(inp: NullableTensor) -> NullableTensor:
    return _apply(inp, lambda x: aten.squeeze.default(x))


@NullableTensor.implements(aten.squeeze.dim)
@preserve_view_inference_mode
def _squeeze_dim(inp: NullableTensor, dim: int) -> NullableTensor:
    return _apply(inp, lambda x: aten.squeeze.dim(x, dim))


@NullableTensor.implements(aten.squeeze.dims)
@preserve_view_inference_mode
def _squeeze_dims(
    inp: NullableTensor,
    dim: Sequence[int],
) -> NullableTensor:
    return _apply(inp, lambda x: aten.squeeze.dims(x, dim))


@NullableTensor.implements(aten.unsqueeze.default)
@preserve_view_inference_mode
def _unsqueeze(inp: NullableTensor, dim: int) -> NullableTensor:
    return _apply(inp, lambda x: aten.unsqueeze.default(x, dim))


@NullableTensor.implements(aten.expand.default)
@preserve_view_inference_mode
def _expand(
    inp: NullableTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> NullableTensor:
    return _apply(
        inp,
        lambda x: aten.expand.default(x, size, implicit=implicit),
    )


@NullableTensor.implements(aten.t.default)
@preserve_view_inference_mode
def _t(inp: NullableTensor) -> NullableTensor:
    return _apply(inp, lambda x: aten.t.default(x))


@NullableTensor.implements(aten.transpose.int)
@preserve_view_inference_mode
def _transpose(inp: NullableTensor, dim0: int, dim1: int) -> NullableTensor:
    return _apply(inp, lambda x: aten.transpose.int(x, dim0, dim1))


@NullableTensor.implements(aten.permute.default)
@preserve_view_inference_mode
def _permute(inp: NullableTensor, dims: Sequence[int]) -> NullableTensor:
    return _apply(inp, lambda x: aten.permute.default(x, dims))


@NullableTensor.implements(aten.movedim.int)
def _movedim_int(
    inp: NullableTensor,
    source: int,
    destination: int,
) -> NullableTensor:
    return cast(
        NullableTensor,
        aten.movedim.int.decompose(inp, source, destination),
    )


@NullableTensor.implements(aten.movedim.intlist)
def _movedim_intlist(
    inp: NullableTensor,
    source: Sequence[int],
    destination: Sequence[int],
) -> NullableTensor:
    return cast(
        NullableTensor,
        aten.movedim.intlist.decompose(inp, source, destination),
    )


@NullableTensor.implements(aten.select.int)
@preserve_view_inference_mode
def _select(inp: NullableTensor, dim: int, index: int) -> NullableTensor:
    return _apply(inp, lambda x: aten.select.int(x, dim, index))


@NullableTensor.implements(aten.slice.Tensor)
@preserve_view_inference_mode
def _slice(
    inp: NullableTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> NullableTensor:
    return _apply(inp, lambda x: aten.slice.Tensor(x, dim, start, end, step))


@NullableTensor.implements(aten.narrow.default)
@preserve_view_inference_mode
def _narrow(
    inp: NullableTensor,
    dim: int,
    start: int,
    length: int,
) -> NullableTensor:
    return _apply(inp, lambda x: aten.narrow.default(x, dim, start, length))


@NullableTensor.implements(aten.unbind.int)
@preserve_view_inference_mode
def _unbind(
    inp: NullableTensor,
    dim: int = 0,
) -> tuple[NullableTensor, ...]:
    return tuple(
        inp.__class__(data, valid)
        for data, valid in zip(
            inp._data.unbind(dim=dim),
            inp._valid.unbind(dim=dim),
        )
    )


@NullableTensor.implements(aten.split.Tensor)
@preserve_view_inference_mode
def _split(
    inp: NullableTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[NullableTensor, ...]:
    return tuple(
        inp.__class__(data, valid)
        for data, valid in zip(
            inp._data.split(split_size, dim=dim),
            inp._valid.split(split_size, dim=dim),
        )
    )


@NullableTensor.implements(aten.split.sizes)
@NullableTensor.implements(aten.split.default)
@NullableTensor.implements(aten.split_with_sizes.default)
@preserve_view_inference_mode
def _split_with_sizes(
    inp: NullableTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[NullableTensor, ...]:
    return tuple(
        inp.__class__(data, valid)
        for data, valid in zip(
            inp._data.split(split_sizes, dim=dim),
            inp._valid.split(split_sizes, dim=dim),
        )
    )


@NullableTensor.implements(aten.masked_select.default)
def _masked_select(inp: NullableTensor, mask: Tensor) -> NullableTensor:
    return _apply(inp, lambda x: aten.masked_select.default(x, mask))


@NullableTensor.implements(aten.index_select.default)
def _index_select(
    inp: NullableTensor,
    dim: int,
    index: Tensor,
) -> NullableTensor:
    return _apply(inp, lambda x: aten.index_select.default(x, dim, index))


@NullableTensor.implements(aten.take.default)
def _take(inp: NullableTensor, index: Tensor) -> NullableTensor:
    return _apply(inp, lambda x: aten.take.default(x, index))


@NullableTensor.implements(aten.index.Tensor)
def _index(
    inp: NullableTensor,
    indices: Sequence[Tensor | None],
) -> NullableTensor:
    return _apply(inp, lambda x: aten.index.Tensor(x, indices))


@NullableTensor.implements(aten.cat.default)
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> NullableTensor:
    if len(tensors) == 0:
        raise ValueError("Expected a non-empty list of Tensors")

    if not isinstance(tensors[0], NullableTensor):
        raise TypeError(
            f"Expected {NullableTensor.__name__!r} as element 0, but got "
            f"{tensors[0].__class__.__name__!r}"
        )

    tensors = cast(Sequence[NullableTensor], tensors)
    return tensors[0].__class__(
        data=torch.cat([tensor._data for tensor in tensors], dim=dim),
        valid=torch.cat([tensor._valid for tensor in tensors], dim=dim),
    )


@NullableTensor.implements(aten.stack.default)
def _stack(tensors: Sequence[Tensor], dim: int = 0) -> NullableTensor:
    out = torch.cat([tensor.unsqueeze(dim) for tensor in tensors], dim=dim)
    return cast(NullableTensor, out)


# Helpers #####################################################################


def _contiguous_stride(size: Sequence[int]) -> tuple[int, ...]:
    value = 1
    stride = []
    for dim_size in reversed(size):
        stride.append(value)
        value *= dim_size
    return tuple(stride[::-1])


def _apply(
    inp: NullableTensor,
    fn: Callable[[Tensor], Tensor],
) -> NullableTensor:
    return inp.__class__(fn(inp._data), fn(inp._valid))

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, SupportsIndex, cast
from typing_extensions import Self

import pyarrow as pa
import torch
from torch import Tensor
from torch.overrides import enable_reentrant_dispatch
from typing_extensions import override

from tabtune.models.kumo._sdm.tensor.io import (
    ARROW_TORCH_DTYPES,
    arrow_as_tensor,
    combine_arrow_chunks,
    to_arrow,
)

aten = torch.ops.aten


def preserve_view_inference_mode(fn: Callable) -> Callable:
    r"""Preserve input inference state for tensor view operations."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.inference_mode(args[0].is_inference()):
            return fn(*args, **kwargs)

    return wrapper


class VarLenTensor(Tensor):
    r"""A :class:`torch.Tensor` for rectangular variable-length values.

    Values are stored in a flat contiguous ``data`` tensor and indexed by an
    ``offset`` tensor.

    .. testcode::

        import torch
        from tabtune.models.kumo._sdm import VarLenTensor

        tensor = VarLenTensor(
            data=torch.tensor([1, 2, 3, 4, 5, 6]),
            offset=torch.tensor([0, 2, 5, 5, 6]),
            valid=None,
            size=(2, 2),
        )

    Args:
        data: Flat contiguous tensor containing all element values.
        offset: One-dimensional offsets into ``data``.
        valid: One-dimensional mask indicating valid, non-null element values.
        size: The shape of the tensor.
        stride: The stride of the tensor.
        storage_offset: The offset into the logical ``offset`` storage.
    """

    ALLOWED_DTYPES: ClassVar[tuple[torch.dtype, ...] | None] = None
    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _data: Tensor
    _offset: Tensor
    _valid: Tensor | None

    # Constructors ############################################################

    def __init__(
        self,
        data: Tensor,
        offset: Tensor,
        valid: Tensor | None,
        size: Sequence[int],
        stride: Sequence[int] | None = None,
        storage_offset: int = 0,
    ) -> None:
        pass

    def __new__(
        cls,
        data: Tensor,
        offset: Tensor,
        valid: Tensor | None,
        size: Sequence[int],
        stride: Sequence[int] | None = None,
        storage_offset: int = 0,
    ) -> Self:
        r"""Create a tensor wrapper."""
        size = tuple(size)
        if any(dim_size < -1 for dim_size in size):
            raise ValueError(f"Invalid shape dimensions (got '{size}')")
        if size.count(-1) > 1:
            raise ValueError("Only one dimension can be inferred")

        if -1 in size:
            if stride is not None:
                raise ValueError("Can't infer size when stride is given")

            numel = offset.numel() - 1 - storage_offset
            known = math.prod(dim_size for dim_size in size if dim_size != -1)
            if numel < 0 or known == 0 or numel % known != 0:
                raise ValueError(
                    f"Shape '{size}' is invalid for input of size {numel}"
                )

            dim = size.index(-1)
            size = (*size[:dim], numel // known, *size[dim + 1 :])

        stride = _contiguous_stride(size) if stride is None else tuple(stride)
        if any(dim_stride < 0 for dim_stride in stride):
            raise ValueError(
                f"Negative strides are not supported (got '{stride}')"
            )

        if (
            cls.ALLOWED_DTYPES is not None
            and data.dtype not in cls.ALLOWED_DTYPES
        ):
            raise ValueError(
                f"Expected 'data' in {cls.__name__!r} to have dtype "
                f"in '{cls.ALLOWED_DTYPES}' (got '{data.dtype}')"
            )
        if data.dim() != 1:
            raise ValueError(
                f"Expected 'data' in {cls.__name__!r} to be one-dimensional "
                f"(got {data.dim()}D tensor)"
            )
        if not data.is_contiguous():
            raise ValueError(
                f"Expected 'data' in {cls.__name__!r} to be contiguous"
            )
        if offset.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"Expected 'offset' in {cls.__name__!r} to have dtype "
                f"'torch.int32' or 'torch.int64' (got '{offset.dtype}')"
            )
        if offset.dim() != 1:
            raise ValueError(
                f"Expected 'offset' in {cls.__name__!r} to be one-dimensional "
                f"(got {offset.dim()}D tensor)"
            )
        if not offset.is_contiguous():
            raise ValueError(
                f"Expected 'offset' in {cls.__name__!r} to be contiguous"
            )
        if data.device != offset.device:
            raise ValueError(
                f"Expected 'data' and 'offset' in {cls.__name__!r} to be on "
                f"the same device (got '{data.device}' and '{offset.device}')"
            )
        if valid is not None:
            if valid.dtype != torch.bool:
                raise ValueError(
                    f"Expected 'valid' in {cls.__name__!r} to have dtype "
                    f"'torch.bool' (got '{valid.dtype}')"
                )
            if valid.dim() != 1:
                raise ValueError(
                    f"Expected 'valid' in {cls.__name__!r} to be "
                    f"one-dimensional (got {valid.dim()}D tensor)"
                )
            if not valid.is_contiguous():
                raise ValueError(
                    f"Expected 'valid' in {cls.__name__!r} to be contiguous"
                )
            if data.device != valid.device:
                raise ValueError(
                    f"Expected 'data' and 'valid' in {cls.__name__!r} to be "
                    f"on the same device (got '{data.device}' and "
                    f"'{valid.device}')"
                )
        if len(size) != len(stride):
            raise ValueError(
                f"Expected 'size' and 'stride' in {cls.__name__!r} to have "
                f"the same length (got {len(size)} and {len(stride)})"
            )
        if storage_offset < 0:
            raise ValueError(
                f"Expected 'storage_offset' in {cls.__name__!r} to be "
                f"non-negative"
            )
        if storage_offset + _span_len(size, stride) >= offset.numel():
            raise ValueError(
                f"'offset' in {cls.__name__!r} is out of bounds (got "
                f"{offset.numel()} entries, but expected at least "
                f"{storage_offset + _span_len(size, stride) + 1} entries)"
            )
        if (
            valid is not None
            and storage_offset + _span_len(size, stride) > valid.numel()
        ):
            raise ValueError(
                f"'valid' in {cls.__name__!r} is out of bounds (got "
                f"{valid.numel()} entries, but expected at least "
                f"{storage_offset + _span_len(size, stride)} entries)"
            )
        if data.numel() > torch.iinfo(offset.dtype).max:
            raise ValueError(
                f"Expected 'offset' in {cls.__name__!r} to represent "
                f"{data.numel()} elements, but '{offset.dtype}' can only "
                f"represent {torch.iinfo(offset.dtype).max} elements"
            )

        # NOTE We do not validate offset values here (e.g., monotonicity) due
        # to device synchronization.

        out = Tensor._make_wrapper_subclass(
            cls,
            size=size,
            strides=stride,
            storage_offset=storage_offset,
            dtype=data.dtype,
            device=data.device,
            requires_grad=False,  # Autograd lives on `_data` only.
        )

        out._data = data
        out._offset = offset
        out._valid = valid

        return out

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        *,
        offset_dtype: torch.dtype = torch.int64,
    ) -> Self:
        r"""Wrap a dense tensor as fixed-size variable-length elements.

        Args:
            tensor: The dense tensor.
            offset_dtype: The dtype of the ``offset`` tensor.
        """
        data = tensor
        if tensor.stride() != (1,) or int(tensor.storage_offset()) != 0:
            span_len = _span_len(tensor.size(), tensor.stride())
            data = torch.as_strided(
                tensor,
                size=(int(data.storage_offset()) + span_len,),
                stride=(1,),
                storage_offset=0,
            )
        offset = torch.arange(
            data.numel() + 1,
            dtype=offset_dtype,
            device=data.device,
        )
        return cls(
            data=data,
            offset=offset,
            valid=None,
            size=tensor.size(),
            stride=tensor.stride(),
            storage_offset=int(tensor.storage_offset()),
        )

    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a list :class:`pyarrow.Array`.

        .. testcode::

            import pyarrow as pa
            from tabtune.models.kumo._sdm import VarLenTensor

            array = pa.array([[1, 2], [3, 4, 5], [], None, [6]])
            tensor = VarLenTensor.from_arrow(array)

        Args:
            array: The list :class:`pyarrow.Array` or
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

        is_list = pa.types.is_list(array.type)
        is_large_list = pa.types.is_large_list(array.type)
        if not is_list and not is_large_list:
            raise TypeError(
                f"Expected 'array' in '{cls.__name__}.from_arrow' to have "
                f"'list' or 'large_list' type (got '{array.type}')"
            )

        if array.values.null_count > 0:
            raise ValueError(
                f"{cls.__name__!r} cannot represent inner null values"
            )

        dtype = ARROW_TORCH_DTYPES.get(array.values.type)
        if dtype is None:
            raise TypeError(f"Unsupported value type '{array.values.type}'")

        buffer = array.values.buffers()[1]
        if buffer is not None and buffer.size > 0:
            if dtype == torch.bool:
                data = arrow_as_tensor(
                    array.values,
                    dtype=dtype,
                    device=device,
                )
            else:
                data = torch.frombuffer(buffer, dtype=dtype)
                start = array.values.offset
                data = data[start : start + len(array.values)].to(device)
        else:
            data = torch.empty(0, dtype=dtype, device=device)

        offset = torch.frombuffer(
            array.buffers()[1],
            dtype=torch.int32 if is_list else torch.int64,
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
            data=data,
            offset=offset.to(device),
            valid=valid,
            size=size,
            storage_offset=storage_offset,
        )

    def to_arrow(self) -> pa.Array:
        r"""Convert this tensor to a flat :class:`pyarrow.Array`."""
        tensor = cast(VarLenTensor, self.detach().contiguous().cpu())
        array = to_arrow(tensor._data)

        return pa.Array.from_buffers(
            pa.list_(array.type)
            if tensor._offset.dtype == torch.int32
            else pa.large_list(array.type),
            length=tensor.numel(),
            buffers=[
                pa.array(tensor._valid.numpy(), type=pa.bool_()).buffers()[1]
                if tensor._valid is not None
                else None,
                pa.py_buffer(tensor._offset.numpy()),
            ],
            null_count=-1,
            children=[array],
            offset=int(tensor.storage_offset()),
        )

    @classmethod
    def from_list(
        cls,
        values: Sequence[Any],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        offset_dtype: torch.dtype = torch.int64,
    ) -> Self:
        r"""Create tensor from a rectangular Python list.

        .. testcode::

            from tabtune.models.kumo._sdm import VarLenTensor

            tensor = VarLenTensor.from_list([
                [[1, 2], [3, 4, 5]],
                [[], [6]],
                [[7, 8], None],
            ])

        Args:
            values: The rectangular Python list.
            dtype: The dtype of the ``value`` tensor.
            device: The device.
            offset_dtype: The dtype of the ``offset`` tensor.
        """
        data: list[Any] = []
        offset: list[int] = [0]
        valid: list[bool] = []

        def is_sequence(value: Any) -> bool:
            return isinstance(value, Sequence) and not isinstance(
                value, str | bytes | bytearray
            )

        def is_leaf(value: Any) -> bool:
            if value is None:
                return True
            if not is_sequence(value):
                return False
            return len(value) == 0 or not is_sequence(value[0])

        def flatten_leaf(value: Any) -> None:
            if value is None:
                offset.append(len(data))
                valid.append(False)
                return

            for item in value:
                if item is None:
                    raise ValueError(
                        f"{cls.__name__!r} cannot represent inner null values"
                    )
                if is_sequence(item):
                    raise ValueError(
                        f"{cls.__name__!r} data must be rectangular"
                    )

            data.extend(value)
            valid.append(True)
            offset.append(len(data))

        def flatten(value: Any) -> tuple[int, ...]:
            if is_leaf(value):
                flatten_leaf(value)
                return ()

            if not is_sequence(value):
                raise ValueError(f"{cls.__name__!r} data must be rectangular")

            child_size: tuple[int, ...] | None = None
            for item in value:
                if is_leaf(item):
                    item_size = ()
                    flatten_leaf(item)
                else:
                    item_size = flatten(item)

                if child_size is None:
                    child_size = item_size
                elif item_size != child_size:
                    raise ValueError(
                        f"{cls.__name__!r} data must be rectangular"
                    )

            assert child_size is not None
            return (len(value), *child_size)

        size = (0,) if len(values) == 0 else flatten(values)

        return cls(
            data=torch.tensor(data, dtype=dtype, device=device),
            offset=torch.tensor(offset, dtype=offset_dtype, device=device),
            valid=torch.tensor(valid, dtype=torch.bool, device=device)
            if False in valid
            else None,
            size=size,
        )

    # Properties ##############################################################

    @property
    def data_offset(self) -> tuple[Tensor, Tensor]:
        r"""Return contiguous data and normalized offsets.

        Returns:
            ``(data, offset)`` tuple.
        """
        if not self.is_contiguous():
            raise RuntimeError(
                f"Can't access 'data_offset' for non-contiguous "
                f"{self.__class__.__name__!r}"
            )

        start = int(self.storage_offset())
        offset = self._offset[start : start + self.numel() + 1]
        data = self._data[offset[0] : offset[-1]]
        if offset.is_cpu and int(offset[0]) == 0:
            return data, offset
        return data, offset - offset[0]

    @property
    def valid(self) -> Tensor | None:
        r"""Return the logical validity mask.

        Returns:
            Boolean mask with shape ``self.size()`` indicating valid,
            non-null tensor elements, or ``None`` when all elements are valid.
        """
        if self._valid is None:
            return None

        return torch.as_strided(
            self._valid,
            size=self.size(),
            stride=self.stride(),
            storage_offset=int(self.storage_offset()),
        )

    @property
    def is_nullable(self) -> bool:
        r"""Whether this tensor has a validity mask."""
        return self._valid is not None

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
        attrs = ["_data", "_offset"]
        if self._valid is not None:
            attrs.append("_valid")
        ctx = (self.__class__, self.storage_offset())
        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        ctx: tuple[Any, ...],
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> VarLenTensor:
        cls, storage_offset = ctx
        return cls(
            data=inner_tensors["_data"],
            offset=inner_tensors["_offset"],
            valid=inner_tensors.get("_valid"),
            size=outer_size,
            stride=outer_stride,
            storage_offset=storage_offset,
        )

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        args = (
            self._data,
            self._offset,
            self._valid,
            tuple(self.size()),
            tuple(self.stride()),
            int(self.storage_offset()),
        )
        return (self.__class__, args)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func is torch.isfinite or func is Tensor.isfinite:
            assert isinstance(args[0], VarLenTensor)
            return _isfinite(args[0])

        with torch._C.DisableTorchFunction():
            return func(*args, **(kwargs or {}))

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if (handler := cls.HANDLED_FUNCTIONS.get(func)) is not None:
            with enable_reentrant_dispatch():  # Record autograd in `_data`.
                return handler(*args, **(kwargs or {}))

        raise NotImplementedError(
            f"'{func}' is not supported for {cls.__name__!r}"
        )

    @override
    def is_shared(self) -> bool:
        is_shared = self._data.is_shared() and self._offset.is_shared()
        return is_shared and (self._valid is None or self._valid.is_shared())

    @override
    def share_memory_(self) -> Self:
        self._data.share_memory_()
        self._offset.share_memory_()
        if self._valid is not None:
            self._valid.share_memory_()
        return self

    @property
    @override
    def grad(self) -> Tensor | None:
        return self._data.grad

    @property
    @override
    def requires_grad(self) -> bool:
        return self._data.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        self._data.requires_grad_(requires_grad)

    @override
    def requires_grad_(self, mode: bool = True) -> Self:
        self._data.requires_grad_(mode)
        return self

    @override
    def detach_(self) -> Self:
        raise RuntimeError(
            f"Can't detach a '{self.__class__.__name__} in-place. Use "
            f"'detach() instead."
        )

    @override
    def tolist(self) -> Any:
        def reshape(values: list[Any], size: tuple[int, ...]) -> Any:
            if len(size) == 0:
                return values[0]
            if len(size) == 1:
                return values

            step = math.prod(size[1:])
            return [
                reshape(values[i * step : (i + 1) * step], size[1:])
                for i in range(size[0])
            ]

        values = self.to_arrow().to_pylist()
        return reshape(values, tuple(self.size()))

    @override
    def item(self) -> list[Any] | None:  # type: ignore
        if self.numel() != 1:
            raise RuntimeError(
                f"{self.__class__.__name__!r} with {self.numel()} "
                f"elements cannot be converted to a single item"
            )
        return self.view(-1).tolist()[0]

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        # TODO Support tensor content printing.
        out = f"{self.__class__.__name__}("
        out += f"size={tuple(self.size())}"
        out += f", dtype={self.dtype}"
        if self.valid is not None:
            out += f", null_count={int((~self.valid).sum())}"
        if not self.is_cpu:
            out += f", device={self.device}"
        if self._data.grad_fn is not None:
            out += f", grad_fn=<{type(self._data.grad_fn).__name__}>"
        elif self.requires_grad:
            out += ", requires_grad=True"
        out += ")"
        return out


@VarLenTensor.implements(aten.alias.default)
@preserve_view_inference_mode
def _alias(inp: VarLenTensor) -> VarLenTensor:
    return inp.__class__(
        data=aten.alias.default(inp._data),
        offset=aten.alias.default(inp._offset),
        valid=aten.alias.default(inp._valid)
        if inp._valid is not None
        else None,
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()),
    )


@VarLenTensor.implements(aten.record_stream.default)
def _record_stream(inp: VarLenTensor, stream: torch.Stream) -> None:
    inp._data.record_stream(stream)
    inp._offset.record_stream(stream)
    if inp._valid is not None:
        inp._valid.record_stream(stream)


@VarLenTensor.implements(aten.to.dtype_layout)
def _to_dtype_layout(
    inp: VarLenTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool | None = None,  # Ignored by PyTorch.
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> VarLenTensor:
    if dtype is None:
        dtype = inp.dtype
    if layout is None:
        layout = inp.layout
    if device is None:
        device = inp.device
    device = torch.device(device)
    if memory_format is None:
        memory_format = torch.preserve_format

    if inp.ALLOWED_DTYPES is not None and dtype not in inp.ALLOWED_DTYPES:
        raise TypeError(
            f"Can't convert {inp.__class__.__name__!r} to dtype '{dtype}'"
        )
    if layout != torch.strided:
        raise TypeError(
            f"Can't convert {inp.__class__.__name__!r} to layout '{layout}'"
        )
    if memory_format not in (torch.preserve_format, torch.contiguous_format):
        raise ValueError(
            f"Unsupported memory format '{memory_format}' for "
            f"'{inp.__class__.__name__}.clone'"
        )

    if (
        not copy
        and dtype == inp.dtype
        and device == inp.device
        and layout == inp.layout
        and (
            memory_format == torch.preserve_format
            or (
                memory_format == torch.contiguous_format
                and inp.is_contiguous()
            )
        )
    ):
        return inp

    # Copying has two cases:
    # 1. Slice when the output layout can reuse the input storage order.
    # 2. Materialize in case of holes, overlaps, or change in memory format.
    use_slice = (
        inp.numel() == 0
        or (
            memory_format == torch.preserve_format
            and torch._debug_has_internal_overlap(_layout_view(inp)) == 0
        )
        or (memory_format == torch.contiguous_format and inp.is_contiguous())
    )
    if not use_slice:
        return _materialize(
            inp,
            lambda x: x.clone(memory_format=memory_format),
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        )

    storage_offset = int(inp.storage_offset())
    span_len = _span_len(inp.size(), inp.stride())
    offset = inp._offset[storage_offset : storage_offset + span_len + 1]
    data = inp._data[offset[0] : offset[-1]].to(
        device=device,
        dtype=dtype,
        non_blocking=non_blocking,
        copy=copy,
    )
    offset = (offset - offset[0]).to(device=device, non_blocking=non_blocking)
    valid: Tensor | None = None
    if inp._valid is not None:
        valid = inp._valid[storage_offset : storage_offset + span_len].to(
            device=device,
            non_blocking=non_blocking,
            copy=copy,
        )

    return inp.__class__(
        data=data,
        offset=offset,
        valid=valid,
        size=inp.size(),
        stride=inp.stride()
        if memory_format == torch.preserve_format
        else None,
        storage_offset=0,
    )


@VarLenTensor.implements(aten.to.dtype)
def _to_dtype(
    inp: VarLenTensor,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> VarLenTensor:
    return _to_dtype_layout(
        inp,
        dtype=dtype,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@VarLenTensor.implements(aten.to.device)
def _to_device(
    inp: VarLenTensor,
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> VarLenTensor:
    return _to_dtype_layout(
        inp,
        dtype=dtype,
        device=device,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@VarLenTensor.implements(aten.to.other)
def _to_other(
    inp: VarLenTensor,
    other: Tensor,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> VarLenTensor:
    return _to_dtype_layout(
        inp,
        dtype=other.dtype,
        layout=other.layout,
        device=other.device,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@VarLenTensor.implements(aten._to_copy.default)
def _to_copy(
    inp: VarLenTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,  # Ignored by PyTorch.
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> VarLenTensor:
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


@VarLenTensor.implements(aten.clone.default)
def _clone(
    inp: VarLenTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> VarLenTensor:
    return _to_dtype_layout(inp, copy=True, memory_format=memory_format)


@VarLenTensor.implements(aten.contiguous.default)
def _contiguous(
    inp: VarLenTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> VarLenTensor:
    return _to_dtype_layout(inp, copy=True, memory_format=memory_format)


@VarLenTensor.implements(aten.detach.default)
@preserve_view_inference_mode
def _detach(inp: VarLenTensor) -> VarLenTensor:
    return inp.__class__(
        data=inp._data.detach(),
        offset=inp._offset,
        valid=inp._valid,
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()),
    )


@VarLenTensor.implements(aten.is_pinned.default)
def _is_pinned(inp: VarLenTensor) -> bool:
    is_pinned = inp._data.is_pinned() and inp._offset.is_pinned()
    return is_pinned and (inp._valid is None or inp._valid.is_pinned())


@VarLenTensor.implements(aten._pin_memory.default)
def _pin_memory(inp: VarLenTensor) -> VarLenTensor:
    return inp.__class__(
        data=inp._data.pin_memory(),
        offset=inp._offset.pin_memory(),
        valid=inp._valid.pin_memory() if inp._valid is not None else None,
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()),
    )


@VarLenTensor.implements(aten.pin_memory.default)
def _pin_memory_composite(
    inp: VarLenTensor,
    device: torch.device | None = None,
) -> VarLenTensor:
    if _is_pinned(inp):
        return inp
    return _pin_memory(inp)


@VarLenTensor.implements(aten.isnan.default)
def _isnan(inp: VarLenTensor) -> Tensor:
    valid = inp.valid
    if valid is None:
        return torch.zeros(
            inp.size(),
            dtype=torch.bool,
            device=inp.device,
        )
    return ~valid


@VarLenTensor.implements(aten.isfinite.default)
def _isfinite(inp: VarLenTensor) -> Tensor:
    valid = inp.valid
    if valid is None:
        return torch.ones(
            inp.size(),
            dtype=torch.bool,
            device=inp.device,
        )
    return valid.clone()


@VarLenTensor.implements(aten.equal.default)
def _equal(inp: VarLenTensor, other: Tensor) -> bool:
    if inp.__class__ is not other.__class__:
        return False
    if inp.size() != other.size():
        return False

    valid1, valid2 = inp.valid, other.valid
    if valid1 is not None and valid2 is not None:
        if not valid1.equal(valid2):
            return False
    elif (valid1 is not None and not valid1.all()) or (
        valid2 is not None and not valid2.all()
    ):
        return False

    inp = cast(
        VarLenTensor,
        inp.contiguous() if valid1 is None else inp[valid1],
    )
    other = cast(
        VarLenTensor,
        other.contiguous() if valid2 is None else other[valid2],
    )

    data1, offset1 = inp.data_offset
    data2, offset2 = other.data_offset

    if not offset1.equal(offset2):
        return False

    return data1.equal(data2)


@VarLenTensor.implements(aten.allclose.default)
def _allclose(
    inp: VarLenTensor,
    other: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    if inp.__class__ is not other.__class__:
        return False
    if inp.size() != other.size():
        return False

    valid1, valid2 = inp.valid, other.valid
    if valid1 is not None and valid2 is not None:
        if not valid1.equal(valid2):
            return False
    elif (valid1 is not None and not valid1.all()) or (
        valid2 is not None and not valid2.all()
    ):
        return False

    inp = cast(
        VarLenTensor,
        inp.contiguous() if valid1 is None else inp[valid1],
    )
    other = cast(
        VarLenTensor,
        other.contiguous() if valid2 is None else other[valid2],
    )

    data1, offset1 = inp.data_offset
    data2, offset2 = other.data_offset

    if not offset1.equal(offset2):
        return False

    if data1.is_floating_point() and data2.is_floating_point():
        return data1.allclose(data2, rtol, atol, equal_nan)

    return data1.equal(data2)


@VarLenTensor.implements(aten.view.default)
@preserve_view_inference_mode
def _view(inp: VarLenTensor, size: Sequence[int]) -> VarLenTensor:
    view = _layout_view(inp).view(tuple(size))
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten._unsafe_view.default)
@preserve_view_inference_mode
def _unsafe_view(inp: VarLenTensor, size: Sequence[int]) -> VarLenTensor:
    view = aten._unsafe_view.default(_layout_view(inp), size)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.reshape.default)
def _reshape(inp: VarLenTensor, size: Sequence[int]) -> VarLenTensor:
    return cast(
        VarLenTensor,
        aten.reshape.default.decompose(inp, size),
    )


@VarLenTensor.implements(aten.flatten.using_ints)
def _flatten(
    inp: VarLenTensor,
    start_dim: int = 0,
    end_dim: int = -1,
) -> VarLenTensor:
    return cast(
        VarLenTensor,
        aten.flatten.using_ints.decompose(inp, start_dim, end_dim),
    )


@VarLenTensor.implements(aten.squeeze.default)
@preserve_view_inference_mode
def _squeeze(inp: VarLenTensor) -> VarLenTensor:
    view = _layout_view(inp).squeeze()
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.squeeze.dim)
@preserve_view_inference_mode
def _squeeze_dim(inp: VarLenTensor, dim: int) -> VarLenTensor:
    view = _layout_view(inp).squeeze(dim)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.squeeze.dims)
@preserve_view_inference_mode
def _squeeze_dims(inp: VarLenTensor, dim: Sequence[int]) -> VarLenTensor:
    view = _layout_view(inp).squeeze(tuple(dim))
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.unsqueeze.default)
@preserve_view_inference_mode
def _unsqueeze(inp: VarLenTensor, dim: int) -> VarLenTensor:
    view = _layout_view(inp).unsqueeze(dim)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.expand.default)
@preserve_view_inference_mode
def _expand(
    inp: VarLenTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> VarLenTensor:
    view = aten.expand.default(_layout_view(inp), size, implicit=implicit)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.t.default)
@preserve_view_inference_mode
def _t(inp: VarLenTensor) -> VarLenTensor:
    view = _layout_view(inp).t()
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.transpose.int)
@preserve_view_inference_mode
def _transpose(inp: VarLenTensor, dim0: int, dim1: int) -> VarLenTensor:
    view = _layout_view(inp).transpose(dim0, dim1)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.permute.default)
@preserve_view_inference_mode
def _permute(inp: VarLenTensor, dims: Sequence[int]) -> VarLenTensor:
    view = _layout_view(inp).permute(tuple(dims))
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.movedim.int)
def _movedim_int(
    inp: VarLenTensor,
    source: int,
    destination: int,
) -> VarLenTensor:
    return cast(
        VarLenTensor,
        aten.movedim.int.decompose(inp, source, destination),
    )


@VarLenTensor.implements(aten.movedim.intlist)
def _movedim_intlist(
    inp: VarLenTensor,
    source: Sequence[int],
    destination: Sequence[int],
) -> VarLenTensor:
    return cast(
        VarLenTensor,
        aten.movedim.intlist.decompose(inp, source, destination),
    )


@VarLenTensor.implements(aten.select.int)
@preserve_view_inference_mode
def _select(inp: VarLenTensor, dim: int, index: int) -> VarLenTensor:
    view = _layout_view(inp).select(dim, index)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.slice.Tensor)
@preserve_view_inference_mode
def _slice(
    inp: VarLenTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> VarLenTensor:
    view = aten.slice.Tensor(_layout_view(inp), dim, start, end, step)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.narrow.default)
@preserve_view_inference_mode
def _narrow(
    inp: VarLenTensor,
    dim: int,
    start: int,
    length: int,
) -> VarLenTensor:
    view = _layout_view(inp).narrow(dim, start, length)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.unbind.int)
@preserve_view_inference_mode
def _unbind(inp: VarLenTensor, dim: int = 0) -> tuple[VarLenTensor, ...]:
    return tuple(
        _from_layout_view(inp, view) for view in _layout_view(inp).unbind(dim)
    )


@VarLenTensor.implements(aten.split.Tensor)
@preserve_view_inference_mode
def _split(
    inp: VarLenTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[VarLenTensor, ...]:
    return tuple(
        _from_layout_view(inp, view)
        for view in _layout_view(inp).split(split_size, dim)
    )


@VarLenTensor.implements(aten.split.sizes)
@VarLenTensor.implements(aten.split.default)
@VarLenTensor.implements(aten.split_with_sizes.default)
@preserve_view_inference_mode
def _split_with_sizes(
    inp: VarLenTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[VarLenTensor, ...]:
    return tuple(
        _from_layout_view(inp, view)
        for view in _layout_view(inp).split(tuple(split_sizes), dim)
    )


@VarLenTensor.implements(aten.masked_select.default)
def _masked_select(inp: VarLenTensor, mask: Tensor) -> VarLenTensor:
    return _materialize(inp, lambda x: x.masked_select(mask))


@VarLenTensor.implements(aten.index_select.default)
def _index_select(
    inp: VarLenTensor,
    dim: int,
    index: Tensor,
) -> VarLenTensor:
    return _materialize(inp, lambda x: x.index_select(dim, index))


@VarLenTensor.implements(aten.take.default)
def _take(inp: VarLenTensor, index: Tensor) -> VarLenTensor:
    return _materialize(inp, lambda x: x.take(index))


@VarLenTensor.implements(aten.index.Tensor)
def _index(
    inp: VarLenTensor,
    indices: Sequence[Tensor | None],
) -> VarLenTensor:
    return _materialize(inp, lambda x: aten.index.Tensor(x, indices))


@VarLenTensor.implements(aten.cat.default)
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> VarLenTensor:
    if len(tensors) == 0:
        raise ValueError("Expected a non-empty list of Tensors")

    if not isinstance(tensors[0], VarLenTensor):
        raise TypeError(
            f"Expected {VarLenTensor.__name__!r} as element 0, but got "
            f"{tensors[0].__class__.__name__!r}"
        )

    tensor_cls = tensors[0].__class__
    for i, tensor in enumerate(tensors):
        if tensor.__class__ is not tensor_cls:
            raise TypeError(
                f"Expected {tensor_cls.__name__!r} as element {i}, but got "
                f"{tensor.__class__.__name__!r}"
            )

    tensors = tuple(
        cast(VarLenTensor, tensor.contiguous()) for tensor in tensors
    )
    data_list, offsets = zip(*(tensor.data_offset for tensor in tensors))
    valid: Tensor | None = None
    if any(tensor._valid is not None for tensor in tensors):
        valid_views = tuple(
            tensor.valid
            if tensor.valid is not None
            else torch.ones(
                tensor.size(),
                dtype=torch.bool,
                device=tensor.device,
            )
            for tensor in tensors
        )
        valid = torch.cat(valid_views, dim=dim).contiguous().view(-1)

    offset_dtype: torch.dtype = torch.int32
    if (
        any(offset.dtype == torch.int64 for offset in offsets)
        or sum(d.numel() for d in data_list) > torch.iinfo(torch.int32).max
    ):
        offset_dtype = torch.int64

    dim_size = 0
    storage_offset = 0
    start_views, end_views = [], []
    for tensor, data, offset in zip(tensors, data_list, offsets):
        offset = offset.to(offset_dtype) + storage_offset
        dim_size += tensor.size(dim)
        storage_offset += data.numel()
        start_views.append(offset[:-1].view(tensor.size()))
        end_views.append(offset[1:].view(tensor.size()))

    size = tensors[0].size()
    dim = dim % len(size)
    size = (*size[:dim], dim_size, *size[dim + 1 :])
    offset = offsets[0].new_empty(math.prod(size) + 1, dtype=offset_dtype)
    start = torch.cat(start_views, dim=dim, out=offset[:-1].view(size))
    data = torch.cat(data_list, dim=0)

    if math.prod(tensors[0].size()[:dim]) == 1:  # Contiguous path:
        offset[-1] = storage_offset
        return tensor_cls(data=data, offset=offset, valid=valid, size=size)

    end = torch.cat(end_views, dim=dim)
    start = torch.as_strided(start, size=(start.numel(),), stride=(1,))
    end = torch.as_strided(end, size=(end.numel(),), stride=(1,))

    offset, index = _compact(start, end)

    return tensor_cls(data=data[index], offset=offset, valid=valid, size=size)


@VarLenTensor.implements(aten.stack.default)
def _stack(tensors: Sequence[Tensor], dim: int = 0) -> VarLenTensor:
    out = torch.cat([tensor.unsqueeze(dim) for tensor in tensors], dim=dim)
    return cast(VarLenTensor, out)


# Helpers #####################################################################


def _contiguous_stride(size: Sequence[int]) -> tuple[int, ...]:
    value = 1
    stride = []
    for dim_size in reversed(size):
        stride.append(value)
        value *= dim_size
    return tuple(stride[::-1])


def _span_len(size: Sequence[int], stride: Sequence[int]) -> int:
    if math.prod(size) == 0:
        return 0
    return 1 + sum(
        (dim_size - 1) * dim_stride
        for dim_size, dim_stride in zip(size, stride)
    )


def _layout_view(inp: VarLenTensor) -> Tensor:
    return torch.as_strided(
        inp._offset,
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()),
    )


def _from_layout_view(inp: VarLenTensor, view: Tensor) -> VarLenTensor:
    return inp.__class__(
        data=inp._data,
        offset=inp._offset,
        valid=inp._valid,
        size=view.size(),
        stride=view.stride(),
        storage_offset=0 if view.numel() == 0 else int(view.storage_offset()),
    )


def _compact(start: Tensor, end: Tensor) -> tuple[Tensor, Tensor]:
    count = end - start

    offset = count.new_empty(count.numel() + 1, dtype=torch.int64)
    offset[0] = 0
    offset[1:] = count.cumsum(dim=0, dtype=torch.int64)

    total = int(offset[-1])
    if start.dtype == torch.int32 and total <= torch.iinfo(torch.int32).max:
        offset = offset.to(torch.int32)

    local = torch.arange(
        end=total,
        dtype=offset.dtype,
        device=offset.device,
    )
    local -= offset[:-1].repeat_interleave(
        count,
        output_size=total,
    )
    index = start.repeat_interleave(count, output_size=total)
    index += local

    return offset, index


def _materialize(
    inp: VarLenTensor,
    function: Callable[[Tensor], Tensor],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    non_blocking: bool = False,
) -> VarLenTensor:
    # Use PyTorch's own memory-format semantics to materialize data:
    start = torch.as_strided(
        inp._offset,
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()),
    )
    start = function(start)
    assert start.storage_offset() == 0
    assert _span_len(start.size(), start.stride()) == start.numel()

    end = torch.as_strided(
        inp._offset,
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()) + 1,
    )
    end = function(end)
    assert end.storage_offset() == 0
    assert _span_len(end.size(), end.stride()) == end.numel()

    size = start.size()
    stride = start.stride()

    start = torch.as_strided(
        start,
        size=(start.numel(),),
        stride=(1,),
        storage_offset=start.storage_offset(),
    )
    end = torch.as_strided(
        end,
        size=(end.numel(),),
        stride=(1,),
        storage_offset=end.storage_offset(),
    )
    valid: Tensor | None = None
    if inp._valid is not None:
        valid = torch.as_strided(
            inp._valid,
            size=inp.size(),
            stride=inp.stride(),
            storage_offset=int(inp.storage_offset()),
        )
        valid = function(valid).contiguous().view(-1)

    offset, index = _compact(start, end)

    return inp.__class__(
        data=inp._data[index].to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        ),
        offset=offset.to(device, non_blocking=non_blocking),
        valid=valid.to(device=device, non_blocking=non_blocking)
        if valid is not None
        else None,
        size=size,
        stride=stride,
        storage_offset=0,
    )

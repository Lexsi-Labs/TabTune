# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from itertools import accumulate, chain
from typing import TYPE_CHECKING, Any, ClassVar, SupportsIndex, cast
from typing_extensions import Self

import pyarrow as pa
import torch
from torch import Tensor
from torch.utils import _pytree as pytree
from typing_extensions import override

from tabtune.models.kumo._sdm.tensor import StringTensor, VarLenTensor
from tabtune.models.kumo._sdm.tensor.io import (
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


class CategoricalTensor(Tensor):
    r"""A :class:`torch.Tensor` for categorical columns.

    A :class:`CategoricalTensor` stores categorical indices in ``code`` and one
    category vector per column in ``categories``.
    Code values are direct indices into the corresponding category vector.
    Negative indices represent missing values.

    .. testcode::

        import torch
        from tabtune.models.kumo._sdm import CategoricalTensor, StringTensor

        tensor = CategoricalTensor(
            code=torch.randint(0, 2, size=(10, 2)),
            categories=(
                StringTensor.from_list(["USA", "GERMANY"]),
                StringTensor.from_list(["enterprise", "startup"]),
            ),
        )

    Args:
        code: The categorical indices of shape ``[..., C]``.
        categories: A tuple of ``C`` category vectors.
    """

    ALLOWED_DTYPES = (torch.int32, torch.int64)
    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _code: Tensor
    _categories: tuple[Tensor, ...]

    # Constructors ############################################################

    def __init__(
        self,
        code: Tensor,
        categories: Sequence[Tensor],
    ) -> None:
        pass

    def __new__(
        cls,
        code: Tensor,
        categories: Sequence[Tensor],
    ) -> Self:
        r"""Create a tensor wrapper."""
        if code.dtype not in cls.ALLOWED_DTYPES:
            raise ValueError(
                f"Expected 'code' in {cls.__name__!r} to have dtype "
                f"in '{cls.ALLOWED_DTYPES}' (got '{code.dtype}')"
            )
        if code.dim() == 0:
            raise ValueError(
                f"Expected {cls.__name__!r} to have at least one dimension"
            )
        if code.size(-1) != len(categories):
            raise ValueError(
                f"Expected the last dimension in {cls.__name__!r} to match "
                f"the number of category vectors (got {code.size(-1)} and "
                f"{len(categories)})"
            )
        for i, category in enumerate(categories):
            if category.dim() != 1:
                raise ValueError(
                    f"Expected category {i} in {cls.__name__!r} to be "
                    f"one-dimensional (got {category.dim()}D)"
                )
            if isinstance(category, VarLenTensor) and category.is_nullable:
                raise ValueError(
                    f"Expected category {i} in {cls.__name__!r} to not "
                    "contain null values"
                )

        out = Tensor._make_wrapper_subclass(
            cls,
            size=code.size(),
            strides=code.stride(),
            storage_offset=code.storage_offset(),
            dtype=code.dtype,
            device=code.device,
            requires_grad=False,
        )

        out._code = code
        out._categories = tuple(categories)

        return out

    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a :class:`pyarrow.Array`.

        .. testcode::

            import pyarrow as pa
            from tabtune.models.kumo._sdm import CategoricalTensor

            array = pa.array(["foo", None, "bar"])
            tensor = CategoricalTensor.from_arrow(array)

            assert tensor.code.tolist() == [[0], [-1], [1]]
            assert tensor.categories[0].tolist() == ["foo", "bar"]

        Args:
            array: The :class:`pyarrow.Array` or
                :class:`pyarrow.ChunkedArray`.
            dtype: The dtype.
            device: The device.
        """
        device = torch.device("cpu" if device is None else device)

        if isinstance(array, pa.ChunkedArray):
            array = combine_arrow_chunks(array)

        encoded = array.dictionary_encode()
        code = arrow_as_tensor(
            encoded.indices.fill_null(-1),
            dtype=dtype,
            device=device,
        ).unsqueeze(-1)
        if dtype is None and code.dtype not in cls.ALLOWED_DTYPES:
            code = code.to(torch.int32)

        dictionary = encoded.dictionary
        is_string = pa.types.is_string(dictionary.type)
        is_large_string = pa.types.is_large_string(dictionary.type)
        if is_string or is_large_string:
            category = StringTensor.from_arrow(dictionary, device=device)
        elif pa.types.is_null(dictionary.type):
            category = torch.empty(0, dtype=torch.int64, device=device)
        else:  # Use regular torch.Tensor for Tensor-compatible dictionaries:
            category = arrow_as_tensor(dictionary, device=device)

        return cls(code=code, categories=(category,))

    def to_arrow(self, names: Sequence[str] | None = None) -> pa.Table:
        r"""Convert this tensor to a two-dimensional :class:`pyarrow.Table`.

        Args:
            names: Column names.
        """
        if names is None:
            names = tuple(str(i) for i in range(self.size(-1)))
        elif len(names) != self.size(-1):
            raise ValueError(
                f"Expected 'columns' to contain {self.size(-1)} entries "
                f"(got {len(names)})"
            )

        arrays = []
        for code, category, mask in zip(
            self.code.movedim(-1, 0).contiguous().cpu().unbind(0),
            self.categories,
            self.isfinite().movedim(-1, 0).contiguous().cpu().unbind(0),
        ):
            arrays.append(
                pa.DictionaryArray.from_arrays(
                    indices=to_arrow(code, mask),
                    dictionary=category.to_arrow()
                    if isinstance(category, StringTensor)
                    else to_arrow(category),
                )
            )

        return pa.Table.from_arrays(arrays, names=names)

    @classmethod
    def from_cudf(
        cls,
        ser: cudf.Series,
        *,
        dtype: torch.dtype = torch.int32,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a categorical :class:`cudf.Series`.

        Args:
            ser: The categorical :class:`cudf.Series`.
            dtype: The dtype.
            device: The device.
        """
        import cudf
        from cudf.api.types import is_string_dtype

        if isinstance(ser.dtype, cudf.CategoricalDtype):
            code_dtype = "int64" if dtype == torch.int64 else "int32"
            codes = ser.cat.codes.astype(code_dtype, copy=False).to_cupy(
                na_value=-1
            )
            categories = ser.cat.categories
        else:
            codes, categories = ser.factorize(
                sort=False,
                use_na_sentinel=True,
            )
        code = torch.from_dlpack(codes).unsqueeze(-1).to(device, dtype)

        if len(categories) == 0:
            category = torch.empty(0, dtype=torch.int64, device=code.device)
        elif is_string_dtype(categories.dtype):
            category = StringTensor.from_cudf(categories, device=device)
        else:
            category = torch.from_dlpack(categories.to_cupy()).to(device)

        return cls(code=code, categories=(category,))

    def to_cudf(self, names: Sequence[str] | None = None) -> cudf.DataFrame:
        r"""Convert this tensor to a two-dimensional :class:`cudf.DataFrame`.

        Args:
            names: Column names.
        """
        import cudf

        if names is None:
            names = tuple(str(i) for i in range(self.size(-1)))
        elif len(names) != self.size(-1):
            raise ValueError(
                f"Expected 'columns' to contain {self.size(-1)} entries "
                f"(got {len(names)})"
            )

        columns = {}
        for name, code, category, mask in zip(
            names,
            self.code.movedim(-1, 0).contiguous().unbind(0),
            self.categories,
            self.isfinite().movedim(-1, 0).contiguous().unbind(0),
        ):
            columns[name] = cudf.CategoricalIndex.from_codes(
                codes=to_cudf(code, mask)._column,
                categories=category.to_cudf()
                if isinstance(category, StringTensor)
                else to_cudf(category),
                ordered=False,
            )

        return cudf.DataFrame(columns)

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> Self:
        r"""Create tensor from a numerical :class:`torch.Tensor`.

        Args:
            tensor: The numerical tensor.
        """
        if tensor.size(-1) == 0:
            return cls(
                code=tensor.to(torch.int64),
                categories=(),
            )

        categories, values = zip(
            *[
                column.unique(return_inverse=True)
                for column in tensor.unbind(dim=-1)
            ]
        )
        return cls(
            code=torch.stack(values, dim=-1),
            categories=categories,
        )

    # Properties ##############################################################

    @property
    def code(self) -> Tensor:
        r"""Return the categorical code tensor."""
        return self._code

    @property
    def categories(self) -> tuple[Tensor, ...]:
        r"""Return category vector for each categorical column."""
        return self._categories

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
        args = (self._code, self._categories)
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
            assert isinstance(args[0], CategoricalTensor)
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
            return handler(*args, **(kwargs or {}))

        # Operate on vanilla tensors for all non-handled functions:
        args = pytree.tree_map_only(CategoricalTensor, lambda x: x._code, args)
        kwargs = pytree.tree_map_only(
            CategoricalTensor, lambda x: x._code, kwargs
        )
        return func(*args, **(kwargs or {}))

    @override
    def is_shared(self) -> bool:
        return self._code.is_shared()

    @override
    def share_memory_(self) -> Self:
        self._code.share_memory_()
        return self

    @override
    def tolist(self) -> Any:
        def apply_na_mask(values: Any, na_mask: Any) -> Any:
            if isinstance(na_mask, bool):
                return None if na_mask else values

            return [
                apply_na_mask(value, isna)
                for value, isna in zip(values, na_mask)
            ]

        def decode_column(tensor: CategoricalTensor) -> Sequence[Any]:
            out = tensor.categories[0][tensor.code.clamp(min=0).squeeze(-1)]
            return apply_na_mask(
                out.tolist(),
                tensor.isnan().squeeze(-1).tolist(),
            )

        def columns_to_rows(
            columns: Sequence[Any],
            size: tuple[int, ...],
        ) -> Any:
            if len(size) == 0:
                return list(columns)

            return [
                columns_to_rows(
                    columns=[column[i] for column in columns],
                    size=size[1:],
                )
                for i in range(size[0])
            ]

        columns = [
            decode_column(cast(CategoricalTensor, column))
            for column in self.split(1, dim=-1)
        ]
        return columns_to_rows(columns, tuple(self.size()[:-1]))


@CategoricalTensor.implements(aten.isnan.default)
def _isnan(inp: CategoricalTensor) -> Tensor:
    return inp._code < 0


@CategoricalTensor.implements(aten.isfinite.default)
def _isfinite(inp: CategoricalTensor) -> Tensor:
    return inp._code >= 0


@CategoricalTensor.implements(aten.alias.default)
@preserve_view_inference_mode
def _alias(inp: CategoricalTensor) -> CategoricalTensor:
    return inp.__class__(aten.alias.default(inp._code), inp._categories)


@CategoricalTensor.implements(aten.record_stream.default)
def _record_stream(inp: CategoricalTensor, stream: torch.Stream) -> None:
    inp._code.record_stream(stream)
    for category in inp._categories:
        category.record_stream(stream)


@CategoricalTensor.implements(aten.to.dtype_layout)
def _to_dtype_layout(
    inp: CategoricalTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool | None = None,  # Ignored by PyTorch.
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> Tensor:

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

    code = aten.to.dtype_layout(
        inp._code,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )
    if code.dtype not in inp.ALLOWED_DTYPES or code.layout != torch.strided:
        return code

    categories = tuple(
        aten.to.dtype_layout(
            category,
            dtype=None,
            layout=None,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            copy=copy,
            memory_format=None,
        )
        for category in inp._categories
    )
    return inp.__class__(code, categories)


@CategoricalTensor.implements(aten.to.dtype)
def _to_dtype(
    inp: CategoricalTensor,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> Tensor:
    return _to_dtype_layout(
        inp,
        dtype=dtype,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@CategoricalTensor.implements(aten.to.device)
def _to_device(
    inp: CategoricalTensor,
    device: torch.device | str,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> Tensor:
    return _to_dtype_layout(
        inp,
        dtype=dtype,
        device=device,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@CategoricalTensor.implements(aten.to.other)
def _to_other(
    inp: CategoricalTensor,
    other: Tensor,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> Tensor:
    return _to_dtype_layout(
        inp,
        dtype=other.dtype,
        layout=other.layout,
        device=other.device,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@CategoricalTensor.implements(aten._to_copy.default)
def _to_copy(
    inp: CategoricalTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,  # Ignored by PyTorch.
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> Tensor:
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


@CategoricalTensor.implements(aten.clone.default)
def _clone(
    inp: CategoricalTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> CategoricalTensor:
    return _to_dtype_layout(inp, copy=True, memory_format=memory_format)


@CategoricalTensor.implements(aten.contiguous.default)
def _contiguous(
    inp: CategoricalTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> CategoricalTensor:
    code = inp._code.contiguous(memory_format=memory_format)
    return inp.__class__(code, inp._categories)


@CategoricalTensor.implements(aten.is_pinned.default)
def _is_pinned(inp: CategoricalTensor) -> bool:
    return inp._code.is_pinned() and all(
        category.is_pinned() for category in inp._categories
    )


@CategoricalTensor.implements(aten._pin_memory.default)
def _pin_memory(inp: CategoricalTensor) -> CategoricalTensor:
    categories = tuple(category.pin_memory() for category in inp._categories)
    return inp.__class__(inp._code.pin_memory(), categories)


@CategoricalTensor.implements(aten.pin_memory.default)
def _pin_memory_composite(
    inp: CategoricalTensor,
    device: torch.device | None = None,
) -> CategoricalTensor:
    if _is_pinned(inp):
        return inp
    return _pin_memory(inp)


@CategoricalTensor.implements(aten.equal.default)
def _equal(inp: CategoricalTensor, other: Tensor) -> bool:
    if inp.__class__ is not other.__class__:
        return False
    if inp.size() != other.size():
        return False

    if not inp._code.equal(other._code):
        return False

    for category1, category2 in zip(inp._categories, other._categories):
        if not category1.equal(category2):
            return False

    return True


@CategoricalTensor.implements(aten.allclose.default)
def _allclose(
    inp: CategoricalTensor,
    other: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    return _equal(inp, other)


@CategoricalTensor.implements(aten.view.default)
@preserve_view_inference_mode
def _view(inp: CategoricalTensor, size: Sequence[int]) -> Tensor:
    return _maybe_wrap(inp, inp._code.view(size))


@CategoricalTensor.implements(aten._unsafe_view.default)
@preserve_view_inference_mode
def _unsafe_view(inp: CategoricalTensor, size: Sequence[int]) -> Tensor:
    return _maybe_wrap(inp, aten._unsafe_view(inp._code, size))


@CategoricalTensor.implements(aten.reshape.default)
def _reshape(inp: CategoricalTensor, size: Sequence[int]) -> Tensor:
    return aten.reshape.default.decompose(inp, size)


@CategoricalTensor.implements(aten.flatten.using_ints)
def _flatten(
    inp: CategoricalTensor,
    start_dim: int = 0,
    end_dim: int = -1,
) -> Tensor:
    return aten.flatten.using_ints.decompose(inp, start_dim, end_dim)


@CategoricalTensor.implements(aten.squeeze.default)
@preserve_view_inference_mode
def _squeeze(inp: CategoricalTensor) -> Tensor:
    return _maybe_wrap(inp, inp._code.squeeze())


@CategoricalTensor.implements(aten.squeeze.dim)
@preserve_view_inference_mode
def _squeeze_dim(inp: CategoricalTensor, dim: int) -> Tensor:
    return _maybe_wrap(inp, inp._code.squeeze(dim))


@CategoricalTensor.implements(aten.squeeze.dims)
@preserve_view_inference_mode
def _squeeze_dims(inp: CategoricalTensor, dim: Sequence[int]) -> Tensor:
    return _maybe_wrap(inp, inp._code.squeeze(tuple(dim)))


@CategoricalTensor.implements(aten.unsqueeze.default)
@preserve_view_inference_mode
def _unsqueeze(inp: CategoricalTensor, dim: int) -> Tensor:
    return _maybe_wrap(inp, inp._code.unsqueeze(dim))


@CategoricalTensor.implements(aten.expand.default)
@preserve_view_inference_mode
def _expand(
    inp: CategoricalTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> Tensor:
    code = aten.expand.default(inp._code, size, implicit=implicit)
    return _maybe_wrap(inp, code)


@CategoricalTensor.implements(aten.transpose.int)
@preserve_view_inference_mode
def _transpose(inp: CategoricalTensor, dim0: int, dim1: int) -> Tensor:
    code = inp._code.transpose(dim0, dim1)
    dim0 %= inp.dim()
    dim1 %= inp.dim()
    if dim0 != dim1 and inp.dim() - 1 in (dim0, dim1):
        return code
    return inp.__class__(code, inp.categories)


@CategoricalTensor.implements(aten.permute.default)
@preserve_view_inference_mode
def _permute(inp: CategoricalTensor, dims: Sequence[int]) -> Tensor:
    code = inp._code.permute(tuple(dims))
    dims = tuple(dim % inp.dim() for dim in dims)
    if dims[-1] != inp.dim() - 1:
        return code
    return inp.__class__(code, inp.categories)


@CategoricalTensor.implements(aten.movedim.int)
def _movedim_int(
    inp: CategoricalTensor,
    source: int,
    destination: int,
) -> Tensor:
    return aten.movedim.int.decompose(inp, source, destination)


@CategoricalTensor.implements(aten.movedim.intlist)
def _movedim_intlist(
    inp: CategoricalTensor,
    source: Sequence[int],
    destination: Sequence[int],
) -> Tensor:
    return aten.movedim.intlist.decompose(inp, source, destination)


@CategoricalTensor.implements(aten.select.int)
@preserve_view_inference_mode
def _select(inp: CategoricalTensor, dim: int, index: int) -> Tensor:
    code = inp._code.select(dim, index)
    dim %= inp.dim()
    if dim == inp.dim() - 1:
        return code
    return inp.__class__(code, inp.categories)


@CategoricalTensor.implements(aten.slice.Tensor)
@preserve_view_inference_mode
def _slice(
    inp: CategoricalTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> CategoricalTensor:
    code = aten.slice.Tensor(inp._code, dim, start, end, step)
    dim %= inp.dim()
    if dim != inp.dim() - 1:
        return inp.__class__(code, inp.categories)
    return inp.__class__(code, inp.categories[slice(start, end, step)])


@CategoricalTensor.implements(aten.narrow.default)
@preserve_view_inference_mode
def _narrow(
    inp: CategoricalTensor,
    dim: int,
    start: int,
    length: int,
) -> CategoricalTensor:
    code = inp._code.narrow(dim, start, length)
    dim %= inp.dim()
    if dim != inp.dim() - 1:
        return inp.__class__(code, inp.categories)
    if start < 0:
        start += inp.size(dim)
    return inp.__class__(code, inp.categories[start : start + length])


@CategoricalTensor.implements(aten.unbind.int)
@preserve_view_inference_mode
def _unbind(inp: CategoricalTensor, dim: int = 0) -> tuple[Tensor, ...]:
    code_list = inp._code.unbind(dim)
    dim %= inp.dim()
    if dim == inp.dim() - 1:
        return code_list
    return tuple(inp.__class__(code, inp.categories) for code in code_list)


@CategoricalTensor.implements(aten.split.Tensor)
@preserve_view_inference_mode
def _split(
    inp: CategoricalTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[CategoricalTensor, ...]:
    code_list = inp._code.split(split_size, dim)
    dim %= inp.dim()
    if dim != inp.dim() - 1:
        return tuple(inp.__class__(code, inp.categories) for code in code_list)
    return tuple(
        inp.__class__(code, inp.categories[i : i + split_size])
        for code, i in zip(code_list, range(0, inp.size(dim), split_size))
    )


@CategoricalTensor.implements(aten.split.sizes)
@CategoricalTensor.implements(aten.split.default)
@CategoricalTensor.implements(aten.split_with_sizes.default)
@preserve_view_inference_mode
def _split_with_sizes(
    inp: CategoricalTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[CategoricalTensor, ...]:
    code_list = inp._code.split(tuple(split_sizes), dim)
    dim %= inp.dim()
    if dim != inp.dim() - 1:
        return tuple(inp.__class__(code, inp.categories) for code in code_list)

    offset = (0, *accumulate(split_sizes))
    return tuple(
        inp.__class__(code, inp.categories[start:end])
        for code, start, end in zip(code_list, offset[:-1], offset[1:])
    )


@CategoricalTensor.implements(aten.index_select.default)
def _index_select(
    inp: CategoricalTensor,
    dim: int,
    index: Tensor,
) -> CategoricalTensor:
    code = inp._code.index_select(dim, index)
    dim %= inp.dim()
    if dim != inp.dim() - 1:
        return inp.__class__(code, inp.categories)
    categories = tuple(inp.categories[i] for i in index.tolist())
    return inp.__class__(code, categories)


@CategoricalTensor.implements(aten.index.Tensor)
def _index(
    inp: CategoricalTensor,
    indices: Sequence[Tensor | None],
) -> Tensor:
    code = aten.index.Tensor(inp._code, indices)

    current_dim = 0
    has_other_index = False
    category_index: Tensor | None = None
    for index in indices:
        if index is None:
            current_dim += 1
            continue

        # Check whether we index the category dimension:
        num_indexed_dims = index.dim() if index.dtype == torch.bool else 1
        if current_dim <= inp.dim() - 1 < current_dim + num_indexed_dims:
            if num_indexed_dims != 1:
                return code
            category_index = index
        else:
            has_other_index = True
        current_dim += num_indexed_dims

    if category_index is None:
        return _maybe_wrap(inp, code)

    if has_other_index or category_index.dim() != 1:
        return code

    if category_index.dtype == torch.bool:
        category_index = category_index.nonzero().view(-1)

    categories = tuple(inp.categories[i] for i in category_index.tolist())
    return inp.__class__(code, categories)


@CategoricalTensor.implements(aten.cat.default)
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    code = torch.cat([_as_tensor(tensor) for tensor in tensors], dim=dim)
    if not all(isinstance(tensor, CategoricalTensor) for tensor in tensors):
        return code

    tensors = cast(Sequence[CategoricalTensor], tensors)
    dim %= tensors[0].dim()
    if dim != tensors[0].dim() - 1:
        # NOTE We trust the user to ensure category compatibility.
        return tensors[0].__class__(code, tensors[0].categories)

    categories = tuple(
        chain.from_iterable(tensor.categories for tensor in tensors)
    )
    return tensors[0].__class__(code, categories)


@CategoricalTensor.implements(aten.stack.default)
def _stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    code = torch.stack([_as_tensor(tensor) for tensor in tensors], dim=dim)
    if not all(isinstance(tensor, CategoricalTensor) for tensor in tensors):
        return code

    tensors = cast(Sequence[CategoricalTensor], tensors)
    dim %= tensors[0].dim() + 1
    if dim >= tensors[0].dim():
        return code

    # NOTE We trust the user to ensure category compatibility.
    return tensors[0].__class__(code, tensors[0].categories)


# Helpers #####################################################################


def _maybe_wrap(inp: CategoricalTensor, code: Tensor) -> Tensor:
    if code.dim() > 0 and code.size(-1) == inp.size(-1):
        return inp.__class__(code, inp.categories)
    return code


def _as_tensor(inp: Tensor) -> Tensor:
    if isinstance(inp, CategoricalTensor):
        return inp._code
    return inp

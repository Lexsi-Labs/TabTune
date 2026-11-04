"""``ArrayOps`` on native torch calls."""

from typing import Any

import torch


class TorchOps:
    """The shared array operations for the ``torch.Tensor`` backend."""

    float32, int8, int32, int64 = torch.float32, torch.int8, torch.int32, torch.int64

    def asarray(self, obj: Any, /, *, dtype: Any = None, device: Any = None) -> torch.Tensor:
        return torch.as_tensor(obj, dtype=dtype, device=device)

    def full(self, shape: tuple[int, ...], fill_value: Any, *, dtype: Any = None, device: Any = None) -> torch.Tensor:
        return torch.full(shape, fill_value, dtype=dtype, device=device)

    def arange(self, start: int, /, *, dtype: Any = None, device: Any = None) -> torch.Tensor:
        return torch.arange(start, dtype=dtype, device=device)

    def zeros_like(self, x: torch.Tensor, /) -> torch.Tensor:
        return torch.zeros_like(x)

    def ones_like(self, x: torch.Tensor, /) -> torch.Tensor:
        return torch.ones_like(x)

    def astype(self, x: torch.Tensor, dtype: Any, /) -> torch.Tensor:
        return x.to(dtype)

    def reshape(self, x: torch.Tensor, /, shape: tuple[int, ...]) -> torch.Tensor:
        return x.reshape(shape)

    def expand_dims(self, x: torch.Tensor, /, *, axis: int) -> torch.Tensor:
        return x.unsqueeze(axis)

    def squeeze(self, x: torch.Tensor, /, axis: int) -> torch.Tensor:
        return x.squeeze(axis)

    def permute_dims(self, x: torch.Tensor, /, axes: tuple[int, ...]) -> torch.Tensor:
        return x.permute(axes)

    def broadcast_to(self, x: torch.Tensor, /, shape: tuple[int, ...]) -> torch.Tensor:
        return x.expand(shape)

    def concat(self, arrays: list[torch.Tensor], /, *, axis: int = 0) -> torch.Tensor:
        return torch.cat(arrays, dim=axis)

    def repeat(self, x: torch.Tensor, repeats: int, /, *, axis: int | None = None) -> torch.Tensor:
        return torch.repeat_interleave(x, repeats, dim=axis)

    def take(self, x: torch.Tensor, indices: torch.Tensor, /, *, axis: int | None = None) -> torch.Tensor:
        if axis is None:
            return torch.take(x, indices)
        return torch.index_select(x, axis, indices)

    def take_along_axis(self, x: torch.Tensor, indices: torch.Tensor, /, *, axis: int = -1) -> torch.Tensor:
        return torch.gather(x, axis, indices.long())

    def where(self, condition: torch.Tensor, x1: Any, x2: Any, /) -> torch.Tensor:
        return torch.where(condition, x1, x2)

    def clip(self, x: torch.Tensor, /, min: Any = None, max: Any = None) -> torch.Tensor:
        return torch.clamp(x, min, max)

    def minimum(self, x1: torch.Tensor, x2: torch.Tensor, /) -> torch.Tensor:
        return torch.minimum(x1, x2)

    def sum(self, x: torch.Tensor, /, *, axis: int | None = None, keepdims: bool = False) -> torch.Tensor:
        return x.sum() if axis is None else x.sum(dim=axis, keepdim=keepdims)

    def any(self, x: torch.Tensor, /, *, axis: int | None = None, keepdims: bool = False) -> torch.Tensor:
        return x.any() if axis is None else x.any(dim=axis, keepdim=keepdims)

    def cumulative_sum(self, x: torch.Tensor, /, *, axis: int | None = None) -> torch.Tensor:
        return torch.cumsum(x, dim=0 if axis is None else axis)

    def argmax(self, x: torch.Tensor, /, *, axis: int | None = None, keepdims: bool = False) -> torch.Tensor:
        return torch.argmax(x, dim=axis, keepdim=keepdims)

    def argsort(self, x: torch.Tensor, /, *, axis: int = -1) -> torch.Tensor:
        return torch.argsort(x, dim=axis, stable=True)

    def isnan(self, x: torch.Tensor, /) -> torch.Tensor:
        return torch.isnan(x)

    def isfinite(self, x: torch.Tensor, /) -> torch.Tensor:
        return torch.isfinite(x)

    def sqrt(self, x: torch.Tensor, /) -> torch.Tensor:
        return torch.sqrt(x)

    def asinh(self, x: torch.Tensor, /) -> torch.Tensor:
        return torch.arcsinh(x)

    def sinh(self, x: torch.Tensor, /) -> torch.Tensor:
        return torch.sinh(x)

    def device(self, x: torch.Tensor, /) -> torch.device:
        return x.device

    def to_device(self, x: torch.Tensor, device: Any, /) -> torch.Tensor:
        return x.to(device)

    def row_sum(self, x: torch.Tensor, /) -> torch.Tensor:
        # One-bucket scatter-add: the summation order the checkpoint's reference numerics use.
        index = torch.zeros(x.shape, dtype=torch.long, device=x.device)
        return torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device).scatter_add_(-1, index, x)

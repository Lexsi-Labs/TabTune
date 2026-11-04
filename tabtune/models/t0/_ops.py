"""How framework-free code reaches arrays: one ``ArrayOps`` object per runtime.

``ArrayOps`` is a strict subset of the Python Array API standard (same names, same
signatures), plus a few marked extensions. Shared code gets the ops for its input with
``ops_for(x)`` and calls ``ops.take(...)`` exactly as it would call ``xp.take(...)`` on a
standard namespace. Each runtime registers its implementation when it is imported
(``t0.torch`` -> ``TorchOps``, ``t0.mlx`` -> ``MLXOps``); this module imports no framework.

Once a runtime's own ``__array_namespace__()`` passes the ops conformance test, ``ops_for``
can serve that namespace for it and its ops class goes, without touching the shared code.
"""

import importlib
from typing import Any, Protocol, TypeVar

__all__ = ["ArrayOps", "ArrayT", "ops_for", "register"]


# A runtime's array — a torch.Tensor or an mx.array — preserved from a function's input to
# its output. Bound to Any: the two frameworks' stubs disagree on indexing and arithmetic,
# so neither a union nor a protocol of both type-checks shared code.
ArrayT = TypeVar("ArrayT", bound=Any)


class ArrayOps(Protocol):
    """The array operations shared code may use. Standard names and signatures only.

    Extensions (not in the standard) are marked, each with its standard fallback.
    """

    float32: Any
    int8: Any
    int32: Any
    int64: Any

    def asarray(self, obj: Any, /, *, dtype: Any = None, device: Any = None) -> Any:
        raise NotImplementedError

    def full(self, shape: tuple[int, ...], fill_value: Any, *, dtype: Any = None, device: Any = None) -> Any:
        raise NotImplementedError

    def arange(self, start: int, /, *, dtype: Any = None, device: Any = None) -> Any:
        raise NotImplementedError

    def zeros_like(self, x: Any, /) -> Any:
        raise NotImplementedError

    def ones_like(self, x: Any, /) -> Any:
        raise NotImplementedError

    def astype(self, x: Any, dtype: Any, /) -> Any:
        raise NotImplementedError

    def reshape(self, x: Any, /, shape: tuple[int, ...]) -> Any:
        raise NotImplementedError

    def expand_dims(self, x: Any, /, *, axis: int) -> Any:
        raise NotImplementedError

    def squeeze(self, x: Any, /, axis: int) -> Any:
        raise NotImplementedError

    def permute_dims(self, x: Any, /, axes: tuple[int, ...]) -> Any:
        raise NotImplementedError

    def broadcast_to(self, x: Any, /, shape: tuple[int, ...]) -> Any:
        raise NotImplementedError

    def concat(self, arrays: list[Any], /, *, axis: int = 0) -> Any:
        raise NotImplementedError

    def repeat(self, x: Any, repeats: int, /, *, axis: int | None = None) -> Any:
        raise NotImplementedError

    def take(self, x: Any, indices: Any, /, *, axis: int | None = None) -> Any:
        raise NotImplementedError

    def take_along_axis(self, x: Any, indices: Any, /, *, axis: int = -1) -> Any:
        raise NotImplementedError

    def where(self, condition: Any, x1: Any, x2: Any, /) -> Any:
        raise NotImplementedError

    def clip(self, x: Any, /, min: Any = None, max: Any = None) -> Any:
        raise NotImplementedError

    def minimum(self, x1: Any, x2: Any, /) -> Any:
        raise NotImplementedError

    def sum(self, x: Any, /, *, axis: int | None = None, keepdims: bool = False) -> Any:
        raise NotImplementedError

    def any(self, x: Any, /, *, axis: int | None = None, keepdims: bool = False) -> Any:
        raise NotImplementedError

    def cumulative_sum(self, x: Any, /, *, axis: int | None = None) -> Any:
        raise NotImplementedError

    def argmax(self, x: Any, /, *, axis: int | None = None, keepdims: bool = False) -> Any:
        raise NotImplementedError

    def argsort(self, x: Any, /, *, axis: int = -1) -> Any:
        raise NotImplementedError

    def isnan(self, x: Any, /) -> Any:
        raise NotImplementedError

    def isfinite(self, x: Any, /) -> Any:
        raise NotImplementedError

    def sqrt(self, x: Any, /) -> Any:
        raise NotImplementedError

    def asinh(self, x: Any, /) -> Any:
        raise NotImplementedError

    def sinh(self, x: Any, /) -> Any:
        raise NotImplementedError

    def device(self, x: Any, /) -> Any:
        """Extension. Standard: the ``x.device`` attribute; ``None`` where arrays have no device."""

    def to_device(self, x: Any, device: Any, /) -> Any:
        """Extension. Standard: ``x.to_device(device)``; the identity where arrays have no device."""

    def row_sum(self, x: Any, /) -> Any:
        """Extension. Sum over the last axis, keeping it. Standard: ``sum(x, axis=-1, keepdims=True)``.

        Lets a runtime keep the summation order its reference numerics were recorded with.
        """


_registry: dict[type, ArrayOps] = {}
_by_type: dict[type, ArrayOps] = {}


def register(array_type: type, ops: ArrayOps) -> None:
    """Serve ``ops`` for arrays of ``array_type`` (and its subclasses)."""
    _registry[array_type] = ops
    _by_type.clear()


# The runtime package that registers the ops for each framework's arrays.
# Vendored in TabTune: runtime packages resolve against this package, and only
# the PyTorch runtime is vendored.
_RUNTIMES = {"torch": __name__.rpartition(".")[0] + ".torch"}


def ops_for(x: Any) -> ArrayOps:
    """The ops for ``x``'s runtime: an exact-type cache, then ``isinstance`` over registered runtimes.

    An array of a framework whose runtime is not imported yet imports it, so the core
    works on a framework's arrays without importing ``t0.torch`` / ``t0.mlx`` first.
    """
    ops = _by_type.get(type(x))
    if ops is None:
        ops = _find(x)
        if ops is None:
            runtime = _RUNTIMES.get(type(x).__module__.split(".")[0])
            if runtime is not None:
                importlib.import_module(runtime)
                ops = _find(x)
        if ops is None:
            raise TypeError(f"no t0 runtime handles {type(x).__name__}")
        _by_type[type(x)] = ops
    return ops


def _find(x: Any) -> ArrayOps | None:
    return next((candidate for array_type, candidate in _registry.items() if isinstance(x, array_type)), None)

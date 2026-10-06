"""Plain tensor containers that can be moved and serialized without rebuilding attention."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class KVPair:
    """Projected keys/values shaped (*batch, heads, context_length, head_width)."""

    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class BlockCache:
    """Layer K/V pairs in forward execution order; query rows never enter this state."""

    layers: tuple[KVPair, ...]


@dataclass(frozen=True)
class CacheResult:
    """Updated training activations plus the context collected while producing them."""

    rows: Tensor
    cache: BlockCache

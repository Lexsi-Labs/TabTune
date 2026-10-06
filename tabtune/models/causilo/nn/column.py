"""Summarize each feature group's training rows and broadcast that context to queries."""

import torch
from torch import Tensor, nn

from .cache import BlockCache, CacheResult, KVPair
from .layers.attention import AttentionLayer


class ColumnLayer(nn.Module):
    """Gather training context into fixed latent slots, then prepare its broadcast K/V."""

    def __init__(self, width: int, heads: int, expansion: int, latents: int) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.empty(latents, width))
        self.gather = AttentionLayer(width, heads, expansion)
        self.broadcast = AttentionLayer(width, heads, expansion, scale_query=False)

    def prepare(self, training: Tensor) -> KVPair:
        """Compress (*batch, training_rows, width) into column_latents context slots."""
        latents = self.latents.expand(*training.shape[:-2], *self.latents.shape)
        latents = self.gather(latents, training)
        return self.broadcast.prepare(latents)


class ColumnBlock(nn.Module):
    """Repeated column-context updates with independent leading feature-group axes."""

    def __init__(self, width: int, heads: int, expansion: int, latents: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(ColumnLayer(width, heads, expansion, latents) for _ in range(depth))

    def forward(self, rows: Tensor, *, context_rows: int) -> Tensor:
        """Update (*batch, training + query rows, width) using only the training prefix."""
        for layer in self.layers:
            keys = layer.prepare(rows[..., :context_rows, :])
            rows = layer.broadcast.query(rows, keys)
        return rows

    def build_context(self, training: Tensor) -> CacheResult:
        """Cache each layer's broadcast K/V while advancing training activations."""
        cache = []
        for layer in self.layers:
            keys = layer.prepare(training)
            training = layer.broadcast.query(training, keys)
            cache.append(keys)
        return CacheResult(training, BlockCache(tuple(cache)))

    def query(self, rows: Tensor, state: BlockCache) -> Tensor:
        """Update query activations using the same layer contexts as direct execution."""
        if len(state.layers) != len(self.layers):
            raise ValueError("Column cache depth does not match the model")
        for layer, keys in zip(self.layers, state.layers):
            rows = layer.broadcast.query(rows, keys)
        return rows

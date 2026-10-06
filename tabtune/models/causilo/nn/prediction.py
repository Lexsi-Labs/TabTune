"""Predict query rows by attending to target-enriched training-row representations."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cache import BlockCache
from .layers.attention import AttentionLayer


class PredictionBlock(nn.Module):
    """Context-only attention: training rows supply K/V at every layer."""

    def __init__(self, width: int, heads: int, expansion: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            AttentionLayer(width, heads, expansion, normalize_heads=True) for _ in range(depth)
        )

    def forward(self, rows: Tensor, *, context_rows: int) -> Tensor:
        """Map (*batch, training + queries, width) to (*batch, queries, width)."""
        for index, layer in enumerate(self.layers):
            keys = layer.prepare(rows[..., :context_rows, :])
            # Earlier layers must advance training rows to prepare the next K/V;
            # the final layer needs only query outputs.
            queries = rows[..., context_rows:, :] if index == len(self.layers) - 1 else rows
            rows = layer.query(queries, keys)
        return rows

    def build_context(self, training: Tensor) -> BlockCache:
        """Save K/V before each training update, omitting the unused final update."""
        cache = []
        for index, layer in enumerate(self.layers):
            keys = layer.prepare(training)
            cache.append(keys)
            if index + 1 < len(self.layers):
                training = layer.query(training, keys)
        return BlockCache(tuple(cache))

    def query(self, rows: Tensor, state: BlockCache) -> Tensor:
        """Run query rows through saved contexts without updating the training state."""
        if len(state.layers) != len(self.layers):
            raise ValueError("Prediction cache depth does not match the model")
        for layer, keys in zip(self.layers, state.layers):
            rows = layer.query(rows, keys)
        return rows


class Head(nn.Module):
    """Produce class logits or regression channels from each query representation."""

    def __init__(self, width: int, outputs: int) -> None:
        super().__init__()
        self.normalize = nn.RMSNorm(width, eps=1e-5)
        self.hidden = nn.Linear(width, width * 2)
        self.output = nn.Linear(width * 2, outputs)

    def forward(self, rows: Tensor) -> Tensor:
        """Return (*batch, queries, outputs), evaluating the output head in FP32."""
        with torch.autocast(device_type=rows.device.type, enabled=False):
            return self.output(F.gelu(self.hidden(self.normalize(rows.float()))))

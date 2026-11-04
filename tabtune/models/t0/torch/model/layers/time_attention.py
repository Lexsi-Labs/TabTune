# Copyright 2026 The Forecasting Company
# Causal time-axis self-attention with RoPE, following Datadog's Toto decoder
# self-attention (https://github.com/DataDog/toto).
# Copyright 2025 Datadog, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time-wise self-attention for flat variate tensors.

Each variate attends to its own temporal history (causal attention).
Uses RoPE for temporal position encoding. Cross-variate attention is
handled separately by ``group_attention.VariateSelfAttention``.
"""

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor

from ....torch.model.layers.norm import RMSNorm
from ....torch.model.layers.rope import TimeAwareRotaryEmbedding


class TimeSelfAttention(nn.Module):
    """Time-wise self-attention for flat variate tensors.

    Per-head RMSNorm is applied to queries and keys (not values) before
    RoPE.

    Input shape: (total_variates, seq_len, embed_dim)
    Output shape: (total_variates, seq_len, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        rotary_emb: TimeAwareRotaryEmbedding,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.rotary_emb = rotary_emb

        self.wQKV = nn.Linear(embed_dim, embed_dim * 3)
        self.wO = nn.Linear(embed_dim, embed_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: Float[Tensor, "variates patches embed"],
        attn_mask: Bool[Tensor, "variates 1 patches patches"],
    ) -> Float[Tensor, "variates patches embed"]:
        qkv = self.wQKV(x)
        qkv = rearrange(qkv, "v s (three h d) -> three v h s d", three=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv.unbind(dim=0)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary_emb.rotate_queries_and_keys(q, k, seq_dim=-2)

        dropout_p = self.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        attn_output = rearrange(attn_output, "v h s d -> v s (h d)")

        return self.wO(attn_output)


class TimeSelfAttentionBlock(nn.Module):
    """Time self-attention with pre-norm and residual connection."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        rotary_emb: TimeAwareRotaryEmbedding,
    ):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.attention = TimeSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            rotary_emb=rotary_emb,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "variates patches embed"],
        attn_mask: Bool[Tensor, "variates 1 patches patches"],
    ) -> Float[Tensor, "variates patches embed"]:
        normed = self.norm(x)
        attn_out = self.attention(normed, attn_mask)
        return x + self.dropout(attn_out)

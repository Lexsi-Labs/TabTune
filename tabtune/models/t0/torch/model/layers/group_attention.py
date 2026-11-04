# Copyright 2026 The Forecasting Company
# Built on Datadog's Toto attention scaffolding
# (https://github.com/DataDog/toto) and inspired by Chronos-2's variate-axis
# (group) self-attention pattern
# (https://github.com/amazon-science/chronos-forecasting): the variate-axis
# rearrange ("batch time d -> time batch d") and no-RoPE choice are both
# kept. Implementation otherwise differs (raw QKV +
# F.scaled_dot_product_attention instead of an MHA class; RMSNorm at the
# block level instead of the upstream LayerNorm inline).
# Copyright 2025 Datadog, Inc.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group-wise self-attention for flat variate tensors.

Variates attend to other variates within the same group (sample). No
RoPE since variates have no natural ordering within a group.
"""

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor

from ....torch.model.layers.norm import RMSNorm


class VariateSelfAttention(nn.Module):
    """Group-wise self-attention for flat variate tensors.

    Per-head RMSNorm is applied to queries and keys (not values) before
    SDPA.

    Input shape: ``(total_variates, seq_len, embed_dim)``.
    Output shape: same.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.wQKV = nn.Linear(embed_dim, embed_dim * 3)
        self.wO = nn.Linear(embed_dim, embed_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: Float[Tensor, "variates patches embed"],
        attn_mask: Bool[Tensor, "patches 1 variates variates"],
    ) -> Float[Tensor, "variates patches embed"]:
        # Flip time and variate axes for attention along variate dimension.
        # Pattern from Chronos-2: "batch time d -> time batch d".
        x_flipped = rearrange(x, "v s d -> s v d")

        qkv = self.wQKV(x_flipped)
        qkv = rearrange(qkv, "s v (three h d) -> three s h v d", three=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv.unbind(dim=0)

        q, k = self.q_norm(q), self.k_norm(k)

        dropout_p = self.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        attn_output = rearrange(attn_output, "s h v d -> s v (h d)")

        attn_output = self.wO(attn_output)

        return rearrange(attn_output, "s v d -> v s d")


class VariateSelfAttentionBlock(nn.Module):
    """Group self-attention with pre-norm and residual connection."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.attention = VariateSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "variates patches embed"],
        attn_mask: Bool[Tensor, "patches 1 variates variates"],
    ) -> Float[Tensor, "variates patches embed"]:
        normed = self.norm(x)
        attn_out = self.attention(normed, attn_mask)
        return x + self.dropout(attn_out)

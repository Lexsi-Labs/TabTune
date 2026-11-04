# Copyright 2026 The Forecasting Company
# Adapted from Datadog's Toto transformer scaffolding
# (https://github.com/DataDog/toto), with alternating variate-axis (group)
# attention layers borrowing Chronos-2's cross-variate pattern
# (https://github.com/amazon-science/chronos-forecasting).
# Copyright 2025 Datadog, Inc.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transformer stack with alternating time and group attention."""

from enum import Enum

import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ....config import XposScaleLayout
from ....mask import build_attention_masks
from ....torch.model.layers.feed_forward import SwiGLU
from ....torch.model.layers.group_attention import VariateSelfAttentionBlock
from ....torch.model.layers.norm import RMSNorm
from ....torch.model.layers.rope import TimeAwareRotaryEmbedding
from ....torch.model.layers.time_attention import TimeSelfAttentionBlock


class AttentionType(Enum):
    """Type of attention for a transformer layer."""

    TIME = "time"
    GROUP = "group"


class TransformerLayer(nn.Module):
    """Single transformer layer with either time or group attention.

    Pre-norm architecture (RMSNorm before each sublayer).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        attention_type: AttentionType,
        rotary_emb: TimeAwareRotaryEmbedding | None = None,
    ):
        super().__init__()
        self.attention_type = attention_type

        if attention_type == AttentionType.TIME:
            if rotary_emb is None:
                raise ValueError("rotary_emb is required for TIME attention")
            self.attention_block: TimeSelfAttentionBlock | VariateSelfAttentionBlock = TimeSelfAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                rotary_emb=rotary_emb,
            )
        else:  # GROUP
            self.attention_block = VariateSelfAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        self.norm = RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * mlp_hidden_dim),
            SwiGLU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Float[Tensor, "variates patches embed"],
        time_attn_mask: Bool[Tensor, "variates 1 patches patches"] | None = None,
        group_attn_mask: Bool[Tensor, "patches 1 variates variates"] | None = None,
    ) -> Float[Tensor, "variates patches embed"]:
        if self.attention_type == AttentionType.TIME:
            if time_attn_mask is None:
                raise ValueError("time_attn_mask is required for TIME attention")
            x = self.attention_block(x, attn_mask=time_attn_mask)
        else:  # GROUP
            if group_attn_mask is None:
                raise ValueError("group_attn_mask is required for GROUP attention")
            x = self.attention_block(x, attn_mask=group_attn_mask)

        x = x + self.mlp(self.norm(x))
        return x


class Transformer(nn.Module):
    """Stack of transformer layers with alternating time and group attention."""

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        group_every_n: int = 2,
        xpos_scale_layout: XposScaleLayout = "interleaved",
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        if group_every_n > 0 and num_layers % group_every_n != 0:
            raise ValueError(f"num_layers ({num_layers}) must be divisible by group_every_n ({group_every_n})")

        self.num_layers = num_layers
        self.group_every_n = group_every_n

        self.rotary_emb = TimeAwareRotaryEmbedding(
            dim=embed_dim // num_heads,
            use_xpos=True,
            cache_if_possible=True,
            seq_before_head_dim=False,
            xpos_scale_layout=xpos_scale_layout,
        )

        layer_types = self._get_layer_types(num_layers, group_every_n)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                    attention_type=layer_type,
                    rotary_emb=self.rotary_emb if layer_type == AttentionType.TIME else None,
                )
                for layer_type in layer_types
            ]
        )

        self.out_norm = RMSNorm(embed_dim)

    def _get_layer_types(
        self,
        num_layers: int,
        group_every_n: int,
    ) -> list[AttentionType]:
        if group_every_n <= 0:
            return [AttentionType.TIME] * num_layers

        block = [AttentionType.TIME] * (group_every_n - 1) + [AttentionType.GROUP]
        n_blocks = num_layers // group_every_n
        return block * n_blocks

    def forward(
        self,
        x: Float[Tensor, "variates patches embed"],
        patched_group_ids: Int[Tensor, "variates patches patch_size"],
        patched_variate_type: Int[Tensor, "variates patches patch_size"],
        patched_mask: Int[Tensor, "variates patches patch_size"],
    ) -> Float[Tensor, "variates patches embed"]:
        time_attn_mask, group_attn_mask = build_attention_masks(patched_group_ids, patched_variate_type, patched_mask)

        for layer in self.layers:
            x = layer(x, time_attn_mask=time_attn_mask, group_attn_mask=group_attn_mask)
        return self.out_norm(x)

    def get_layer_type_counts(self) -> dict[str, int]:
        time_count = sum(1 for layer in self.layers if layer.attention_type == AttentionType.TIME)
        group_count = len(self.layers) - time_count
        return {"time": time_count, "group": group_count}

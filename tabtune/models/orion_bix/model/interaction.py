"""
Row-wise interaction module for Tabular FM Experiments.

This module provides the RowInteraction class, which applies row-wise attention and aggregation.
"""
# --- model/interaction.py ---
# This file handles row-wise interaction using the new Encoder.
from __future__ import annotations

from typing import Optional
from collections import OrderedDict

import torch
from torch import nn, Tensor

from .encoders import Encoder # Updated to import the new Encoder
from .inference import InferenceManager # Assuming InferenceManager remains the same
from .inference_config import MgrConfig # Assuming MgrConfig remains the same


class RowInteraction(nn.Module):
    """
    Row-wise interaction module using a stack of linear attention blocks.

    Args:
        embed_dim (int): Embedding dimension.
        num_blocks (int): Number of attention blocks.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Feedforward network dimension.
        num_cls (int): Number of class tokens.
        rope_base (float): Base for rotary positional encoding.
        dropout (float): Dropout rate.
        activation (str or callable): Activation function.
        norm_first (bool): Whether to use pre-norm.
        feature_map (str): Feature map type ("elu", "identity", "hedgehog").
    """
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        attention_type: str = "linear",
        feature_map: str = "elu",  # "elu", "identity", "hedgehog"
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cls = num_cls
        self.norm_first = norm_first
        self.debug = debug
        self.attention_type = attention_type
        self.feature_map = feature_map

        self.tf_row = Encoder(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            attention_type=attention_type,
            feature_map=feature_map,
            use_rope=True,
            rope_base=rope_base,
            debug=debug,
        )

        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.out_ln = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()
        self.inference_mgr = InferenceManager(enc_name="tf_row", out_dim=embed_dim * self.num_cls, out_no_seq=True)

    def _aggregate_embeddings(self, embeddings: Tensor, key_mask: Optional[Tensor] = None) -> Tensor:
        """
        Aggregate class token embeddings after row-wise attention.

        Args:
            embeddings (Tensor): Input embeddings.
            key_mask (Optional[Tensor]): Optional mask for padding.
        Returns:
            Tensor: Aggregated class token representations.
        """
        outputs = self.tf_row(embeddings, key_padding_mask=key_mask)
        cls_outputs = outputs[..., : self.num_cls, :].clone()
        del outputs

        cls_outputs = self.out_ln(cls_outputs)

        return cls_outputs.flatten(-2)

    def _train_forward(self, embeddings: Tensor, d: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for training mode.

        Args:
            embeddings (Tensor): Input embeddings.
            d (Optional[Tensor]): Optional feature dimension tensor.
        Returns:
            Tensor: Row representations for training.
        """
        B, T, HC, E = embeddings.shape
        device = embeddings.device

        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)

        if d is None:
            key_mask = None
        else:
            d = d + self.num_cls
            indices = torch.arange(HC, device=device).view(1, 1, HC).expand(B, T, HC)
            key_mask = indices >= d.view(B, 1, 1)

        representations = self._aggregate_embeddings(embeddings, key_mask)
        return representations

    def _inference_forward(self, embeddings: Tensor, mgr_config: MgrConfig = None) -> Tensor:
        """
        Forward pass for inference mode.

        Args:
            embeddings (Tensor): Input embeddings.
            mgr_config (MgrConfig): Inference manager configuration.
        Returns:
            Tensor: Row representations for inference.
        """
        if mgr_config is None:
            mgr_config = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload=False,
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        self.inference_mgr.configure(**mgr_config)

        B, T = embeddings.shape[:2]
        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)
        representations = self.inference_mgr(
            self._aggregate_embeddings, inputs=OrderedDict([("embeddings", embeddings)])
        )

        return representations

    def forward(self, embeddings: Tensor, d: Optional[Tensor] = None, mgr_config: MgrConfig = None) -> Tensor:
        """
        Unified forward pass for both training and inference.

        Args:
            embeddings (Tensor): Input embeddings.
            d (Optional[Tensor]): Optional feature dimension tensor.
            mgr_config (MgrConfig): Inference manager configuration.
        Returns:
            Tensor: Row representations.
        """
        if self.training:
            representations = self._train_forward(embeddings, d)
        else:
            representations = self._inference_forward(embeddings, mgr_config)

        return representations
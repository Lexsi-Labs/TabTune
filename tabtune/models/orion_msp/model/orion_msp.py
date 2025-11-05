# orion_msp.py
from __future__ import annotations

from typing import Optional, List
import torch
from torch import nn, Tensor

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ICLearning
from .inference_config import InferenceConfig


class OrionMSP(nn.Module):
    """ORION-MSP with hierarchical + block-sparse RowInteraction and optional Perceiver memory."""

    def __init__(
        self,
        max_classes: int = 10,
        embed_dim: int = 128,
        # TFcol
        col_num_blocks: int = 3,
        col_nhead: int = 4,
        col_num_inds: int = 128,
        # Multi-Scale Sparse TFrow 
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        # Enhanced ICL
        icl_num_blocks: int = 12,
        icl_nhead: int = 4,
        # shared
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        # globals & sparsity / multi-scale
        row_num_global: int = 2,
        row_scales: tuple[int, ...] = (1, 4, 8),
        row_window: int = 4,
        row_num_random: int = 0,
        row_group_mode: str = "contiguous",  # or "pma"
        # Perceiver memory knobs (0 disables)
        perc_num_latents: int = 16,
        perc_layers: int = 2,
    ):
        super().__init__()
        self.max_classes = max_classes
        self.embed_dim = embed_dim

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            # reserve slots for [CLS + GLOBAL] at the front
            reserve_cls_tokens=row_num_cls + row_num_global,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            dim_feedforward=embed_dim * ff_factor,
            num_cls=row_num_cls,
            num_global=row_num_global,
            rope_base=row_rope_base,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            scales=tuple(row_scales),
            window=int(row_window),
            num_random=int(row_num_random),
            group_mode=row_group_mode.lower(),
        )

        icl_dim = embed_dim * row_num_cls
        self.icl_predictor = ICLearning(
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            perc_num_latents=perc_num_latents,
            perc_layers=perc_layers,
        )

    # ---- training / inference entrypoints ---- #

    def _train_forward(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None, embed_with_test: bool = False
    ) -> Tensor:
        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"

        # if d is degenerate (same value == H), ignore it
        if d is not None and (d.numel() == 1 or (d == H).all()):
            d = None

        emb = self.col_embedder(X, d=d, train_size=None if embed_with_test else train_size)
        R = self.row_interactor(emb, d=d)
        return self.icl_predictor(R, y_train=y_train)

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig | None = None,
    ) -> Tensor:
        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

        if inference_config is None:
            inference_config = InferenceConfig()

        emb = self.col_embedder(
            X,
            train_size=None if embed_with_test else train_size,
            feature_shuffles=feature_shuffles,
            mgr_config=inference_config.COL_CONFIG,
        )
        R = self.row_interactor(emb, mgr_config=inference_config.ROW_CONFIG)
        return self.icl_predictor(
            R,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            mgr_config=inference_config.ICL_CONFIG,
        )

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig | None = None,
    ) -> Tensor:
        if self.training:
            return self._train_forward(X, y_train, d=d, embed_with_test=embed_with_test)
        else:
            return self._inference_forward(
                X,
                y_train,
                feature_shuffles=feature_shuffles,
                embed_with_test=embed_with_test,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                inference_config=inference_config,
            )

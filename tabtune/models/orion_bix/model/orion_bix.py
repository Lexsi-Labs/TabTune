"""
Main model definition for Tabular BiAxial In-Context Learning (OrionBix).

This module defines the OrionBix class, which orchestrates column embedding, row interaction, and final prediction using linear attention mechanisms at each stage. The model is highly configurable for different attention feature maps, embedding dimensions, and architectural hyperparameters.
"""
# --- model/orion_bix.py ---
# This is the main model that orchestrates all the components.
from __future__ import annotations


from typing import Optional, List, Literal
from torch import nn, Tensor

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ICLearning
from .inference_config import InferenceConfig
from .layers import BiAxialAttention, LinearAttentionBlock



class OrionBix(nn.Module):
    """
    Main model for Tabular In-Context Learning (OrionBix).

    This model orchestrates column embedding, row interaction, and final prediction
    using linear attention mechanisms at each stage. It is highly configurable
    for different attention feature maps, embedding dimensions, and architectural
    hyperparameters.

    Args:
        max_classes (int): Maximum number of output classes.
        embed_dim (int): Embedding dimension for all layers.
        col_num_blocks (int): Number of attention blocks in the column embedder.
        col_nhead (int): Number of attention heads for column attention.
        row_num_blocks (int): Number of attention blocks in the row interactor.
        row_nhead (int): Number of attention heads for row attention.
        row_num_cls (int): Number of class tokens for row interaction.
        row_rope_base (float): Base for rotary positional encoding in row attention.
        icl_num_blocks (int): Number of attention blocks in the ICL predictor.
        icl_nhead (int): Number of attention heads for ICL attention.
        ff_factor (int): Feedforward network expansion factor.
        dropout (float): Dropout rate.
        activation (str or callable): Activation function.
        norm_first (bool): Whether to use pre-norm in attention blocks.
        col_feature_map (str): Feature map for column attention ("elu", "identity", "hedgehog").
        row_feature_map (str): Feature map for row attention.
        icl_feature_map (str): Feature map for ICL attention.
    """
    def __init__(
        self,
        max_classes: int = 10,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 4,
        col_num_inds: int = 128, #check
        col_attention_type: Literal["linear", "standard"] = "linear",
        col_feature_map: str = "elu",
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        row_attention_type: Literal["linear", "standard", "bi_axial"] = "bi_axial",
        row_feature_map: str = "elu",
        icl_num_blocks: int = 12,
        icl_nhead: int = 4,
        icl_attention_type: Literal["linear", "standard"] = "standard",
        icl_feature_map: str = "elu",
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        debug: bool = False,
        **kwargs  # This catches any extra parameters
    ):
        """
        Initialize the OrionBix model and its submodules.
        """
        super().__init__()
        self.max_classes = max_classes
        self.embed_dim = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead = col_nhead
        self.col_num_inds = col_num_inds
        self.row_num_blocks = row_num_blocks
        self.row_nhead = row_nhead
        self.row_num_cls = row_num_cls
        self.row_rope_base = row_rope_base
        self.icl_num_blocks = icl_num_blocks
        self.icl_nhead = icl_nhead
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first

        # Store attention configurations
        self.col_attention_type = col_attention_type
        self.col_feature_map = col_feature_map
        self.row_attention_type = row_attention_type
        self.row_feature_map = row_feature_map
        self.icl_attention_type = icl_attention_type
        self.icl_feature_map = icl_feature_map

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            attention_type=col_attention_type,
            feature_map=col_feature_map,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            reserve_cls_tokens=row_num_cls,
            debug=debug,
        )
        if self.row_attention_type == "bi_axial":
            self.row_interactor = None
            self.bi_axial_attention = BiAxialAttention(
                embed_dim=embed_dim,
                num_blocks=row_num_blocks,
                nhead=row_nhead,
                dim_feedforward=embed_dim * ff_factor,
                num_cls=row_num_cls,
                rope_base=row_rope_base,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            )
        else:
            self.row_interactor = RowInteraction(
                embed_dim=embed_dim,
                num_blocks=row_num_blocks,
                nhead=row_nhead,
                num_cls=row_num_cls,
                rope_base=row_rope_base,
                attention_type=row_attention_type,
                feature_map=row_feature_map,
                dim_feedforward=embed_dim * ff_factor,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                debug=debug,
            )
            self.bi_axial_attention = None

        icl_dim = embed_dim * row_num_cls
        self.icl_predictor = ICLearning(
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            attention_type=icl_attention_type,
            feature_map=icl_feature_map,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

    def _train_forward(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None, embed_with_test: bool = False
    ) -> Tensor:
        """
        Forward pass for training mode.

        Args:
            X (Tensor): Input tensor of shape (B, T, H).
            y_train (Tensor): Training labels.
            d (Optional[Tensor]): Optional feature dimension tensor.
            embed_with_test (bool): Whether to embed with test samples.
        Returns:
            Tensor: Model output.
        """
        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"

        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None
        
        col_embeddings = self.col_embedder(X, d=d, train_size=None if embed_with_test else train_size)
        if self.row_attention_type == "bi_axial":
            representations = self.bi_axial_attention(col_embeddings, d=d)
        else:
            representations = self.row_interactor(col_embeddings, d=d)

        out = self.icl_predictor(representations, y_train=y_train)

        return out

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
    ) -> Tensor:
        """
        Forward pass for inference mode.

        Args:
            X (Tensor): Input tensor of shape (B, T, H).
            y_train (Tensor): Training labels.
            feature_shuffles (Optional[List[List[int]]]): Feature shuffle patterns.
            embed_with_test (bool): Whether to embed with test samples.
            return_logits (bool): Whether to return logits.
            softmax_temperature (float): Softmax temperature for output.
            inference_config (InferenceConfig): Inference configuration.
        Returns:
            Tensor: Model output.
        """
        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

        if inference_config is None:
            inference_config = InferenceConfig()
        
        col_embeddings = self.col_embedder(
            X,
            train_size=None if embed_with_test else train_size,
            feature_shuffles=feature_shuffles,
            mgr_config=inference_config.COL_CONFIG,
        )

        if self.row_attention_type == "bi_axial":
            representations = self.bi_axial_attention(
                col_embeddings,
                mgr_config=inference_config.ROW_CONFIG,
            )
        else:
            representations = self.row_interactor(
                col_embeddings,
                mgr_config=inference_config.ROW_CONFIG,
            )

        out = self.icl_predictor(
            representations,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            mgr_config=inference_config.ICL_CONFIG,
        )

        return out

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
    ) -> Tensor:
        """
        Unified forward pass for both training and inference.

        Args:
            X (Tensor): Input tensor of shape (B, T, H).
            y_train (Tensor): Training labels.
            d (Optional[Tensor]): Optional feature dimension tensor.
            feature_shuffles (Optional[List[List[int]]]): Feature shuffle patterns.
            embed_with_test (bool): Whether to embed with test samples.
            return_logits (bool): Whether to return logits.
            softmax_temperature (float): Softmax temperature for output.
            inference_config (InferenceConfig): Inference configuration.
        Returns:
            Tensor: Model output.
        """
        if self.training:
            out = self._train_forward(X, y_train, d=d, embed_with_test=embed_with_test)
        else:
            out = self._inference_forward(
                X,
                y_train,
                feature_shuffles=feature_shuffles,
                embed_with_test=embed_with_test,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                inference_config=inference_config,
            )
        return out
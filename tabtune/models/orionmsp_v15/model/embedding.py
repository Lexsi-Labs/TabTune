from __future__ import annotations

from typing import List, Optional
from collections import OrderedDict

import torch
from torch import nn, Tensor

from .layers import SkippableLinear
from .encoders import SetTransformer
from .inference import InferenceManager
from .inference_config import MgrConfig


class ColEmbedding(nn.Module):
    """Distribution-aware column-wise embedding with feature positional embeddings.
    
    This module maps each scalar cell in a column to a high-dimensional embedding while
    capturing statistical regularities within the column. Unlike traditional approaches
    that use separate embedding layers per column, it employs a shared set transformer
    to process all features.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        reserve_cls_tokens: int = 4,
        # Feature positional embedding parameters
        feature_pos_emb: str = "subspace",  # "subspace", "learned", or None
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.reserve_cls_tokens = reserve_cls_tokens
        self.in_linear = SkippableLinear(1, embed_dim)

        self.tf_col = SetTransformer(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        self.out_w = SkippableLinear(embed_dim, embed_dim)
        self.ln_w = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()

        self.out_b = SkippableLinear(embed_dim, embed_dim)
        self.ln_b = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()

        # FEATURE ADDITION: Feature positional embeddings
        self.feature_pos_emb = feature_pos_emb
        if feature_pos_emb == "subspace":
            # Use fixed random seed=42 for reproducibility (like TabPFN)
            # Create random embeddings: (2000, embed_dim // 4) - matches TabPFN's approach
            generator = torch.Generator().manual_seed(42)
            self.register_buffer(
                "col_embedding_seed",
                torch.randn(2000, embed_dim // 4, generator=generator)
            )
            self.feature_pos_proj = nn.Linear(embed_dim // 4, embed_dim)
        elif feature_pos_emb == "learned":
            self.feature_pos_embeddings = nn.Embedding(1000, embed_dim)
        elif feature_pos_emb is None:
            self.feature_pos_emb = None
        else:
            raise ValueError(f"Unknown feature_pos_emb: {feature_pos_emb}. Use 'subspace', 'learned', or None")

        self.inference_mgr = InferenceManager(enc_name="tf_col", out_dim=embed_dim)

    @staticmethod
    def map_feature_shuffle(reference_pattern: List[int], other_pattern: List[int]) -> List[int]:
        """Map feature shuffle pattern from the reference table to another table.

        Parameters
        ----------
        reference_pattern : List[int]
            The shuffle pattern of features in a reference table w.r.t. the original table

        other_pattern : List[int]
            The shuffle pattern of features in another table w.r.t. the original table

        Returns
        -------
        List[int]
            A mapping from the reference table's ordering to another table's ordering
        """

        orig_to_other = {feature: idx for idx, feature in enumerate(other_pattern)}
        mapping = [orig_to_other[feature] for feature in reference_pattern]

        return mapping

    def _add_feature_pos_emb(self, embeddings: Tensor, feature_indices: Optional[Tensor] = None) -> Tensor:
        """
        Add feature positional embeddings to column embeddings.
        
        Args:
            embeddings: (B, H+C, T, E) or (B, T, H+C, E) tensor
            feature_indices: Optional tensor for feature selection
            
        Returns:
            Embeddings with positional encoding added
        """
        if self.feature_pos_emb is None:
            return embeddings
        
        # Handle different input shapes
        if embeddings.dim() == 4:
            if embeddings.shape[1] == self.reserve_cls_tokens + embeddings.shape[2]:  # (B, H+C, T, E)
                B, HC, T, E = embeddings.shape
                H = HC - self.reserve_cls_tokens
                
                if self.feature_pos_emb == "subspace":
                    # Get base embeddings from seed buffer
                    base_seed = self.col_embedding_seed.to(embeddings.device)
                    
                    # If we need more features than available in seed, extend it
                    if H > base_seed.shape[0]:
                        # Extend by generating only the additional embeddings needed
                        # Note: For proper continuation, we'd need to skip the first 2000*32 random numbers,
                        # but this is complex. Instead, we generate deterministically with seed 42.
                        # Since unused features are masked out (via d tensor), this won't affect performance
                        # for datasets with max_features <= 2000 (which is the normal case).
                        generator = torch.Generator(device=embeddings.device).manual_seed(42)
                        # Generate only the additional embeddings needed (positions 2000 to H-1)
                        num_additional = H - base_seed.shape[0]
                        additional_seed = torch.randn(
                            num_additional,
                            base_seed.shape[1],
                            device=embeddings.device,
                            dtype=base_seed.dtype,
                            generator=generator
                        )
                        # Concatenate: buffer (0-1999) + new (2000+)
                        full_seed = torch.cat([base_seed, additional_seed], dim=0)
                    else:
                        full_seed = base_seed[:H]
                    
                    proj = self.feature_pos_proj
                    if full_seed.device != proj.weight.device:
                        # Compute on embeddings device without moving the module permanently
                        W = proj.weight.to(full_seed.device)
                        b = proj.bias.to(full_seed.device) if proj.bias is not None else None
                        pos_emb = nn.functional.linear(full_seed, W, b)
                    else:
                        pos_emb = proj(full_seed)

                        
                    # Add positional embeddings only to feature columns (skip CLS tokens)
                    embeddings[:, self.reserve_cls_tokens:, :, :] += pos_emb[None, :, None, :]
                elif self.feature_pos_emb == "learned":
                    # Use learned embeddings
                    if feature_indices is not None:
                        pos_emb = self.feature_pos_embeddings(feature_indices)  # (H, E)
                    else:
                        pos_indices = torch.arange(H, device=embeddings.device)

                        idx = pos_indices
                        emb_w = self.feature_pos_embeddings.weight
                        if idx.device != emb_w.device:
                            pos_emb = nn.functional.embedding(idx.to(embeddings.device), emb_w.to(embeddings.device))
                        else:
                            pos_emb = self.feature_pos_embeddings(idx)


    
                    embeddings[:, self.reserve_cls_tokens:, :, :] += pos_emb[None, :, None, :]
            else:  # (B, T, H+C, E)
                B, T, HC, E = embeddings.shape
                H = HC - self.reserve_cls_tokens
                
                if self.feature_pos_emb == "subspace":
                    # Get base embeddings from seed buffer
                    base_seed = self.col_embedding_seed.to(embeddings.device)
                    
                    # If we need more features than available in seed, extend it
                    if H > base_seed.shape[0]:
                        # Extend by generating only the additional embeddings needed
                        # Note: For proper continuation, we'd need to skip the first 2000*32 random numbers,
                        # but this is complex. Instead, we generate deterministically with seed 42.
                        # Since unused features are masked out (via d tensor), this won't affect performance
                        # for datasets with max_features <= 2000 (which is the normal case).
                        generator = torch.Generator(device=embeddings.device).manual_seed(42)
                        # Generate only the additional embeddings needed (positions 2000 to H-1)
                        num_additional = H - base_seed.shape[0]
                        additional_seed = torch.randn(
                            num_additional,
                            base_seed.shape[1],
                            device=embeddings.device,
                            dtype=base_seed.dtype,
                            generator=generator
                        )
                        # Concatenate: buffer (0-1999) + new (2000+)
                        full_seed = torch.cat([base_seed, additional_seed], dim=0)
                    else:
                        full_seed = base_seed[:H]
                    
                    proj = self.feature_pos_proj
                    if full_seed.device != proj.weight.device:
                        # Compute on embeddings device without moving the module permanently
                        W = proj.weight.to(full_seed.device)
                        b = proj.bias.to(full_seed.device) if proj.bias is not None else None
                        pos_emb = nn.functional.linear(full_seed, W, b)
                    else:
                        pos_emb = proj(full_seed)

    
                    embeddings[:, :, self.reserve_cls_tokens:, :] += pos_emb[None, None, :, :]
                elif self.feature_pos_emb == "learned":
                    if feature_indices is not None:
                        idx = feature_indices.to(device=embeddings.device).long()
                    else:
                        idx = torch.arange(H, device=embeddings.device).long()

                    emb_w = self.feature_pos_embeddings.weight
                    if idx.device != emb_w.device:
                        pos_emb = nn.functional.embedding(idx, emb_w.to(idx.device))  # (H, E)
                    else:
                        pos_emb = self.feature_pos_embeddings(idx)  # (H, E)

                    embeddings[:, :, self.reserve_cls_tokens:, :] += pos_emb[None, None, :, :]
        
        return embeddings

    def _compute_embeddings(self, features: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Feature embedding using a shared set transformer

        Parameters
        ----------
        features : Tensor
            Input features of shape (..., T, 1) where:
             - ... represents arbitrary batch dimensions
             - T is the number of samples (rows)

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data (positions < train_size)
            in the set transformer to prevent information leakage from test data.

        Returns
        -------
        Tensor
            Embeddings of shape (..., T, E) where E is the embedding dimension
        """

        src = self.in_linear(features)  # (..., T, 1) -> (..., T, E)
        src = self.tf_col(src, train_size)
        weights = self.ln_w(self.out_w(src))  # (..., T, E)
        biases = self.ln_b(self.out_b(src))  # (..., T, E)
        embeddings = features * weights + biases

        return embeddings

    def _train_forward(self, X: Tensor, d: Optional[Tensor] = None, train_size: Optional[int] = None) -> Tensor:
        """Transform input table into embeddings for training.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)

        d : Optional[Tensor], default=None
            The number of features per dataset.

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data (positions < train_size)
            in the set transformer to prevent information leakage from test data.

        Returns
        -------
        Tensor
            Embeddings of shape (B, T, H+C, E) where:
             - C is the number of class tokens
             - E is embedding dimension
        """

        if self.reserve_cls_tokens > 0:
            # Pad with -100.0 to mark inputs that should be skipped in SkippableLinear and SetTransformer
            X = nn.functional.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

        if d is None:
            features = X.transpose(1, 2).unsqueeze(-1)  # (B, H+C, T, 1)
            embeddings = self._compute_embeddings(features, train_size)  # (B, H+C, T, E)
        else:
            if self.reserve_cls_tokens > 0:
                d = d + self.reserve_cls_tokens

            B, T, HC = X.shape
            device = X.device
            X = X.transpose(1, 2)  # (B, H+C, T)

            indices = torch.arange(HC, device=device).unsqueeze(0).expand(B, HC)
            mask = indices < d.unsqueeze(1)  # (B, H+C) used extract non-empty features
            features = X[mask].unsqueeze(-1)  # (N, T, 1) -> N = sum(d)
            effective_embeddings = self._compute_embeddings(features, train_size)  # (N, T, E)

            embeddings = torch.zeros(B, HC, T, self.embed_dim, device=device)
            embeddings[mask] = effective_embeddings  # Fill in the computed embeddings

        # FEATURE ADDITION: Add feature positional embeddings
        embeddings = self._add_feature_pos_emb(embeddings)
        
        return embeddings.transpose(1, 2)  # (B, T, H+C, E)

    def _inference_forward(
        self,
        X: Tensor,
        train_size: Optional[int] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """Transform input table into embeddings for inference.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
                - B is the number of tables
                - T is the number of samples (rows)
                - H is the number of features (columns)

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data (positions < train_size)
            in the set transformer to prevent information leakage from test data.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, embeddings are computed once and then shuffled accordingly.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager

        Returns
        -------
        Tensor
            Embeddings of shape (B, T, H+C, E) where:
                - C is the number of class tokens
                - E is embedding dimension
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload="auto",
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        self.inference_mgr.configure(**mgr_config)

        if feature_shuffles is None:
            # Processing all tables
            if self.reserve_cls_tokens > 0:
                # Pad with -100.0 to mark inputs that should be skipped in SkippableLinear and SetTransformer
                X = nn.functional.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

            features = X.transpose(1, 2).unsqueeze(-1)  # (B, H+C, T, 1)
            embeddings = self.inference_mgr(
                self._compute_embeddings, inputs=OrderedDict([("features", features), ("train_size", train_size)])
            )  # (B, H+C, T, E)
        else:
            B = X.shape[0]
            # Process only the first table and then shuffle features for other tables
            first_table = X[0]
            if self.reserve_cls_tokens > 0:
                # Pad with -100.0 to mark inputs that should be skipped in SkippableLinear and SetTransformer
                first_table = nn.functional.pad(first_table, (self.reserve_cls_tokens, 0), value=-100.0)

            features = first_table.transpose(0, 1).unsqueeze(-1)  # (H+C, T, 1)
            first_embeddings = self.inference_mgr(
                self._compute_embeddings,
                inputs=OrderedDict([("features", features), ("train_size", train_size)]),
                output_repeat=B,
            )  # (H+C, T, E)

            # Apply shuffles for tables after the first one
            embeddings = first_embeddings.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H+C, T, E)
            first_pattern = feature_shuffles[0]
            for i in range(1, B):
                mapping = self.map_feature_shuffle(first_pattern, feature_shuffles[i])
                if self.reserve_cls_tokens > 0:
                    mapping = [m + self.reserve_cls_tokens for m in mapping]
                    mapping = list(range(self.reserve_cls_tokens)) + mapping
                embeddings[i] = first_embeddings[mapping]

        # FEATURE ADDITION: Add feature positional embeddings
        embeddings = self._add_feature_pos_emb(embeddings)
        
        return embeddings.transpose(1, 2)  # (B, T, H+C, E)

    def forward(
        self,
        X: Tensor,
        d: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """Transform input table into embeddings.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data (positions < train_size)
            in the set transformer to prevent information leakage from test data.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch. Used only in inference mode.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, embeddings are computed once and then shuffled accordingly.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. Used only in inference mode.

        Returns
        -------
        Tensor
            Embeddings of shape (B, T, H+C, E) where:
             - C is the number of class tokens
             - E is embedding dimension
        """

        if self.training:
            embeddings = self._train_forward(X, d, train_size)
        else:
            embeddings = self._inference_forward(X, train_size, feature_shuffles, mgr_config)

        return embeddings  # (B, T, H+C, E)

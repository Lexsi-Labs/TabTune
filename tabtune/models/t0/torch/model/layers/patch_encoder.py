# Copyright 2026 The Forecasting Company
# The masked patch input embedding (the [values ‖ time_index ‖ validity]
# triple projection) is derived from Chronos-2
# (https://github.com/amazon-science/chronos-forecasting), adapted for a
# decoder model. Notable differences from the upstream:
#   - causal: t0-alpha attends to the past only; Chronos-2's encoder is bidirectional
#   - within-patch indexing: each patch carries a normalised time index
#   - learned type embeddings (target / historical / future) added on top
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch encoder for the t0-alpha decoder model.

Each patch is encoded as ``[values ‖ time_index ‖ validity_mask]`` (a triple
projection inspired by Chronos-2), then summed with a learned type embedding
(target / historical / future).
"""

from collections.abc import Callable

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from ....data import MaskType
from ....mask import reduce_patch_metadata
from ....torch.model.layers.mlp import ResidualBlock


class PatchEncoder(nn.Module):
    """Patch encoder with [values ‖ time_index ‖ validity_mask] triple projection.

    Input: per-patch tensors ``(total_variates, n_patches, patch_size)``
    Output: ``(total_variates, n_patches, embed_dim)``
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 3x patch_size: values, normalised time index, validity bit.
        self.projection = ResidualBlock(
            input_size=patch_size * 3,
            hidden_size=embed_dim,
            output_size=embed_dim,
            activation=activation,
        )

        # 0=target, 1=historical, 2=future
        self.type_embeddings = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)

        # Cached normalised time index, populated lazily on first forward.
        self.register_buffer("_time_index", torch.empty(0), persistent=False)

    def forward(
        self,
        values: Float[Tensor, "variates patches patch_size"],
        mask: Int[Tensor, "variates patches patch_size"],
        variate_type: Int[Tensor, "variates patches patch_size"],
    ) -> Float[Tensor, "variates patches embed"]:
        total_variates, n_patches, _ = values.shape

        if self._time_index.shape[0] != self.patch_size or self._time_index.dtype != values.dtype:
            self._time_index = torch.arange(self.patch_size, device=values.device, dtype=values.dtype) / self.patch_size

        validity = (mask == MaskType.VALID).to(dtype=values.dtype)
        t = self._time_index.unsqueeze(0).unsqueeze(0).expand(total_variates, n_patches, -1)

        embedded = self.projection(torch.cat([values, t, validity], dim=-1))
        # Left padding must not retype a covariate row as a target. Fully padded
        # patches reduce to the -1 sentinel, clamped to 0 here; masked downstream.
        patch_type = reduce_patch_metadata(variate_type, mask).clamp(min=0)
        return embedded + self.type_embeddings(patch_type)

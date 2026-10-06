"""
This module implements the fusion of variate label embeddings with input embeddings in the TOTO model.
It prepends trainable variate label embeddings, allowing the model to distinguish between target and exogenous input features.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float


class Fusion(torch.nn.Module):
    """
    Prepends variate label embeddings to the input embeddings along the sequence dimension.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        embeddings: Float[torch.Tensor, "batch variate seq_len embed_dim"],
        variate_label_embeds: Optional[Float[torch.Tensor, "batch variate 1 embed_dim"]] = None,
    ) -> Float[torch.Tensor, "batch variate new_seq_len embed_dim"]:

        # Nothing to fuse
        if variate_label_embeds is None:
            return embeddings

        processed_embeddings = F.normalize(variate_label_embeds, p=2, dim=-1)

        # Prepend along sequence dimension
        return torch.cat(
            [processed_embeddings.to(dtype=embeddings.dtype, device=embeddings.device, non_blocking=True), embeddings],
            dim=2,
        )

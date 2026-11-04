# Copyright 2026 The Forecasting Company
# RMSNorm adapted from Datadog's Toto (https://github.com/DataDog/toto).
# Copyright 2025 Datadog, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``RMSNorm`` for the time / group attention blocks."""

import torch
from jaxtyping import Float


class RMSNorm(torch.nn.Module):
    """RMS Norm (https://arxiv.org/abs/1910.07467)."""

    def __init__(self, dim: int, include_weight: bool = True, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        if include_weight:
            self.scale: torch.nn.Parameter | None = torch.nn.Parameter(torch.ones(dim))
        else:
            self.scale = None

    def forward(self, x: Float[torch.Tensor, "*batch dim"]) -> Float[torch.Tensor, "*batch dim"]:
        x_normed = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x_normed if self.scale is None else x_normed * self.scale

# Copyright 2026 The Forecasting Company
# SwiGLU is a direct port of Datadog's Toto feed-forward activation
# (https://github.com/DataDog/toto/blob/main/toto/model/feed_forward.py).
# Copyright 2025 Datadog, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from jaxtyping import Float


class SwiGLU(torch.nn.Module):
    """SwiGLU activation (https://arxiv.org/abs/2002.05202); input must be 2x the output width."""

    def forward(self, x: Float[torch.Tensor, "*batch double_dim"]) -> Float[torch.Tensor, "*batch dim"]:
        # Note this ordering is unusual, but is done so to match xFormers
        gate, x = x.chunk(2, dim=-1)
        return F.silu(gate) * x

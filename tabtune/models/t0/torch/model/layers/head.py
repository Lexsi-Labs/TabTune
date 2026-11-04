"""Quantile head: the model's only output-shaping layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class QuantileHead(nn.Module):
    """Direct-quantile head with monotonicity enforced via cumsum-of-softplus.

    Enforces monotonicity, e.g. ``q_0.1 ≤ q_0.5 ≤ q_0.9`` using the
    cumsum-of-softplus trick:

        q_0 = raw_0 (unconstrained)
        q_i = q_0 + Σ_{j=1}^i softplus(raw_j) for i > 0

    The forward pass maps raw decoder outputs to quantile values; the last
    dimension indexes ``quantile_levels``.

    Args:
        quantile_levels: List of quantile levels (e.g. ``[0.1, 0.5, 0.9]``).
        enforce_monotonicity: Whether to apply the cumsum trick. Default True.
    """

    def __init__(
        self,
        quantile_levels: list[float],
        enforce_monotonicity: bool = True,
    ):
        super().__init__()
        self.enforce_monotonicity = enforce_monotonicity
        self.register_buffer("quantile_levels", torch.tensor(sorted(quantile_levels), dtype=torch.float32))
        self.quantile_levels: torch.Tensor

    @property
    def n_quantiles(self) -> int:
        return int(self.quantile_levels.numel())

    def forward(self, x: Float[torch.Tensor, "*batch quantiles"]) -> Float[torch.Tensor, "*batch quantiles"]:
        if not self.enforce_monotonicity:
            return x
        first_quantile = x[..., :1]
        if x.shape[-1] == 1:
            return first_quantile
        increments = F.softplus(x[..., 1:])
        remaining = first_quantile + torch.cumsum(increments, dim=-1)
        return torch.cat([first_quantile, remaining], dim=-1)

"""Embed grouped scalar features and known targets into the model's hidden space."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FeatureEmbedding(nn.Module):
    """Combine learned sinusoidal features with explicit missing-value embeddings."""

    def __init__(self, width: int, group_size: int, frequencies: int) -> None:
        super().__init__()
        self.group_size = group_size
        self.frequencies = nn.Parameter(torch.empty(group_size, frequencies))
        self.missing = nn.Parameter(torch.empty(group_size, width))
        self.projection = nn.Linear(2 * frequencies, width)

    def forward(self, table: Tensor) -> Tensor:
        """Map (..., rows, features) to (..., rows, groups, width).

        Pad the feature axis to a multiple of group_size, then sum per-feature
        signals within each group. Padding is a numeric zero, not a missing
        value. Scaling by sqrt(group_size) is the fixed embedding convention.
        """
        padding = (-table.shape[-1]) % self.group_size
        grouped = F.pad(table, (0, padding)).unflatten(-1, (-1, self.group_size))
        missing = grouped.isnan()
        phase = torch.nan_to_num(grouped, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)
        phase = phase * (2 * math.pi) * self.frequencies
        waves = torch.cat((phase.sin(), phase.cos()), dim=-1).masked_fill(missing.unsqueeze(-1), 0)
        del phase, grouped
        magnitude = math.sqrt(self.group_size)
        # The projection is shared by the features in a group. Sum its inputs
        # first so we never materialize (..., groups, group_size, width).
        # Each original projection contributed its bias, including missing and
        # padded features, so retain group_size copies of that bias.
        signal = F.linear(
            waves.sum(-2), self.projection.weight, self.projection.bias * self.group_size
        )
        del waves
        # Both buffers are newly allocated here; reuse them for scaling and
        # combination instead of keeping extra full feature grids alive.
        signal.div_(magnitude)
        absence = missing.to(signal.dtype) @ self.missing.to(signal.dtype)
        absence.div_(magnitude)
        return signal.add_(absence)


class TargetEmbedding(nn.Module):
    """Project one-hot classes or a standardized regression scalar to hidden width."""

    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.classes = classes
        self.projection = nn.Linear(classes if classes else 1, width)

    def forward(self, targets: Tensor) -> Tensor:
        """Map (batch, training_rows) targets to (batch, training_rows, width)."""
        values = F.one_hot(targets.long(), self.classes).float() if self.classes else targets.unsqueeze(-1)
        return self.projection(values)

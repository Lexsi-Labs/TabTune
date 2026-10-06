"""Gated feedforward transformation shared by attention and row blocks."""

from torch import Tensor, nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """SwiGLU-style expansion and contraction along the final hidden-width axis."""

    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        self.value = nn.Linear(width, width * expansion)
        self.gate = nn.Linear(width, width * expansion)
        self.output = nn.Linear(width * expansion, width)

    def forward(self, rows: Tensor) -> Tensor:
        return self.output(F.silu(self.gate(rows)) * self.value(rows))

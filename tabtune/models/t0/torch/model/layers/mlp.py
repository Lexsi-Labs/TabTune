"""Implementation of MLP and ResidualBlock used by the patch encoder and decoder."""

from collections.abc import Callable

import torch
import torch.nn as nn
from jaxtyping import Float


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ):
        """Classical MLP.

        Args:
            input_size: Size of the input.
            hidden_size: Size of the hidden layer.
            output_size: Size of the output.
            dropout: Dropout rate.
            activation: Zero-arg factory returning a fresh ``nn.Module`` activation.
                Defaults to ``nn.ReLU``.
        """
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x: Float[torch.Tensor, "*batch input_size"]) -> Float[torch.Tensor, "*batch output_size"]:
        out = self.hidden_layer(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ):
        """Residual block.

        Args:
            input_size: Size of the input.
            hidden_size: Size of the hidden layer.
            output_size: Size of the output.
            dropout: Dropout rate.
            activation: Zero-arg factory returning a fresh ``nn.Module`` activation.
                Defaults to ``nn.ReLU``.
        """
        super().__init__()
        self.mlp = MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            activation=activation,
        )
        self.residual_layer = nn.Linear(input_size, output_size)

    def forward(self, x: Float[torch.Tensor, "*batch input_size"]) -> Float[torch.Tensor, "*batch output_size"]:
        out = self.mlp(x)
        residual = self.residual_layer(x)
        return out + residual

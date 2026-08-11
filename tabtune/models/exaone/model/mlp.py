"""A small, bias-free feed-forward transformation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


class FeedForward(nn.Module):
    """Expand and project vectors along the last tensor dimension."""

    def __init__(
        self,
        width: int,
        expanded_width: int,
        *,
        activation: str = "gelu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        zero_output: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(width, int) or isinstance(width, bool):
            raise TypeError("width must be an integer")
        if not isinstance(expanded_width, int) or isinstance(expanded_width, bool):
            raise TypeError("expanded_width must be an integer")
        if width <= 0 or expanded_width <= 0:
            raise ValueError("width and expanded_width must be positive")
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in {"gelu", "relu"}:
            raise ValueError("activation must be 'gelu' or 'relu'")
        if not isinstance(zero_output, bool):
            raise TypeError("zero_output must be a bool")
        if dtype is not None and (
            not isinstance(dtype, torch.dtype) or not dtype.is_floating_point
        ):
            raise ValueError("dtype must be a floating-point torch dtype")

        self.width = width
        self.expanded_width = expanded_width
        self.activation = activation

        factory_options = {"device": device, "dtype": dtype}
        self.expansion_weight = nn.Parameter(
            torch.empty((expanded_width, width), **factory_options)
        )
        self.projection_weight = nn.Parameter(
            torch.empty((width, expanded_width), **factory_options)
        )

        nn.init.kaiming_uniform_(self.expansion_weight)
        if zero_output:
            nn.init.zeros_(self.projection_weight)
        else:
            nn.init.kaiming_uniform_(self.projection_weight)

    def forward(
        self, tensor: torch.Tensor, *, add_residual: bool = False
    ) -> torch.Tensor:
        """Apply the learned transformation and optionally add its input."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if tensor.dtype not in _INPUT_DTYPES:
            raise TypeError("tensor has an unsupported dtype")
        if not isinstance(add_residual, bool):
            raise TypeError("add_residual must be a bool")
        if tensor.ndim == 0:
            raise ValueError("tensor must have at least one dimension")
        if tensor.shape[-1] != self.width:
            raise ValueError("the final tensor dimension does not match width")
        if (
            tensor.dtype != self.expansion_weight.dtype
            or tensor.device != self.expansion_weight.device
            or tensor.dtype != self.projection_weight.dtype
            or tensor.device != self.projection_weight.device
        ):
            raise RuntimeError("tensor and parameters must share dtype and device")

        expanded = F.linear(tensor, self.expansion_weight)
        if self.activation == "gelu":
            expanded = F.gelu(expanded, approximate="none")
        else:
            expanded = F.relu(expanded)
        contribution = F.linear(expanded, self.projection_weight)
        return tensor + contribution if add_residual else contribution


__all__ = ["FeedForward"]

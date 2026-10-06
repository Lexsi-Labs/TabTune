# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""sLSTM layers and configuration helpers (FlashRNN "vanilla" recurrence in pure PyTorch)."""

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
from torch import nn

from .slstm_kernel import slstm_forward
from .xlstm_layers import (
    CausalConv1d,
    CausalConv1dConfig,
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
    MultiHeadLayerNorm,
    small_init_init_,
)
from .xlstm_mixed_config import xLSTMMixedConfig


@dataclass
class FlashRNNLayerConfig:
    """Configuration for FlashRNN-based sLSTM layers used inside TiRex."""

    embedding_dim: int = -1
    num_heads: int = 4  # this must divide the embedding_dim
    conv1d_kernel_size: int = 0  # 0 means no convolution included
    group_norm_weight: bool = True
    dropout: float = 0.0

    # Cell specific inits
    recurrent_weight_init: str = "standard"
    bias_init: str = "powerlaw_blockdependent"

    # sLSTM: four gates (i, f, z, o) and four states (y, c, n, m)
    num_gates_i: int = 4
    num_states: int = 4

    def __post_init__(self):
        """Validate dimensions and derive head information."""
        self.hidden_dim = self.embedding_dim
        assert self.embedding_dim % self.num_heads == 0
        self.head_dim = self.embedding_dim // self.num_heads


class _FlashRNNLayer(nn.Module):
    """Base class holding the gate projections and the output norm of an sLSTM layer."""

    config_class = FlashRNNLayerConfig

    def __init__(self, config: FlashRNNLayerConfig):
        super().__init__()
        self.config = config

        if self.config.conv1d_kernel_size > 0:
            self.conv1d = CausalConv1d(
                config=CausalConv1dConfig(
                    feature_dim=self.config.embedding_dim,
                    kernel_size=self.config.conv1d_kernel_size,
                )
            )
            self.conv_act_fn = nn.SiLU()

        self.fgate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.igate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.zgate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.ogate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )

        self.group_norm = MultiHeadLayerNorm(
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            eps=1e-6,
            use_weight=self.config.group_norm_weight,
            use_bias=False,
            force_float32_reductions=True,
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def get_R(self):
        """Return the recurrent weight tensor ``[H, P, G, D]``."""
        raise NotImplementedError

    def get_bias(self):
        """Return the gate bias tensor ``[H, G, D]``."""
        raise NotImplementedError

    def reset_parameters(self):
        """Reset parameters."""
        small_init_init_(self.igate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.fgate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.zgate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.ogate.weight, dim=self.config.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Process a full sequence through the sLSTM recurrence."""
        batch_size, _, _ = x.shape
        if self.config.conv1d_kernel_size > 0:
            x_conv = self.conv1d(x)
            x_conv = self.conv_act_fn(x_conv)
        else:
            x_conv = x

        f_gate = self.fgate(x_conv)
        i_gate = self.igate(x_conv)
        zgate = self.zgate(x)
        ogate = self.ogate(x)
        gates = (
            f_gate,
            i_gate,
            zgate,
            ogate,
        )

        Wx = torch.stack(gates, dim=2)
        Wx = Wx.reshape(*Wx.shape[:-1], self.config.num_heads, -1)
        y = slstm_forward(Wx, self.get_R(), self.get_bias())

        y = self.dropout(y)

        out = self.group_norm(y)
        return out


class sLSTMFlashRNNLayer(_FlashRNNLayer):
    """sLSTM layer with custom parameter initializers."""

    def __init__(self, config: FlashRNNLayerConfig, block_idx: int, num_blocks: int):
        super().__init__(config)
        assert self.config.recurrent_weight_init in ["zeros", "standard"]
        assert self.config.bias_init in ["powerlaw_blockdependent", "zeros"]
        self._block_idx = block_idx
        self._num_blocks = num_blocks

        self._recurrent_kernel_ = nn.Parameter(
            torch.empty(
                self.config.num_heads,
                self.config.head_dim,
                self.config.num_gates_i,
                self.config.head_dim,
            )
        )
        # Upstream checkpoints store the recurrent kernel under both names (same tensor).
        self.recurrent_kernel = self._recurrent_kernel_
        self._bias_ = nn.Parameter(torch.empty(self.config.num_heads, self.config.num_gates_i, self.config.head_dim))

        self.reset_parameters()

    def reset_weights(self):
        """Reset recurrent kernels according to the chosen scheme."""
        with torch.no_grad():
            if self.config.recurrent_weight_init == "zeros":
                nn.init.zeros_(self._recurrent_kernel_)
            elif self.config.recurrent_weight_init == "standard":
                bound = 1.0 / sqrt(self.config.hidden_dim)
                nn.init.uniform_(self._recurrent_kernel_, -bound, bound)

    def reset_bias(self):
        """Reset gate biases with power-law schedule for the forget gate."""
        with torch.no_grad():
            nn.init.zeros_(self._bias_)
            if self.config.bias_init == "powerlaw_blockdependent":
                ratio_0_to_1 = self._block_idx / (self._num_blocks - 1) if self._num_blocks > 1 else 0.0
                init_values = -(
                    -5.0
                    + 12.0
                    * (torch.arange(self.config.head_dim) / (self.config.head_dim - 1)) ** (0.3 + 1.3 * ratio_0_to_1)
                )
                # gate order (i, f, z, o): only the forget gate gets a non-zero bias
                self._bias_[:, 1, :] = init_values

    def reset_parameters(self):
        """Reset projections, recurrent weights, and biases."""
        super().reset_parameters()
        self.reset_weights()
        self.reset_bias()

    def get_R(self):
        """Return the recurrent kernel."""
        return self._recurrent_kernel_

    def get_bias(self):
        """Return the bias tensor."""
        return self._bias_


def init_cell(config: xLSTMMixedConfig, block_idx: int, num_blocks: int, device: Literal["cpu", "cuda", "mps"]):
    """Instantiate an sLSTM cell. The pure-PyTorch recurrence is the same on every device."""
    return sLSTMFlashRNNLayer(
        FlashRNNLayerConfig(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_slstm_heads,
            conv1d_kernel_size=config.conv1d_kernel_size,  # 0 means no convolution included
            group_norm_weight=True,
            dropout=0,
            recurrent_weight_init="zeros",
            bias_init="powerlaw_blockdependent",
        ),
        block_idx=block_idx,
        num_blocks=num_blocks,
    )

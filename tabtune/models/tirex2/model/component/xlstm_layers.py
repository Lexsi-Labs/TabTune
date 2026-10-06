# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Building blocks ported from xlstm 2.0.6 (https://github.com/NX-AI/xlstm).

Sources: ``xlstm/components/{conv,init,linear_headwise}.py`` and
``xlstm/xlstm_large/{components,model,utils}.py``. See ATTRIBUTION.md.
"""

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from .mlstm_kernel import mLSTMBackendConfig

WeightModeType = Literal["single", "fused"]


def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    """Rounds up x to the next multiple of multiple_of."""
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


def soft_cap(values: torch.Tensor, cap_value: float | torch.Tensor | None) -> torch.Tensor:
    """
    Soft caps a tensor to a value.

    Performs a tanh operation on the logits and scales the result to the cap value. Common technique in attention
    and output language heads to prevent large logits from dominating the softmax. See for example Gemma2:
    https://arxiv.org/abs/2408.00118

    Args:
        values: The tensor to cap.
        cap_value: The value to cap the values to. If None, no cap is applied.

    Returns:
        The capped values.
    """
    if cap_value is None:
        return values
    return cap_value * torch.tanh(values / cap_value)


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


@dataclass
class CausalConv1dConfig:
    feature_dim: int = None  # F
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    conv1d_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    config_class = CausalConv1dConfig

    def __init__(self, config: CausalConv1dConfig):
        super().__init__()
        self.config = config
        self.groups = self.config.feature_dim
        if self.config.channel_mixing:
            self.groups = 1
        if self.config.kernel_size == 0:
            self.conv = None  # Noop
        else:
            self.pad = self.config.kernel_size - 1  # padding of this size assures temporal causality.
            self.conv = nn.Conv1d(
                in_channels=self.config.feature_dim,
                out_channels=self.config.feature_dim,
                kernel_size=self.config.kernel_size,
                padding=self.pad,
                groups=self.groups,
                bias=self.config.causal_conv_bias,
                **self.config.conv1d_kwargs,
            )
        # B, C, L
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        if self.conv is not None:
            self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.kernel_size == 0:
            return x
        y = x.transpose(2, 1)  # (B,F,T) tensor - now in the right shape for conv layer.
        y = self.conv(y)  # (B,F,T+pad) tensor
        return y[:, :, : -self.pad].transpose(2, 1)


@dataclass
class LinearHeadwiseExpandConfig:
    in_features: int = 0
    # this is the number of heads that the in_features are split into
    # if num_heads=1, this is a normal linear layer
    # if num_heads>1, the in_features are split into num_heads and each head is projected separately
    # if num_heads=in_features, each feature is projected separately
    num_heads: int = -1
    expand_factor_up: float = 1

    # this is internally computed
    # but can be overwritten if you want to use a different output dimension
    # if > 0 the expand factor is ignored
    _out_features: int = -1

    bias: bool = True
    trainable_weight: bool = True
    trainable_bias: bool = True

    def __post_init__(self):
        assert self.num_heads > 0, "num_heads must be set"
        assert self.num_heads <= self.in_features, "num_heads must be <= in_features"
        assert self.in_features % self.num_heads == 0, "in_features must be a multiple of num_heads"

        if self._out_features < 0:
            self._out_features = round(self.expand_factor_up * self.in_features)


class LinearHeadwiseExpand(nn.Module):
    """This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    config_class = LinearHeadwiseExpandConfig

    def __init__(self, config: LinearHeadwiseExpandConfig):
        super().__init__()
        self.config = config
        in_features = self.config.in_features
        num_heads = self.config.num_heads
        out_features_per_head = config._out_features // num_heads
        self.weight = nn.Parameter(
            torch.empty(num_heads, out_features_per_head, in_features // num_heads),
            requires_grad=config.trainable_weight,
        )
        if config.bias:
            self.bias = nn.Parameter(torch.empty(config._out_features), requires_grad=config.trainable_bias)
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        # small init
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], self.config.num_heads, -1)
        x = torch.einsum("...hd,hod->...ho", x, self.weight)
        x = x.reshape(*shape[:-1], -1)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"in_features={self.config.in_features}, "
            f"num_heads={self.config.num_heads}, "
            f"expand_factor_up={self.config.expand_factor_up}, "
            f"bias={self.config.bias}, "
            f"trainable_weight={self.config.trainable_weight}, "
            f"trainable_bias={self.config.trainable_bias}, "
        )


class NormLayer(nn.Module):
    """Base class for normalization layers.
    This class contains optional learnable weight and bias parameters.

    Args:
        num_features: The number of features in the input tensor.
        eps: A small value to avoid division by zero.
        use_weight: Whether to use a learnable weight.
        use_bias: Whether to use a learnable bias.
        force_float32_reductions: Whether to force float32 reductions.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.force_float32_reductions = force_float32_reductions

        if use_weight:
            self.weight = nn.Parameter(torch.ones(num_features))
        else:
            self.weight = None

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.bias = None

    def _apply_weight_bias(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class RMSNorm(NormLayer):
    """Root mean square normalization layer implementation similar
    to https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html.

    It normalizes the input tensor by the root mean square of the last dimension.
    """

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        # apply rms norm over the last dimension, i.e. D dimension
        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(in_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        x = self._rms_normalize(x)
        x = self._apply_weight_bias(x)
        return x


class LayerNorm(NormLayer):
    """Layer normalization layer implementation similar to
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html.

    The layer normalization is applied over the last dimension of the input tensor.
    """

    def _layer_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        # apply layer norm over the last dimension, i.e. D dimension
        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y = x_centered * torch.rsqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        return y.to(in_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        x = self._layer_normalize(x)
        x = self._apply_weight_bias(x)
        return x


class MultiHeadLayerNorm(LayerNorm):
    """Multi-head version of the LayerNorm layer.

    The input is assumed to have the shape (B, S, NH, DH). The normalization is applied over the last
    dimension (DH) of the input tensor; the result has the shape (B, S, NH * DH).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__(
            num_features=num_heads * head_dim,
            eps=eps,
            use_weight=use_weight,
            use_bias=use_bias,
            force_float32_reductions=force_float32_reductions,
        )
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        x: torch.Tensor,  # (B, S, NH, DH)
    ) -> torch.Tensor:  # (B, S, NH * DH)
        B, S, NH, DH = x.shape
        assert NH == self.num_heads, f"Expected {self.num_heads} heads, got {NH}, input shape: {x.shape}"
        assert DH == self.head_dim, f"Expected {self.head_dim} head dimension, got {DH}, input shape: {x.shape}"

        x = self._layer_normalize(x)
        x = x.reshape(B, S, -1)
        x = self._apply_weight_bias(x)
        return x


@dataclass
class xLSTMLargeConfig:
    embedding_dim: int
    """Embedding dimension of the model."""
    num_heads: int
    """Number of heads."""
    num_blocks: int
    """Number of blocks."""
    vocab_size: int
    """Vocabulary size."""
    use_bias: bool = False
    """Whether to use bias in linear layers."""
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in the normalization layers."""
    norm_reduction_force_float32: bool = True
    """Whether to force float32 reductions in the normalization layers."""

    # mlstm layer
    qk_dim_factor: float = 0.5
    """The factor to determine the dimension of the query and key tensors."""
    v_dim_factor: float = 1.0
    """The factor to determine the dimension of the value tensor."""

    # mlstm backend
    mode: Literal["train", "train_with_padding"] = "train"
    """The mode of operation for the backend. 'train_with_padding' pads the input to multiples of `chunk_size`."""
    chunk_size: int = 64
    """The chunk size of the chunkwise kernel."""
    return_last_states: bool = False
    """Whether to return the last states of the sequence in training mode."""
    eps: float = 1e-6
    """Epsilon value for numerical stability in the kernel."""

    # feedforward
    ffn_proj_factor: float = 2.6667
    """The factor to determine the dimension of the intermediate projection in the feedforward layer."""
    ffn_round_up_to_multiple_of: int = 64
    """Round the intermediate projection dimension to the next multiple of this value."""

    # capping
    gate_soft_cap: float = 15.0
    """Soft cap value for the gates."""

    weight_mode: WeightModeType = "single"
    """The weight mode to use for the mLSTM layer.
    Mode 'single' uses separate weights for the query, key, value, and gates.
    Mode 'fused' uses a single weight matrix for the query, key, value, and gates.
    """


class FeedForward(nn.Module):
    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config

        self.up_proj_dim = round_up_to_next_multiple_of(
            config.embedding_dim * config.ffn_proj_factor,
            config.ffn_round_up_to_multiple_of,
        )

        if self.config.weight_mode == "single":
            self.proj_up_gate = nn.Linear(
                in_features=config.embedding_dim,
                out_features=self.up_proj_dim,
                bias=self.config.use_bias,
            )
            self.proj_up = nn.Linear(
                in_features=config.embedding_dim,
                out_features=self.up_proj_dim,
                bias=self.config.use_bias,
            )
        elif self.config.weight_mode == "fused":
            self.proj_up_gate_z = nn.Linear(
                in_features=config.embedding_dim,
                out_features=2 * self.up_proj_dim,
                bias=self.config.use_bias,
            )

        self.proj_down = nn.Linear(
            in_features=self.up_proj_dim,
            out_features=config.embedding_dim,
            bias=self.config.use_bias,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.weight_mode == "single":
            x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
        elif self.config.weight_mode == "fused":
            x = self.proj_up_gate_z(x)
            gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z

        y = self.proj_down(x)
        return y


@dataclass
class mLSTMLayerConfig:
    embedding_dim: int
    """Embedding dimension of the model."""
    num_heads: int
    """Number of heads."""
    use_bias: bool = False
    """Whether to use bias in linear layers."""
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in the normalization layers."""
    norm_reduction_force_float32: bool = True
    """Whether to force float32 reductions in the normalization layers."""

    qk_dim_factor: float = 0.5
    """The factor to determine the dimension of the query and key tensors."""
    v_dim_factor: float = 1.0
    """The factor to determine the dimension of the value tensor."""
    gate_soft_cap: float = 15.0
    """Soft cap value for the gates."""

    mlstm_backend: mLSTMBackendConfig = field(default_factory=mLSTMBackendConfig)
    """Configuration of the mLSTM backend."""

    weight_mode: WeightModeType = "single"
    """The weight mode to use for the mLSTM layer."""

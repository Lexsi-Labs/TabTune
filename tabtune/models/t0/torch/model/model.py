# Copyright 2026 The Forecasting Company
# The architecture is inspired by Datadog's Toto patched-transformer backbone
# (https://github.com/DataDog/toto).
# Copyright 2025 Datadog, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Transformer backbone for the open-weights t0-alpha model.

``T0Forecaster.forward`` runs one differentiable pass; ``T0Forecaster.predict`` wraps it
with scaling, rollout and inference mode.
"""

import contextlib
import sys
from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float, Int
from torch import Tensor

from ...config import ScalerEpsMode, T0Config, XposScaleLayout
from ...data import TimeSeries
from ...forecast import Forecast
from ...forecaster import DEFAULT_MAX_HORIZON, ForecasterBase
from ...torch.model.layers import PatchEncoder, QuantileHead, ResidualBlock, Transformer

if sys.version_info >= (3, 11):
    pass
else:
    pass

__all__ = ["DEFAULT_MAX_HORIZON", "T0Forecaster"]


class T0Forecaster(
    ForecasterBase,
    nn.Module,
    PyTorchModelHubMixin,
    library_name="tfc-t0",
    repo_url="https://github.com/theforecastingcompany/tfc-t0",
    pipeline_tag="time-series-forecasting",
    license="apache-2.0",
    tags=["time-series", "forecasting", "foundation-models", "pretrained-models", "safetensors"],
):
    """Open-weights t0-alpha forecasting backbone.

    Construct with explicit hyperparameters, or via ``from_config`` /
    ``from_pretrained``. ``embed_dim`` must be divisible by ``num_heads`` and
    ``group_every_n`` must divide ``num_layers``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        patch_size: int,
        group_every_n: int,
        dropout: float,
        quantile_levels: Sequence[float],
        scaler_use_arcsinh: bool = True,
        scaler_eps: float = 0.1,
        scaler_eps_mode: ScalerEpsMode = "variance_offset",
        xpos_scale_layout: XposScaleLayout = "interleaved",
        scaler_zero_no_value: bool = True,
        # Forwarded by huggingface_hub 1.x's from_pretrained (renamed from
        # torch_dtype). bf16/fp16 keep fp32 weights and autocast the forward
        # in predict(); other dtypes run in fp32 with no autocast.
        dtype: torch.dtype | None = None,
        **_: object,
    ):
        super().__init__()
        # Building the config validates every hyperparameter.
        self._init_shared(
            T0Config(
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                patch_size=patch_size,
                group_every_n=group_every_n,
                dropout=dropout,
                quantile_levels=tuple(quantile_levels),
                scaler_use_arcsinh=scaler_use_arcsinh,
                scaler_eps=scaler_eps,
                scaler_eps_mode=scaler_eps_mode,
                xpos_scale_layout=xpos_scale_layout,
                scaler_zero_no_value=scaler_zero_no_value,
            )
        )
        self.head = QuantileHead(quantile_levels=list(quantile_levels))
        self.patch_encoder = PatchEncoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            activation=nn.ReLU,
        )
        self.transformer = Transformer(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            group_every_n=group_every_n,
            xpos_scale_layout=xpos_scale_layout,
        )
        self.decoder = ResidualBlock(
            input_size=embed_dim,
            hidden_size=embed_dim,
            output_size=patch_size * self.head.n_quantiles,
            activation=nn.ReLU,
        )
        # bf16/fp16 autocast the forward in predict() (weights stay fp32);
        # anything else runs in fp32.
        self._amp_dtype: torch.dtype | None = dtype if dtype in (torch.float16, torch.bfloat16) else None

    def forward(self, model_input: TimeSeries) -> Float[Tensor, "variates patches patch_size quantiles"]:
        """Predict per-patch quantiles for every patch position (one differentiable pass)."""
        return self._backbone(model_input)

    @overload
    def predict(
        self,
        model_input: TimeSeries,
        horizon: int,
        quantile_levels: Sequence[float] = ...,
        *,
        context_length: int | None = ...,
        trim_leading_missing: bool = ...,
        symmetric_averaging: bool = ...,
        clip_nonnegative: bool = ...,
        snap_integers: bool = ...,
    ) -> Forecast[Tensor]: ...

    @overload
    def predict(
        self,
        model_input: Float[Tensor, "batch time"]
        | Float[Tensor, "batch variates time"]
        | Float[np.ndarray, "batch time"]
        | Float[np.ndarray, "batch variates time"],
        horizon: int,
        quantile_levels: Sequence[float] = ...,
        *,
        future_covariates: Float[Tensor, "batch future_variates context_plus_horizon"]
        | Float[np.ndarray, "batch future_variates context_plus_horizon"]
        | None = ...,
        mask: Int[Tensor, "*batch time"] | Int[np.ndarray, "*batch time"] | None = ...,
        group_ids: Int[Tensor, " rows"] | Int[np.ndarray, " rows"] | None = ...,
        trim_leading_missing: bool = ...,
        symmetric_averaging: bool = ...,
        clip_nonnegative: bool = ...,
        snap_integers: bool = ...,
    ) -> Forecast[Tensor]: ...

    @torch.inference_mode()
    def predict(
        self,
        model_input: Any,
        horizon: int,
        quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
        *,
        context_length: int | None = None,
        future_covariates: Any = None,
        mask: Any = None,
        group_ids: Any = None,
        trim_leading_missing: bool = False,
        symmetric_averaging: bool = False,
        clip_nonnegative: bool = False,
        snap_integers: bool = False,
    ) -> Forecast[Tensor]:
        return super().predict(
            model_input,
            horizon,
            quantile_levels,
            context_length=context_length,
            future_covariates=future_covariates,
            mask=mask,
            group_ids=group_ids,
            trim_leading_missing=trim_leading_missing,
            symmetric_averaging=symmetric_averaging,
            clip_nonnegative=clip_nonnegative,
            snap_integers=snap_integers,
        )

    predict.__doc__ = ForecasterBase.predict.__doc__

    def _as_array(self, value: Any, *, floating: bool) -> Tensor:
        tensor = torch.as_tensor(value)
        if floating:
            return tensor.to(device=self._device(), dtype=torch.float32)
        return tensor.to(device=self._device())

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _predict_scope(self) -> contextlib.AbstractContextManager:
        if self._amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self._device().type, dtype=self._amp_dtype)

# Copyright 2026 The Forecasting Company
# The Welford-style cumulative causal statistics borrow from Datadog's Toto
# scaler (https://github.com/DataDog/toto).
# Copyright 2025 Datadog, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Causal scaler for time series normalization, shared by every runtime."""

import dataclasses
from typing import Any, Generic

from jaxtyping import Bool, Float

from ._ops import ArrayT, ops_for
from .config import ScalerEpsMode
from .types import MaskType, VariateType

__all__ = ["EPS", "CausalScaler", "LocScale", "compute_causal_stats", "compute_global_stats"]

# Same epsilon as upstream Toto: https://github.com/DataDog/toto/blob/b4d4c9f3e121701fb02f65d525a435bf551d2582/toto/model/scaler.py#L307
EPS = 1e-1


@dataclasses.dataclass
class LocScale(Generic[ArrayT]):
    """Per-row, per-time-step location and scale."""

    loc: Float[ArrayT, "variates time"]
    scale: Float[ArrayT, "variates time"]


def compute_causal_stats(
    x: Float[ArrayT, "variates time"],
    invalid: Bool[ArrayT, "variates time"],
    eps: float = EPS,
    eps_mode: ScalerEpsMode = "variance_offset",
) -> tuple[Float[ArrayT, "variates time"], Float[ArrayT, "variates time"]]:
    """Welford causal mean/std along the time axis, one series per row.

    ``invalid`` cells contribute nothing; ``eps_mode`` selects how ``eps`` keeps the
    std away from zero — see ``T0Config.scaler_eps_mode``.
    """
    ops = ops_for(x)
    valid = ops.astype(~invalid, x.dtype)
    count = ops.clip(ops.cumulative_sum(valid, axis=-1), 1.0, None)
    masked = ops.where(invalid, ops.zeros_like(x), x)
    means = ops.cumulative_sum(masked, axis=-1) / count
    shifted_means = ops.concat([ops.zeros_like(means[:, :1]), means[:, :-1]], axis=-1)
    m_2 = ops.clip(ops.cumulative_sum((masked - shifted_means) * (masked - means) * valid, axis=-1), 0.0, None)
    variance = m_2 / ops.clip(count - 1.0, 1.0, None)
    if eps_mode == "variance_offset":
        return means, ops.sqrt(variance + eps)
    return means, ops.clip(ops.sqrt(variance), eps, None)


def compute_global_stats(
    x: Float[ArrayT, "variates time"],
    invalid: Bool[ArrayT, "variates time"],
    eps: float = EPS,
) -> tuple[Float[ArrayT, "variates time"], Float[ArrayT, "variates time"]]:
    """Per-row global (non-causal) mean and std, broadcast back to ``(V, T)``.

    ``eps`` is always applied as a lower bound on the standard deviation.
    """
    ops = ops_for(x)
    valid = ops.astype(~invalid, x.dtype)
    count = ops.row_sum(valid)
    masked = ops.where(invalid, ops.zeros_like(x), x)
    mean = ops.row_sum(masked) / ops.clip(count, 1.0, None)
    squared = ops.where(invalid, ops.zeros_like(x), (x - mean) * (x - mean))
    std = ops.clip(ops.sqrt(ops.row_sum(squared) / ops.clip(count, 2.0, None)), eps, None)
    return ops.broadcast_to(mean, x.shape), ops.broadcast_to(std, x.shape)


class CausalScaler:
    """Per-row causal scaler used by ``T0Forecaster``.

    Targets and historicals get causal stats (Welford); futures get per-row global
    stats. Every time step is standardized by its own running statistics, matching
    how the published checkpoints were trained; forecasts are rescaled with the stats
    at each model patch's last time step. Optionally applies arcsinh after the standard
    ``(x - loc) / scale`` step (novel to t0-alpha, helps with extreme outliers).

    ``eps`` and ``eps_mode`` must match what the checkpoint was trained with;
    ``T0Forecaster`` takes both from its ``T0Config``. Stateless, and branch-free on
    array values, so it traces into compiled graphs.
    """

    def __init__(
        self,
        use_arcsinh: bool = False,
        eps: float = EPS,
        eps_mode: ScalerEpsMode = "variance_offset",
        zero_no_value: bool = True,
    ):
        self.use_arcsinh = use_arcsinh
        self.eps = eps
        self.eps_mode = eps_mode
        self.zero_no_value = zero_no_value

    def scale_input(self, series: Any) -> tuple[Any, LocScale]:
        """Standardize a ``TimeSeries``; returns the scaled series and the statistics used."""
        x = series.variates
        ops = ops_for(x)
        invalid = (series.mask != MaskType.VALID) | ops.isnan(x)
        if not self.zero_no_value:
            # A cell without an observation carries zero into the scaling below.
            x = ops.where(ops.isnan(x), ops.zeros_like(x), x)
        causal_loc, causal_scale = compute_causal_stats(x, invalid, self.eps, self.eps_mode)
        future_loc, future_scale = compute_global_stats(x, invalid, self.eps)
        real = series.group_ids >= 0
        is_future = real & (series.variate_type == VariateType.FUTURE)
        is_causal = real & (
            (series.variate_type == VariateType.TARGET) | (series.variate_type == VariateType.HISTORICAL)
        )
        loc = ops.where(is_future, future_loc, ops.where(is_causal, causal_loc, ops.zeros_like(x)))
        scale = ops.where(is_future, future_scale, ops.where(is_causal, causal_scale, ops.ones_like(x)))
        scaled = (x - loc) / scale
        if self.use_arcsinh:
            scaled = ops.asinh(scaled)
        # Cells that hold no observation read as zero, whatever placeholder they carried;
        # censored cells keep their (scaled) bound.
        if self.zero_no_value:
            no_value = (
                (series.mask == MaskType.PAD) | (series.mask == MaskType.MISSING) | (series.mask == MaskType.WITHHELD)
            )
            scaled = ops.where(no_value, ops.zeros_like(scaled), scaled)
        return dataclasses.replace(series, variates=scaled), LocScale(loc=loc, scale=scale)

    def rescale_predictions(
        self,
        predictions: Float[ArrayT, "variates patches *event"],
        loc_scale: LocScale,
        model_patch_size: int,
    ) -> Float[ArrayT, "variates patches *event"]:
        """Inverse-transform per-patch predictions back to data space."""
        ops = ops_for(predictions)
        loc = loc_scale.loc[:, model_patch_size - 1 :: model_patch_size]
        scale = loc_scale.scale[:, model_patch_size - 1 :: model_patch_size]
        for _ in range(predictions.ndim - 2):
            loc, scale = ops.expand_dims(loc, axis=-1), ops.expand_dims(scale, axis=-1)
        values = ops.sinh(predictions) if self.use_arcsinh else predictions
        return values * scale + loc

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.processing import InvertibleMixin, Processor
from tabtune.models.kumo._sdm.processing.numerical._stats import (
    _constant_feature_mask,
    _isfinite,
)

# Keep GPU execution batched; adaptive per-column stopping would resynchronize.
# For float32 overflow-safe bounds, 44 golden steps reaches ~1.48e-8.
_YEOJOHNSON_OPTIMIZATION_STEPS = 44


def _yeojohnson_transform(
    inp: Tensor,
    lambdas: Tensor,
    *,
    magnitude_log: Tensor | None = None,
    positive: Tensor | None = None,
    exponents: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    eps = torch.finfo(inp.dtype).eps
    two_minus_lambda = 2 - lambdas
    if magnitude_log is None:
        magnitude_log = inp.abs().log1p_()
    if positive is None:
        positive = inp >= 0
    exponents = torch.where(positive, lambdas, two_minus_lambda, out=exponents)
    out = torch.mul(exponents, magnitude_log, out=out)
    out.expm1_().div_(exponents)
    zero_exponent = torch.where(
        positive, lambdas.abs() < eps, two_minus_lambda.abs() < eps
    )
    torch.where(zero_exponent, magnitude_log, out, out=out)
    return out.copysign_(inp)


def _yeojohnson_inverse_transform(inp: Tensor, lambdas: Tensor) -> Tensor:
    positive = inp >= 0
    eps = torch.finfo(inp.dtype).eps

    positive_power = ((inp * lambdas).log1p() / lambdas).expm1()
    positive_log = inp.expm1()
    positive_out = torch.where(
        lambdas.abs() < eps,
        positive_log,
        positive_power,
    )

    two_minus_lambda = 2 - lambdas
    negative_power = -(
        (-two_minus_lambda * inp).log1p() / two_minus_lambda
    ).expm1()
    negative_log = -(-inp).expm1()
    negative_out = torch.where(
        two_minus_lambda.abs() < eps,
        negative_log,
        negative_power,
    )

    return torch.where(positive, positive_out, negative_out)


def _yeojohnson_bounds(inp: Tensor) -> tuple[Tensor, Tensor]:
    missing = inp.isnan()
    max_abs = inp.abs().nan_to_num_(nan=0.0).amax(dim=-2, keepdim=True)
    log1p_max_x = (20 * max_abs).log1p()
    log1p_max_x = torch.where(
        max_abs == 0,
        torch.ones_like(log1p_max_x),
        log1p_max_x,
    )
    finfo = torch.finfo(inp.dtype)
    log_eps = math.log(finfo.eps)
    log_tiny_float = (math.log(finfo.tiny) - log_eps) / 2
    log_max_float = (math.log(finfo.max) + log_eps) / 2

    lower_bound = log_tiny_float / log1p_max_x
    upper_bound = log_max_float / log1p_max_x
    positive_lower = lower_bound
    positive_upper = upper_bound

    negative = inp < 0
    all_negative = (negative | missing).all(dim=-2, keepdim=True)
    any_negative = negative.any(dim=-2, keepdim=True)

    mixed_lower = torch.maximum(2 - positive_upper, positive_lower)
    mixed_upper = torch.minimum(2 - mixed_lower, positive_upper)
    lower_bound = torch.where(any_negative, mixed_lower, positive_lower)
    upper_bound = torch.where(any_negative, mixed_upper, positive_upper)

    negative_lower = 2 - positive_upper
    negative_upper = 2 - positive_lower
    lower_bound = torch.where(all_negative, negative_lower, lower_bound)
    upper_bound = torch.where(all_negative, negative_upper, upper_bound)

    zero = max_abs == 0
    return (
        torch.where(zero, torch.ones_like(lower_bound), lower_bound),
        torch.where(zero, torch.ones_like(upper_bound), upper_bound),
    )


def _yeojohnson_log_likelihood(
    inp: Tensor,
    lambdas: Tensor,
    magnitude_log: Tensor,
    positive: Tensor,
    log_jacobian: Tensor,
    count: Tensor,
    exponents: Tensor,
    transformed: Tensor,
) -> Tensor:
    transformed = _yeojohnson_transform(
        inp=inp,
        lambdas=lambdas,
        magnitude_log=magnitude_log,
        positive=positive,
        exponents=exponents,
        out=transformed,
    )
    mean = transformed.nansum(dim=-2, keepdim=True).div_(count)
    variance = transformed.sub_(mean).square_().nansum(dim=-2, keepdim=True)
    variance /= count
    tiny = torch.finfo(inp.dtype).tiny
    valid = variance.isfinite() & (variance >= tiny)
    loglike = variance.log_().mul_(-count / 2)
    loglike.add_((lambdas - 1) * log_jacobian)
    return loglike.masked_fill_(~valid, -math.inf)


def _optimize_lambdas(
    inp: Tensor,
    constant_features: Tensor,
    *,
    count: Tensor,
) -> Tensor:
    left, right = _yeojohnson_bounds(inp)
    left = left.masked_fill(constant_features, 1.0)
    right = right.masked_fill(constant_features, 1.0)

    # Reuse full-table workspaces throughout the golden-section search.
    magnitude_log = inp.abs().log1p_()
    positive = inp >= 0
    log_jacobian = magnitude_log.copysign(inp).nansum(dim=-2, keepdim=True)
    exponents = torch.empty_like(inp)
    transformed = torch.empty_like(inp)

    invphi = (math.sqrt(5) - 1) / 2
    span = (right - left).mul_(invphi)
    c = right - span
    d = left + span
    fc = _yeojohnson_log_likelihood(
        inp=inp,
        lambdas=c,
        magnitude_log=magnitude_log,
        positive=positive,
        log_jacobian=log_jacobian,
        count=count,
        exponents=exponents,
        transformed=transformed,
    )
    fd = _yeojohnson_log_likelihood(
        inp=inp,
        lambdas=d,
        magnitude_log=magnitude_log,
        positive=positive,
        log_jacobian=log_jacobian,
        count=count,
        exponents=exponents,
        transformed=transformed,
    )
    c_next = torch.empty_like(c)
    d_next = torch.empty_like(d)
    new_point = torch.empty_like(c)
    choose_right = torch.empty_like(c, dtype=torch.bool)

    for _ in range(_YEOJOHNSON_OPTIMIZATION_STEPS):
        choose_right = torch.lt(fc, fd, out=choose_right)
        # Keep the search fully vectorized: each feature independently
        # chooses its next interval without per-column Python branching.
        left = torch.where(choose_right, c, left, out=left)
        right = torch.where(choose_right, right, d, out=right)
        span = torch.sub(right, left, out=span).mul_(invphi)
        c_next = torch.sub(right, span, out=c_next)
        c_next = torch.where(choose_right, d, c_next, out=c_next)
        d_next = torch.add(left, span, out=d_next)
        d_next = torch.where(choose_right, d_next, c, out=d_next)
        new_point = torch.where(choose_right, d_next, c_next, out=new_point)
        new_score = _yeojohnson_log_likelihood(
            inp=inp,
            lambdas=new_point,
            magnitude_log=magnitude_log,
            positive=positive,
            log_jacobian=log_jacobian,
            count=count,
            exponents=exponents,
            transformed=transformed,
        )
        fc = torch.where(choose_right, new_score, fc, out=fc)
        fd = torch.where(choose_right, fd, new_score, out=fd)
        fc, fd = fd, fc
        c, c_next = c_next, c
        d, d_next = d_next, d

    lambdas = (left + right).div_(2)
    return lambdas.masked_fill_(constant_features, 1.0)


class PowerTransform(Processor, InvertibleMixin):
    """Apply a feature-wise Yeo-Johnson power transform.

    NaN and infinite values are left out of the fitted statistics. NaN values
    are preserved during the transform.

    Args:
        standardize: If ``True``, zero-mean and unit-variance the transformed
            features using statistics fitted after the power transform.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        standardize: bool = True,
    ) -> None:
        super().__init__()
        self.standardize = standardize
        self.register_buffer("lambdas", torch.empty(0))
        self.register_buffer("max", torch.empty(0))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def _optimize_lambdas(
        self,
        inp: Tensor,
        constant_features: Tensor,
        *,
        count: Tensor,
    ) -> Tensor:
        return _optimize_lambdas(inp, constant_features, count=count)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        finite = _isfinite(table.numerical)
        finite_or_nan = table.numerical.masked_fill(~finite, torch.nan)
        count = finite.sum(dim=-2, keepdim=True).clamp_(min=1)

        mean = finite_or_nan.nansum(-2, keepdim=True).div_(count)
        var = finite_or_nan.sub(mean).square_().nansum(-2, keepdim=True)
        var /= count
        constant_features = _constant_feature_mask(
            var,
            mean,
            num_samples=count,
        )
        del var

        # ``mean`` never exceeds the column maximum, and is zero for an
        # entirely missing column, whose max must be zero.
        self.max = torch.where(finite, table.numerical, mean).amax(
            dim=-2,
            keepdim=True,
        )
        del finite

        self.lambdas = self._optimize_lambdas(
            finite_or_nan,
            constant_features,
            count=count,
        )

        if self.standardize:
            transformed = _yeojohnson_transform(finite_or_nan, self.lambdas)
            del finite_or_nan
            mean = transformed.nansum(dim=-2, keepdim=True).div_(count)
            var = transformed.sub_(mean).square_().nansum(-2, keepdim=True)
            var /= count
            scale = var.sqrt()
            scale[_constant_feature_mask(var, mean, count)] = 1.0
            self.mean = mean
            self.scale = scale
        else:
            self.mean = torch.zeros_like(self.lambdas)
            self.scale = torch.ones_like(self.lambdas)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` with fitted Yeo-Johnson parameters."""
        transformed = _yeojohnson_transform(table.numerical, self.lambdas)
        numerical = transformed.sub_(self.mean).div_(self.scale)
        # The fitted lambdas only keep the fitted range representable, so a
        # query far outside it can overflow.
        bound = torch.finfo(numerical.dtype).max
        return table.replace_blocks(
            numerical=numerical.clamp_(min=-bound, max=bound)
        )

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        unscaled = table.numerical * self.scale + self.mean
        inverse = _yeojohnson_inverse_transform(unscaled, self.lambdas)

        # Above the fitted upper bound the inverse diverges, either to
        # infinity or, past the asymptote, to NaN.
        diverged = ~_isfinite(inverse) & ~unscaled.isnan()
        return table.replace_blocks(
            numerical=torch.where(
                diverged,
                torch.fmin(inverse, self.max),
                inverse,
            )
        )

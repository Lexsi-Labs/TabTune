# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# ---------------------------------------------------------------------------
# TabTune vendoring note.
#
# Toto 1.0's model core imports four symbols from gluonts; this module supplies
# them so that gluonts itself is not a TabTune dependency. That matters for a
# concrete reason rather than tidiness: gluonts 0.16.2 pins "numpy<2.2,>=1.16",
# so installing it would downgrade numpy for all sixteen tabular models too.
#
# Copied from gluonts 0.16.2 (Apache-2.0), keeping Amazon's header above:
#
#   * lazy_property     from gluonts/util.py
#   * AffineTransformed from gluonts/torch/distributions/affine_transformed.py
#   * StudentT          from gluonts/torch/distributions/studentT.py
#   * Scaler            from gluonts/torch/scaler.py
#
# Only "validated" is not upstream's code. The real
# gluonts.core.component.validated builds a pydantic model from the decorated
# __init__'s signature, coerces the arguments against it and records them on the
# instance for gluonts' own serde. Toto uses it on the three scalers in
# model/scaler.py, which TabTune only ever constructs from a checkpoint's
# config.json with already-correct types and never serialises back, so the
# passthrough below is equivalent for inference. tests/test_toto1_adapter.py
# pins that claim rather than trusting it.
# ---------------------------------------------------------------------------

from __future__ import annotations

import functools
from typing import Union

import torch
from scipy.stats import t as ScipyStudentT
from torch.distributions import (
    AffineTransform,
    Distribution,
    TransformedDistribution,
)
from torch.distributions import StudentT as TorchStudentT

__all__ = ["AffineTransformed", "Scaler", "StudentT", "lazy_property", "validated"]


def lazy_property(method):
    """
    Property that is lazily evaluated.

    This is the same as::

        @property
        @lru_cache(1)
        def my_property(self):
            ...

    In addition to be more concise, `lazy_property` also works with mypy,
    since it poses to be just `property` when type checked.

    This implementation follows the recipe from the `functools`
    documentation, and mimics `functools.cached_property` which was
    introduced in Python 3.8.
    """
    return property(functools.lru_cache(1)(method))


class AffineTransformed(TransformedDistribution):
    """
    Represents the distribution of an affinely transformed random variable.

    This is the distribution of ``Y = scale * X + loc``, where ``X`` is a
    random variable distributed according to ``base_distribution``.
    """

    def __init__(self, base_distribution: Distribution, loc=None, scale=None):
        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc

        super().__init__(base_distribution, [AffineTransform(self.loc, self.scale)])

    @property
    def mean(self):
        """Returns the mean of the distribution."""
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        """Returns the variance of the distribution."""
        return self.base_dist.variance * self.scale**2

    @property
    def stddev(self):
        """Returns the standard deviation of the distribution."""
        return self.variance.sqrt()


class StudentT(TorchStudentT):
    """
    Student's t-distribution parametrized by degree of freedom `df`, mean `loc`
    and scale `scale`.

    Based on torch.distributions.StudentT, with added `cdf` and `icdf` methods.
    """

    def __init__(
        self,
        df: Union[float, torch.Tensor],
        loc: Union[float, torch.Tensor] = 0.0,
        scale: Union[float, torch.Tensor] = 1.0,
        validate_args=None,
    ):
        super().__init__(df=df, loc=loc, scale=scale, validate_args=validate_args)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        result = self.scipy_student_t.cdf(value.detach().cpu().numpy())
        return torch.tensor(result, device=value.device, dtype=value.dtype)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        result = self.scipy_student_t.ppf(value.detach().cpu().numpy())
        return torch.tensor(result, device=value.device, dtype=value.dtype)

    @lazy_property
    def scipy_student_t(self):
        return ScipyStudentT(
            df=self.df.detach().cpu().numpy(),
            loc=self.loc.detach().cpu().numpy(),
            scale=self.scale.detach().cpu().numpy(),
        )


class Scaler:
    def __call__(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def validated(base_model=None):
    """Stand in for ``gluonts.core.component.validated`` as a passthrough.

    See the vendoring note at the top of this module for why this is equivalent
    for inference and where that equivalence is tested.
    """

    def decorator(init):
        @functools.wraps(init)
        def wrapper(self, *args, **kwargs):
            return init(self, *args, **kwargs)

        return wrapper

    return decorator

"""The forecast every runtime returns."""

import dataclasses
import logging
from typing import Generic

from jaxtyping import Float

from ._ops import ArrayT
from .quantile import interpolate_quantiles

__all__ = ["Forecast"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Forecast(Generic[ArrayT]):
    """Quantile forecast.

    ``quantiles`` is ``(B, horizon, Q)`` or ``(B, V, horizon, Q)``, last axis
    ordered like ``quantile_levels``; an array of the runtime that produced it.
    """

    quantiles: Float[ArrayT, "batch horizon quantiles"] | Float[ArrayT, "batch variates horizon quantiles"]
    quantile_levels: tuple[float, ...]

    @property
    def median(self) -> Float[ArrayT, "batch horizon"] | Float[ArrayT, "batch variates horizon"]:
        """The 0.5 quantile — exact when requested, otherwise interpolated from ``quantiles``."""
        if 0.5 in self.quantile_levels:
            return self.quantiles[..., self.quantile_levels.index(0.5)]
        logger.debug("0.5 not among quantile_levels %s — interpolating the median", self.quantile_levels)
        return interpolate_quantiles((0.5,), self.quantile_levels, self.quantiles)[..., 0]

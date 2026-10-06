"""Training-fitted transforms with explicit missing masks and compressed outliers."""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler


@dataclass
class Rank2Gaussian:
    """Interpolate training mid-ranks and map them through the Gaussian quantile function."""

    knots: tuple[tuple[np.ndarray, np.ndarray], ...]

    @classmethod
    def fit(cls, table):
        """Store unique values and tie-aware probabilities per feature column."""
        knots = []
        for column in table.T:
            values, counts = np.unique(column, return_counts=True)
            probabilities = (np.cumsum(counts) - counts / 2) / len(column)
            knots.append((values, probabilities))
        return cls(tuple(knots))

    def transform(self, table):
        """Reuse fitted ranks; interpolation clamps unseen extremes to endpoint ranks."""
        return np.column_stack([
            ndtri(np.interp(column, values, probabilities))
            for column, (values, probabilities) in zip(table.T, self.knots)
        ])


@dataclass
class PowerNormalizer:
    """Keep undefined power outputs missing before and after standard scaling."""

    power: PowerTransformer
    scale: StandardScaler

    @staticmethod
    def _power_values(power, table):
        with np.errstate(over="ignore", invalid="ignore"):
            values = power.transform(table)
        values[~np.isfinite(values)] = np.nan
        return values

    @classmethod
    def fit(cls, table):
        power = PowerTransformer(standardize=False).fit(table)
        values = cls._power_values(power, table)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scale = StandardScaler(copy=False).fit(values)
        return cls(power, scale)

    def transform(self, table):
        values = self._power_values(self.power, table)
        with np.errstate(over="ignore", invalid="ignore"):
            values = self.scale.transform(values)
        values[~np.isfinite(values)] = np.nan
        return values


@dataclass
class Normalizer:
    """Normalize encoded features using statistics fitted only on training rows.

    Missing entries are temporarily mean-filled to fit/apply transforms, then
    restored to NaN for the model's missing-value embedding. Method ``none``
    skips the optional distribution transform, but still standardizes columns
    and compresses outlier tails. Rank-to-Gaussian uses its own final scaling.
    """

    fill: np.ndarray
    center: np.ndarray
    spread: np.ndarray
    normalizer: object | None
    quantile_scale: StandardScaler | None
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def fit(cls, table: np.ndarray, method: str) -> "Normalizer":
        """Fit filling, scaling, optional distribution transform, and tail bounds."""
        missing = np.isnan(table)
        counts = (~missing).sum(axis=0)
        fill = np.divide(
            np.where(missing, 0, table).sum(axis=0), counts, out=np.zeros(table.shape[1]), where=counts > 0
        )
        filled = np.where(missing, fill, table)
        if method == "rank2gaussian":
            center = np.zeros(filled.shape[1], dtype=filled.dtype)
            spread = np.ones_like(center)
        else:
            center, spread = filled.mean(axis=0), filled.std(axis=0)
            tolerance = np.finfo(filled.dtype).eps * np.maximum(np.abs(center), 1)
            spread = np.where(spread <= tolerance, 1.0, spread)
        scaled = (filled - center) / spread
        normalizer, quantile_scale = None, None
        if method == "power":
            normalizer = PowerNormalizer.fit(scaled)
        elif method == "robust":
            normalizer = RobustScaler(unit_variance=True).fit(scaled)
        elif method == "rank2gaussian":
            normalizer = Rank2Gaussian.fit(scaled)
        elif method != "none":
            raise ValueError(f"Unknown internal normalization: {method}")
        normalized = scaled if normalizer is None else normalizer.transform(scaled)
        if method == "rank2gaussian":
            quantile_scale = StandardScaler().fit(normalized)
            normalized = quantile_scale.transform(normalized)
        lower, upper = outlier_bounds(normalized)
        return cls(
            fill=fill,
            center=center,
            spread=spread,
            normalizer=normalizer,
            quantile_scale=quantile_scale,
            lower=lower,
            upper=upper,
        )

    def transform(self, table: np.ndarray) -> np.ndarray:
        """Return float32 features with the original NaN mask and unchanged fitted state."""
        missing = np.isnan(table)
        values = (np.where(missing, self.fill, table) - self.center) / self.spread
        if self.normalizer is not None:
            values = self.normalizer.transform(values)
        if self.quantile_scale is not None:
            values = self.quantile_scale.transform(values)
        values = compress_tails(values, self.lower, self.upper)
        return np.where(missing, np.nan, values).astype(np.float32)


def outlier_bounds(values):
    """Estimate four-sigma bounds after excluding preliminary extreme values.

    These thresholds are part of the fixed preprocessing policy. A second pass
    limits how much extreme training values can inflate the final bounds.
    """
    if np.isnan(values).any():
        lower, upper = np.zeros(values.shape[1]), np.zeros(values.shape[1])
        for index, column in enumerate(values.T):
            observed = column[~np.isnan(column)]
            if observed.size:
                bounds = outlier_bounds(observed[:, None])
                lower[index], upper[index] = bounds[0][0], bounds[1][0]
        return lower, upper
    correction = int(len(values) > 1)
    preliminary = np.maximum(values.std(axis=0, ddof=correction), 1e-6)
    mean = values.mean(axis=0)
    central = np.where(np.abs(values - mean) > 4 * preliminary, np.nan, values)
    center = np.nan_to_num(np.nanmean(central, axis=0), nan=0)
    spread = np.maximum(np.nan_to_num(np.nanstd(central, axis=0, ddof=correction), nan=1), 1e-6)
    return center - 4 * spread, center + 4 * spread


def compress_tails(values, lower, upper):
    """Leave central values unchanged and smoothly compress both tails with arcsinh."""
    values = np.where(values < lower, lower - np.arcsinh(lower - values), values)
    return np.where(values > upper, upper + np.arcsinh(values - upper), values)

"""Interpolate native quantiles and extrapolate exponential tails."""

import numpy as np


def validate_quantiles(quantiles) -> np.ndarray:
    """Require a nonempty vector of finite probabilities strictly between zero and one."""
    levels = np.asarray(quantiles)
    if levels.ndim != 1 or not levels.size or levels.dtype.kind not in 'fiu':
        raise ValueError('quantiles must be a nonempty one-dimensional numeric sequence')
    levels = levels.astype(np.float64)
    if not np.isfinite(levels).all() or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError('quantiles must be finite probabilities strictly between 0 and 1')
    return levels


def interpolate_quantiles(raw: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Preserve requested order; interpolate centrally and use log probability in the tails."""
    q = np.asarray(raw, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 2:
        raise ValueError('At least two native quantiles per row are required')
    native = np.arange(1, q.shape[1] + 1, dtype=np.float64) / (q.shape[1] + 1)
    left = levels < native[0]
    right = levels > native[-1]
    middle = ~(left | right)
    result = np.empty((len(q), len(levels)), dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        upper = np.searchsorted(native, levels[middle]).clip(1, len(native) - 1)
        lower = upper - 1
        weight = (levels[middle] - native[lower]) / (native[upper] - native[lower])
        result[:, middle] = q[:, lower] * (1 - weight) + q[:, upper] * weight
        if left.any():
            scale = (q[:, 1] - q[:, 0]) / np.log(native[1] / native[0])
            result[:, left] = q[:, :1] + scale[:, None] * np.log(levels[left] / native[0])
        if right.any():
            scale = (q[:, -1] - q[:, -2]) / np.log((1 - native[-2]) / (1 - native[-1]))
            result[:, right] = q[:, -1:] - scale[:, None] * np.log((1 - levels[right]) / (1 - native[-1]))
    if not np.isfinite(result).all():
        raise ValueError('Quantile interpolation or extrapolation produced nonfinite target values')
    return result

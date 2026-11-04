"""Shared scalar-temperature fitting.

The scalar-T NLL minimisation already existed twice in TabTune - in
:func:`tabtune.distillation.strategies._temperature_calibration` for distilled
students and in :class:`tabtune.ensemble.strategies.TemperatureScaledBlending`
for ensemble members. :class:`tabtune.uncertainty.Recalibrator` would have been
the third copy, so the math lives here instead.

The distillation implementation is the reference: this module reproduces it
exactly - log-probability "logits", the same clipping constants, the same
bounded scalar search - so a recalibrated pipeline and a calibrated distilled
student agree to the last digit. Those two call sites still hold their own
copies; pointing them here is a safe follow-up that changes no behaviour, and
:func:`tabtune.uncertainty._scaling.fit_temperature` is the intended home when
that happens.

The fit itself: given predicted probabilities ``p`` and integer targets, find
the scalar ``T`` minimising the negative log-likelihood of
``softmax(log(p) / T)``. ``T > 1`` softens over-confident predictions,
``T < 1`` sharpens under-confident ones, and ``T = 1`` is the identity - a
single degree of freedom, so accuracy and the argmax ranking are untouched.

.. versionadded:: 0.2.0
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["apply_temperature", "fit_temperature"]

#: Probabilities are clipped here before taking logs, matching the historical
#: distillation implementation bit-for-bit.
_EPS = 1e-10

#: Search bounds for the scalar temperature. Wide enough for any realistic
#: miscalibration; a fit that lands on a bound is logged.
_BOUNDS = (0.1, 10.0)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale a probability matrix by a scalar temperature.

    Computes ``softmax(log(p) / T)`` row-wise. The argmax of every row is
    preserved for any ``T > 0``, so this changes confidence, never predictions.

    Args:
        probabilities: Predicted probabilities, shape ``(n, n_classes)``.
        temperature: Scalar ``T > 0``. Values above 1 soften the distribution.

    Returns:
        Rescaled probabilities of the same shape, rows summing to 1.

    Raises:
        ValueError: If ``temperature`` is not strictly positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    proba = np.asarray(probabilities, dtype=float)
    scaled = np.exp(np.log(np.clip(proba, _EPS, 1.0)) / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(
    probabilities: np.ndarray,
    y_index: np.ndarray,
    *,
    bounds: tuple[float, float] = _BOUNDS,
) -> tuple[float, float]:
    """Fit the scalar temperature minimising NLL on held-out predictions.

    Args:
        probabilities: Predicted probabilities on the calibration split, shape
            ``(n, n_classes)``.
        y_index: Integer class indices in ``[0, n_classes)`` aligned with the
            probability columns, shape ``(n,)``.
        bounds: Search interval for ``T``.

    Returns:
        ``(temperature, nll)``: the optimal temperature and the negative
        log-likelihood it achieves.

    Raises:
        ValueError: On empty input or out-of-range ``y_index``.
    """
    from scipy.optimize import minimize_scalar

    proba = np.asarray(probabilities, dtype=float)
    index = np.asarray(y_index, dtype=int)
    if proba.ndim != 2 or proba.shape[0] == 0:
        raise ValueError(
            f"probabilities must be a non-empty (n, n_classes) matrix, got shape {proba.shape}"
        )
    if index.shape[0] != proba.shape[0]:
        raise ValueError(
            f"y_index has {index.shape[0]} rows but probabilities has {proba.shape[0]}"
        )
    if index.min() < 0 or index.max() >= proba.shape[1]:
        raise ValueError(
            f"y_index values must lie in [0, {proba.shape[1]}), got "
            f"[{index.min()}, {index.max()}]"
        )

    rows = np.arange(len(index))

    def nll(temperature: float) -> float:
        scaled = np.exp(np.log(np.clip(proba, _EPS, 1.0)) / temperature)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        picked = np.clip(scaled[rows, index], _EPS, 1.0)
        return -float(np.mean(np.log(picked)))

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    temperature = float(result.x)
    if np.isclose(temperature, bounds[0]) or np.isclose(temperature, bounds[1]):
        logger.debug(
            "[Temperature] Fit landed on the search bound T=%.3f; the model may "
            "be extremely miscalibrated.",
            temperature,
        )
    return temperature, float(result.fun)

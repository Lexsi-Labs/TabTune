"""Post-hoc probability recalibration for any fitted pipeline.

TabTune could *measure* miscalibration (``evaluate_calibration`` reports
ECE/MCE/Brier) but, before 0.2.0, could only *fix* it on distilled students and
inside ensembles. :class:`Recalibrator` closes that gap for a plain pipeline:
it wraps a fitted classifier, learns a correction on a held-out split, and then
behaves exactly like the pipeline it wraps - ``predict_proba``, ``predict`` and
``classes_`` all work - so it drops into anything that accepts a pipeline,
including :class:`~tabtune.uncertainty.ConformalClassifier`.

Both corrections leave the argmax ranking of temperature-scaled probabilities
untouched and neither retrains the model; they reshape confidence, not
decisions.

.. versionadded:: 0.2.0
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._scaling import apply_temperature, fit_temperature
from .conformal import _label_indices, _proba_matrix, _resolve_classes

logger = logging.getLogger(__name__)

__all__ = ["Recalibrator"]


class Recalibrator:
    """Learn a post-hoc probability correction on a held-out split.

    Two methods:

    * ``'temperature'``: a single scalar ``T`` minimising NLL on the
      calibration split, applied as ``softmax(log(p) / T)``. One parameter, so
      it cannot overfit a small split, and it preserves every prediction; it
      can only fix *global* over- or under-confidence. This is the same fit
      used by :func:`tabtune.distillation.strategies.calibrate_student`,
      via the shared :mod:`tabtune.uncertainty._scaling`.
    * ``'isotonic'``: per-class one-vs-rest isotonic regression followed by
      row renormalisation, matching the distillation implementation. Each
      class's mapping is monotone (so per-class ranking is preserved), and it
      can fix shape as well as scale - at the cost of needing more calibration
      data to avoid overfitting the step function.

    The wrapper is duck-type compatible with the pipeline it wraps, so it
    composes: ``ConformalClassifier(Recalibrator(pipe).fit(X_cal, y_cal))``
    conformalizes the *recalibrated* probabilities. Use disjoint splits for the
    two fits - reusing one split makes the conformal threshold optimistic.

    Args:
        pipeline: Fitted object with ``predict_proba`` - a
            :class:`~tabtune.TabularPipeline.pipeline.TabularPipeline` or any
            scikit-learn classifier.
        method: ``'temperature'`` or ``'isotonic'``.

    Attributes:
        temperature_: The fitted scalar (temperature method only).
        calibrators_: Per-class isotonic regressors (isotonic method only).
        classes_: Class labels in probability-column order, set by :meth:`fit`.

    Example:
        >>> recal = Recalibrator(pipeline).fit(X_cal, y_cal)     # doctest: +SKIP
        >>> recal.predict_proba(X_test)                          # doctest: +SKIP
        >>> recal.temperature_                                   # doctest: +SKIP
        2.31

    .. versionadded:: 0.2.0
    """

    _METHODS = ("temperature", "isotonic")

    def __init__(self, pipeline: Any, method: str = "temperature") -> None:
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown recalibration method {method!r}. Supported: {list(self._METHODS)}."
            )
        if not hasattr(pipeline, "predict_proba"):
            raise TypeError(
                f"Recalibrator needs a fitted object with predict_proba; "
                f"{type(pipeline).__name__} has none."
            )
        self.pipeline = pipeline
        self.method = method
        self.temperature_: float | None = None
        self.calibrators_: list[Any] | None = None
        self._classes: np.ndarray | None = None

    # -------------------------------------------------------------- fitting

    def fit(self, X_cal: Any, y_cal: Any) -> Recalibrator:
        """Learn the correction on a held-out calibration split.

        The split must be disjoint from the pipeline's training data: a model
        is usually far better calibrated on data it has memorised, so fitting
        the correction there under-corrects everywhere else.

        Args:
            X_cal: Calibration features.
            y_cal: Calibration labels in the original label space.

        Returns:
            ``self``, fitted.
        """
        from .conformal import _reject_training_frame

        _reject_training_frame(self.pipeline, X_cal, caller="Recalibrator.fit")
        proba = _proba_matrix(self.pipeline, X_cal)
        classes = _resolve_classes(self.pipeline)
        if classes is None:
            classes = np.unique(np.asarray(getattr(y_cal, "values", y_cal)))
            logger.warning(
                "[Recalibrator] The pipeline exposes no classes_; assuming "
                "probability columns follow sorted-unique calibration labels %s.",
                classes.tolist(),
            )
        if proba.shape[1] != len(classes):
            raise ValueError(
                f"predict_proba returned {proba.shape[1]} columns but the "
                f"pipeline reports {len(classes)} classes ({classes.tolist()})."
            )
        y_index = _label_indices(y_cal, classes)

        if self.method == "temperature":
            self.temperature_, nll = fit_temperature(proba, y_index)
            logger.info(
                "[Recalibrator] Fitted temperature T=%.3f (NLL=%.4f, n_cal=%d)",
                self.temperature_,
                nll,
                len(y_index),
            )
        else:
            from sklearn.isotonic import IsotonicRegression

            calibrators = []
            for column in range(proba.shape[1]):
                target = (y_index == column).astype(float)
                isotonic = IsotonicRegression(out_of_bounds="clip")
                isotonic.fit(proba[:, column], target)
                calibrators.append(isotonic)
            self.calibrators_ = calibrators
            logger.info(
                "[Recalibrator] Fitted isotonic calibration for %d classes (n_cal=%d)",
                proba.shape[1],
                len(y_index),
            )

        self._classes = classes
        return self

    def _require_fitted(self) -> None:
        if self._classes is None:
            raise RuntimeError(
                "This Recalibrator has not been fitted. Call fit(X_cal, y_cal) "
                "on a held-out split first."
            )

    # ----------------------------------------------------------- prediction

    @property
    def classes_(self) -> np.ndarray:
        """Class labels ordering the probability columns."""
        if self._classes is None:
            raise AttributeError(
                "Recalibrator has no classes_ until fit() has been called."
            )
        return self._classes

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return recalibrated probabilities, columns ordered as :attr:`classes_`.

        Args:
            X: Features to predict on.

        Returns:
            Array of shape ``(n, n_classes)``, rows summing to 1.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted()
        proba = _proba_matrix(self.pipeline, X)
        if self.method == "temperature":
            return apply_temperature(proba, self.temperature_)

        calibrated = np.column_stack(
            [
                calibrator.predict(proba[:, column])
                for column, calibrator in enumerate(self.calibrators_)
            ]
        )
        row_sums = calibrated.sum(axis=1, keepdims=True)
        # A row can renormalise to zero when every one-vs-rest mapping outputs
        # zero at that confidence; fall back to uniform rather than divide by 0.
        degenerate = row_sums[:, 0] < 1e-10
        if np.any(degenerate):
            calibrated[degenerate] = 1.0 / calibrated.shape[1]
            row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / row_sums

    def predict(self, X: Any) -> np.ndarray:
        """Return argmax labels in the original label space.

        For the temperature method these are identical to the wrapped
        pipeline's predictions (scaling preserves the argmax); isotonic can
        change them, since each class is remapped independently.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

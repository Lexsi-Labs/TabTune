"""Uncertainty quantification: conformal prediction and post-hoc recalibration.

Tabular foundation models win benchmarks on accuracy and lose them on
uncertainty: their probabilities are systematically over-confident out of the
box, and their conditional coverage under distribution shift is markedly worse
than a GBDT's. TabTune could already *measure* this (``evaluate_calibration``
reports ECE/MCE/Brier) but not fix it on a plain pipeline. This package adds
the two standard fixes as thin wrappers around any fitted pipeline:
:class:`Recalibrator` reshapes the probabilities, and
:class:`ConformalClassifier` / :class:`ConformalRegressor` wrap predictions in
sets and intervals with a distribution-free guarantee. Wrappers compose around
the pipeline rather than mutating it, so ``save()``/``load()`` and picklability
are untouched, and they work over anything with ``predict_proba``/``predict`` -
a :class:`~tabtune.TabularPipeline.pipeline.TabularPipeline` or a plain
scikit-learn estimator.

Split conformal prediction is the engine. Hold out a calibration split the
model never trained on, score how "nonconforming" each calibration example is
(for classification: how little probability the model gave the true label),
and take the :math:`\\lceil (n+1)(1-\\alpha) \\rceil`-th smallest score as a
threshold. A test point's prediction set is every label whose score clears
that threshold. The guarantee this buys is precise and worth stating honestly:
**marginal** coverage :math:`P(y \\in C(x)) \\ge 1 - \\alpha`, on average over
exchangeable draws of the calibration and test data. It is *not* conditional
coverage - no distribution-free method can promise 90% on every slice of the
input space - and the size-stratified coverage score in
:func:`uncertainty_report` exists precisely to show how far conditional
coverage falls short of the marginal number.

Example:
    >>> from tabtune.uncertainty import ConformalClassifier, Recalibrator
    >>> recal = Recalibrator(pipeline).fit(X_cal1, y_cal1)         # doctest: +SKIP
    >>> conformal = ConformalClassifier(recal, alpha=0.1)          # doctest: +SKIP
    >>> sets = conformal.calibrate(X_cal2, y_cal2).predict_set(X)  # doctest: +SKIP

.. versionadded:: 0.2.0
"""

from __future__ import annotations

from .conformal import ConformalClassifier, ConformalRegressor
from .recalibration import Recalibrator
from .report import size_stratified_coverage, uncertainty_report

__all__ = [
    "ConformalClassifier",
    "ConformalRegressor",
    "Recalibrator",
    "size_stratified_coverage",
    "uncertainty_report",
]

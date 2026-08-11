"""Split conformal prediction over any fitted pipeline.

Split conformal turns point predictions into prediction *sets* (classification)
or *intervals* (regression) with a finite-sample coverage guarantee. It needs
nothing from the model but predictions on a held-out calibration split: a
nonconformity score is computed for every calibration example, and the
:math:`\\lceil (n+1)(1-\\alpha) \\rceil`-th smallest score becomes the threshold
``q_hat_``. At test time, every candidate label (or value) whose score clears
the threshold joins the set.

The guarantee is **marginal** and holds under **exchangeability** of the
calibration and test points: averaged over draws of the calibration and test
data, :math:`P(y \\in C(x)) \\ge 1 - \\alpha`. It is *not* conditional coverage -
the 90% is an average over the whole population, and specific slices (a class,
a region of feature space, the hard examples) can sit well below it. Use the
size-stratified coverage score from :func:`tabtune.uncertainty.uncertainty_report`
to see how far.

Both wrappers duck-type their pipeline argument: anything with
``predict_proba`` (classification) or ``predict`` (regression) works - a fitted
:class:`~tabtune.TabularPipeline.pipeline.TabularPipeline`, a scikit-learn
estimator, or a fitted :class:`~tabtune.uncertainty.Recalibrator`.

.. versionadded:: 0.2.0
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ConformalClassifier", "ConformalRegressor"]


# ------------------------------------------------------------------ helpers


def _check_alpha(alpha: float) -> float:
    """Validate the miscoverage level."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}")
    return float(alpha)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample conformal quantile of calibration scores.

    This is the :math:`k`-th smallest score with
    :math:`k = \\lceil (n+1)(1-\\alpha) \\rceil` - the exact quantile from the
    split-conformal coverage theorem, not an interpolated one. When
    :math:`k > n` (too few calibration points for the requested ``alpha``) the
    threshold is ``+inf`` and prediction sets become maximal; that is the
    honest answer, and it is logged.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("Cannot compute a conformal quantile from zero calibration scores.")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        logger.warning(
            "[Conformal] n_cal=%d is too small for alpha=%.3f "
            "(needs n_cal >= %d); the threshold is +inf and every prediction "
            "set will contain all classes.",
            n,
            alpha,
            math.ceil((1.0 - alpha) / alpha),
        )
        return float("inf")
    return float(np.sort(scores)[k - 1])


def _resolve_classes(pipeline: Any) -> np.ndarray | None:
    """Recover the class labels ordering ``predict_proba``'s columns.

    Tries, in order: a ``classes_`` attribute on the pipeline itself (sklearn
    estimators, :class:`Recalibrator`), the underlying model's ``classes_``
    (``TabularPipeline`` keeps it on ``pipeline.model``), and TabularPipeline's
    own ``_get_model_class_labels`` resolver, which also covers models that
    delegate label encoding to their preprocessor.
    """
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        classes = getattr(getattr(pipeline, "model", None), "classes_", None)
    if classes is None:
        getter = getattr(pipeline, "_get_model_class_labels", None)
        if callable(getter):
            try:
                classes = getter()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("[Conformal] Class-label resolution failed: %s", exc)
                classes = None
    if classes is None:
        return None
    return np.asarray(classes)


def _label_indices(y: Any, classes: np.ndarray) -> np.ndarray:
    """Map original-space labels to probability-column indices.

    Args:
        y: Labels in the original space (strings allowed), shape ``(n,)``.
        classes: Class labels in probability-column order.

    Returns:
        Integer indices into the probability columns, shape ``(n,)``.

    Raises:
        ValueError: If a label does not appear in ``classes``.
    """
    labels = np.asarray(getattr(y, "values", y)).ravel()
    index = {label: i for i, label in enumerate(classes.tolist())}
    try:
        return np.array([index[label] for label in labels.tolist()], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"Label {exc.args[0]!r} does not appear in the pipeline's classes "
            f"{classes.tolist()}. Calibration labels must come from the same "
            f"label space the pipeline was fitted on."
        ) from None


def _proba_matrix(pipeline: Any, X: Any) -> np.ndarray:
    """Fetch probabilities as a float matrix, upcasting a 1-D binary column."""
    proba = np.asarray(pipeline.predict_proba(X), dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    return proba


# ------------------------------------------------------------- classification


def _reject_training_frame(pipeline: Any, X_cal: Any, *, caller: str) -> None:
    """Raise if the calibration frame is byte-identical to the training frame.

    Split conformal's guarantee needs the calibration split to be disjoint from
    the rows the model was fitted on; scoring the training data produces
    optimistically small thresholds and a coverage claim that is simply false.
    A full row-level overlap check would need to hash every row, so this guard
    catches the one mistake that is both common and always wrong: passing the
    training frame itself. ``TabularPipeline`` keeps that frame as
    ``X_raw_train``; the comparison reuses the caching module's content
    fingerprint (values, columns, dtypes, shape, index). Anything short of an
    exact match - a subsample, a shuffle - passes, on the principle that a
    guard must never produce false alarms.
    """
    reference = getattr(pipeline, "X_raw_train", None)
    if reference is None:
        return
    shape = getattr(reference, "shape", None)
    if shape is None or shape != getattr(X_cal, "shape", None):
        return
    try:
        from ..caching import fingerprint_data
    except ImportError:  # pragma: no cover - caching is a hard dependency
        return
    if fingerprint_data(reference) == fingerprint_data(X_cal):
        raise ValueError(
            f"{caller} received the pipeline's own training frame as the "
            "calibration split. Split conformal requires a held-out split the "
            "model never fitted on; calibrating on training rows yields an "
            "optimistically small threshold and a void coverage guarantee. "
            "Hold out a separate calibration set (e.g. train_test_split the "
            "data three ways) and pass that instead."
        )


class ConformalClassifier:
    """Split conformal prediction sets over any probabilistic classifier.

    Wraps a fitted pipeline - anything exposing ``predict_proba`` - and, after
    :meth:`calibrate` on a held-out split, produces prediction sets whose
    marginal coverage is at least ``1 - alpha`` under exchangeability. The
    wrapper holds no reference into the pipeline's internals and adds no state
    to it, so the pipeline stays picklable and reusable.

    Two scores are implemented:

    * ``'lac'`` (least ambiguous set-valued classifier): score
      :math:`1 - p_{y}`. Produces the smallest sets on average when the model
      ranks well, but a set can be **empty** when the threshold ``q_hat_`` is
      smaller than :math:`1 - \\max_c p_c` for a test point - a confident
      prediction combined with a small calibration quantile. Empty sets are
      returned as-is rather than silently patched, because patching would make
      reported set sizes lie; the marginal guarantee is unaffected.
    * ``'aps'`` (adaptive prediction sets): score is the cumulative probability
      mass of all classes ranked at or above the true class. Sets adapt to
      ambiguity - hard examples get larger sets - at the cost of larger sets on
      average. The randomized tie-breaking of the original paper is **off by
      default** for determinism; with ``randomized=False`` the top-ranked class
      is always included, so sets are never empty. Pass ``randomized=True`` for
      the exact-coverage randomized variant, which may produce empty sets.

    Args:
        pipeline: Fitted object with ``predict_proba``. A
            :class:`~tabtune.TabularPipeline.pipeline.TabularPipeline`, a
            scikit-learn classifier, or a fitted
            :class:`~tabtune.uncertainty.Recalibrator`.
        method: ``'lac'`` or ``'aps'``.
        alpha: Target miscoverage, ``0 < alpha < 1``. ``alpha=0.1`` asks for
            90% coverage.
        randomized: APS only - use the randomized score and set rule.
        random_state: Seed for the APS randomization.

    Attributes:
        classes_: Class labels in probability-column order, set by
            :meth:`calibrate`.
        n_cal_: Number of calibration examples.
        q_hat_: The fitted conformal threshold.

    Example:
        >>> conformal = ConformalClassifier(pipeline, alpha=0.1)   # doctest: +SKIP
        >>> conformal.calibrate(X_cal, y_cal)                      # doctest: +SKIP
        >>> sets = conformal.predict_set(X_test)                   # doctest: +SKIP
        >>> sets.shape                                             # doctest: +SKIP
        (1000, 3)

    .. versionadded:: 0.2.0
    """

    _METHODS = ("lac", "aps")

    def __init__(
        self,
        pipeline: Any,
        method: str = "lac",
        alpha: float = 0.1,
        *,
        randomized: bool = False,
        random_state: int | None = None,
    ) -> None:
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown conformal method {method!r}. Supported: "
                f"{list(self._METHODS)}. RAPS is not implemented; use 'aps' "
                f"for adaptive sets."
            )
        if not hasattr(pipeline, "predict_proba"):
            raise TypeError(
                f"ConformalClassifier needs a fitted object with predict_proba; "
                f"{type(pipeline).__name__} has none. For regression use "
                f"ConformalRegressor."
            )
        self.pipeline = pipeline
        self.method = method
        self.alpha = _check_alpha(alpha)
        self.randomized = bool(randomized)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.classes_: np.ndarray | None = None
        self.n_cal_: int | None = None
        self.q_hat_: float | None = None

    # ------------------------------------------------------------ calibration

    def calibrate(self, X_cal: Any, y_cal: Any) -> ConformalClassifier:
        """Fit the conformal threshold on a held-out calibration split.

        The split must be disjoint from the data the pipeline was fitted on -
        scoring the training data produces optimistically small thresholds and
        voids the guarantee.

        Args:
            X_cal: Calibration features.
            y_cal: Calibration labels in the **original** label space (strings
                allowed).

        Returns:
            ``self``, calibrated.

        Raises:
            ValueError: If labels fall outside the pipeline's classes, or the
                probability matrix width does not match the class count.
        """
        _reject_training_frame(self.pipeline, X_cal, caller="ConformalClassifier.calibrate")
        proba = _proba_matrix(self.pipeline, X_cal)
        classes = _resolve_classes(self.pipeline)
        if classes is None:
            classes = np.unique(np.asarray(getattr(y_cal, "values", y_cal)))
            logger.warning(
                "[Conformal] The pipeline exposes no classes_; assuming "
                "probability columns follow sorted-unique calibration labels %s.",
                classes.tolist(),
            )
        if proba.shape[1] != len(classes):
            raise ValueError(
                f"predict_proba returned {proba.shape[1]} columns but the "
                f"pipeline reports {len(classes)} classes ({classes.tolist()}); "
                f"cannot map labels to columns."
            )

        y_index = _label_indices(y_cal, classes)
        scores = self._scores(proba, y_index)

        self.classes_ = classes
        self.n_cal_ = int(len(scores))
        self.q_hat_ = _conformal_quantile(scores, self.alpha)
        logger.info(
            "[Conformal] Calibrated %s on n_cal=%d: q_hat=%.4f (alpha=%.3f)",
            self.method,
            self.n_cal_,
            self.q_hat_,
            self.alpha,
        )
        return self

    def _scores(self, proba: np.ndarray, y_index: np.ndarray) -> np.ndarray:
        """Nonconformity score of the true label for every row."""
        rows = np.arange(proba.shape[0])
        if self.method == "lac":
            return 1.0 - proba[rows, y_index]
        # APS: cumulative probability of all classes ranked at or above the
        # true class (including it), optionally minus u * p_true.
        order = np.argsort(-proba, axis=1)
        sorted_proba = np.take_along_axis(proba, order, axis=1)
        cumulative = np.cumsum(sorted_proba, axis=1)
        rank_of = np.empty_like(order)
        np.put_along_axis(rank_of, order, np.arange(proba.shape[1])[None, :], axis=1)
        true_rank = rank_of[rows, y_index]
        scores = cumulative[rows, true_rank]
        if self.randomized:
            u = self._rng.uniform(size=proba.shape[0])
            scores = scores - u * proba[rows, y_index]
        return scores

    # ------------------------------------------------------------- prediction

    def _require_calibrated(self) -> None:
        if self.q_hat_ is None:
            raise RuntimeError(
                "This ConformalClassifier has not been calibrated. Call "
                "calibrate(X_cal, y_cal) on a held-out split first."
            )

    def predict_set(self, X: Any) -> np.ndarray:
        """Return boolean membership sets of shape ``(n, n_classes)``.

        Columns follow :attr:`classes_`. ``sets[i, j]`` is ``True`` when class
        ``classes_[j]`` is in the prediction set for row ``i``.

        Args:
            X: Features to predict on.

        Returns:
            Boolean array ``(n, n_classes)``.

        Raises:
            RuntimeError: If called before :meth:`calibrate`.
        """
        self._require_calibrated()
        proba = _proba_matrix(self.pipeline, X)
        if proba.shape[1] != len(self.classes_):
            raise ValueError(
                f"predict_proba returned {proba.shape[1]} columns but "
                f"calibration saw {len(self.classes_)}."
            )
        if self.method == "lac":
            # score(x, y) = 1 - p_y <= q_hat  <=>  p_y >= 1 - q_hat
            return proba >= (1.0 - self.q_hat_)

        order = np.argsort(-proba, axis=1)
        sorted_proba = np.take_along_axis(proba, order, axis=1)
        cumulative = np.cumsum(sorted_proba, axis=1)
        if self.randomized:
            u = self._rng.uniform(size=(proba.shape[0], 1))
            include_sorted = (cumulative - u * sorted_proba) <= self.q_hat_
        else:
            include_sorted = cumulative <= self.q_hat_
            # Deterministic APS always keeps the top-ranked class: enlarging a
            # set can only raise coverage, and it keeps sets non-empty.
            include_sorted[:, 0] = True
        sets = np.empty_like(include_sorted)
        np.put_along_axis(sets, order, include_sorted, axis=1)
        return sets

    def set_sizes(self, X: Any) -> np.ndarray:
        """Return the prediction-set size for every row of ``X``."""
        return self.predict_set(X).sum(axis=1)

    def coverage(self, X: Any, y: Any) -> dict[str, float]:
        """Measure empirical coverage and mean set size on labelled data.

        Args:
            X: Features.
            y: True labels in the original label space.

        Returns:
            ``{'coverage', 'avg_set_size', 'n'}``.
        """
        self._require_calibrated()
        sets = self.predict_set(X)
        y_index = _label_indices(y, self.classes_)
        covered = sets[np.arange(len(y_index)), y_index]
        return {
            "coverage": float(np.mean(covered)),
            "avg_set_size": float(np.mean(sets.sum(axis=1))),
            "n": int(len(y_index)),
        }


# ----------------------------------------------------------------- regression


class ConformalRegressor:
    """Split conformal prediction intervals over any regressor.

    Two methods:

    * ``'absolute'``: score :math:`|y - \\hat{y}|` around the point prediction,
      giving symmetric intervals of constant width :math:`2 \\hat{q}`. Works
      for **any** regressor with ``predict``.
    * ``'cqr'`` (conformalized quantile regression): scores measure how far the
      truth escapes the model's own ``[alpha/2, 1 - alpha/2]`` quantile band,
      and the band is widened (or tightened) by the conformal quantile. Interval
      widths adapt to the input, but the pipeline must expose
      ``predict_quantiles`` - among TabTune models only the TabPFN regressor
      family does today (see
      :meth:`~tabtune.TabularPipeline.pipeline.TabularPipeline.predict_quantiles`).

    Args:
        pipeline: Fitted regressor. ``'absolute'`` needs ``predict``; ``'cqr'``
            additionally needs ``predict_quantiles(X, quantiles=[...])``
            returning a mapping from quantile to array.
        method: ``'absolute'`` or ``'cqr'``.
        alpha: Target miscoverage, ``0 < alpha < 1``.

    Attributes:
        n_cal_: Number of calibration examples.
        q_hat_: The fitted conformal threshold.

    .. versionadded:: 0.2.0
    """

    _METHODS = ("absolute", "cqr")

    def __init__(self, pipeline: Any, method: str = "absolute", alpha: float = 0.1) -> None:
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown conformal method {method!r}. Supported: {list(self._METHODS)}."
            )
        if not hasattr(pipeline, "predict"):
            raise TypeError(
                f"ConformalRegressor needs a fitted object with predict; "
                f"{type(pipeline).__name__} has none."
            )
        if method == "cqr" and not hasattr(pipeline, "predict_quantiles"):
            raise ValueError(
                f"method='cqr' needs a pipeline exposing predict_quantiles, and "
                f"{type(pipeline).__name__} does not. In TabTune only the "
                f"TabPFN regressor family supports native quantiles; for any "
                f"other model use method='absolute', which needs only predict()."
            )
        self.pipeline = pipeline
        self.method = method
        self.alpha = _check_alpha(alpha)
        self.n_cal_: int | None = None
        self.q_hat_: float | None = None

    def _quantile_band(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Fetch the model's own lower/upper quantile predictions."""
        lo_q, hi_q = self.alpha / 2.0, 1.0 - self.alpha / 2.0
        band = self.pipeline.predict_quantiles(X, quantiles=[lo_q, hi_q])
        return np.asarray(band[lo_q], dtype=float).ravel(), np.asarray(
            band[hi_q], dtype=float
        ).ravel()

    def calibrate(self, X_cal: Any, y_cal: Any) -> ConformalRegressor:
        """Fit the conformal threshold on a held-out calibration split.

        Args:
            X_cal: Calibration features.
            y_cal: Calibration targets.

        Returns:
            ``self``, calibrated.
        """
        _reject_training_frame(self.pipeline, X_cal, caller="ConformalRegressor.calibrate")
        y = np.asarray(getattr(y_cal, "values", y_cal), dtype=float).ravel()
        if self.method == "absolute":
            predictions = np.asarray(self.pipeline.predict(X_cal), dtype=float).ravel()
            scores = np.abs(y - predictions)
        else:
            lower, upper = self._quantile_band(X_cal)
            scores = np.maximum(lower - y, y - upper)
        self.n_cal_ = int(len(scores))
        self.q_hat_ = _conformal_quantile(scores, self.alpha)
        logger.info(
            "[Conformal] Calibrated %s on n_cal=%d: q_hat=%.4f (alpha=%.3f)",
            self.method,
            self.n_cal_,
            self.q_hat_,
            self.alpha,
        )
        return self

    def _require_calibrated(self) -> None:
        if self.q_hat_ is None:
            raise RuntimeError(
                "This ConformalRegressor has not been calibrated. Call "
                "calibrate(X_cal, y_cal) on a held-out split first."
            )

    def predict_interval(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` interval bounds for every row of ``X``.

        Args:
            X: Features to predict on.

        Returns:
            Two arrays of shape ``(n,)``. Marginal coverage of
            ``[lower, upper]`` is at least ``1 - alpha`` under exchangeability.

        Raises:
            RuntimeError: If called before :meth:`calibrate`.
        """
        self._require_calibrated()
        if self.method == "absolute":
            predictions = np.asarray(self.pipeline.predict(X), dtype=float).ravel()
            return predictions - self.q_hat_, predictions + self.q_hat_
        lower, upper = self._quantile_band(X)
        return lower - self.q_hat_, upper + self.q_hat_

    def coverage(self, X: Any, y: Any) -> dict[str, float]:
        """Measure empirical coverage and mean interval width on labelled data.

        Args:
            X: Features.
            y: True targets.

        Returns:
            ``{'coverage', 'avg_width', 'n'}``.
        """
        self._require_calibrated()
        truth = np.asarray(getattr(y, "values", y), dtype=float).ravel()
        lower, upper = self.predict_interval(X)
        inside = (truth >= lower) & (truth <= upper)
        return {
            "coverage": float(np.mean(inside)),
            "avg_width": float(np.mean(upper - lower)),
            "n": int(len(truth)),
        }

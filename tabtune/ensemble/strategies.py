"""
TabTune Ensemble Strategies
============================


Strategies
----------
1. **WeightedAveraging** — probability/prediction averaging with uniform,
   performance-based, or inverse-error weights.
2. **GreedyEnsembleSelection** — iterative forward selection with
   replacement .
3. **StackingEnsemble** — meta-learner (LR / Ridge / GBDT / MLP) trained
   on cross-validated out-of-fold predictions .
4. **TemperatureScaledBlending** — per-model temperature calibration
    followed by weighted combination.
5. **CascadeStackingEnsemble** — multi-level cascade stacking with skip
   connections and final GES aggregation .
6. **RandomInitEnsemble** — deep ensembles via multiple random seeds
   per model .

References
----------
- Tanna, A. et al. (2025). *TabTune.* arXiv:2511.02802.
- Caruana, R. et al. (2004). *Ensemble Selection from Libraries of Models.* ICML.
- Wolpert, D. H. (1992). *Stacked Generalization.* Neural Networks 5, 241-259.
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Erickson, N. et al. (2020). *AutoGluon-Tabular.* arXiv:2003.06505.
- Lakshminarayanan, B. et al. (2017). *Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles.* NeurIPS.

"""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax as sp_softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)


# ======================================================================
# Utility helpers
# ======================================================================

_CLS_METRICS: Dict[str, Tuple[Any, bool]] = {
    "accuracy": (accuracy_score, True),
    "log_loss": (log_loss, False),
    "f1_score": (lambda y, p: f1_score(y, p, average="weighted"), True),
}

_REG_METRICS: Dict[str, Tuple[Any, bool]] = {
    "mse": (mean_squared_error, False),
    "rmse": (lambda y, p: float(np.sqrt(mean_squared_error(y, p))), False),
    "r2": (r2_score, True),
    "mae": (mean_absolute_error, False),
}


def _get_metric_fn(
    metric_name: str, task_type: str = "classification"
) -> Tuple[Any, bool]:
    """Return ``(metric_fn, higher_is_better)`` for *metric_name*.

    Parameters
    ----------
    metric_name : str
        One of ``accuracy``, ``log_loss``, ``f1_score`` (classification) or
        ``mse``, ``rmse``, ``r2``, ``mae`` (regression).
    task_type : str
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    tuple
        ``(callable, bool)`` where *callable* is ``fn(y_true, y_pred)`` and
        *bool* indicates whether higher values are better.

    Raises
    ------
    ValueError
        If *metric_name* is not recognised for the given *task_type*.
    """
    metrics = _CLS_METRICS if task_type == "classification" else _REG_METRICS
    if metric_name not in metrics:
        raise ValueError(
            f"Unknown metric '{metric_name}' for task_type='{task_type}'. "
            f"Available: {list(metrics.keys())}"
        )
    return metrics[metric_name]


def _probas_to_preds(probas: np.ndarray) -> np.ndarray:
    """Convert a ``(n, k)`` probability matrix to ``(n,)`` class predictions."""
    if probas.ndim != 2:
        raise ValueError(
            f"Expected 2-D probability array, got shape {probas.shape}."
        )
    return np.argmax(probas, axis=1)


def _validate_model_outputs(
    model_outputs: Dict[str, np.ndarray],
    *,
    min_models: int = 1,
    context: str = "",
) -> None:
    """Sanity-check *model_outputs* dict.

    Raises
    ------
    ValueError
        If the dict is empty, has < *min_models* entries, or contains
        arrays with inconsistent first-dimension sizes.
    """
    tag = f"[{context}] " if context else ""
    if not model_outputs:
        raise ValueError(f"{tag}model_outputs is empty -- nothing to ensemble.")
    if len(model_outputs) < min_models:
        raise ValueError(
            f"{tag}Expected at least {min_models} models, "
            f"got {len(model_outputs)}."
        )
    sizes = {name: arr.shape[0] for name, arr in model_outputs.items()}
    unique_sizes = set(sizes.values())
    if len(unique_sizes) > 1:
        raise ValueError(
            f"{tag}Inconsistent sample counts across model outputs: {sizes}"
        )


def _to_numpy(y: Any) -> np.ndarray:
    """Coerce *y* (Series / list / array) to a 1-D numpy array."""
    if hasattr(y, "values"):
        return np.asarray(y.values)
    return np.asarray(y)


# ======================================================================
# Strategy 1: Weighted Averaging
# ======================================================================

class WeightedAveraging:
    """Combine model outputs via weighted averaging.

    Supports three weighting schemes:

    * **uniform** -- equal weight for every model (``weight_scheme="uniform"``).
    * **performance** -- weight proportional to validation score
      (``weight_scheme="performance"``, default when *weights* is ``None``).
    * **inverse_error** -- weight proportional to ``1 / (1 - score + eps)``
      (``weight_scheme="inverse_error"``).
    * **manual** -- supply an explicit *weights* dict in :meth:`fit`.

    For classification the strategy averages probability matrices; for
    regression it averages point predictions.

    Parameters
    ----------
    task_type : str
        ``"classification"`` or ``"regression"``.
    weight_scheme : str
        ``"uniform"``, ``"performance"`` (default), or ``"inverse_error"``.

    Attributes
    ----------
    weights_ : dict[str, float] | None
        Normalised per-model weights (available after :meth:`fit`).
    model_names_ : list[str] | None
        Model identifiers in insertion order.
    """

    def __init__(
        self,
        task_type: str = "classification",
        weight_scheme: str = "performance",
    ) -> None:
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', "
                f"got '{task_type}'."
            )
        if weight_scheme not in ("uniform", "performance", "inverse_error"):
            raise ValueError(
                f"weight_scheme must be 'uniform', 'performance', or "
                f"'inverse_error', got '{weight_scheme}'."
            )
        self.task_type = task_type
        self.weight_scheme = weight_scheme
        self.weights_: Optional[Dict[str, float]] = None
        self.model_names_: Optional[List[str]] = None

    def fit(
        self,
        model_outputs: Dict[str, np.ndarray],
        y_val: np.ndarray,
        metric: str = "accuracy",
        weights: Optional[Dict[str, float]] = None,
    ) -> "WeightedAveraging":
        """Learn per-model weights from validation performance.

        Parameters
        ----------
        model_outputs : dict[str, ndarray]
            ``{model_name: predictions_or_probabilities}`` on the validation
            set.  For classification with probability outputs each value
            should be ``(n, k)``; for regression ``(n,)``.
        y_val : array-like of shape ``(n,)``
            True validation labels / targets.
        metric : str
            Metric for computing performance-based weights.
        weights : dict[str, float] or None
            If provided, use directly (will be normalised to sum to 1).

        Returns
        -------
        self
        """
        _validate_model_outputs(model_outputs, context="WeightedAveraging.fit")
        self.model_names_ = list(model_outputs.keys())
        y_val = _to_numpy(y_val)

        # Manual weights
        if weights is not None:
            missing = set(self.model_names_) - set(weights.keys())
            if missing:
                raise ValueError(
                    f"[WeightedAveraging] Manual weights missing for: {missing}"
                )
            total = sum(weights[k] for k in self.model_names_)
            if total <= 0:
                raise ValueError(
                    "[WeightedAveraging] Sum of manual weights must be > 0."
                )
            self.weights_ = {k: weights[k] / total for k in self.model_names_}
            logger.info("[WeightedAveraging] Using manual weights: %s", self.weights_)
            return self

        # Uniform weights
        if self.weight_scheme == "uniform":
            n = len(self.model_names_)
            self.weights_ = {k: 1.0 / n for k in self.model_names_}
            logger.info("[WeightedAveraging] Uniform weights assigned.")
            return self

        # Performance / inverse-error weights
        metric_fn, higher_is_better = _get_metric_fn(metric, self.task_type)
        scores: Dict[str, float] = {}
        for name, output in model_outputs.items():
            try:
                if self.task_type == "classification" and output.ndim == 2:
                    preds = _probas_to_preds(output)
                    scores[name] = float(metric_fn(y_val, preds))
                else:
                    scores[name] = float(metric_fn(y_val, output))
            except Exception as exc:
                logger.warning(
                    "[WeightedAveraging] Could not score model '%s': %s. "
                    "Assigning zero weight.", name, exc,
                )
                scores[name] = 0.0

        if self.weight_scheme == "inverse_error":
            eps = 1e-6
            if higher_is_better:
                raw = {k: 1.0 / (1.0 - v + eps) for k, v in scores.items()}
            else:
                raw = {k: 1.0 / (v + eps) for k, v in scores.items()}
        else:  # "performance"
            if not higher_is_better:
                max_score = max(scores.values()) + 1e-8
                raw = {k: max_score - v for k, v in scores.items()}
            else:
                raw = dict(scores)

        total = sum(raw.values()) + 1e-10
        self.weights_ = {k: v / total for k, v in raw.items()}
        logger.info(
            "[WeightedAveraging] Weights (%s): %s",
            self.weight_scheme, self.weights_,
        )
        return self

    def predict_proba(self, model_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return weighted average of probability matrices.

        Returns
        -------
        ndarray of shape ``(n, k)``
        """
        if self.weights_ is None:
            raise RuntimeError("Strategy not fitted. Call fit() first.")
        _validate_model_outputs(model_outputs, context="WeightedAveraging.predict_proba")
        combined: Optional[np.ndarray] = None
        for name in self.model_names_:
            if name not in model_outputs:
                logger.warning("[WeightedAveraging] Model '%s' missing at predict time.", name)
                continue
            w = self.weights_.get(name, 0.0)
            contribution = w * model_outputs[name]
            combined = contribution if combined is None else combined + contribution
        if combined is None:
            raise RuntimeError("[WeightedAveraging] No valid model outputs to combine.")
        return combined

    def predict(self, model_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return final predictions (class labels or point estimates)."""
        if self.task_type == "classification":
            return _probas_to_preds(self.predict_proba(model_outputs))
        if self.weights_ is None:
            raise RuntimeError("Strategy not fitted. Call fit() first.")
        combined: Optional[np.ndarray] = None
        for name in self.model_names_:
            if name not in model_outputs:
                continue
            w = self.weights_.get(name, 0.0)
            contribution = w * model_outputs[name]
            combined = contribution if combined is None else combined + contribution
        if combined is None:
            raise RuntimeError("[WeightedAveraging] No model outputs to combine.")
        return combined


# ======================================================================
# Strategy 2: Greedy Ensemble Selection (Caruana et al., ICML 2004)
# ======================================================================

class GreedyEnsembleSelection:
    """Iterative forward ensemble selection with replacement.

    At each iteration the model whose inclusion most improves the ensemble
    is added.  Because selection is *with replacement*, a model can be
    picked multiple times, giving it higher effective weight.

    

    Parameters
    ----------
    ensemble_size : int
        Number of greedy iterations (default 50).
    task_type : str
        ``"classification"`` or ``"regression"``.
    metric : str
        Validation metric to maximise / minimise.
    with_replacement : bool
        If ``True`` (default) a model can be selected more than once.

    Attributes
    ----------
    weights_ : dict[str, float] | None
        Normalised per-model weights derived from selection counts.
    selection_history_ : list[tuple[str, float]]
        ``(model_name, ensemble_score)`` at each iteration.
    """

    def __init__(
        self,
        ensemble_size: int = 50,
        task_type: str = "classification",
        metric: str = "accuracy",
        with_replacement: bool = True,
    ) -> None:
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1.")
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', "
                f"got '{task_type}'."
            )
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.with_replacement = with_replacement
        self.weights_: Optional[Dict[str, float]] = None
        self.model_names_: Optional[List[str]] = None
        self.selection_history_: List[Tuple[str, float]] = []

    def fit(
        self,
        model_outputs: Dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> "GreedyEnsembleSelection":
        """Greedily select models to maximise validation performance.

        Parameters
        ----------
        model_outputs : dict[str, ndarray]
        y_val : array-like of shape ``(n,)``

        Returns
        -------
        self
        """
        _validate_model_outputs(
            model_outputs, min_models=1, context="GreedyEnsembleSelection.fit"
        )
        self.model_names_ = list(model_outputs.keys())
        y_val = _to_numpy(y_val)
        metric_fn, higher_is_better = _get_metric_fn(self.metric, self.task_type)

        selected: List[str] = []
        self.selection_history_ = []

        is_cls_proba = (
            self.task_type == "classification"
            and next(iter(model_outputs.values())).ndim == 2
        )

        # Incremental running sum for O(ensemble_size * n_models) complexity
        running_sum: Optional[np.ndarray] = None
        n_selected = 0

        def _score(avg: np.ndarray) -> float:
            if is_cls_proba:
                return float(metric_fn(y_val, _probas_to_preds(avg)))
            return float(metric_fn(y_val, avg))

        for step in range(self.ensemble_size):
            best_name: Optional[str] = None
            best_score = -np.inf if higher_is_better else np.inf

            for name in self.model_names_:
                if not self.with_replacement and name in selected:
                    continue

                candidate_sum = (
                    model_outputs[name]
                    if running_sum is None
                    else running_sum + model_outputs[name]
                )
                candidate_avg = candidate_sum / (n_selected + 1)

                try:
                    score = _score(candidate_avg)
                except Exception:
                    continue

                improved = (
                    (higher_is_better and score > best_score)
                    or (not higher_is_better and score < best_score)
                )
                if improved:
                    best_score = score
                    best_name = name

            if best_name is None:
                logger.warning(
                    "[GreedyEnsembleSelection] No improving candidate at "
                    "step %d -- stopping early.", step,
                )
                break

            selected.append(best_name)
            if running_sum is None:
                running_sum = model_outputs[best_name].copy()
            else:
                running_sum += model_outputs[best_name]
            n_selected += 1
            self.selection_history_.append((best_name, best_score))

            if (step + 1) % 10 == 0:
                logger.info(
                    "[GreedyEnsembleSelection] Step %d/%d  %s=%.4f  picked=%s",
                    step + 1, self.ensemble_size,
                    self.metric, best_score, best_name,
                )

        # Derive effective weights from selection counts
        counts = Counter(selected)
        total = sum(counts.values())
        self.weights_ = {
            name: counts.get(name, 0) / total for name in self.model_names_
        }
        logger.info("[GreedyEnsembleSelection] Final weights: %s", self.weights_)
        if self.selection_history_:
            logger.info(
                "[GreedyEnsembleSelection] Final score: %.4f",
                self.selection_history_[-1][1],
            )
        return self

    def predict_proba(self, model_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted-average probability predictions."""
        if self.weights_ is None:
            raise RuntimeError("Strategy not fitted. Call fit() first.")
        combined: Optional[np.ndarray] = None
        for name, w in self.weights_.items():
            if w > 0 and name in model_outputs:
                contribution = w * model_outputs[name]
                combined = contribution if combined is None else combined + contribution
        if combined is None:
            raise RuntimeError("[GreedyEnsembleSelection] No outputs to combine.")
        return combined

    def predict(self, model_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return class labels (classification) or point estimates (regression)."""
        if self.task_type == "classification":
            return _probas_to_preds(self.predict_proba(model_outputs))
        combined: Optional[np.ndarray] = None
        for name, w in self.weights_.items():
            if w > 0 and name in model_outputs:
                contribution = w * model_outputs[name]
                combined = contribution if combined is None else combined + contribution
        if combined is None:
            raise RuntimeError("[GreedyEnsembleSelection] No outputs to combine.")
        return combined


# ======================================================================
# Strategy 3: Stacking (Wolpert, 1992)
# ======================================================================

class StackingEnsemble:
    """Meta-learner trained on base-model predictions.

    Base-model predictions are stacked as meta-features and a second-level
    learner is trained to combine them.  When using cross-validated OOF
    (out-of-fold) predictions as input, data leakage is avoided.

    Parameters
    ----------
    meta_learner : str
        ``"lr"`` (LogisticRegression / Ridge), ``"gbdt"`` (LightGBM, falls
        back to sklearn GBT), or ``"mlp"`` (MLPClassifier / MLPRegressor).
    task_type : str
    n_folds : int
        Number of CV folds for OOF generation (used externally by
        :class:`~tabtune.ensemble.TabularEnsemble`).
    use_original_features : bool
        If ``True``, append the raw features alongside meta-features.

    Attributes
    ----------
    meta_model_ : object
        Fitted sklearn-compatible meta-learner.
    model_names_ : list[str]
    """

    def __init__(
        self,
        meta_learner: str = "lr",
        task_type: str = "classification",
        n_folds: int = 5,
        use_original_features: bool = False,
    ) -> None:
        if meta_learner not in ("lr", "gbdt", "mlp"):
            raise ValueError(
                f"meta_learner must be 'lr', 'gbdt', or 'mlp', "
                f"got '{meta_learner}'."
            )
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', "
                f"got '{task_type}'."
            )
        self.meta_learner = meta_learner
        self.task_type = task_type
        self.n_folds = n_folds
        self.use_original_features = use_original_features
        self.meta_model_: Any = None
        self.model_names_: Optional[List[str]] = None

    def _build_meta_model(self, n_features: int) -> Any:
        """Instantiate the meta-learner."""
        if self.task_type == "classification":
            if self.meta_learner == "lr":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(max_iter=1000, C=1.0)
            elif self.meta_learner == "gbdt":
                try:
                    from lightgbm import LGBMClassifier
                    return LGBMClassifier(n_estimators=100, max_depth=3, verbose=-1)
                except ImportError:
                    from sklearn.ensemble import GradientBoostingClassifier
                    logger.info(
                        "[StackingEnsemble] LightGBM unavailable, "
                        "falling back to sklearn GradientBoosting."
                    )
                    return GradientBoostingClassifier(n_estimators=100, max_depth=3)
            else:  # mlp
                from sklearn.neural_network import MLPClassifier
                return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
        else:  # regression
            if self.meta_learner == "lr":
                from sklearn.linear_model import Ridge
                return Ridge(alpha=1.0)
            elif self.meta_learner == "gbdt":
                try:
                    from lightgbm import LGBMRegressor
                    return LGBMRegressor(n_estimators=100, max_depth=3, verbose=-1)
                except ImportError:
                    from sklearn.ensemble import GradientBoostingRegressor
                    logger.info(
                        "[StackingEnsemble] LightGBM unavailable, "
                        "falling back to sklearn GradientBoosting."
                    )
                    return GradientBoostingRegressor(n_estimators=100, max_depth=3)
            else:  # mlp
                from sklearn.neural_network import MLPRegressor
                return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)

    def _build_meta_features(
        self,
        model_outputs: Dict[str, np.ndarray],
        X_original: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Stack base-model predictions into a meta-feature matrix."""
        features: List[np.ndarray] = []
        for name in self.model_names_:
            output = model_outputs[name]
            features.append(output if output.ndim == 2 else output.reshape(-1, 1))
        meta_X = np.hstack(features)

        if self.use_original_features and X_original is not None:
            if hasattr(X_original, "values"):
                X_original = X_original.values
            meta_X = np.hstack([meta_X, np.asarray(X_original, dtype=float)])

        return meta_X

    def fit(
        self,
        model_outputs: Dict[str, np.ndarray],
        y_val: np.ndarray,
        X_original: Optional[np.ndarray] = None,
    ) -> "StackingEnsemble":
        """Fit the stacking meta-learner.

        Parameters
        ----------
        model_outputs : dict[str, ndarray]
            Out-of-fold (or validation) predictions from each base model.
        y_val : array-like
            True labels / targets corresponding to *model_outputs*.
        X_original : ndarray or None
            Original feature matrix (only when ``use_original_features=True``).

        Returns
        -------
        self
        """
        _validate_model_outputs(model_outputs, context="StackingEnsemble.fit")
        self.model_names_ = list(model_outputs.keys())
        y_val = _to_numpy(y_val)
        meta_X = self._build_meta_features(model_outputs, X_original)
        self.meta_model_ = self._build_meta_model(meta_X.shape[1])

        try:
            self.meta_model_.fit(meta_X, y_val)
        except Exception as exc:
            raise RuntimeError(
                f"[StackingEnsemble] Meta-learner fitting failed: {exc}"
            ) from exc

        logger.info(
            "[StackingEnsemble] Meta-learner (%s) trained on %d features "
            "from %d base models.",
            self.meta_learner, meta_X.shape[1], len(self.model_names_),
        )
        return self

    def predict_proba(
        self,
        model_outputs: Dict[str, np.ndarray],
        X_original: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return meta-learner probability predictions."""
        if self.meta_model_ is None:
            raise RuntimeError("Strategy not fitted. Call fit() first.")
        meta_X = self._build_meta_features(model_outputs, X_original)
        if hasattr(self.meta_model_, "predict_proba"):
            return self.meta_model_.predict_proba(meta_X)
        return self.meta_model_.predict(meta_X)

    def predict(
        self,
        model_outputs: Dict[str, np.ndarray],
        X_original: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return meta-learner class labels or point estimates."""
        if self.meta_model_ is None:
            raise RuntimeError("Strategy not fitted. Call fit() first.")
        meta_X = self._build_meta_features(model_outputs, X_original)
        return self.meta_model_.predict(meta_X)


# ======================================================================
# Strategy 4: Temperature-Scaled Blending (Guo et al., ICML 2017)
# ======================================================================

class TemperatureScaledBlending:
    """Per-model temperature calibration followed by weighted combination.

    For each model a single temperature parameter *T* is learned by
    minimising NLL on the validation set:
    ``calibrated_p = softmax(log(p) / T)``.

    After calibration the models are combined with uniform weights.  For
    regression, the strategy falls back to :class:`WeightedAveraging`.

    Parameters
    ----------
    task_type : str

    Attributes
    ----------
    temperatures_ : dict[str, float]
        Learned temperatures per model.
    weights_ : dict[str, float]
        Combination weights (uniform after calibration).
    """

    def __init__(self, task_type: str = "classification") -> None:
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', "
                f"got '{task_type}'."
            )
        self.task_type = task_type
        self.temperatures_: Dict[str, float] = {}
        self.weights_: Optional[Dict[str, float]] = None
        self.model_names_: Optional[List[str]] = None

    @staticmethod
    def _optimize_temperature(probas: np.ndarray, y: np.ndarray) -> float:
        """Find optimal *T* by minimising NLL via L-BFGS-B in log-space.

        Parameters
        ----------
        probas : ndarray of shape ``(n, k)``
        y : ndarray of shape ``(n,)`` (integer class labels)

        Returns
        -------
        float
            Optimal temperature (always positive, clamped to [0.01, 20.0]).
        """
        logits = np.log(np.clip(probas, 1e-10, 1.0))
        y_int = y.astype(int)

        def _nll(log_T: np.ndarray) -> float:
            T = float(np.exp(log_T[0]))
            scaled = sp_softmax(logits / T, axis=1)
            return float(log_loss(y_int, scaled))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(_nll, x0=np.array([0.0]), method="L-BFGS-B")

        T_opt = float(np.exp(result.x[0]))
        return float(np.clip(T_opt, 0.01, 20.0))

    def _apply_temperature(
        self, probas: np.ndarray, temperature: float
    ) -> np.ndarray:
        """Apply temperature scaling to a probability array."""
        if temperature == 1.0 or probas.ndim != 2:
            return probas
        logits = np.log(np.clip(probas, 1e-10, 1.0))
        return sp_softmax(logits / temperature, axis=1)

    def fit(
        self,
        model_outputs: Dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> "TemperatureScaledBlending":
        """Learn per-model temperatures and set uniform combination weights.

        Parameters
        ----------
        model_outputs : dict[str, ndarray]
        y_val : array-like

        Returns
        -------
        self
        """
        _validate_model_outputs(
            model_outputs, context="TemperatureScaledBlending.fit"
        )
        self.model_names_ = list(model_outputs.keys())
        y_val = _to_numpy(y_val)

        if self.task_type != "classification":
            wa = WeightedAveraging(task_type="regression")
            wa.fit(model_outputs, y_val, metric="mse")
            self.weights_ = wa.weights_
            return self

        for name, probas in model_outputs.items():
            if probas.ndim == 2:
                try:
                    t = self._optimize_temperature(probas, y_val)
                except Exception as exc:
                    logger.warning(
                        "[TempScaled] Temperature optimisation failed for "
                        "'%s': %s. Using T=1.0.", name, exc,
                    )
                    t = 1.0
                self.temperatures_[name] = t
                logger.info("[TempScaled] %s: T=%.4f", name, t)
            else:
                self.temperatures_[name] = 1.0

        n = len(self.model_names_)
        self.weights_ = {name: 1.0 / n for name in self.model_names_}
        return self

    def predict_proba(self, model_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine temperature-calibrated probability matrices."""
        if self.weights_ is None:
            raise RuntimeError("Strategy not fitted. Call fit() first.")
        combined: Optional[np.ndarray] = None
        for name in self.model_names_:
            if name not in model_outputs:
                continue
            probas = self._apply_temperature(
                model_outputs[name], self.temperatures_.get(name, 1.0)
            )
            w = self.weights_[name]
            contribution = w * probas
            combined = contribution if combined is None else combined + contribution
        if combined is None:
            raise RuntimeError("[TempScaled] No outputs to combine.")
        return combined

    def predict(self, model_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return class labels (classification) or point estimates (regression)."""
        if self.task_type == "classification":
            return _probas_to_preds(self.predict_proba(model_outputs))
        combined: Optional[np.ndarray] = None
        for name, w in self.weights_.items():
            if w > 0 and name in model_outputs:
                contribution = w * model_outputs[name]
                combined = contribution if combined is None else combined + contribution
        if combined is None:
            raise RuntimeError("[TempScaled] No outputs to combine.")
        return combined


# ======================================================================
# Strategy 5: Cascade Stacking 
# ======================================================================

class CascadeStackingEnsemble:
    """Multi-level cascade stacking with skip connections and final GES.

    At each level, K-fold OOF predictions from all base models are
    concatenated with the original features (skip connection) and fed as
    input to the next stacking level.  After all levels, Greedy Ensemble
    Selection (GES) is applied across every stacker output from every
    level to produce the final combination.

    

    Parameters
    ----------
    n_levels : int
        Number of stacking levels (default 2).
    n_folds : int
        K-fold splits per level for OOF generation.
    ges_size : int
        Number of GES iterations in the final aggregation layer.
    task_type : str
    random_state : int
        Seed for K-fold splitting reproducibility.

    Attributes
    ----------
    weights_ : dict[str, float] | None
        GES-derived weights for the final combination (keyed by
        ``"L{level}_{model_name}"``).
    level_info_ : list[dict]
        Per-level metadata (input dim, output dim, etc.).
    """

    def __init__(
        self,
        n_levels: int = 2,
        n_folds: int = 5,
        ges_size: int = 50,
        task_type: str = "classification",
        random_state: int = 42,
    ) -> None:
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1.")
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2.")
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', "
                f"got '{task_type}'."
            )
        self.n_levels = n_levels
        self.n_folds = n_folds
        self.ges_size = ges_size
        self.task_type = task_type
        self.random_state = random_state
        self.weights_: Optional[Dict[str, float]] = None
        self.model_names_: Optional[List[str]] = None
        self.level_info_: List[Dict[str, Any]] = []
        self._stacker_names: List[str] = []

    @staticmethod
    def _greedy_select_blocks(
        val_blocks: List[np.ndarray],
        test_blocks: List[np.ndarray],
        y_val: np.ndarray,
        ensemble_size: int = 50,
        task_type: str = "classification",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply GES over a list of prediction blocks.

        For classification each block is ``(n, n_classes)`` probabilities.
        For regression each block is ``(n,)`` or ``(n, 1)`` point estimates.

        Returns
        -------
        val_blend : ndarray
        test_blend : ndarray
        weights : ndarray of floats summing to 1
        """
        n_cands = len(val_blocks)
        if n_cands == 0:
            raise ValueError("[CascadeStacking] No candidate blocks to select from.")

        is_reg = task_type == "regression"

        # Flatten regression blocks to 1-D for scoring
        if is_reg:
            val_blocks = [b.ravel() if b.ndim > 1 else b for b in val_blocks]
            test_blocks = [b.ravel() if b.ndim > 1 else b for b in test_blocks]

        counts = np.zeros(n_cands, dtype=int)
        running_sum = np.zeros_like(val_blocks[0], dtype=float)
        n_selected = 0

        for _ in range(ensemble_size):
            best_score, best_k = -np.inf, -1
            for k in range(n_cands):
                candidate = (running_sum + val_blocks[k]) / (n_selected + 1)
                if is_reg:
                    # Negative MSE (higher = better)
                    score = -float(mean_squared_error(y_val, candidate))
                else:
                    score = float(accuracy_score(y_val, candidate.argmax(axis=1)))
                if score > best_score:
                    best_score, best_k = score, k
            if best_k < 0:
                break
            counts[best_k] += 1
            running_sum += val_blocks[best_k]
            n_selected += 1

        total = counts.sum()
        w = counts / total if total > 0 else np.ones(n_cands) / n_cands
        val_blend = sum(w[k] * val_blocks[k] for k in range(n_cands))
        test_blend = sum(w[k] * test_blocks[k] for k in range(n_cands))
        return val_blend, test_blend, w

    def fit(
        self,
        model_outputs_per_level: List[Dict[str, np.ndarray]],
        y_val: np.ndarray,
        model_names: Optional[List[str]] = None,
    ) -> "CascadeStackingEnsemble":
        """Fit the cascade by applying final GES across all level outputs.

        This method expects *pre-computed* stacker outputs for each level
        (typically produced by :class:`TabularEnsemble` during its
        pipeline-level orchestration).

        Parameters
        ----------
        model_outputs_per_level : list[dict[str, ndarray]]
            One dict per level.  Each dict maps stacker names to their
            ``(n_val, n_classes)`` probability arrays on the validation set.
        y_val : array-like
        model_names : list[str] or None
            Base model names (used for labelling).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If no stacker outputs are provided.
        """
        y_val = _to_numpy(y_val)
        self.model_names_ = model_names or []
        self.level_info_ = []
        self._stacker_names = []

        val_blocks: List[np.ndarray] = []

        for level_idx, outputs in enumerate(model_outputs_per_level):
            _validate_model_outputs(
                outputs,
                context=f"CascadeStacking.fit Level {level_idx + 1}",
            )
            for name, arr in outputs.items():
                label = f"L{level_idx + 1}_{name}"
                self._stacker_names.append(label)
                val_blocks.append(arr)
            self.level_info_.append({
                "level": level_idx + 1,
                "n_stackers": len(outputs),
            })

        if not val_blocks:
            raise ValueError("[CascadeStackingEnsemble] No stacker outputs provided.")

        _, _, ges_weights = self._greedy_select_blocks(
            val_blocks, val_blocks, y_val, self.ges_size,
            task_type=self.task_type,
        )

        self.weights_ = {
            self._stacker_names[i]: float(ges_weights[i])
            for i in range(len(self._stacker_names))
        }
        logger.info("[CascadeStacking] GES weights: %s", self.weights_)
        return self

    def predict_proba(
        self, stacker_outputs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Combine stacker outputs using learned GES weights.

        Parameters
        ----------
        stacker_outputs : dict[str, ndarray]
            Keys must match the ``L{level}_{model_name}`` labels produced
            during :meth:`fit`.
        """
        if self.weights_ is None:
            raise RuntimeError("Strategy not fitted. Call fit() first.")
        combined: Optional[np.ndarray] = None
        for name, w in self.weights_.items():
            if w > 0 and name in stacker_outputs:
                contribution = w * stacker_outputs[name]
                combined = contribution if combined is None else combined + contribution
        if combined is None:
            raise RuntimeError("[CascadeStacking] No outputs to combine.")
        return combined

    def predict(self, stacker_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return class labels or point estimates."""
        if self.task_type == "classification":
            return _probas_to_preds(self.predict_proba(stacker_outputs))
        combined: Optional[np.ndarray] = None
        for name, w in self.weights_.items():
            if w > 0 and name in stacker_outputs:
                contribution = w * stacker_outputs[name]
                combined = contribution if combined is None else combined + contribution
        if combined is None:
            raise RuntimeError("[CascadeStacking] No outputs to combine.")
        return combined


# ======================================================================
# Strategy 6: Random-Initialisation Ensemble (Deep Ensembles)
# (Lakshminarayanan et al., NeurIPS 2017)
# ======================================================================

class RandomInitEnsemble:
    """Average multiple runs of the same model(s) with different seeds.

    Each model is trained *M* times with distinct random seeds, producing
    diverse predictions that are then averaged.  Cross-model averaging
    uses performance-based weights.

    This also yields a free epistemic uncertainty estimate: the variance
    across ensemble members per sample.

    Parameters
    ----------
    n_seeds : int
        Number of random seeds per model (default 5).
    base_seed : int
        Starting seed; subsequent seeds are ``base_seed + 1, ...``.
    task_type : str

    Attributes
    ----------
    weights_ : dict[str, float] | None
        Cross-model combination weights.
    uncertainty_ : ndarray or None
        Per-sample epistemic uncertainty from the last :meth:`predict_proba`
        call.
    """

    def __init__(
        self,
        n_seeds: int = 5,
        base_seed: int = 42,
        task_type: str = "classification",
    ) -> None:
        if n_seeds < 1:
            raise ValueError("n_seeds must be >= 1.")
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', "
                f"got '{task_type}'."
            )
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.task_type = task_type
        self.weights_: Optional[Dict[str, float]] = None
        self.model_names_: Optional[List[str]] = None
        self.uncertainty_: Optional[np.ndarray] = None
        self._per_model_avg: Dict[str, np.ndarray] = {}

    def fit(
        self,
        model_outputs_per_seed: Dict[str, List[np.ndarray]],
        y_val: np.ndarray,
    ) -> "RandomInitEnsemble":
        """Compute per-model averages and cross-model weights.

        Parameters
        ----------
        model_outputs_per_seed : dict[str, list[ndarray]]
            ``{model_name: [seed_0_output, seed_1_output, ...]}`` where each
            output is ``(n, k)`` for classification or ``(n,)`` for
            regression.
        y_val : array-like

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If no valid model outputs are provided.
        RuntimeError
            If no models produced valid averages.
        """
        if not model_outputs_per_seed:
            raise ValueError("[RandomInitEnsemble] model_outputs_per_seed is empty.")
        y_val = _to_numpy(y_val)
        self.model_names_ = list(model_outputs_per_seed.keys())
        self._per_model_avg = {}

        val_scores: Dict[str, float] = {}
        for name, seed_outputs in model_outputs_per_seed.items():
            if not seed_outputs:
                logger.warning("[RandomInitEnsemble] No outputs for model '%s'.", name)
                continue
            stacked = np.stack(seed_outputs, axis=0)
            avg = stacked.mean(axis=0)
            self._per_model_avg[name] = avg

            try:
                if self.task_type == "classification" and avg.ndim == 2:
                    val_scores[name] = float(
                        accuracy_score(y_val, _probas_to_preds(avg))
                    )
                else:
                    val_scores[name] = float(r2_score(y_val, avg))
            except Exception as exc:
                logger.warning(
                    "[RandomInitEnsemble] Could not score '%s': %s", name, exc
                )
                val_scores[name] = 0.0

            logger.info(
                "[RandomInitEnsemble] %s: %d seeds, avg_score=%.4f",
                name, len(seed_outputs), val_scores[name],
            )

        active = list(self._per_model_avg.keys())
        if not active:
            raise RuntimeError("[RandomInitEnsemble] No valid model averages.")

        accs_arr = np.array([max(val_scores.get(n, 0.0), 0.0) for n in active])
        total = accs_arr.sum()
        if total <= 0:
            total = float(len(active))
            accs_arr = np.ones(len(active))
        self.weights_ = {n: float(accs_arr[i] / total) for i, n in enumerate(active)}
        logger.info("[RandomInitEnsemble] Cross-model weights: %s", self.weights_)
        return self

    def predict_proba(
        self,
        model_outputs_per_seed: Dict[str, List[np.ndarray]],
    ) -> np.ndarray:
        """Return weighted-average probabilities with uncertainty estimation.

        Side-effect: populates ``self.uncertainty_`` with per-sample
        epistemic uncertainty (mean squared deviation across all seed
        instances).

        Parameters
        ----------
        model_outputs_per_seed : dict[str, list[ndarray]]

        Returns
        -------
        ndarray of shape ``(n, k)``
        """
        if self.weights_ is None:
            raise RuntimeError("Strategy not fitted. Call fit() first.")

        all_instances: List[np.ndarray] = []
        per_model_avg: Dict[str, np.ndarray] = {}

        for name, seed_outputs in model_outputs_per_seed.items():
            if name not in self.weights_ or not seed_outputs:
                continue
            stacked = np.stack(seed_outputs, axis=0)
            per_model_avg[name] = stacked.mean(axis=0)
            all_instances.extend(seed_outputs)

        # Weighted cross-model combination
        combined: Optional[np.ndarray] = None
        for name, w in self.weights_.items():
            if w > 0 and name in per_model_avg:
                contribution = w * per_model_avg[name]
                combined = contribution if combined is None else combined + contribution

        if combined is None:
            raise RuntimeError("[RandomInitEnsemble] No outputs to combine.")

        # Epistemic uncertainty (Lakshminarayanan et al. eq. 6)
        if all_instances:
            try:
                all_stack = np.stack(all_instances, axis=0)
                mean_pred = all_stack.mean(axis=0)
                if all_stack.ndim == 3:
                    # (M, n, k) -> per-sample: mean over seeds and classes
                    self.uncertainty_ = (
                        (all_stack - mean_pred[None, :, :]) ** 2
                    ).mean(axis=(0, 2))
                else:
                    self.uncertainty_ = (
                        (all_stack - mean_pred[None, :]) ** 2
                    ).mean(axis=0)
            except Exception as exc:
                logger.warning(
                    "[RandomInitEnsemble] Could not compute uncertainty: %s", exc
                )
                self.uncertainty_ = None

        return combined

    def predict(
        self,
        model_outputs_per_seed: Dict[str, List[np.ndarray]],
    ) -> np.ndarray:
        """Return class labels or point estimates."""
        probas = self.predict_proba(model_outputs_per_seed)
        if self.task_type == "classification" and probas.ndim == 2:
            return _probas_to_preds(probas)
        return probas


# ======================================================================
# Strategy factory
# ======================================================================

STRATEGY_MAP: Dict[str, type] = {
    "weighted_averaging": WeightedAveraging,
    "greedy_selection": GreedyEnsembleSelection,
    "stacking": StackingEnsemble,
    "temperature_scaled": TemperatureScaledBlending,
    "cascade_stacking": CascadeStackingEnsemble,
    "random_init": RandomInitEnsemble,
}


def get_strategy(name: str, **kwargs: Any) -> Any:
    """Instantiate an ensemble strategy by name.

    Parameters
    ----------
    name : str
        One of the keys in :data:`STRATEGY_MAP`:
        ``weighted_averaging``, ``greedy_selection``, ``stacking``,
        ``temperature_scaled``, ``cascade_stacking``, ``random_init``.
    **kwargs
        Forwarded to the strategy constructor.

    Returns
    -------
    object
        Initialised (but unfitted) strategy instance.

    Raises
    ------
    ValueError
        If *name* is not recognised or *kwargs* are incompatible.
    """
    if name not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(STRATEGY_MAP.keys())}"
        )
    try:
        return STRATEGY_MAP[name](**kwargs)
    except TypeError as exc:
        raise ValueError(
            f"Invalid kwargs for strategy '{name}': {exc}"
        ) from exc
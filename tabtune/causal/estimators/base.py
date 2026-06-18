"""
tabtune.causal.estimators.base
==============================

Abstract base class for all causal estimators in :mod:`tabtune.causal`.

Every concrete estimator (DML, S/T/X/R-Learners, Causal Forest) implements
this interface, which mirrors :class:`tabtune.TabularPipeline` in spirit:

* Construct with explicit configuration.
* ``.fit(X, treatment, outcome)`` fits the nuisance learners and estimates
  the ATE.
* ``.ate()`` returns a dict with ``ate`` / ``std_error`` / ``ci_lower`` /
  ``ci_upper``.
* ``.cate(X_query)`` returns per-row treatment effects (heterogeneous).
* ``.counterfactual(row, intervention)`` answers single-row what-if queries.

Concrete estimators do not have to implement every method -- DML provides
ATE robustly, meta-learners provide CATE, etc. Methods that aren't
implemented should raise :class:`NotImplementedError` with a clear message.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseCausalEstimator(ABC):
    """
    Abstract base class for causal estimators.

    Parameters
    ----------
    treatment : str
        Column name of the treatment variable T.
    outcome : str
        Column name of the outcome variable Y.
    confounders : list of str
        Column names of the confounders X.
    outcome_model : sklearn-compatible estimator
        The fitted-on-demand nuisance learner that predicts Y from X.
        Built from a :class:`tabtune.causal.adapters._TabTuneSklearnAdapter`
        in normal usage.
    treatment_model : sklearn-compatible estimator
        The nuisance learner that predicts the treatment-propensity
        distribution from X.
    estimator_params : dict, optional
        Estimator-specific configuration (n_folds, score function, etc.).
    """

    name: str = "base"

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: list[str],
        outcome_model: Any,
        treatment_model: Any,
        estimator_params: dict | None = None,
    ):
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = list(confounders)
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.estimator_params = estimator_params or {}
        self._fitted = False
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseCausalEstimator":
        """
        Fit the nuisance learners and compute the causal estimate.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe containing the treatment, outcome, and confounder
            columns. Concrete subclasses should not mutate the input.

        Returns
        -------
        self : BaseCausalEstimator
        """
        raise NotImplementedError

    @abstractmethod
    def ate(self, confidence_level: float = 0.95) -> dict:
        """
        Return the Average Treatment Effect with a confidence interval.

        Returns
        -------
        result : dict
            Keys: ``ate``, ``std_error``, ``ci_lower``, ``ci_upper``,
            ``p_value``, ``confidence_level``, ``n_samples``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional interface (default = NotImplementedError)
    # ------------------------------------------------------------------
    def cate(self, X_query: pd.DataFrame, return_ci: bool = False):
        """
        Return per-row Conditional Average Treatment Effects.

        Subclasses that support heterogeneous-effect estimation (X/R-Learner,
        Causal Forest) override this method. The default raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement CATE estimation."
        )

    # ------------------------------------------------------------------
    # Prediction surface
    # ------------------------------------------------------------------
    # IMPORTANT: ``self.outcome_model`` is the *unfitted* adapter handed to
    # DoubleML / EconML. Those libraries clone it internally for cross-
    # fitting, so the original instance is never fitted directly. To answer
    # downstream "what does the model think Y is for these rows?" queries
    # (counterfactual fairness, single-row predictions), we therefore fit a
    # lightweight *surrogate* during ``fit`` on (X, T, Y) and expose it via
    # ``.predict``. The surrogate is identical in shape to the outcome
    # model used as a nuisance learner.
    def _fit_predict_surrogate(self, df: pd.DataFrame) -> None:
        """Fit a self-contained predict-Y model on (X, T) using a clone of
        ``self.outcome_model``. Called by subclasses' ``fit`` methods."""
        from sklearn.base import clone

        try:
            surrogate = clone(self.outcome_model)
        except Exception:
            # If clone fails for any reason, fall back to direct reuse;
            # we will re-fit it below either way.
            surrogate = self.outcome_model
        X_full = df[self.confounders + [self.treatment]]
        y = df[self.outcome]
        try:
            surrogate.fit(X_full, y)
        except Exception as exc:
            logger.warning(
                "[Causal] predict-surrogate fit failed (%s); "
                "predict()/counterfactual()/counterfactual-fairness audits "
                "will be unavailable for this estimator.",
                exc,
            )
            surrogate = None
        self._predict_surrogate = surrogate

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict outcomes for rows under their *observed* treatment.

        Uses a lightweight surrogate fitted at ``.fit`` time. See
        ``_fit_predict_surrogate`` for details. This method exists so the
        sensitive-attribute audits (and any downstream consumer that wants
        a "what does the model think Y is?" answer) can call ``predict``
        on a fitted estimator without depending on the underlying
        DoubleML/EconML library exposing one.
        """
        self._require_fit()
        if getattr(self, "_predict_surrogate", None) is None:
            raise RuntimeError(
                f"{self.__class__.__name__} could not build a predict "
                "surrogate during fit. Check the warning logs from .fit()."
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError("predict() expects a pandas DataFrame.")
        cols = self.confounders + [self.treatment]
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"predict() missing required columns: {missing}"
            )
        return np.asarray(self._predict_surrogate.predict(X[cols]))

    def counterfactual(self, row: pd.DataFrame, intervention: dict) -> dict:
        """
        Predict the counterfactual outcome for a single row under an
        intervention on one or more features.

        Uses the predict surrogate built at ``.fit`` time.
        """
        self._require_fit()
        if not isinstance(row, pd.DataFrame):
            raise TypeError("counterfactual() expects a single-row DataFrame.")
        if getattr(self, "_predict_surrogate", None) is None:
            raise RuntimeError(
                "predict surrogate is not available. See .fit() warnings."
            )
        original_X = row[self.confounders + [self.treatment]].copy()
        modified_X = original_X.copy()
        for col, val in intervention.items():
            if col not in modified_X.columns:
                raise ValueError(
                    f"Intervention column '{col}' not present in row features."
                )
            modified_X[col] = val
        try:
            pred_original = float(
                np.asarray(self._predict_surrogate.predict(original_X))[0]
            )
            pred_modified = float(
                np.asarray(self._predict_surrogate.predict(modified_X))[0]
            )
        except Exception as exc:
            raise RuntimeError(
                "Outcome model failed during counterfactual prediction: "
                f"{exc}"
            ) from exc
        return {
            "original": pred_original,
            "counterfactual": pred_modified,
            "delta": pred_modified - pred_original,
            "intervention": intervention,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be .fit() before this call."
            )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = {self.treatment, self.outcome, *self.confounders}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in input DataFrame: {sorted(missing)}"
            )

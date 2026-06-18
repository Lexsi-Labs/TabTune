"""
tabtune.causal.estimators.meta_learners
=======================================

S-/T-/X-/R-Learner meta-estimators for heterogeneous treatment effect
estimation.

A *meta-learner* turns a generic supervised-learning algorithm into a CATE
estimator by composing one or more sub-models. We expose four variants:

* **S-Learner** -- a single model with treatment as an extra feature.
  Simplest; biased when treatment effect is small relative to outcome
  variance.
* **T-Learner** -- two models, one per treatment arm. Robust when both
  arms have plenty of data; degrades on imbalanced treatments.
* **X-Learner** -- uses propensity-weighted combinations of T-learner
  residuals. Designed for imbalanced treatments.
* **R-Learner** -- residualises both T and Y first (DML-style), then fits
  a single CATE model on the residuals. State-of-the-art for many
  benchmarks.

All four route through EconML's reference implementations
(``econml.metalearners`` for S/T/X; ``econml.dml.NonParamDML`` for R).
Nuisance learners arrive as sklearn-compatible TabTune adapters.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseCausalEstimator

logger = logging.getLogger(__name__)


def _try_import_econml():
    try:
        import econml  # noqa: F401
        return econml
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internal common machinery
# ---------------------------------------------------------------------------
class _MetaLearnerBase(BaseCausalEstimator):
    """Shared plumbing for S/T/X/R-Learners."""

    name = "meta_learner"

    def _binary_check(self, t: pd.Series) -> None:
        levels = sorted(t.dropna().unique().tolist())
        if levels != [0, 1] and not (
            len(levels) == 2 and all(v in (0, 1, True, False) for v in levels)
        ):
            raise ValueError(
                "Meta-learners currently require a binary {0, 1} treatment. "
                f"Got levels: {levels}. For multi-level treatments use the "
                "'dml' estimator."
            )

    def ate(self, confidence_level: float = 0.95) -> dict:
        self._require_fit()
        cate_vec = self._cate_train_.flatten()
        n = len(cate_vec)
        ate = float(np.mean(cate_vec))
        se = float(np.std(cate_vec, ddof=1) / np.sqrt(n))
        # Normal-approx CI; not strictly orthogonal but consistent for ATE
        # under T-/S-/X-/R-Learner. EconML's BootstrapInference can replace
        # this when ``bootstrap=True`` is set in estimator_params.
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + confidence_level / 2.0))
        return {
            "ate": ate,
            "std_error": se,
            "ci_lower": ate - z * se,
            "ci_upper": ate + z * se,
            "p_value": float(2.0 * (1.0 - norm.cdf(abs(ate) / se))) if se > 0 else None,
            "confidence_level": confidence_level,
            "n_samples": n,
            "estimator": f"econml.{self.name}",
        }

    def cate(self, X_query: pd.DataFrame, return_ci: bool = False):
        self._require_fit()
        X = X_query[self.confounders].to_numpy()
        tau = np.asarray(self.backend_.effect(X))
        return tau


# ---------------------------------------------------------------------------
# S-Learner
# ---------------------------------------------------------------------------
class SLearner(_MetaLearnerBase):
    """Single-model meta-learner with treatment as a feature."""

    name = "s_learner"

    def fit(self, df: pd.DataFrame) -> "SLearner":
        self._validate_columns(df)
        self._df = df.copy()
        if _try_import_econml() is None:
            raise ImportError(
                "SLearner requires the 'econml' package. "
                "Install with `pip install econml`."
            )
        from econml.metalearners import SLearner as _SLearner

        self._binary_check(df[self.treatment])
        X = df[self.confounders].to_numpy()
        T = df[self.treatment].to_numpy()
        Y = df[self.outcome].to_numpy()

        self.backend_ = _SLearner(overall_model=self.outcome_model)
        self.backend_.fit(Y, T, X=X)
        self._cate_train_ = np.asarray(self.backend_.effect(X))
        self._fit_predict_surrogate(self._df)
        self._fitted = True
        logger.info("[Causal] S-Learner fit complete.")
        return self


# ---------------------------------------------------------------------------
# T-Learner
# ---------------------------------------------------------------------------
class TLearner(_MetaLearnerBase):
    """Two-model meta-learner, one per treatment arm."""

    name = "t_learner"

    def fit(self, df: pd.DataFrame) -> "TLearner":
        self._validate_columns(df)
        self._df = df.copy()
        if _try_import_econml() is None:
            raise ImportError("TLearner requires 'econml'.")
        from econml.metalearners import TLearner as _TLearner
        from sklearn.base import clone

        self._binary_check(df[self.treatment])
        X = df[self.confounders].to_numpy()
        T = df[self.treatment].to_numpy()
        Y = df[self.outcome].to_numpy()

        # Two clones of the outcome model, one per arm.
        models = [clone(self.outcome_model), clone(self.outcome_model)]
        self.backend_ = _TLearner(models=models)
        self.backend_.fit(Y, T, X=X)
        self._cate_train_ = np.asarray(self.backend_.effect(X))
        self._fit_predict_surrogate(self._df)
        self._fitted = True
        logger.info("[Causal] T-Learner fit complete.")
        return self


# ---------------------------------------------------------------------------
# X-Learner
# ---------------------------------------------------------------------------
class XLearner(_MetaLearnerBase):
    """Propensity-weighted X-Learner; strong for imbalanced treatments."""

    name = "x_learner"

    def fit(self, df: pd.DataFrame) -> "XLearner":
        self._validate_columns(df)
        self._df = df.copy()
        if _try_import_econml() is None:
            raise ImportError("XLearner requires 'econml'.")
        from econml.metalearners import XLearner as _XLearner
        from sklearn.base import clone

        self._binary_check(df[self.treatment])
        X = df[self.confounders].to_numpy()
        T = df[self.treatment].to_numpy()
        Y = df[self.outcome].to_numpy()

        models = [clone(self.outcome_model), clone(self.outcome_model)]
        # X-Learner needs a propensity model and per-arm effect models too.
        propensity_model = self.treatment_model
        cate_models = [clone(self.outcome_model), clone(self.outcome_model)]
        self.backend_ = _XLearner(
            models=models,
            propensity_model=propensity_model,
            cate_models=cate_models,
        )
        self.backend_.fit(Y, T, X=X)
        self._cate_train_ = np.asarray(self.backend_.effect(X))
        self._fit_predict_surrogate(self._df)
        self._fitted = True
        logger.info("[Causal] X-Learner fit complete.")
        return self


# ---------------------------------------------------------------------------
# R-Learner
# ---------------------------------------------------------------------------
class RLearner(_MetaLearnerBase):
    """
    R-Learner via :class:`econml.dml.NonParamDML`.

    First residualises Y and T against X using the supplied nuisance
    learners, then fits a final regressor on the residuals to recover the
    CATE.
    """

    name = "r_learner"

    def fit(self, df: pd.DataFrame) -> "RLearner":
        self._validate_columns(df)
        self._df = df.copy()
        if _try_import_econml() is None:
            raise ImportError("RLearner requires 'econml'.")
        from econml.dml import NonParamDML
        from sklearn.base import clone
        from sklearn.ensemble import GradientBoostingRegressor

        self._binary_check(df[self.treatment])
        X = df[self.confounders].to_numpy()
        T = df[self.treatment].to_numpy()
        Y = df[self.outcome].to_numpy()

        # NonParamDML expects:
        #  - model_y  : Y ~ X
        #  - model_t  : T ~ X
        #  - model_final : residual_Y ~ residual_T * f(X)
        final_model = self.estimator_params.get(
            "final_model", GradientBoostingRegressor(n_estimators=100)
        )
        self.backend_ = NonParamDML(
            model_y=clone(self.outcome_model),
            model_t=clone(self.treatment_model),
            model_final=final_model,
            discrete_treatment=True,
            cv=int(self.estimator_params.get("n_folds", 5)),
        )
        self.backend_.fit(Y, T, X=X)
        self._cate_train_ = np.asarray(self.backend_.effect(X))
        self._fit_predict_surrogate(self._df)
        self._fitted = True
        logger.info("[Causal] R-Learner fit complete.")
        return self

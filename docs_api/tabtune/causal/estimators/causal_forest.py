"""
tabtune.causal.estimators.causal_forest
=======================================

Honest Causal Forest estimator for non-parametric heterogeneous treatment
effect estimation.

Causal Forests grow an ensemble of trees where every split is chosen to
maximise heterogeneity in the treatment effect (not the outcome itself).
*Honesty* means the data used to choose the splits is disjoint from the
data used to estimate effects within each leaf, yielding asymptotically
unbiased CATE estimates with valid confidence intervals.

This estimator wraps :class:`econml.grf.CausalForest`. Nuisance learners
arrive as sklearn-compatible TabTune adapters; the forest uses them
internally for the orthogonal-score residualisation step.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseCausalEstimator

logger = logging.getLogger(__name__)


def _try_import_econml_grf():
    try:
        from econml.grf import CausalForest as _CF  # noqa: F401
        return _CF
    except Exception:
        return None


class CausalForestEstimator(BaseCausalEstimator):
    """
    Honest Causal Forest for heterogeneous treatment effects.

    Parameters consumed via ``estimator_params``:

    * ``n_estimators`` (int, default 200)
    * ``max_depth`` (int, default None)
    * ``min_samples_leaf`` (int, default 5)
    * ``honest`` (bool, default True)
    * ``confidence_level`` (float, default 0.95) -- for ATE / CATE CIs
      produced via ``backend_.predict(..., interval=True)``.
    """

    name = "causal_forest"

    def fit(self, df: pd.DataFrame) -> "CausalForestEstimator":
        self._validate_columns(df)
        self._df = df.copy()
        _CF = _try_import_econml_grf()
        if _CF is None:
            raise ImportError(
                "CausalForestEstimator requires 'econml'. "
                "Install with `pip install econml`."
            )

        X = df[self.confounders].to_numpy()
        T = df[self.treatment].to_numpy().reshape(-1, 1)
        Y = df[self.outcome].to_numpy().reshape(-1, 1)

        self.backend_ = _CF(
            n_estimators=int(self.estimator_params.get("n_estimators", 200)),
            max_depth=self.estimator_params.get("max_depth", None),
            min_samples_leaf=int(self.estimator_params.get("min_samples_leaf", 5)),
            honest=bool(self.estimator_params.get("honest", True)),
            random_state=int(self.estimator_params.get("random_state", 0)),
        )
        # econml.grf.CausalForest signature: fit(X, T, y)
        self.backend_.fit(X, T, Y)
        # Cache training CATEs for ATE summary.
        self._cate_train_ = np.asarray(self.backend_.predict(X)).flatten()
        self._fit_predict_surrogate(self._df)
        self._fitted = True
        logger.info("[Causal] Causal Forest fit complete.")
        return self

    def ate(self, confidence_level: float = 0.95) -> dict:
        self._require_fit()
        cate_vec = self._cate_train_
        ate = float(np.mean(cate_vec))
        # CI via per-row prediction intervals averaged.
        try:
            X_train = self._df[self.confounders].to_numpy()
            point, lb, ub = self.backend_.predict(
                X_train, interval=True, alpha=1.0 - confidence_level
            )
            ci_lower = float(np.mean(np.asarray(lb).flatten()))
            ci_upper = float(np.mean(np.asarray(ub).flatten()))
            se = (ci_upper - ci_lower) / (2.0 * 1.96)  # normal approx
        except Exception:
            # Fallback if the EconML version doesn't return intervals.
            n = len(cate_vec)
            se = float(np.std(cate_vec, ddof=1) / np.sqrt(n))
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se
        return {
            "ate": ate,
            "std_error": float(se),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value": None,
            "confidence_level": confidence_level,
            "n_samples": int(len(self._df)),
            "estimator": "econml.grf.CausalForest",
        }

    def cate(self, X_query: pd.DataFrame, return_ci: bool = False):
        self._require_fit()
        X = X_query[self.confounders].to_numpy()
        if return_ci:
            point, lb, ub = self.backend_.predict(X, interval=True)
            return {
                "point": np.asarray(point).flatten(),
                "ci_lower": np.asarray(lb).flatten(),
                "ci_upper": np.asarray(ub).flatten(),
            }
        return np.asarray(self.backend_.predict(X)).flatten()

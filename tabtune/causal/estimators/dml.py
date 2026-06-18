"""
tabtune.causal.estimators.dml
=============================

Double / Debiased Machine Learning estimator.

This is the **default estimator** for :mod:`tabtune.causal`. It uses two
nuisance learners -- one for the outcome ``E[Y | X]`` (``ml_g``) and one
for the treatment-propensity ``E[T | X]`` (``ml_m``) -- to construct
Neyman-orthogonal residuals, then regresses the residualised Y on the
residualised T to recover the ATE.

Implementation
--------------
* Binary or continuous treatment  -> ``doubleml.DoubleMLPLR``
  (partially linear regression).
* Discrete multi-level treatment  -> ``doubleml.DoubleMLAPOS``
  (Average Potential Outcomes), with the causal contrast against the
  reference level used as the ATE.
* Cross-fitting with ``n_folds`` partitions; default 5.
* Confidence interval reported at ``confidence_level`` (default 0.95).

Both nuisance learners arrive as sklearn-compatible objects, normally
instances of :class:`tabtune.causal.adapters._TabTuneSklearnAdapter`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseCausalEstimator

logger = logging.getLogger(__name__)


def _try_import_doubleml():
    try:
        import doubleml as dml  # noqa: F401
        return dml
    except Exception:
        return None


class DMLEstimator(BaseCausalEstimator):
    """
    Double Machine Learning estimator for the Average Treatment Effect.

    Parameters
    ----------
    treatment, outcome, confounders : see :class:`BaseCausalEstimator`.
    outcome_model : sklearn-compatible regressor for ``E[Y | X]``.
    treatment_model : sklearn-compatible classifier (binary/multi-level T)
        or regressor (continuous T) for ``E[T | X]``.
    estimator_params : dict, optional
        Keys consumed:

        * ``n_folds`` (int, default 5): cross-fitting folds.
        * ``n_rep`` (int, default 1): number of repeated cross-fits.
        * ``confidence_level`` (float, default 0.95).
        * ``score`` (str, default ``'partialling out'``): PLR score for
          continuous/binary treatment.
        * ``reference_level``: only used for multi-level treatments;
          defaults to the smallest observed treatment value.

    Attributes
    ----------
    backend_ : object
        The underlying ``DoubleMLPLR`` or ``DoubleMLAPOS`` instance after
        ``.fit()``.
    treatment_levels_ : numpy.ndarray
        Sorted unique treatment values observed during fit.
    is_discrete_ : bool
        True if the discrete-treatment APOS pathway was used.
    """

    name = "dml"

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "DMLEstimator":
        self._validate_columns(df)
        self._df = df.copy()

        dml_lib = _try_import_doubleml()
        if dml_lib is None:
            raise ImportError(
                "DMLEstimator requires the 'doubleml' package. "
                "Install with `pip install doubleml`."
            )

        # ── Detect treatment type ────────────────────────────────────────
        t_values = df[self.treatment].dropna().unique()
        self.treatment_levels_ = np.array(sorted(t_values))
        n_unique = len(self.treatment_levels_)
        self.is_discrete_ = (
            n_unique > 2
            and np.issubdtype(df[self.treatment].dtype, np.number)
            and np.all(np.equal(np.mod(t_values, 1), 0))
        )

        dml_data = dml_lib.DoubleMLData(
            data=self._df,
            y_col=self.outcome,
            d_cols=self.treatment,
            x_cols=self.confounders,
        )

        n_folds = int(self.estimator_params.get("n_folds", 5))
        n_rep = int(self.estimator_params.get("n_rep", 1))

        if self.is_discrete_:
            logger.info(
                "[Causal] DML: multi-level discrete treatment detected "
                "(%d levels). Using DoubleMLAPOS.",
                n_unique,
            )
            self.backend_ = dml_lib.DoubleMLAPOS(
                dml_data,
                ml_g=self.outcome_model,
                ml_m=self.treatment_model,
                treatment_levels=self.treatment_levels_,
                n_folds=n_folds,
                n_rep=n_rep,
            )
        else:
            score = self.estimator_params.get("score", "partialling out")
            logger.info(
                "[Causal] DML: binary/continuous treatment. Using DoubleMLPLR "
                "(score=%s, n_folds=%d).",
                score,
                n_folds,
            )
            self.backend_ = dml_lib.DoubleMLPLR(
                dml_data,
                ml_l=self.outcome_model,
                ml_m=self.treatment_model,
                score=score,
                n_folds=n_folds,
                n_rep=n_rep,
            )

        self.backend_.fit()
        # Fit the predict-surrogate so downstream consumers (counterfactual
        # fairness audit, single-row predictions) can use this estimator
        # as a regular predict-Y model. See BaseCausalEstimator.predict().
        self._fit_predict_surrogate(self._df)
        self._fitted = True
        logger.info("[Causal] DML fit complete.")
        return self

    # ------------------------------------------------------------------
    # ATE
    # ------------------------------------------------------------------
    def ate(self, confidence_level: float = 0.95) -> dict:
        self._require_fit()

        if self.is_discrete_:
            # For multi-level treatments, report the average contrast vs.
            # the reference level (default = smallest value).
            ref = self.estimator_params.get(
                "reference_level", float(self.treatment_levels_[0])
            )
            contrast = self.backend_.causal_contrast(reference_levels=ref)
            ci = contrast.confint(level=confidence_level)
            ates = np.asarray(contrast.thetas)
            ses = np.asarray(contrast.ses)
            # Overall ATE = average over non-reference contrasts.
            ate = float(np.mean(ates))
            se = float(np.sqrt(np.mean(ses ** 2)))
            lo = float(np.mean(ci.iloc[:, 0].values))
            hi = float(np.mean(ci.iloc[:, 1].values))
            return {
                "ate": ate,
                "std_error": se,
                "ci_lower": lo,
                "ci_upper": hi,
                "p_value": None,
                "confidence_level": confidence_level,
                "n_samples": int(len(self._df)),
                "treatment_levels": [float(l) for l in self.treatment_levels_],
                "per_level": {
                    float(self.treatment_levels_[i + 1]): {
                        "ate": float(ates[i]),
                        "std_error": float(ses[i]),
                        "ci_lower": float(ci.iloc[i, 0]),
                        "ci_upper": float(ci.iloc[i, 1]),
                    }
                    for i in range(len(ates))
                },
                "estimator": "doubleml.DoubleMLAPOS",
            }

        # Binary / continuous treatment via PLR.
        ate = float(self.backend_.coef[0])
        se = float(self.backend_.se[0])
        ci = self.backend_.confint(level=confidence_level)
        return {
            "ate": ate,
            "std_error": se,
            "ci_lower": float(ci.iloc[0, 0]),
            "ci_upper": float(ci.iloc[0, 1]),
            "p_value": float(self.backend_.pval[0]),
            "confidence_level": confidence_level,
            "n_samples": int(len(self._df)),
            "estimator": "doubleml.DoubleMLPLR",
        }

    # ------------------------------------------------------------------
    # CATE
    # ------------------------------------------------------------------
    def cate(self, X_query: pd.DataFrame, return_ci: bool = False):
        """
        Approximate CATE for PLR/APOS via per-row outcome-model contrast.

        DoubleML's PLR/APOS provide ATE but not native per-row CATE. We
        approximate the CATE by evaluating the outcome model at each row
        under T = treated vs. T = control. For richer CATE estimation use
        the X-Learner, R-Learner, or Causal Forest backends.
        """
        self._require_fit()
        if self.is_discrete_:
            raise NotImplementedError(
                "Per-row CATE for multi-level treatments is not supported by "
                "DMLEstimator. Use 'causal_forest' or 'x_learner' instead."
            )
        # Binary treatment approximation. We evaluate the fitted predict
        # surrogate (built in BaseCausalEstimator.fit via _fit_predict_surrogate)
        # under T = 1 vs T = 0. The surrogate is a clone of the outcome model
        # that was actually fitted on (X + T -> Y); the raw self.outcome_model
        # is only a prototype handed to DoubleML for internal cross-fitting and
        # is never fitted directly, so it must not be used here.
        X = X_query[self.confounders].copy()
        X_t1 = X.copy()
        X_t1[self.treatment] = 1
        X_t0 = X.copy()
        X_t0[self.treatment] = 0
        try:
            y1 = self.predict(X_t1)
            y0 = self.predict(X_t0)
        except Exception as exc:
            raise RuntimeError(
                f"Outcome model failed during CATE approximation: {exc}"
            ) from exc
        return y1 - y0
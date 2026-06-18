"""
tabtune.causal.leaderboard
==========================

Run several ``(estimator, TFM)`` combinations on the same data and rank
them by stability and refuter pass-rate.

Mirrors :class:`tabtune.TabularLeaderboard` in spirit: add entries with
:meth:`add_model`, then call :meth:`run` to fit each entry on the same
``(X, y)`` and produce a ranked DataFrame.

Use cases
---------
* Model selection -- which TFM gives the most stable ATE on this dataset?
* Sensitivity to estimator choice -- does the ATE move across DML,
  X-Learner, and Causal Forest?
* Compliance -- a leaderboard becomes part of the audit trail showing
  the headline estimate is not artifactual.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np
import pandas as pd

from .analysis import CausalAnalysis

logger = logging.getLogger(__name__)


_RANK_BY_OPTIONS = (
    "refuter_pass_rate",
    "ate_stability",
    "ci_width",
)


# ---------------------------------------------------------------------------
# CausalLeaderboard
# ---------------------------------------------------------------------------
class CausalLeaderboard:
    """
    Fit multiple causal-analysis configurations and produce a ranked table.

    Example
    -------
    >>> from tabtune.causal import CausalLeaderboard
    >>> lb = CausalLeaderboard(
    ...     treatment='T', outcome='Y', confounders=['x0', 'x1', 'x2'],
    ... )
    >>> lb.add_model('TabPFNv26', task_type='regression', estimator='dml')
    >>> lb.add_model('TabICLv2',  task_type='regression', estimator='dml')
    >>> lb.add_model('TabPFNv26', task_type='regression', estimator='x_learner')
    >>> ranked = lb.run(X_train, y_train, rank_by='refuter_pass_rate')
    """

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: Iterable[str],
        sensitive: Iterable[str] | None = None,
        tuning_params: dict | None = None,
    ):
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = list(confounders)
        self.sensitive = list(sensitive or [])
        self.tuning_params = tuning_params or {}
        self._entries: list[dict] = []

    # ------------------------------------------------------------------
    # Building the entry list
    # ------------------------------------------------------------------
    def add_model(
        self,
        model_name: str,
        task_type: str,
        estimator: str = "dml",
        tuning_strategy: str = "inference",
        treatment_model: dict | None = None,
        estimator_params: dict | None = None,
        label: str | None = None,
    ) -> "CausalLeaderboard":
        """Add one ``(estimator, TFM)`` configuration to the leaderboard."""
        self._entries.append(
            {
                "label": label or f"{model_name}/{estimator}",
                "model_name": model_name,
                "task_type": task_type,
                "tuning_strategy": tuning_strategy,
                "estimator": estimator,
                "treatment_model": treatment_model,
                "estimator_params": estimator_params or {},
            }
        )
        return self

    # ------------------------------------------------------------------
    # Running the leaderboard
    # ------------------------------------------------------------------
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        rank_by: str = "refuter_pass_rate",
        include_refutation: bool = True,
        include_proxy: bool = False,
        include_counterfactual_fairness: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Fit every registered entry and return a ranked DataFrame.

        Parameters
        ----------
        rank_by : str
            One of ``'refuter_pass_rate'`` (descending), ``'ate_stability'``
            (ascending |bootstrap_std|), or ``'ci_width'`` (ascending).
        """
        if rank_by not in _RANK_BY_OPTIONS:
            raise ValueError(
                f"Unknown rank_by='{rank_by}'. "
                f"Available: {_RANK_BY_OPTIONS}."
            )

        rows: list[dict] = []
        include = ["effect"]
        if include_refutation:
            include.append("refutation")
        if include_proxy and self.sensitive:
            include.append("proxy")
        if include_counterfactual_fairness and self.sensitive:
            include.append("counterfactual_fairness")

        for i, entry in enumerate(self._entries):
            label = entry["label"]
            if verbose:
                logger.info(
                    "[Causal] Leaderboard [%d/%d]: fitting %s",
                    i + 1,
                    len(self._entries),
                    label,
                )
            t0 = time.time()
            try:
                ca = CausalAnalysis(
                    model_name=entry["model_name"],
                    task_type=entry["task_type"],
                    tuning_strategy=entry["tuning_strategy"],
                    treatment=self.treatment,
                    outcome=self.outcome,
                    confounders=self.confounders,
                    sensitive=self.sensitive,
                    estimator=entry["estimator"],
                    treatment_model=entry["treatment_model"],
                    estimator_params=entry["estimator_params"],
                    tuning_params=self.tuning_params,
                    verbose=False,
                )
                ca.fit(X, y)
                report = ca.evaluate(include=tuple(include), output_format="silent")
            except Exception as exc:  # pragma: no cover
                logger.warning("[Causal] Leaderboard entry '%s' failed: %s", label, exc)
                rows.append(
                    {
                        "label": label,
                        "model_name": entry["model_name"],
                        "estimator": entry["estimator"],
                        "ate": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "ci_width": np.nan,
                        "refuter_pass_rate": np.nan,
                        "ate_stability": np.nan,
                        "fit_seconds": time.time() - t0,
                        "error": str(exc),
                    }
                )
                continue

            effect = report.get("effect", {})
            ate = float(effect.get("ate", np.nan))
            ci_l = float(effect.get("ci_lower", np.nan))
            ci_u = float(effect.get("ci_upper", np.nan))
            ci_w = ci_u - ci_l if (ci_l == ci_l and ci_u == ci_u) else np.nan

            ref = report.get("refutation") or {}
            passed = ref.get("refuters_passed", 0)
            failed = ref.get("refuters_failed", 0)
            denom = passed + failed
            pass_rate = float(passed / denom) if denom else float("nan")
            # ATE stability: width of bootstrap CI when available
            ate_stab = float("nan")
            boot = ref.get("bootstrap") or {}
            if "ci_lower" in boot and "ci_upper" in boot:
                ate_stab = float(boot["ci_upper"] - boot["ci_lower"])

            rows.append(
                {
                    "label": label,
                    "model_name": entry["model_name"],
                    "estimator": entry["estimator"],
                    "ate": ate,
                    "ci_lower": ci_l,
                    "ci_upper": ci_u,
                    "ci_width": ci_w,
                    "refuter_pass_rate": pass_rate,
                    "ate_stability": ate_stab,
                    "fit_seconds": time.time() - t0,
                    "error": None,
                }
            )

        df = pd.DataFrame(rows)

        # Sort
        ascending = rank_by != "refuter_pass_rate"
        df = df.sort_values(
            by=rank_by, ascending=ascending, na_position="last"
        ).reset_index(drop=True)
        return df

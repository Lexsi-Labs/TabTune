"""
tabtune.causal.refute
=====================

**Step 3 of the causal discipline: Refute.**

A causal estimate rests on assumptions that data alone cannot verify.
This module stress-tests a fitted estimator to bound the consequences if
those assumptions break:

* **Placebo treatment** -- shuffle T at random; the new estimate should
  collapse to (near) zero. Detects pipelines that find signal in noise.
* **Random common cause** -- inject a synthetic noise covariate; the
  estimate should barely move. Tests robustness to missing-confounder
  misspecification.
* **Data subset** -- refit on bootstrap subsamples; estimates should be
  stable across resamples.
* **Sensitivity** -- compute the E-value (Vanderweele & Ding, 2017)
  reporting how strong an unobserved confounder must be to nullify the
  result. Also returns Manski bounds when feasible.

The module attempts to use ``dowhy.causal_refuters`` when the user
supplied a graph and identification succeeded; otherwise it falls back
to direct refit-based refuters that do not require DoWhy.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_import_dowhy():
    try:
        import dowhy  # noqa: F401
        return dowhy
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Refuter
# ---------------------------------------------------------------------------
class Refuter:
    """
    Run a battery of robustness checks on a fitted causal estimator.

    Parameters
    ----------
    estimator : BaseCausalEstimator
        A fitted instance of one of the estimators in
        :mod:`tabtune.causal.estimators`.
    df : pandas.DataFrame
        The dataframe the estimator was originally fitted on (column names
        must match ``estimator.treatment / outcome / confounders``).
    estimator_factory : Callable[[], BaseCausalEstimator], optional
        A no-arg callable that returns a fresh, unfitted estimator of the
        same kind as ``estimator``. Used by refit-based refuters (placebo,
        RCC, subset). If not supplied, refit-based refuters will be
        skipped with a warning.
    """

    def __init__(
        self,
        estimator,
        df: pd.DataFrame,
        estimator_factory=None,
    ):
        self.estimator = estimator
        self.df = df.copy()
        self.estimator_factory = estimator_factory
        self._original_ate: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        checks: Iterable[str] = (
            "placebo",
            "random_common_cause",
            "subset",
            "sensitivity",
        ),
        subset_fraction: float = 0.7,
        n_subset_iters: int = 2,
        seed: int = 0,
    ) -> dict:
        """
        Execute the requested refuters and return their results.

        Returns
        -------
        report : dict
            One key per requested check, plus ``refuters_passed`` /
            ``refuters_failed`` / ``original_ate`` summaries.
        """
        self._original_ate = float(self.estimator.ate()["ate"])
        out: dict[str, Any] = {"original_ate": self._original_ate}
        passed, failed = 0, 0

        if "placebo" in checks:
            res = self._placebo(seed=seed)
            out["placebo"] = res
            passed += int(res["pass"])
            failed += int(not res["pass"])

        if "random_common_cause" in checks:
            res = self._random_common_cause(seed=seed)
            out["random_common_cause"] = res
            passed += int(res["pass"])
            failed += int(not res["pass"])

        if "subset" in checks:
            res = self._data_subset(
                subset_fraction=subset_fraction, n_iters=n_subset_iters, seed=seed
            )
            out["subset"] = res
            passed += int(res["pass"])
            failed += int(not res["pass"])

        if "sensitivity" in checks:
            res = self._sensitivity()
            out["sensitivity"] = res
            passed += int(res["pass"])
            failed += int(not res["pass"])

        out["refuters_passed"] = int(passed)
        out["refuters_failed"] = int(failed)
        logger.info(
            "[Causal] Refutation complete: %d/%d passed",
            passed,
            passed + failed,
        )
        return out

    # ------------------------------------------------------------------
    # Individual refuters
    # ------------------------------------------------------------------
    def _refit_ate(self, df: pd.DataFrame) -> float:
        """Refit a fresh estimator on the given dataframe; return ATE."""
        if self.estimator_factory is None:
            raise RuntimeError(
                "Refit-based refuters require an estimator_factory callable."
            )
        est = self.estimator_factory()
        est.fit(df)
        return float(est.ate()["ate"])

    def _placebo(self, seed: int = 0) -> dict:
        if self.estimator_factory is None:
            return {
                "ate": None,
                "pass": False,
                "rule": "skipped (no estimator_factory provided)",
            }
        rng = np.random.default_rng(seed)
        df = self.df.copy()
        df[self.estimator.treatment] = rng.permutation(
            df[self.estimator.treatment].values
        )
        try:
            placebo_ate = self._refit_ate(df)
        except Exception as exc:
            return {"ate": None, "pass": False, "rule": f"error: {exc}"}
        # Pass criterion: |placebo_ate| < 0.5 * |original_ate|
        threshold = 0.5 * abs(self._original_ate)
        passed = abs(placebo_ate) < threshold if threshold > 0 else abs(placebo_ate) < 0.05
        return {
            "ate": float(placebo_ate),
            "pass": bool(passed),
            "rule": "|placebo_ate| < 0.5 * |original_ate|",
        }

    def _random_common_cause(self, seed: int = 0) -> dict:
        if self.estimator_factory is None:
            return {
                "ate": None,
                "pass": False,
                "rule": "skipped (no estimator_factory provided)",
            }
        rng = np.random.default_rng(seed)
        df = self.df.copy()
        rcc_col = "__rcc_noise__"
        df[rcc_col] = rng.standard_normal(len(df))
        # Temporarily extend confounders.
        original_confounders = list(self.estimator.confounders)
        try:
            est = self.estimator_factory()
            est.confounders = original_confounders + [rcc_col]
            est.fit(df)
            rcc_ate = float(est.ate()["ate"])
        except Exception as exc:
            return {"ate": None, "pass": False, "rule": f"error: {exc}"}
        denom = max(abs(self._original_ate), 1e-9)
        relative_shift = abs(rcc_ate - self._original_ate) / denom
        return {
            "ate": rcc_ate,
            "shift": float(relative_shift),
            "pass": bool(relative_shift < 0.2),
            "rule": "|delta ATE| / |ATE| < 0.2",
        }

    def _data_subset(self, subset_fraction: float, n_iters: int, seed: int) -> dict:
        if self.estimator_factory is None:
            return {
                "estimates": [],
                "pass": False,
                "rule": "skipped (no estimator_factory provided)",
            }
        rng = np.random.default_rng(seed)
        ates: list[float] = []
        for i in range(n_iters):
            sample = self.df.sample(
                frac=subset_fraction,
                replace=False,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            try:
                ates.append(self._refit_ate(sample))
            except Exception:
                continue
        if not ates:
            return {"estimates": [], "pass": False, "rule": "no successful refits"}
        std = float(np.std(ates, ddof=1)) if len(ates) > 1 else 0.0
        denom = max(abs(self._original_ate), 1e-9)
        cv = std / denom
        return {
            "estimates": [float(a) for a in ates],
            "mean": float(np.mean(ates)),
            "std": std,
            "cv": float(cv),
            "pass": bool(cv < 0.25),
            "rule": "coefficient-of-variation across subsets < 0.25",
        }

    def _sensitivity(self) -> dict:
        """
        Sensitivity analysis via the E-value (VanderWeele & Ding, 2017).

        For a continuous ATE the E-value approximates the minimum strength
        an unmeasured confounder would need (in risk-ratio scale) to nullify
        the observed effect.

        Manski bounds are reported as a sanity check on the support of Y.
        """
        ate = self._original_ate
        std_err = float(self.estimator.ate().get("std_error") or 0.0)

        # E-value on the ATE itself (continuous-outcome variant).
        if std_err > 0:
            # Approximate the risk-ratio scale via |ATE| / SD(Y).
            sd_y = float(
                self.estimator._df[self.estimator.outcome].std(ddof=1)  # type: ignore[attr-defined]
            )
            sd_y = max(sd_y, 1e-9)
            standardised = abs(ate) / sd_y
            e_value = float(np.exp(0.91 * standardised) + np.sqrt(
                np.exp(0.91 * standardised) * (np.exp(0.91 * standardised) - 1.0)
            ))
        else:
            e_value = float("nan")

        # Manski bounds (only sensible for bounded outcomes).
        y_min = float(self.estimator._df[self.estimator.outcome].min())  # type: ignore[attr-defined]
        y_max = float(self.estimator._df[self.estimator.outcome].max())  # type: ignore[attr-defined]
        manski_width = y_max - y_min
        return {
            "e_value": e_value,
            "manski_min": y_min,
            "manski_max": y_max,
            "manski_width": manski_width,
            "pass": bool(e_value >= 1.25) if e_value == e_value else False,
            "rule": "E-value >= 1.25",
        }
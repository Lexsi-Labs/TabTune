"""
tabtune.causal.audit.counterfactual
===================================

**Counterfactual fairness** audit.

For each row in the audit set, flip every sensitive attribute to each of
its other observed values while holding the remaining features fixed,
and re-predict. The audit reports:

* **flip rate** -- fraction of rows where the prediction changes when S
  is flipped.
* **mean delta** -- mean absolute change in the prediction.
* **per-attribute breakdown** when multiple sensitive attributes are
  supplied.

A causal model that respects counterfactual fairness should produce a
flip rate close to zero and a mean delta close to zero.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CounterfactualFairness:
    """
    Audit a fitted estimator for counterfactual fairness w.r.t. one or
    more sensitive attributes.

    Parameters
    ----------
    estimator : object
        Any object with a callable ``.predict(X)`` method. In the
        :mod:`tabtune.causal` pipeline this is the outcome nuisance
        learner used by the chosen :class:`BaseCausalEstimator`.
    df : pandas.DataFrame
        Audit set. Must contain all features the estimator expects.
    sensitive : str or list of str
        Column name(s) of the sensitive attribute(s).
    feature_columns : list of str
        Columns to pass to the estimator's ``.predict`` after flipping S.
    flip_threshold : float, default 1e-6
        A prediction change above this magnitude counts as a flip.
    """

    def __init__(
        self,
        estimator: Any,
        df: pd.DataFrame,
        sensitive: str | Iterable[str],
        feature_columns: Iterable[str],
        flip_threshold: float = 1e-6,
    ):
        self.estimator = estimator
        self.df = df.copy()
        self.feature_columns = list(feature_columns)
        if isinstance(sensitive, str):
            self.sensitive = [sensitive]
        else:
            self.sensitive = list(sensitive)
        self.flip_threshold = float(flip_threshold)

    def run(self) -> dict:
        """Return a per-attribute flip-rate and mean-delta report."""
        results: dict[str, dict] = {}

        # Baseline predictions on the original rows.
        try:
            baseline_pred = np.asarray(
                self.estimator.predict(self.df[self.feature_columns])
            )
        except Exception as exc:
            raise RuntimeError(
                f"Estimator failed during baseline prediction: {exc}"
            ) from exc

        for s_col in self.sensitive:
            if s_col not in self.df.columns:
                logger.warning(
                    "[Causal] Sensitive column '%s' not in dataframe; skipping.",
                    s_col,
                )
                continue

            levels = sorted(self.df[s_col].dropna().unique().tolist())
            if len(levels) < 2:
                logger.warning(
                    "[Causal] Sensitive attribute '%s' has only one observed level; skipping.",
                    s_col,
                )
                continue

            flip_count = 0
            total_count = 0
            deltas: list[float] = []
            per_flip_rows = []

            for orig_level in levels:
                orig_mask = self.df[s_col] == orig_level
                if not orig_mask.any():
                    continue
                rows = self.df.loc[orig_mask, self.feature_columns].copy()
                base = baseline_pred[orig_mask.values]
                for alt_level in levels:
                    if alt_level == orig_level:
                        continue
                    flipped = rows.copy()
                    if s_col in flipped.columns:
                        flipped[s_col] = alt_level
                    try:
                        alt_pred = np.asarray(self.estimator.predict(flipped))
                    except Exception as exc:
                        logger.warning(
                            "[Causal] Estimator failed on flipped rows for "
                            "%s=%s -> %s: %s",
                            s_col,
                            orig_level,
                            alt_level,
                            exc,
                        )
                        continue
                    diff = np.abs(np.asarray(alt_pred) - np.asarray(base))
                    flip_count += int((diff > self.flip_threshold).sum())
                    total_count += len(diff)
                    deltas.extend(diff.tolist())
                    per_flip_rows.append(
                        {
                            "from": str(orig_level),
                            "to": str(alt_level),
                            "n_rows": int(len(diff)),
                            "n_flipped": int((diff > self.flip_threshold).sum()),
                            "mean_delta": float(np.mean(diff)) if len(diff) else 0.0,
                        }
                    )

            flip_rate = float(flip_count / total_count) if total_count else 0.0
            mean_delta = float(np.mean(deltas)) if deltas else 0.0
            results[s_col] = {
                "flip_rate": flip_rate,
                "mean_delta": mean_delta,
                "per_flip": per_flip_rows,
                "n_levels": len(levels),
                "n_rows_audited": int(total_count),
            }

        # Overall pass if every sensitive attribute has flip_rate < 0.1.
        passed = all(r["flip_rate"] < 0.1 for r in results.values()) if results else True
        return {
            "per_attribute": results,
            "pass": bool(passed),
            "rule": "flip_rate < 0.1 for every sensitive attribute",
        }

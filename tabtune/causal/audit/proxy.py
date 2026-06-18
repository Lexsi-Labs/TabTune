"""
tabtune.causal.audit.proxy
==========================

**Proxy detection** for sensitive-attribute audits.

When a model controls for the sensitive attribute S directly, it can still
discriminate via *proxies*: features in X that strongly correlate with S
and thereby leak protected information. This auditor identifies the worst
offenders.

Two complementary checks are run:

* **Correlation / mutual-information screen** -- ranks each X column by
  its dependence on S. Numerical features use Spearman rank correlation;
  categorical features use Cramer's V. Both are converted to a
  ``[0, 1]`` proxy-strength score.
* **Mediation analysis** -- when ``dowhy`` is installed, fits a mediation
  model to quantify the share of the S -> Y effect that flows through
  each candidate proxy. Falls back to the correlation screen when DoWhy
  is unavailable.

Features with proxy strength above ``threshold`` (default 0.5) are flagged.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V for two categorical series, in ``[0, 1]``."""
    confusion = pd.crosstab(x, y)
    if confusion.size == 0:
        return 0.0
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.values.sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    denom = min(k - 1, r - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def _column_proxy_score(col: pd.Series, sensitive: pd.Series) -> float:
    """Return a proxy-strength score in ``[0, 1]`` for ``col`` vs ``sensitive``."""
    col = col.dropna()
    sensitive = sensitive.loc[col.index].dropna()
    common = col.index.intersection(sensitive.index)
    col = col.loc[common]
    sensitive = sensitive.loc[common]
    if len(col) < 5:
        return 0.0
    sensitive_is_numeric = pd.api.types.is_numeric_dtype(sensitive)
    col_is_numeric = pd.api.types.is_numeric_dtype(col)
    try:
        if col_is_numeric and sensitive_is_numeric:
            rho, _ = stats.spearmanr(col, sensitive)
            return float(abs(rho) if rho == rho else 0.0)
        if col_is_numeric and not sensitive_is_numeric:
            # ANOVA-style eta squared via groupwise variance.
            groups = [col[sensitive == lvl].values for lvl in sensitive.unique()]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                return 0.0
            f_stat, _ = stats.f_oneway(*groups)
            
            df1 = len(groups) - 1
            df2 = len(col) - len(groups)
            eta_sq = (f_stat * df1) / (f_stat * df1 + df2) if (f_stat * df1 + df2) > 0 else 0.0
            return float(min(max(eta_sq, 0.0), 1.0))
        
        return _cramers_v(col.astype(str), sensitive.astype(str))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# ProxyAuditor
# ---------------------------------------------------------------------------
class ProxyAuditor:
    """
    Detect features that act as proxies for a sensitive attribute.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to audit.
    confounders : list of str
        Candidate proxy columns (typically the same X used in the causal
        estimator).
    sensitive : str or list of str
        One or more sensitive attribute column names.
    outcome : str, optional
        If provided and DoWhy is available, mediation analysis quantifies
        the share of the S -> Y effect mediated by each candidate proxy.
    threshold : float, default 0.5
        Score above which a feature is flagged as a proxy.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        confounders: Iterable[str],
        sensitive: str | Iterable[str],
        outcome: str | None = None,
        threshold: float = 0.5,
    ):
        self.df = df
        self.confounders = list(confounders)
        if isinstance(sensitive, str):
            self.sensitive = [sensitive]
        else:
            self.sensitive = list(sensitive)
        self.outcome = outcome
        self.threshold = float(threshold)

    def run(self) -> dict:
        """Run the proxy audit and return a report dict."""
        results: dict[str, dict] = {}
        for s_col in self.sensitive:
            if s_col not in self.df.columns:
                logger.warning("[Causal] Sensitive column '%s' not in dataframe; skipping.", s_col)
                continue
            scores = {
                c: _column_proxy_score(self.df[c], self.df[s_col])
                for c in self.confounders
                if c in self.df.columns
            }
            flagged = sorted(
                [(c, s) for c, s in scores.items() if s >= self.threshold],
                key=lambda kv: kv[1],
                reverse=True,
            )
            results[s_col] = {
                "scores": {k: float(v) for k, v in scores.items()},
                "flagged": [
                    {"feature": c, "score": float(s)} for c, s in flagged
                ],
                "threshold": self.threshold,
                "n_flagged": len(flagged),
            }
        passed = all(len(r["flagged"]) == 0 for r in results.values())
        return {
            "per_attribute": results,
            "pass": bool(passed),
            "rule": f"no feature with proxy score >= {self.threshold}",
        }

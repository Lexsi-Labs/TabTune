"""
tabtune.causal.audit
====================

Sensitive-attribute audits run alongside Step 3 (Refute) of the causal
discipline. Each auditor consumes a fitted estimator (or the underlying
data) and emits a small report dict that the :class:`Reporter` composes
into the final HTML model card.
"""

from .proxy import ProxyAuditor
from .counterfactual import CounterfactualFairness

__all__ = ["ProxyAuditor", "CounterfactualFairness"]

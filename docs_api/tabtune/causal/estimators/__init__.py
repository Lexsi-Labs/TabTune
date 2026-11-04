"""
tabtune.causal.estimators
=========================

Registry of causal-effect estimators. Each estimator subclasses
:class:`BaseCausalEstimator` and exposes a common ``.fit(df) / .ate()``
contract; some additionally implement ``.cate(X)`` and
``.counterfactual(row, intervention)``.

The :data:`ESTIMATOR_REGISTRY` dict drives the ``estimator='...'``
argument of :class:`tabtune.causal.CausalAnalysis`.
"""

from .base import BaseCausalEstimator
from .dml import DMLEstimator
from .meta_learners import SLearner, TLearner, XLearner, RLearner
from .causal_forest import CausalForestEstimator

ESTIMATOR_REGISTRY: dict[str, type[BaseCausalEstimator]] = {
    "dml": DMLEstimator,
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "r_learner": RLearner,
    "causal_forest": CausalForestEstimator,
}

__all__ = [
    "BaseCausalEstimator",
    "DMLEstimator",
    "SLearner",
    "TLearner",
    "XLearner",
    "RLearner",
    "CausalForestEstimator",
    "ESTIMATOR_REGISTRY",
]

"""
tabtune.causal
==============

Causal-inference module for TabTune. Mirrors :class:`TabularPipeline` in
shape and conventions; uses TabTune TFMs as nuisance learners under
DoubleML, EconML, and DoWhy.

Public API
----------
:class:`CausalAnalysis`
    The user-facing pipeline. Holds T/Y/X/S, picks an estimator, runs the
    full identify -> estimate -> refute -> audit discipline.

:class:`CausalLeaderboard`
    Multi-configuration runner; ranks ``(estimator, TFM)`` combinations
    by stability and refuter pass-rate.

The remaining components (graph builders, individual estimators,
refuters, audits, and the HTML reporter) are exposed for advanced users
and library extension.
"""

from .adapters import as_sklearn, _TabTuneSklearnAdapter
from .analysis import CausalAnalysis
from .audit.counterfactual import CounterfactualFairness
from .audit.proxy import ProxyAuditor
from .estimators import (
    BaseCausalEstimator,
    CausalForestEstimator,
    DMLEstimator,
    ESTIMATOR_REGISTRY,
    RLearner,
    SLearner,
    TLearner,
    XLearner,
)
from .graph import GraphBuilder, Identifier
from .leaderboard import CausalLeaderboard
from .refute import Refuter
from .reporter import Reporter

__all__ = [
    # User-facing
    "CausalAnalysis",
    "CausalLeaderboard",
    # Bridge
    "as_sklearn",
    "_TabTuneSklearnAdapter",
    # Step 1
    "GraphBuilder",
    "Identifier",
    # Step 2
    "BaseCausalEstimator",
    "DMLEstimator",
    "SLearner",
    "TLearner",
    "XLearner",
    "RLearner",
    "CausalForestEstimator",
    "ESTIMATOR_REGISTRY",
    # Step 3 + audits
    "Refuter",
    "ProxyAuditor",
    "CounterfactualFairness",
    # Reporting
    "Reporter",
]

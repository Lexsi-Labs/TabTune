"""
TabTune Ensemble Module
========================
End-to-end ensembling of tabular foundation models.

Supports six ensemble strategies:

1. **Weighted Averaging** — performance-proportional or manual weights
2. **Greedy Ensemble Selection** — Caruana et al. 2004
3. **Stacking** — CV-based OOF predictions with LR/GBDT/MLP meta-learners
4. **Temperature-Scaled Blending** — calibrate then combine (Guo et al. 2017)
5. **Cascade Stacking** — multi-level stack + GES final layer
6. **Random-Init Ensemble** — deep ensembles via multiple random seeds

Quick Start
-----------
::

    from tabtune.ensemble import TabularEnsemble

    ensemble = TabularEnsemble(
        models=[
            {"model_name": "TabPFN",   "tuning_strategy": "inference"},
            {"model_name": "TabICLv2", "tuning_strategy": "inference"},
            {"model_name": "OrionMSP", "tuning_strategy": "inference"},
        ],
        ensemble_strategy="greedy_selection",
    )
    ensemble.fit(X_train, y_train)
    ensemble.evaluate(X_test, y_test)
"""

from .tabular_ensemble import TabularEnsemble
from .strategies import (
    WeightedAveraging,
    GreedyEnsembleSelection,
    StackingEnsemble,
    TemperatureScaledBlending,
    CascadeStackingEnsemble,
    RandomInitEnsemble,
    get_strategy,
    STRATEGY_MAP,
)

__all__ = [
    "TabularEnsemble",
    "WeightedAveraging",
    "GreedyEnsembleSelection",
    "StackingEnsemble",
    "TemperatureScaledBlending",
    "CascadeStackingEnsemble",
    "RandomInitEnsemble",
    "get_strategy",
    "STRATEGY_MAP",
]
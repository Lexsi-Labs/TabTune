"""
TabTune Distillation Module
=============================
Model-agnostic knowledge distillation for tabular foundation models.

Usage:
    from tabtune.distillation import TabDistiller

    d = TabDistiller(teachers="TabPFN", student="mlp")
    d.fit(X_train, y_train)
    d.compare(X_test, y_test)
"""

from .distiller import TabDistiller, preprocess_for_baseline

__all__ = ["TabDistiller", "preprocess_for_baseline"]

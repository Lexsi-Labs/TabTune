"""Kumo Tabular regression wrapper for the TabTune pipeline."""
from __future__ import annotations

from tabtune.models.kumo.tabtune_support import KumoTabularRegressor


class KumoTabularRegressorWrapper(KumoTabularRegressor):

    def fit(self, X, y):
        if self.tuning_strategy not in ("inference", "finetune", "peft"):
            raise ValueError(
                "Regression supports 'inference', 'finetune' or 'peft'. "
                f"Got: {self.tuning_strategy!r}"
            )
        return super().fit(X, y)

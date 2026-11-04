"""Kumo Tabular preprocessor for TabTune's DataProcessor.

Kumo Tabular infers column types and runs its own per-estimator preprocessing
recipe, so features pass through untouched.
"""

from .tabldm_preprocessor import TabLDMPreprocessor


class KumoTabularPreprocessor(TabLDMPreprocessor):
    pass

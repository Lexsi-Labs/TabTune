"""Bridges between TabTune's time series and tabular models.

The forecasting direction (tabular models as forecasters) is served by the
``TabPFN-TS`` and ``TabularTS-*`` time series models. This package holds the
other direction: time series histories turned into features for tabular
models.

* :class:`SeriesFeaturizer`: leak-safe, per-row history features (summary
  statistics and time series model embeddings), scikit-learn compatible.
* :func:`add_series_features`: the one-call form.
"""

from .series_features import SUMMARY_NAMES, SeriesFeaturizer, add_series_features, summarise_history

__all__ = ["SUMMARY_NAMES", "SeriesFeaturizer", "add_series_features", "summarise_history"]

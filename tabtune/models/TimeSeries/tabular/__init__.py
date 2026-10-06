"""Tabular foundation models as time series forecasters.

* :class:`TabPFNTSAdapter` serves ``TabPFN-TS`` (Hoo et al., 2025), a
  re-implementation on TabTune's vendored TabPFN regressors.
* :class:`TabularForecasterAdapter` serves ``TabularTS-<model>`` for every
  TabTune tabular model with a regression head, with TabPFN-TS's time features
  or a pooled direct multi-horizon lag design, and the ``TabularTS-GBM``
  gradient-boosting baseline.

Importing this package does not import torch or any model code.
"""

from .backends import (
    NATIVE_QUANTILE_MODELS,
    TABPFN_VERSIONS,
    PipelineBackend,
    SklearnBackend,
    TabPFNBackend,
    TabularRegressorBackend,
    gbm_backend,
    rearrange_quantiles,
)
from .features import (
    CALENDAR_SEASONALITIES,
    LagDesign,
    calendar_features,
    find_seasonal_periods,
    lag_design,
    time_design,
)
from .forecaster import GBM_CHECKPOINT, TabularForecasterAdapter
from .tabpfn_ts import TABPFN_TS_VARIANTS, TabPFNTSAdapter, handle_missing

__all__ = [
    "CALENDAR_SEASONALITIES",
    "GBM_CHECKPOINT",
    "LagDesign",
    "NATIVE_QUANTILE_MODELS",
    "PipelineBackend",
    "SklearnBackend",
    "TABPFN_TS_VARIANTS",
    "TABPFN_VERSIONS",
    "TabPFNBackend",
    "TabPFNTSAdapter",
    "TabularForecasterAdapter",
    "TabularRegressorBackend",
    "calendar_features",
    "find_seasonal_periods",
    "gbm_backend",
    "handle_missing",
    "lag_design",
    "rearrange_quantiles",
    "time_design",
]

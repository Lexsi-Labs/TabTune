"""
tabtune.causal.adapters
=======================

Sklearn-compatible adapter that wraps any TabTune :class:`TabularPipeline`
as a drop-in estimator for causal-inference libraries (DoubleML, EconML).

The adapter is the single piece of glue that lets every TFM in TabTune's
registry (TabPFN, TabPFNv26, TabICL, TabICLv2, Mitra, OrionMSP, OrionBix,
TabDPT, ConTextTab, Limix, ...) serve as a nuisance learner inside causal
estimators that expect a scikit-learn-style ``.fit / .predict / .predict_proba``
interface.

Design notes
------------
* sklearn's ``clone(estimator)`` performs an *identity* check on every
  constructor parameter: ``self.<name> is <value_passed_in>`` must hold.
  Defaulting ``self.tuning_params = tuning_params or {}`` would break that
  check because ``{}`` is a brand-new object. DoubleML calls ``clone()``
  internally during cross-fitting, so this would fail at fit time with a
  cryptic ``RuntimeError``. We therefore store constructor arguments
  exactly as received and only materialise defaults inside ``.fit()``.

* The adapter accepts either ``numpy`` arrays or ``pandas`` objects. Causal
  libraries internally hand off numpy arrays during cross-fitting; the
  TabTune pipeline expects DataFrames. We convert on the way in.

* For classification, ``predict_proba`` is delegated to the underlying
  pipeline. We also expose ``classes_`` after fit, as sklearn-consumer code
  relies on it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


try:
    from ..TabularPipeline import TabularPipeline  # type: ignore
except Exception:  # pragma: no cover
    try:
        from tabtune import TabularPipeline  # type: ignore
    except Exception:
        TabularPipeline = None  # type: ignore


logger = logging.getLogger(__name__)


class _TabTuneSklearnAdapter(BaseEstimator):
    """
    Wrap a :class:`TabularPipeline` as a scikit-learn-compatible estimator.

    The wrapped pipeline is lazily constructed inside ``fit`` so that
    sklearn's ``clone`` can copy parameters by value while preserving
    identity semantics on stored attributes.

    Parameters
    ----------
    model_name : str
        Identifier in the TabTune model registry (e.g. ``'TabPFNv26'``,
        ``'TabICLv2'``, ``'Mitra'``, ``'OrionMSP'``, ``'TabDPT'``).
    task_type : str
        Either ``'classification'`` or ``'regression'``. Causal estimators
        typically need a regressor for the outcome model ``g(X)`` and a
        classifier for the treatment-propensity model ``m(X)``.
    tuning_strategy : str, default ``'inference'``
        One of ``'inference' | 'finetune' | 'peft'``. Mirrors the
        :class:`TabularPipeline` argument.
    tuning_params : dict, optional
        Forwarded to :class:`TabularPipeline.tuning_params`. Common keys
        include ``device`` and finetune-specific options.
    processor_params : dict, optional
        Forwarded to :class:`TabularPipeline.processor_params`. Common keys
        include context-sampling controls.
    model_params : dict, optional
        Forwarded to :class:`TabularPipeline.model_params`.

    Attributes
    ----------
    classes_ : ndarray
        Available after ``fit`` for classification tasks. Required by some
        sklearn-consumer code in EconML and DoubleML.
    """

    def __init__(
        self,
        model_name: str | None = None,
        task_type: str | None = None,
        tuning_strategy: str = "inference",
        tuning_params: dict | None = None,
        processor_params: dict | None = None,
        model_params: dict | None = None,
    ):
        
        self.model_name = model_name
        self.task_type = task_type
        self.tuning_strategy = tuning_strategy
        self.tuning_params = tuning_params
        self.processor_params = processor_params
        self.model_params = model_params

    
    @property
    def _estimator_type(self) -> str:
        return "classifier" if self.task_type == "classification" else "regressor"

    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = self._estimator_type
        
        if hasattr(tags, "target_tags"):
            tags.target_tags.required = True
        if self.task_type == "classification":
            if hasattr(tags, "classifier_tags") and tags.classifier_tags is not None:
                
                pass
        return tags

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_dataframe(X: Any) -> pd.DataFrame:
        """Coerce numpy arrays to DataFrames for TabTune compatibility."""
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        # Last resort: try DataFrame constructor.
        return pd.DataFrame(X)

    @staticmethod
    def _as_series(y: Any) -> pd.Series:
        """Coerce y to a pandas Series."""
        if isinstance(y, pd.Series):
            return y
        return pd.Series(np.asarray(y).ravel())

    def _build_pipeline(self) -> "TabularPipeline":
        """Materialise a fresh TabularPipeline with effective parameters."""
        if TabularPipeline is None:
            raise ImportError(
                "TabularPipeline could not be imported. Ensure TabTune is installed."
            )
        tp = self.tuning_params if self.tuning_params is not None else {}
        pp = self.processor_params if self.processor_params is not None else {}
        mp = self.model_params if self.model_params is not None else {}
        return TabularPipeline(
            model_name=self.model_name,
            task_type=self.task_type,
            tuning_strategy=self.tuning_strategy,
            tuning_params=tp,
            processor_params=pp,
            model_params=mp,
        )

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "_TabTuneSklearnAdapter":
        """Fit the underlying TabTune pipeline on ``(X, y)``."""
        X_df = self._as_dataframe(X)
        y_ser = self._as_series(y)
        self._pipe = self._build_pipeline()
        self._pipe.fit(X_df, y_ser)
        if self.task_type == "classification":
            self.classes_ = np.array(sorted(pd.unique(y_ser)))
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict point estimates."""
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                "_TabTuneSklearnAdapter must be .fit() before .predict()."
            )
        X_df = self._as_dataframe(X)
        return np.asarray(self._pipe.predict(X_df))

   
    def _predict_proba_impl(self, X: Any) -> np.ndarray:
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                "_TabTuneSklearnAdapter must be .fit() before .predict_proba()."
            )
        X_df = self._as_dataframe(X)
        return np.asarray(self._pipe.predict_proba(X_df))

    def __getattr__(self, name: str):
        
        if name == "predict_proba":
            # Access __dict__ directly to avoid recursing through __getattr__.
            tt = self.__dict__.get("task_type")
            if tt == "classification":
                return self._predict_proba_impl
            raise AttributeError(
                "_TabTuneSklearnAdapter has no attribute 'predict_proba' "
                "(regression adapter)."
            )
        raise AttributeError(
            f"_TabTuneSklearnAdapter has no attribute {name!r}"
        )


def as_sklearn(
    model_name: str,
    task_type: str,
    tuning_strategy: str = "inference",
    tuning_params: dict | None = None,
    processor_params: dict | None = None,
    model_params: dict | None = None,
) -> _TabTuneSklearnAdapter:
    """
    Convenience factory that returns a sklearn-adapted TabTune pipeline.

    Mirrors the :class:`TabularPipeline` constructor so existing TabTune
    users can lift their nuisance-learner config directly.

    Examples
    --------
    >>> ml_g = as_sklearn('TabPFNv26', 'regression',
    ...                   tuning_strategy='inference',
    ...                   tuning_params={'device': 'cuda'})
    >>> ml_m = as_sklearn('TabICLv2', 'classification',
    ...                   tuning_strategy='inference')
    """
    return _TabTuneSklearnAdapter(
        model_name=model_name,
        task_type=task_type,
        tuning_strategy=tuning_strategy,
        tuning_params=tuning_params,
        processor_params=processor_params,
        model_params=model_params,
    )
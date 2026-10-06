"""scikit-learn entry points; estimator construction never loads model weights."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError

from .engine import Engine
from .quantiles import interpolate_quantiles, validate_quantiles
from .serialization import export_estimator, import_estimator


def fit_adapter(estimator, X, y, task):
    """Invalidate old fitted attributes before replacing the engine's context."""
    for name in ("classes_", "n_features_in_", "feature_names_in_"):
        estimator.__dict__.pop(name, None)
    engine = getattr(estimator, "_engine", None)
    if engine is not None:
        engine.state = None
    if engine is None or engine.device_request != estimator.device:
        engine = Engine(task, estimator.device)
        estimator._engine = engine
    engine.fit(
        X,
        y,
        n_estimators=estimator.n_estimators,
        use_kv_cache=estimator.use_kv_cache,
        retain_preprocessing=estimator.retain_preprocessing,
        random_state=estimator.random_state,
    )
    state = engine.require_state().dataset
    estimator.n_features_in_ = state.encoder.width
    if state.encoder.names is not None and all(isinstance(name, str) for name in state.encoder.names):
        estimator.feature_names_in_ = np.asarray(state.encoder.names, dtype=object)
    if task == "classification":
        estimator.classes_ = state.target_encoder.classes_
    return estimator


def fitted_engine(estimator):
    engine = getattr(estimator, "_engine", None)
    if engine is None:
        raise NotFittedError("Call fit successfully before prediction")
    engine.require_state()
    return engine


def supported_input_tags(tags):
    tags.input_tags.allow_nan = True
    tags.input_tags.categorical = True
    return tags


class CausiloClassifier(ClassifierMixin, BaseEstimator):
    """Classify tables using a fixed pretrained model and training-row context.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of preprocessing/permutation ensemble members, sharing weights.
        For many-class classification, every output-code row uses this ensemble.
    random_state : int, default=42
        Nonnegative seed for feature and class permutations; None is unsupported.
    device : str, default="auto"
        CPU or one CUDA device. Auto selects CUDA when available, otherwise CPU.
    use_kv_cache : bool, default=False
        Precompute attention context during fit, retaining it on the device.
    retain_preprocessing : bool, default=True
        Keep transformed training tables instead of recomputing them for prediction.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Original class labels in predict_proba column order.
    n_features_in_ : int
        Input width before constant features are removed.
    feature_names_in_ : numpy.ndarray
        Fitted column names, available when all DataFrame names are strings.
    """

    __getstate__ = export_estimator
    __setstate__ = import_estimator

    def __sklearn_tags__(self):
        return supported_input_tags(super().__sklearn_tags__())

    def __init__(
        self,
        n_estimators: int = 8,
        *,
        random_state: int = 42,
        device: str = "auto",
        use_kv_cache: bool = False,
        retain_preprocessing: bool = True,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self.use_kv_cache = use_kv_cache
        self.retain_preprocessing = retain_preprocessing

    def fit(self, X, y):
        """Prepare a context from X (rows, features) and y (rows,); return self.

        No gradient training occurs. Feature NaNs are supported, target NaNs
        are rejected. The native head uses error-correcting output
        codes when there are more than ten target classes.
        Parameter changes take effect on refit; a failed refit clears old state.
        """
        return fit_adapter(self, X, y, "classification")

    def predict_proba(self, X):
        """Return (rows, classes) probabilities using the fitted feature schema."""
        return fitted_engine(self).predict(X)

    def predict(self, X):
        """Return the original label with highest probability for each input row."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


class CausiloRegressor(RegressorMixin, BaseEstimator):
    """Predict one continuous target using a fixed pretrained context model.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of preprocessing/permutation ensemble members, sharing weights.
    random_state : int, default=42
        Nonnegative seed for feature permutations; None is unsupported.
    device : str, default="auto"
        CPU or one CUDA device. Auto selects CUDA when available, otherwise CPU.
    use_kv_cache : bool, default=False
        Precompute and retain attention context during fit for repeated prediction.
    retain_preprocessing : bool, default=True
        Retain transformed training tables; encoded training data is always kept.

    Fitted attributes are n_features_in_ and, for DataFrames with string column
    names, feature_names_in_. Change settings before fit or refit to apply them.
    """

    __getstate__ = export_estimator
    __setstate__ = import_estimator

    def __sklearn_tags__(self):
        return supported_input_tags(super().__sklearn_tags__())

    def __init__(
        self,
        n_estimators: int = 8,
        *,
        random_state: int = 42,
        device: str = "auto",
        use_kv_cache: bool = False,
        retain_preprocessing: bool = True,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self.use_kv_cache = use_kv_cache
        self.retain_preprocessing = retain_preprocessing

    def fit(self, X, y):
        """Prepare X (rows, features) and finite y (rows,) without weight updates.

        Return self. Feature NaNs are supported; a failed refit clears old state.
        """
        return fit_adapter(self, X, y, "regression")

    def predict(self, X, *, output_type: str = "mean", quantiles=None):
        """Predict in target units using the selected output mode.

        mean (default) and median return (rows,); raw returns (rows, 999).
        quantiles returns (rows, requested levels) for probabilities in (0, 1).
        """
        if not isinstance(output_type, str) or output_type not in ("mean", "median", "raw", "quantiles"):
            raise ValueError("output_type must be 'mean', 'median', 'raw', or 'quantiles'")
        if output_type != "quantiles" and quantiles is not None:
            raise ValueError("quantiles is only valid with output_type='quantiles'")
        levels = validate_quantiles(quantiles) if output_type == "quantiles" else None
        engine = fitted_engine(self)
        if output_type == "mean":
            return engine.predict(X)
        raw = engine.predict_raw(X)
        if output_type == "median":
            return raw[:, raw.shape[1] // 2]
        return raw if output_type == "raw" else interpolate_quantiles(raw, levels)

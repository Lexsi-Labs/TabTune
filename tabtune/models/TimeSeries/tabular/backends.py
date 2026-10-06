"""Tabular regressors behind one ``fit`` / ``predict(levels)`` interface.

* :class:`TabPFNBackend`: TabTune's vendored TabPFN regressors (v2, v2.6,
  v3, v3.5, v3.5-fast), called as TabPFN-TS calls ``tabpfn``:
  ``predict(X, output_type="main", quantiles=levels)``, with the point taken
  from the ``"median"``, ``"mean"`` or ``"mode"`` output.
* :class:`PipelineBackend`: any TabTune tabular model with a regression head,
  through :class:`~tabtune.TabularPipeline.pipeline.TabularPipeline`
  (``task_type="regression"``, zero-shot ``inference``), with native quantiles
  where :meth:`TabularPipeline.predict_quantiles` exposes them.
* :class:`SklearnBackend`: any scikit-learn regressor, with optional
  per-level quantile regressors (the gradient-boosting baseline).

Every backend is re-fitted per call. Nothing here imports torch at module level.
"""

from __future__ import annotations

import copy
import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ....registry.errors import ConfigError
from ....registry.TimeSeriesCatalog import NATIVE_QUANTILE_TABULAR_MODELS

__all__ = [
    "PipelineBackend",
    "SklearnBackend",
    "TabPFNBackend",
    "TabularRegressorBackend",
    "TABPFN_VERSIONS",
    "NATIVE_QUANTILE_MODELS",
    "rearrange_quantiles",
]

#: Vendored TabPFN regressor classes by version.
TABPFN_VERSIONS: dict[str, tuple[str, str]] = {
    "v2": ("tabtune.models.tabpfn", "TabPFNRegressor"),
    "v2.6": ("tabtune.models.tabpfnv26", "TabPFNv26Regressor"),
    "v3": ("tabtune.models.tabpfnv3", "TabPFNv3Regressor"),
    "v3.5": ("tabtune.models.tabpfnv35", "TabPFNv35Regressor"),
    "v3.5-fast": ("tabtune.models.tabpfnv35", "TabPFNv35FastRegressor"),
}

#: TabTune tabular models whose regression head returns quantiles natively.
NATIVE_QUANTILE_MODELS: frozenset[str] = NATIVE_QUANTILE_TABULAR_MODELS


def rearrange_quantiles(quantiles: np.ndarray) -> np.ndarray:
    """Sort quantiles along the last axis (Chernozhukov et al., 2010, rearrangement).

    Independently fitted quantile regressors can cross. Sorting each row gives
    monotone quantiles and never increases the pinball loss.
    """
    return np.sort(np.asarray(quantiles, dtype=float), axis=-1)


def _constant_output(y: np.ndarray, n: int, levels: Sequence[float]) -> tuple[np.ndarray, np.ndarray | None]:
    value = float(y[0]) if len(y) else 0.0
    point = np.full(n, value)
    return point, (np.full((n, len(levels)), value) if levels else None)


class TabularRegressorBackend(ABC):
    """``fit(X, y)`` then ``predict(X, levels) -> (point [n], quantiles [n, q] | None)``."""

    #: Whether :meth:`predict` returns quantiles.
    native_quantiles: bool = False
    #: What the point forecast estimates (reported in forecast metadata).
    point_statistic: str = "mean"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> TabularRegressorBackend:
        """Fit on a table; returns ``self``."""

    @abstractmethod
    def predict(
        self, X: pd.DataFrame, levels: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Point forecast ``[n]`` and quantiles ``[n, len(levels)]`` (or ``None``)."""

    def clone(self) -> TabularRegressorBackend:
        """An unfitted copy (the default deep-copies the configuration)."""
        return copy.deepcopy(self)


class TabPFNBackend(TabularRegressorBackend):
    """A vendored TabPFN regressor, called as TabPFN-TS calls it.

    Its point forecast is the median of TabPFN's predictive distribution.

    Args:
        version: Key of :data:`TABPFN_VERSIONS`.
        model_path: Checkpoint file name or path (``None``: the version's
            default). A bare file name resolves to TabPFN's cache directory
            and is downloaded on first use.
        output_selection: Which ``output_type="main"`` statistic is the point
            forecast: ``"median"`` (TabPFN-TS's default), ``"mean"`` or ``"mode"``.
        device: Torch device string.
        estimator_kwargs: Extra constructor arguments (``n_estimators``,
            ``random_state``, ...). None are set by default, so the
            estimator uses TabPFN's defaults, as TabPFN-TS does.
        factory: Test hook: a callable returning an estimator with TabPFN's
            ``fit`` / ``predict(output_type="main", quantiles=...)`` API.
    """

    native_quantiles = True

    point_statistic = "median"

    def __init__(
        self,
        version: str = "v3.5",
        *,
        model_path: str | None = None,
        output_selection: str = "median",
        device: str = "auto",
        estimator_kwargs: Mapping[str, Any] | None = None,
        factory: Callable[..., Any] | None = None,
    ) -> None:
        if version not in TABPFN_VERSIONS:
            raise ConfigError(
                f"TabPFN version must be one of {list(TABPFN_VERSIONS)}, got {version!r}."
            )
        if output_selection not in ("median", "mean", "mode"):
            raise ConfigError(
                f"output_selection must be 'median', 'mean' or 'mode', got {output_selection!r}."
            )
        self.version = version
        self.model_path = model_path
        self.output_selection = output_selection
        self.device = device
        self.estimator_kwargs = dict(estimator_kwargs or {})
        self.factory = factory
        self._estimator: Any = None
        self._constant: np.ndarray | None = None

    def _make(self) -> Any:
        kwargs = dict(self.estimator_kwargs)
        if self.model_path is not None:
            kwargs["model_path"] = self.model_path
        kwargs.setdefault("device", self.device)
        if self.factory is not None:
            return self.factory(**kwargs)
        module, name = TABPFN_VERSIONS[self.version]
        return getattr(importlib.import_module(module), name)(**kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> TabPFNBackend:
        y = np.asarray(y, dtype=float)
        # TabPFN standardises the target; a constant target has no scale.
        self._constant = y if y.size == 0 or np.ptp(y) == 0 else None
        if self._constant is None:
            self._estimator = self._make()
            self._estimator.fit(np.asarray(X, dtype=float), y)
        return self

    def predict(self, X: pd.DataFrame, levels: Sequence[float]) -> tuple[np.ndarray, np.ndarray | None]:
        levels = [float(q) for q in levels]
        if self._constant is not None:
            return _constant_output(self._constant, len(X), levels)
        request = levels or [0.5]
        out = self._estimator.predict(
            np.asarray(X, dtype=float), output_type="main", quantiles=request
        )
        point = np.asarray(out[self.output_selection], dtype=float)
        quantiles = np.stack([np.asarray(q, dtype=float) for q in out["quantiles"]], axis=-1)
        return point, (rearrange_quantiles(quantiles) if levels else None)

    def clone(self) -> TabPFNBackend:
        return TabPFNBackend(
            self.version,
            model_path=self.model_path,
            output_selection=self.output_selection,
            device=self.device,
            estimator_kwargs=self.estimator_kwargs,
            factory=self.factory,
        )


class PipelineBackend(TabularRegressorBackend):
    """Any TabTune tabular model with a regression head, via ``TabularPipeline``.

    Args:
        model_name: Registered tabular model name (``"TabICLv2"``, ``"Causilo"``, ...).
        model_params, processor_params: Forwarded to ``TabularPipeline``.
        device: Put into ``model_params["device"]`` unless given there.
        license_mode, envelope_mode: Forwarded, so a forecaster obeys the
            same license and envelope checks as the tabular pipeline.
    """

    def __init__(
        self,
        model_name: str,
        *,
        model_params: Mapping[str, Any] | None = None,
        processor_params: Mapping[str, Any] | None = None,
        device: str | None = None,
        license_mode: str = "research",
        envelope_mode: str = "warn",
    ) -> None:
        from ....registry import resolve_model_name

        self.model_name = resolve_model_name(model_name)
        self.model_params = dict(model_params or {})
        if device is not None:
            self.model_params.setdefault("device", device)
        self.processor_params = dict(processor_params or {})
        self.license_mode = license_mode
        self.envelope_mode = envelope_mode
        self.native_quantiles = self.model_name in NATIVE_QUANTILE_MODELS
        # One loaded TabularPipeline shared by every clone (see clone()).
        self._shared: dict[str, Any] = {"pipeline": None}
        self._constant: np.ndarray | None = None

    @property
    def _pipeline(self) -> Any:
        return self._shared["pipeline"]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> PipelineBackend:
        from ....TabularPipeline.pipeline import TabularPipeline

        y = np.asarray(y, dtype=float)
        self._constant = y if y.size == 0 or np.ptp(y) == 0 else None
        if self._constant is not None:
            return self
        if self._shared["pipeline"] is None:
            self._shared["pipeline"] = TabularPipeline(
                model_name=self.model_name,
                task_type="regression",
                tuning_strategy="inference",
                model_params=dict(self.model_params),
                processor_params=dict(self.processor_params),
                license_mode=self.license_mode,
                envelope_mode=self.envelope_mode,
            )
        self._pipeline.fit(pd.DataFrame(X).reset_index(drop=True), pd.Series(y))
        return self

    def predict(self, X: pd.DataFrame, levels: Sequence[float]) -> tuple[np.ndarray, np.ndarray | None]:
        levels = [float(q) for q in levels]
        if self._constant is not None:
            return _constant_output(self._constant, len(X), levels)
        frame = pd.DataFrame(X).reset_index(drop=True)
        point = np.asarray(self._pipeline.predict(frame), dtype=float).reshape(-1)
        if not (levels and self.native_quantiles):
            return point, None
        result = self._pipeline.predict_quantiles(frame, levels)
        quantiles = np.stack([np.asarray(result[q], dtype=float).reshape(-1) for q in levels], axis=-1)
        return point, rearrange_quantiles(quantiles)

    def clone(self) -> PipelineBackend:
        """A backend sharing this one's loaded pipeline.

        A clone's predictions are valid only until another clone is fitted.
        """
        twin = copy.copy(self)  # shallow: the holder dict is shared
        twin._constant = None
        return twin


class SklearnBackend(TabularRegressorBackend):
    """A scikit-learn regressor, with optional per-level quantile regressors.

    Args:
        point_factory: Returns an unfitted point regressor.
        quantile_factory: ``level -> unfitted regressor`` for that quantile, or
            ``None`` for point forecasts only.
        anchor_levels: If given, quantile regressors are fitted only at these
            levels and every requested level is interpolated between them
            linearly in probit (normal-quantile) space, extrapolating from the
            two outermost anchors. This costs ``len(anchor_levels)`` fits
            instead of one per level. It is exact at the anchors and assumes
            a locally Gaussian shape elsewhere. ``None`` fits every level.
    """

    def __init__(
        self,
        point_factory: Callable[[], Any],
        quantile_factory: Callable[[float], Any] | None = None,
        anchor_levels: Sequence[float] | None = None,
    ) -> None:
        self.point_factory = point_factory
        self.quantile_factory = quantile_factory
        self.anchor_levels = None if anchor_levels is None else sorted(float(q) for q in anchor_levels)
        if self.anchor_levels is not None and (
            len(self.anchor_levels) < 2 or not all(0 < q < 1 for q in self.anchor_levels)
        ):
            raise ConfigError("anchor_levels needs at least two levels in (0, 1).")
        self.native_quantiles = quantile_factory is not None
        self._X: pd.DataFrame | None = None
        self._y: np.ndarray | None = None
        self._point: Any = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SklearnBackend:
        self._X = pd.DataFrame(X).reset_index(drop=True)
        self._y = np.asarray(y, dtype=float)
        self._point = self.point_factory().fit(self._X, self._y) if len(self._y) else None
        return self

    def predict(self, X: pd.DataFrame, levels: Sequence[float]) -> tuple[np.ndarray, np.ndarray | None]:
        levels = [float(q) for q in levels]
        frame = pd.DataFrame(X).reset_index(drop=True)
        if self._point is None or np.ptp(self._y) == 0:
            return _constant_output(self._y if self._y is not None else np.zeros(0), len(frame), levels)
        point = np.asarray(self._point.predict(frame), dtype=float)
        if not (levels and self.quantile_factory is not None):
            return point, None
        fitted_levels = levels if self.anchor_levels is None else self.anchor_levels
        columns = np.stack(
            [
                np.asarray(self.quantile_factory(q).fit(self._X, self._y).predict(frame), dtype=float)
                for q in fitted_levels
            ],
            axis=-1,
        )
        columns = rearrange_quantiles(columns)
        if self.anchor_levels is not None:
            columns = _probit_interpolate(columns, self.anchor_levels, levels)
        return point, rearrange_quantiles(columns)

    def clone(self) -> SklearnBackend:
        return SklearnBackend(self.point_factory, self.quantile_factory, self.anchor_levels)


def _probit_interpolate(
    values: np.ndarray, anchors: Sequence[float], levels: Sequence[float]
) -> np.ndarray:
    """Quantiles at ``levels`` from quantiles at ``anchors`` (``[n, len(anchors)]``).

    Linear in probit space between neighbouring anchors, and linear
    extrapolation from the outermost pair beyond them.
    """
    from scipy.stats import norm

    za = norm.ppf(np.asarray(anchors, dtype=float))
    out = np.empty((values.shape[0], len(levels)))
    for j, level in enumerate(levels):
        z = norm.ppf(level)
        k = int(np.clip(np.searchsorted(za, z) - 1, 0, len(za) - 2))
        w = (z - za[k]) / (za[k + 1] - za[k])
        out[:, j] = values[:, k] + w * (values[:, k + 1] - values[:, k])
    return out


def gbm_backend(
    seed: int | None = 0,
    max_iter: int = 100,
    anchor_levels: Sequence[float] | None = (0.1, 0.5, 0.9),
) -> SklearnBackend:
    """``HistGradientBoostingRegressor`` point model plus quantile-loss models.

    The M5-style gradient-boosting baseline for tabular forecasting. It
    handles ``NaN`` natively, so missing lags need no imputation. By default
    quantile models are fitted at 0.1, 0.5 and 0.9 and other levels are
    probit-interpolated (see :class:`SklearnBackend`). Pass
    ``anchor_levels=None`` to fit one model per requested level.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    def point() -> Any:
        return HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)

    def quantile(level: float) -> Any:
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=level, max_iter=max_iter, random_state=seed
        )

    return SklearnBackend(point, quantile, anchor_levels)

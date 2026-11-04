"""Registry for time series foundation models (TSFMs): lookup and validation.

Specs are declared in :mod:`~tabtune.registry.TimeSeriesSpec` and the built-in ones
listed in :mod:`~tabtune.registry.TimeSeriesCatalog`; this module turns them
into behaviour: name resolution, discovery, request validation and the
forecast envelope check. Time series models live in their own registry dict,
separate from the tabular ``MODEL_REGISTRY``, and a name may never
refer to both a tabular and a time series model.

"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal

from .errors import (
    ConfigError,
    EnvelopeError,
    ModelNotFoundError,
    UnsupportedStrategyError,
    UnsupportedTaskError,
)
from .registry import _ALIASES as _TABULAR_ALIASES
from .spec import EnvelopeViolation, normalise_name
from .TimeSeriesCatalog import TS_MODEL_SPECS
from .TimeSeriesSpec import TIME_SERIES_STRATEGIES, TIME_SERIES_TASKS, TimeSeriesModelSpec

logger = logging.getLogger(__name__)

__all__ = [
    "TimeSeriesModelSpec",
    "TS_MODEL_REGISTRY",
    "TS_MODEL_SPECS",
    "register_time_series_model",
    "resolve_time_series_model_name",
    "get_time_series_model_spec",
    "list_time_series_models",
    "validate_time_series_request",
    "check_forecast_envelope",
    "check_schema_support",
]

TS_MODEL_REGISTRY: dict[str, TimeSeriesModelSpec] = {}

_TS_ALIASES: dict[str, str] = {}

def register_time_series_model(
    spec: TimeSeriesModelSpec, *, overwrite: bool = False
) -> TimeSeriesModelSpec:
    """Register a time series model specification.

    Args:
        spec: The specification to register.
        overwrite: Allow replacing an existing registration of the same name.

    Returns:
        The registered spec.

    Raises:
        TypeError: If ``spec`` is not a :class:`TimeSeriesModelSpec`.
        ValueError: If the name is already registered and ``overwrite`` is
            ``False``, or if a name or alias is already claimed by another time
            series model or by a tabular model.
    """
    if not isinstance(spec, TimeSeriesModelSpec):
        raise TypeError(f"expected TimeSeriesModelSpec, got {type(spec).__name__}")

    if spec.name in TS_MODEL_REGISTRY and not overwrite:
        raise ValueError(
            f"Time series model {spec.name!r} is already registered. "
            f"Pass overwrite=True to replace it."
        )

    for candidate in (spec.name, *spec.aliases):
        key = normalise_name(candidate)
        tabular_owner = _TABULAR_ALIASES.get(key)
        if tabular_owner is not None:
            raise ValueError(
                f"Alias {candidate!r} is already claimed by tabular model {tabular_owner!r}"
            )
        owner = _TS_ALIASES.get(key)
        if owner is not None and owner != spec.name:
            raise ValueError(
                f"Alias {candidate!r} is already claimed by time series model {owner!r}"
            )

    unknown_tasks = spec.tasks - TIME_SERIES_TASKS
    if unknown_tasks:
        raise ValueError(
            f"{spec.name} declares unknown task(s) {sorted(unknown_tasks)}; "
            f"supported: {sorted(TIME_SERIES_TASKS)}"
        )
    unknown_strategies = spec.strategies - TIME_SERIES_STRATEGIES
    if unknown_strategies:
        raise ValueError(
            f"{spec.name} declares unknown strategy(ies) {sorted(unknown_strategies)}; "
            f"supported: {sorted(TIME_SERIES_STRATEGIES)}"
        )

    TS_MODEL_REGISTRY[spec.name] = spec
    for candidate in (spec.name, *spec.aliases):
        _TS_ALIASES[normalise_name(candidate)] = spec.name
    return spec


for _spec in TS_MODEL_SPECS:
    register_time_series_model(_spec)
del _spec


def resolve_time_series_model_name(name: str) -> str:
    """Resolve a user-supplied time series model name to its canonical form.

    Raises:
        ModelNotFoundError: If the name matches no registered time series model.
    """
    canonical = _TS_ALIASES.get(normalise_name(name))
    if canonical is None:
        raise ModelNotFoundError(name, sorted(TS_MODEL_REGISTRY))
    return canonical


def get_time_series_model_spec(name: str) -> TimeSeriesModelSpec:
    """Return the :class:`TimeSeriesModelSpec` for ``name``.

    Raises:
        ModelNotFoundError: If the name matches no registered time series model.
    """
    return TS_MODEL_REGISTRY[resolve_time_series_model_name(name)]


def list_time_series_models(
    *,
    task: str | None = None,
    strategy: str | None = None,
    commercial_ok: bool | None = None,
    family: str | None = None,
    include_unverified_licenses: bool = False,
) -> list[TimeSeriesModelSpec]:
    """Return registered time series models matching the filters, sorted by name.

    Args:
        task: Keep only models implementing this task.
        strategy: Keep only models supporting this tuning strategy.
        commercial_ok: ``True`` keeps models whose weights are cleared for
            commercial use, ``False`` those explicitly restricted.
        family: Keep only models of this architecture family.
        include_unverified_licenses: When filtering with ``commercial_ok=True``,
            also include models whose license TabTune has not verified
            (``commercial_use_ok=None``). Off by default.
    """
    out = []
    for spec in TS_MODEL_REGISTRY.values():
        if task is not None and task not in spec.tasks:
            continue
        if strategy is not None and strategy not in spec.strategies:
            continue
        if family is not None and spec.family != family:
            continue
        if commercial_ok is not None:
            flag = spec.license.commercial_use_ok
            if commercial_ok:
                if not (flag is True or (flag is None and include_unverified_licenses)):
                    continue
            elif flag is not False:
                continue
        out.append(spec)
    return sorted(out, key=lambda s: s.name)


def validate_time_series_request(
    model_name: str, task_type: str, tuning_strategy: str
) -> TimeSeriesModelSpec:
    """Validate a (model, task, strategy) request before any weights are loaded.

    Returns:
        The resolved spec.

    Raises:
        ModelNotFoundError: Unknown model.
        UnsupportedTaskError: The model does not implement ``task_type``.
        UnsupportedStrategyError: The model does not implement ``tuning_strategy``.
    """
    spec = get_time_series_model_spec(model_name)

    if task_type not in spec.tasks:
        raise UnsupportedTaskError(spec.name, task_type, sorted(spec.tasks))

    if tuning_strategy not in spec.strategies:
        alternatives = [
            other.name
            for other in list_time_series_models(task=task_type, strategy=tuning_strategy)
        ]
        raise UnsupportedStrategyError(
            spec.name, tuning_strategy, task_type, sorted(spec.strategies), alternatives
        )
    return spec


def check_forecast_envelope(
    spec: TimeSeriesModelSpec,
    *,
    prediction_length: int,
    mode: Literal["error", "warn", "ignore"] = "warn",
) -> list[EnvelopeViolation]:
    """Check a forecast request against the model's documented limits.

    Returns:
        The violations found, whether or not they were escalated.

    Raises:
        EnvelopeError: If a violation is escalated to an error.
    """
    if mode not in ("error", "warn", "ignore"):
        raise ValueError(
            f"envelope mode must be 'error', 'warn' or 'ignore', got {mode!r}"
        )
    if mode == "ignore":
        return []

    violations: list[EnvelopeViolation] = []
    if spec.max_horizon is not None and prediction_length > spec.max_horizon:
        violations.append(
            EnvelopeViolation(
                constraint="max_horizon",
                limit=spec.max_horizon,
                actual=prediction_length,
                severity="warn",
                message=(
                    f"is documented for horizons up to {spec.max_horizon} steps "
                    f"(requested {prediction_length}); expect degraded accuracy"
                ),
            )
        )
    if not violations:
        return []

    if mode == "error":
        raise EnvelopeError(spec.name, violations)

    for violation in violations:
        warnings.warn(f"{spec.name} {violation.message}", UserWarning, stacklevel=3)
        logger.warning("[Registry] %s %s", spec.name, violation.message)
    return violations


def check_schema_support(
    spec: TimeSeriesModelSpec, schema: Any, *, panel: Any = None
) -> None:
    """Check a data schema against what the model can actually read.

    ``schema`` is duck-typed (it needs ``target_names``, ``past_covariates``
    and ``known_covariates``). Covariate dtypes are only known once the data
    has been read, so pass ``panel`` to also check them.

    Raises:
        ConfigError: Naming the unsupported feature and the models that do
            support it.
    """
    targets = tuple(getattr(schema, "target_names", ()))
    covariates = (
        *getattr(schema, "past_covariates", ()),
        *getattr(schema, "known_covariates", ()),
    )

    if len(targets) > 1 and not spec.supports_multivariate:
        raise ConfigError(
            f"{spec.name} forecasts one target at a time, but the schema names "
            f"{len(targets)} ({list(targets)}). Fit one pipeline per target, or use a "
            f"multivariate model: {_models_supporting('supports_multivariate')}."
        )

    if covariates and not spec.supports_covariates:
        raise ConfigError(
            f"{spec.name} does not use covariates, but the schema names "
            f"{list(covariates)}. Drop them from the schema (they would be ignored), "
            f"or use a covariate-aware model: "
            f"{_models_supporting('supports_covariates')}."
        )

    if covariates and panel is not None and not spec.supports_categorical_covariates:
        categorical = _categorical_covariates(panel)
        if categorical:
            raise ConfigError(
                f"{spec.name} only reads numeric covariates, but {list(categorical)} "
                f"are categorical. Encode them numerically first, or use "
                f"{_models_supporting('supports_categorical_covariates')}."
            )


def _categorical_covariates(panel: Any) -> tuple[str, ...]:
    """Covariate names in ``panel`` whose values are not numeric."""
    rows = (*getattr(panel, "past_covariates", ()), *getattr(panel, "future_covariates", ()))
    names: dict[str, None] = {}
    for row in rows:
        for name, values in row.items():
            if getattr(values, "dtype", None) is not None and values.dtype.kind not in "fiu":
                names[name] = None
    return tuple(names)


def _models_supporting(flag: str) -> str:
    """Names of registered models with a capability flag set, for error messages."""
    names = [s.name for s in TS_MODEL_REGISTRY.values() if getattr(s, flag)]
    return ", ".join(names) if names else "none of the registered models"

"""Model registry: lookup, filtering and request validation.

This module turns the declarative catalog into behaviour. It answers three
kinds of question:

1. *Resolution* - "what does the string ``'tabpfn-v2.6'`` refer to?"
2. *Discovery*  - "which models support regression and can be shipped
   commercially?"
3. *Validation* - "is this (model, task, strategy, data) combination valid,
   and if not, what should the user do instead?"

The registry is deliberately free of heavy imports. Importing it costs a few
milliseconds and does not pull in torch, so the CLI, documentation build and
license checks all stay fast.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from typing import Any, Literal

from .catalog import MODEL_SPECS
from .errors import (
    EnvelopeError,
    LicenseError,
    ModelNotFoundError,
    UnsupportedStrategyError,
    UnsupportedTaskError,
)
from .spec import EnvelopeViolation, LicenseSpec, ModelSpec, normalise_name

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "get_model_spec",
    "resolve_model_name",
    "list_models",
    "list_model_names",
    "models_dataframe",
    "validate_request",
    "check_envelope",
    "check_license",
    "EnvelopeMode",
    "LicenseMode",
]

EnvelopeMode = Literal["error", "warn", "ignore"]
LicenseMode = Literal["research", "commercial", "ignore"]

#: Canonical name -> spec. Populated at import from :data:`MODEL_SPECS`.
MODEL_REGISTRY: dict[str, ModelSpec] = {}

#: Normalised alias -> canonical name.
_ALIASES: dict[str, str] = {}


def register_model(spec: ModelSpec, *, overwrite: bool = False) -> ModelSpec:
    """Register a model specification.

    Third-party packages can call this at import time to make their own models
    visible to TabTune's validation, discovery and reporting layers.

    Args:
        spec: The specification to register.
        overwrite: Allow replacing an existing registration of the same name.

    Returns:
        The registered spec, so this can be used as a decorator-ish one-liner.

    Raises:
        ValueError: If the name is already registered and ``overwrite`` is
            ``False``, or if one of the aliases is claimed by another model.
    """
    if not isinstance(spec, ModelSpec):
        raise TypeError(f"expected ModelSpec, got {type(spec).__name__}")

    if spec.name in MODEL_REGISTRY and not overwrite:
        raise ValueError(
            f"Model {spec.name!r} is already registered. Pass overwrite=True to replace it."
        )

    for candidate in (spec.name, *spec.aliases):
        key = normalise_name(candidate)
        owner = _ALIASES.get(key)
        if owner is not None and owner != spec.name:
            raise ValueError(
                f"Alias {candidate!r} is already claimed by model {owner!r}"
            )

    MODEL_REGISTRY[spec.name] = spec
    for candidate in (spec.name, *spec.aliases):
        _ALIASES[normalise_name(candidate)] = spec.name
    return spec


for _spec in MODEL_SPECS:
    register_model(_spec)
del _spec


# --------------------------------------------------------------------- lookup


def resolve_model_name(name: str) -> str:
    """Resolve a user-supplied model name to its canonical form.

    Matching ignores case, hyphens, underscores, dots and spaces, so
    ``"TabPFN-v2.6"``, ``"tabpfnv26"`` and ``"TabPFN v26"`` all resolve to
    ``"TabPFNv26"``.

    Args:
        name: A model name or alias.

    Returns:
        The canonical model name.

    Raises:
        ModelNotFoundError: If the name does not match any registered model.
    """
    key = normalise_name(name)
    canonical = _ALIASES.get(key)
    if canonical is None:
        raise ModelNotFoundError(name, sorted(MODEL_REGISTRY))
    return canonical


def get_model_spec(name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``name``.

    Args:
        name: A model name or alias.

    Raises:
        ModelNotFoundError: If the name does not match any registered model.
    """
    return MODEL_REGISTRY[resolve_model_name(name)]


def list_model_names() -> list[str]:
    """Return every registered canonical model name, sorted."""
    return sorted(MODEL_REGISTRY)


def list_models(
    *,
    task: str | None = None,
    strategy: str | None = None,
    commercial_ok: bool | None = None,
    family: str | None = None,
    include_unverified_licenses: bool = False,
) -> list[ModelSpec]:
    """Return registered models matching the given filters.

    Args:
        task: Keep only models supporting ``"classification"`` or
            ``"regression"``.
        strategy: Keep only models supporting this tuning strategy. Combined
            with ``task`` when both are given.
        commercial_ok: When ``True``, keep only models whose weights are
            explicitly cleared for commercial use. When ``False``, keep only
            those explicitly restricted.
        family: Keep only models in this architecture family.
        include_unverified_licenses: When filtering with ``commercial_ok=True``,
            also include models whose license TabTune has not verified. Off by
            default so the result is conservative.

    Returns:
        Matching specs, sorted by canonical name.

    Example:
        >>> ", ".join(s.name for s in list_models(task="regression", commercial_ok=True))
        'Mitra, TabICLv2'
    """
    out: list[ModelSpec] = []
    for spec in MODEL_REGISTRY.values():
        if task is not None and task not in spec.tasks:
            continue
        if strategy is not None:
            if task is not None:
                if strategy not in spec.strategies_for(task):
                    continue
            elif not (
                strategy in spec.classification_strategies
                or strategy in spec.regression_strategies
            ):
                continue
        if family is not None and spec.family != family:
            continue
        if commercial_ok is not None:
            flag = spec.license.commercial_use_ok
            if commercial_ok:
                if flag is True:
                    pass
                elif flag is None and include_unverified_licenses:
                    pass
                else:
                    continue
            else:
                if flag is not False:
                    continue
        out.append(spec)
    return sorted(out, key=lambda s: s.name)


def models_dataframe(**filters: Any):
    """Return the registry as a :class:`pandas.DataFrame`.

    Used by the documentation build and the leaderboard report to render the
    supported-models table without hand-maintaining it in Markdown.

    Args:
        **filters: Forwarded to :func:`list_models`.

    Returns:
        A DataFrame with one row per model.
    """
    import pandas as pd

    rows = []
    for spec in list_models(**filters):
        rows.append(
            {
                "Model": spec.name,
                "Family": spec.family,
                "Tasks": ", ".join(spec.tasks),
                "Classification strategies": ", ".join(sorted(spec.classification_strategies)) or "-",
                "Regression strategies": ", ".join(sorted(spec.regression_strategies)) or "-",
                "Max classes": spec.envelope.max_classes,
                "Max rows": spec.envelope.max_rows,
                "Max features": spec.envelope.max_features,
                "License": spec.license.name,
                "Commercial": spec.license.badge,
                "Paper": spec.paper,
            }
        )
    return pd.DataFrame(rows)


def __iter__() -> Iterator[ModelSpec]:  # pragma: no cover - module-level sugar
    return iter(MODEL_REGISTRY.values())


# ----------------------------------------------------------------- validation


def validate_request(
    model_name: str,
    task_type: str,
    tuning_strategy: str,
    finetune_mode: str | None = None,
    *,
    strict_finetune_mode: bool = False,
) -> ModelSpec:
    """Validate a (model, task, strategy) request before any weights are loaded.

    Failing here costs milliseconds; failing after a multi-gigabyte checkpoint
    download costs minutes, so every entry point should call this first.

    Args:
        model_name: Model name or alias.
        task_type: ``"classification"`` or ``"regression"``.
        tuning_strategy: ``"inference"``, ``"finetune"`` or ``"peft"``.
        finetune_mode: Optional fine-tuning algorithm to validate.
        strict_finetune_mode: Raise instead of warning when ``finetune_mode``
            is not one the model implements.

    Returns:
        The resolved :class:`ModelSpec`.

    Raises:
        ModelNotFoundError: Unknown model.
        UnsupportedTaskError: Model has no head for this task.
        UnsupportedStrategyError: Model does not implement this strategy.
    """
    spec = get_model_spec(model_name)

    if task_type not in ("classification", "regression"):
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got {task_type!r}"
        )

    if task_type not in spec.tasks:
        raise UnsupportedTaskError(spec.name, task_type, spec.tasks)

    supported = spec.strategies_for(task_type)
    if tuning_strategy not in supported:
        alternatives = [
            other.name
            for other in list_models(task=task_type, strategy=tuning_strategy)
        ]
        raise UnsupportedStrategyError(
            spec.name, tuning_strategy, task_type, supported, alternatives
        )

    if spec.is_experimental(tuning_strategy):
        warnings.warn(
            f"{spec.name} support for tuning_strategy={tuning_strategy!r} is "
            f"experimental and may produce unstable predictions. Consider "
            f"tuning_strategy='finetune' instead.",
            UserWarning,
            stacklevel=3,
        )

    if finetune_mode is not None and tuning_strategy in ("finetune", "peft"):
        if spec.finetune_modes and finetune_mode not in spec.finetune_modes:
            message = (
                f"{spec.name} does not implement finetune_mode={finetune_mode!r}. "
                f"Implemented: {sorted(spec.finetune_modes)}."
            )
            if strict_finetune_mode:
                raise UnsupportedStrategyError(
                    spec.name, finetune_mode, task_type, spec.finetune_modes
                )
            warnings.warn(message, UserWarning, stacklevel=3)

    return spec


def check_envelope(
    spec: ModelSpec | str,
    *,
    n_rows: int | None = None,
    n_features: int | None = None,
    n_classes: int | None = None,
    mode: EnvelopeMode = "warn",
) -> list[EnvelopeViolation]:
    """Check a dataset shape against a model's capability envelope.

    Hard constraints (architectural limits such as the number of output
    classes) always raise unless ``mode="ignore"``, because no amount of extra
    hardware makes them work. Soft constraints (row and feature counts) warn by
    default, since exceeding them degrades quality rather than being invalid.

    Args:
        spec: A model spec or name.
        n_rows: Training row count.
        n_features: Feature column count.
        n_classes: Distinct class count, for classification.
        mode: ``"error"`` escalates every violation, ``"warn"`` is the default,
            ``"ignore"`` disables checking entirely.

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

    if isinstance(spec, str):
        spec = get_model_spec(spec)

    violations = spec.envelope.check(
        n_rows=n_rows, n_features=n_features, n_classes=n_classes
    )
    if not violations:
        return []

    hard = [v for v in violations if v.severity == "error"]
    escalate = hard if mode == "warn" else violations

    if escalate:
        compatible = _compatible_models(spec, n_classes=n_classes)
        error = EnvelopeError(spec.name, escalate)
        if compatible:
            error.args = (
                error.args[0]
                + "\n  Compatible models for this data: "
                + ", ".join(compatible),
            )
        raise error

    for violation in violations:
        warnings.warn(f"{spec.name} {violation.message}", UserWarning, stacklevel=3)
        logger.warning("[Registry] %s %s", spec.name, violation.message)

    return violations


def check_license(
    spec: ModelSpec | str,
    mode: LicenseMode = "research",
) -> LicenseSpec:
    """Check a model's weight license against the intended usage mode.

    Args:
        spec: A model spec or name.
        mode: ``"research"`` (default) permits everything and stays quiet;
            ``"commercial"`` raises for restricted weights and warns for
            unverified ones; ``"ignore"`` disables the check.

    Returns:
        The model's :class:`LicenseSpec`.

    Raises:
        LicenseError: If ``mode="commercial"`` and the weights are explicitly
            restricted.
    """
    if mode not in ("research", "commercial", "ignore"):
        raise ValueError(
            f"license mode must be 'research', 'commercial' or 'ignore', got {mode!r}"
        )
    if isinstance(spec, str):
        spec = get_model_spec(spec)
    if mode in ("ignore", "research"):
        return spec.license

    flag = spec.license.commercial_use_ok
    if flag is False:
        raise LicenseError(
            spec.name,
            spec.license.name,
            url=spec.license.url,
            alternatives=spec.commercial_alternatives,
            notes=spec.license.notes,
        )
    if flag is None:
        warnings.warn(
            f"TabTune has not verified whether {spec.name} weights "
            f"({spec.license.name}) permit commercial use. "
            f"{spec.license.notes or 'Confirm the terms upstream before deploying.'}"
            + (f" See {spec.license.url}" if spec.license.url else ""),
            UserWarning,
            stacklevel=3,
        )
    elif spec.license.requires_attribution:
        logger.info(
            "[Registry] %s weights are %s and require attribution when deployed.",
            spec.name,
            spec.license.name,
        )
    return spec.license


#: How many alternatives an envelope error names before it gets unhelpfully long.
_MAX_SUGGESTIONS = 6


def _compatible_models(spec: ModelSpec, *, n_classes: int | None) -> list[str]:
    """Return models whose envelope would accept data that ``spec`` rejected.

    Ordered by headroom - models with no declared ceiling first, then the
    highest ceilings - rather than alphabetically. Sorting by name and then
    truncating meant a model late in the alphabet was never suggested no matter
    how well it fit, which is how xRFM went missing from 120-class suggestions.

    Args:
        spec: The model that rejected the data.
        n_classes: The class count that triggered the rejection.

    Returns:
        Up to :data:`_MAX_SUGGESTIONS` model names, best fit first, followed by
        a "(+N more)" marker when the list was truncated.
    """
    if n_classes is None:
        return []

    candidates: list[tuple[int, str]] = []
    for candidate in MODEL_REGISTRY.values():
        if candidate.name == spec.name or not candidate.supports_classification:
            continue
        limit = candidate.envelope.max_classes
        if limit is None:
            headroom = -1  # unlimited sorts first
        elif limit >= n_classes:
            headroom = limit
        else:
            continue
        candidates.append((headroom, candidate.name))

    # (-1, ...) first for unlimited, then descending limit, then name for stability.
    candidates.sort(key=lambda item: (item[0] != -1, -item[0], item[1]))
    names = [name for _, name in candidates]
    if len(names) <= _MAX_SUGGESTIONS:
        return names
    # Never truncate silently: an error that lists six options reads as "these
    # are your options". Say how many were withheld and where to find them.
    hidden = len(names) - _MAX_SUGGESTIONS
    return names[:_MAX_SUGGESTIONS] + [
        f"(+{hidden} more - see tabtune.registry.list_models)"
    ]


def infer_data_shape(X: Any, y: Any = None, task_type: str = "classification") -> dict[str, int | None]:
    """Best-effort extraction of ``(n_rows, n_features, n_classes)`` from inputs.

    Accepts DataFrames, numpy arrays and anything exposing ``.shape``. Returns
    ``None`` for any quantity that cannot be determined rather than guessing,
    so envelope checks silently no-op on exotic inputs instead of misfiring.

    Args:
        X: Feature matrix.
        y: Target vector, optional.
        task_type: Used to decide whether class counting is meaningful.

    Returns:
        Mapping with keys ``n_rows``, ``n_features`` and ``n_classes``.
    """
    n_rows: int | None = None
    n_features: int | None = None
    n_classes: int | None = None

    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) >= 1:
        n_rows = int(shape[0])
        if len(shape) >= 2:
            n_features = int(shape[1])
    elif X is not None:
        try:
            n_rows = len(X)
        except TypeError:
            n_rows = None

    if y is not None and task_type == "classification":
        try:
            import numpy as np

            values = getattr(y, "values", y)
            n_classes = int(len(np.unique(np.asarray(values))))
        except Exception:  # pragma: no cover - defensive; exotic target types
            n_classes = None

    return {"n_rows": n_rows, "n_features": n_features, "n_classes": n_classes}

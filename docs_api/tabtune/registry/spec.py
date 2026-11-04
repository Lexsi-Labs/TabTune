"""Declarative descriptions of the tabular foundation models TabTune wraps.

A :class:`ModelSpec` is the single source of truth for everything TabTune knows
about a model that is *not* its implementation: which tasks and tuning
strategies it supports, the architectural limits of its pretrained checkpoint,
the license its weights ship under, and which preprocessor it needs.

The design goal is that adding a model becomes "write a ``ModelSpec``" rather
than "edit five dispatch chains". The specs are frozen dataclasses with no
third-party dependencies beyond the standard library, so importing the registry
is cheap and does not pull in torch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "CapabilityEnvelope",
    "EnvelopeViolation",
    "LicenseSpec",
    "ModelSpec",
    "Severity",
]

Severity = Literal["error", "warn"]

#: Constraints that reflect a hard architectural property of the pretrained
#: checkpoint. Exceeding one of these cannot be worked around by allocating
#: more memory, so they escalate to an error by default.
HARD_CONSTRAINTS: frozenset[str] = frozenset({"max_classes", "min_rows"})


@dataclass(frozen=True)
class EnvelopeViolation:
    """A single way in which a dataset falls outside a model's envelope.

    Attributes:
        constraint: Name of the envelope field that was violated.
        limit: The declared limit.
        actual: The observed value.
        severity: ``"error"`` for architectural limits, ``"warn"`` for limits
            that are about resources rather than correctness.
        message: Human-readable, actionable description.
    """

    constraint: str
    limit: Any
    actual: Any
    severity: Severity
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


@dataclass(frozen=True)
class CapabilityEnvelope:
    """Architectural and practical limits of a pretrained checkpoint.

    ``None`` means "no declared limit"; it is deliberately distinct from a
    large number, because TabTune should not invent limits it has not verified.

    Note that ``max_cells`` is checked in addition to ``max_rows`` and
    ``max_features`` rather than instead of them. Several recent checkpoints
    (TabPFN-3 in particular) advertise a *budget* that trades rows against
    features, so a single scalar row cap would misrepresent them.
    """

    max_classes: int | None = None
    max_features: int | None = None
    max_rows: int | None = None
    max_cells: int | None = None
    min_rows: int = 2
    native_nan: bool = False
    native_text: bool = False
    native_categorical: bool = False
    notes: str = ""

    def check(
        self,
        *,
        n_rows: int | None = None,
        n_features: int | None = None,
        n_classes: int | None = None,
    ) -> list[EnvelopeViolation]:
        """Return every way the given data shape falls outside this envelope.

        Args:
            n_rows: Number of training rows, if known.
            n_features: Number of feature columns, if known.
            n_classes: Number of distinct target classes, for classification.

        Returns:
            A list of violations, empty when the data fits. Ordered with hard
            (``"error"``) violations first so callers can report the most
            important problem first.
        """
        violations: list[EnvelopeViolation] = []

        if n_classes is not None and self.max_classes is not None and n_classes > self.max_classes:
            violations.append(
                EnvelopeViolation(
                    constraint="max_classes",
                    limit=self.max_classes,
                    actual=n_classes,
                    severity="error",
                    message=(
                        f"supports at most {self.max_classes} classes "
                        f"(found {n_classes})"
                    ),
                )
            )

        if n_rows is not None and n_rows < self.min_rows:
            violations.append(
                EnvelopeViolation(
                    constraint="min_rows",
                    limit=self.min_rows,
                    actual=n_rows,
                    severity="error",
                    message=f"needs at least {self.min_rows} rows (found {n_rows})",
                )
            )

        if n_features is not None and self.max_features is not None and n_features > self.max_features:
            violations.append(
                EnvelopeViolation(
                    constraint="max_features",
                    limit=self.max_features,
                    actual=n_features,
                    severity="warn",
                    message=(
                        f"is documented up to {self.max_features:,} features "
                        f"(found {n_features:,}); expect degraded accuracy or "
                        f"out-of-memory errors"
                    ),
                )
            )

        if n_rows is not None and self.max_rows is not None and n_rows > self.max_rows:
            violations.append(
                EnvelopeViolation(
                    constraint="max_rows",
                    limit=self.max_rows,
                    actual=n_rows,
                    severity="warn",
                    message=(
                        f"is documented up to {self.max_rows:,} rows "
                        f"(found {n_rows:,}); consider context sampling via "
                        f"processor_params={{'context_sampling_strategy': 'stratified', "
                        f"'context_size': {self.max_rows}}}"
                    ),
                )
            )

        if n_rows is not None and n_features is not None and self.max_cells is not None:
            cells = n_rows * n_features
            if cells > self.max_cells:
                violations.append(
                    EnvelopeViolation(
                        constraint="max_cells",
                        limit=self.max_cells,
                        actual=cells,
                        severity="warn",
                        message=(
                            f"has a cell budget of ~{self.max_cells:,} "
                            f"(rows x features); this dataset needs {cells:,}"
                        ),
                    )
                )

        violations.sort(key=lambda v: 0 if v.severity == "error" else 1)
        return violations

    def describe(self) -> str:
        """Return a compact one-line human summary of the declared limits."""
        parts: list[str] = []
        if self.max_classes is not None:
            parts.append(f"<={self.max_classes} classes")
        if self.max_rows is not None:
            parts.append(f"<={self.max_rows:,} rows")
        if self.max_features is not None:
            parts.append(f"<={self.max_features:,} features")
        if self.max_cells is not None:
            parts.append(f"<={self.max_cells:,} cells")
        native = [
            label
            for label, flag in (
                ("NaN", self.native_nan),
                ("text", self.native_text),
                ("categorical", self.native_categorical),
            )
            if flag
        ]
        if native:
            parts.append("native " + "/".join(native))
        return "; ".join(parts) if parts else "no declared limits"


@dataclass(frozen=True)
class LicenseSpec:
    """License metadata for a model's *weights*.

    ``commercial_use_ok`` is deliberately tri-state:

    * ``True``  - upstream explicitly permits commercial use.
    * ``False`` - upstream explicitly forbids it, or requires a separate grant.
    * ``None``  - TabTune has not verified it. Treated as "unknown": TabTune
      warns rather than blocking, because inventing a restriction is as wrong
      as ignoring one.
    """

    name: str
    commercial_use_ok: bool | None = None
    url: str = ""
    requires_attribution: bool = False
    notes: str = ""

    @property
    def badge(self) -> str:
        """Return a short symbol suitable for tables and CLI output."""
        if self.commercial_use_ok is True:
            return "yes" if not self.requires_attribution else "yes (attribution)"
        if self.commercial_use_ok is False:
            return "no"
        return "unverified"


@dataclass(frozen=True)
class ModelSpec:
    """Everything TabTune knows about a model other than its implementation.

    Attributes:
        name: Canonical model name, as accepted by :class:`TabularPipeline`.
        aliases: Alternative spellings that resolve to ``name``. Matching is
            case-insensitive and ignores ``-``, ``_``, ``.`` and spaces.
        family: Coarse architecture family, used for grouping in reports.
        classification_strategies: Tuning strategies valid for classification.
        regression_strategies: Tuning strategies valid for regression. Empty
            when the model has no regression head.
        finetune_modes: Fine-tuning algorithms this model implements.
        preprocessor_key: Key into ``DataProcessor``'s preprocessor factory.
        envelope: Architectural limits of the checkpoint.
        license: License of the *weights*.
        commercial_alternatives: Models to suggest when a license check fails.
        paper: Canonical paper or technical report URL.
        weights: Where the weights come from (HF repo id or URL).
    """

    name: str
    family: str
    aliases: tuple[str, ...] = ()
    classification_strategies: frozenset[str] = frozenset({"inference"})
    regression_strategies: frozenset[str] = frozenset()
    finetune_modes: frozenset[str] = frozenset()
    preprocessor_key: str | None = None
    envelope: CapabilityEnvelope = field(default_factory=CapabilityEnvelope)
    license: LicenseSpec = field(default_factory=lambda: LicenseSpec(name="unknown"))
    commercial_alternatives: tuple[str, ...] = ()
    paper: str = ""
    weights: str = ""
    summary: str = ""
    experimental: frozenset[str] = frozenset()

    # ------------------------------------------------------------------ tasks

    @property
    def supports_classification(self) -> bool:
        return bool(self.classification_strategies)

    @property
    def supports_regression(self) -> bool:
        return bool(self.regression_strategies)

    @property
    def tasks(self) -> tuple[str, ...]:
        """Return the task types this model implements."""
        out: list[str] = []
        if self.supports_classification:
            out.append("classification")
        if self.supports_regression:
            out.append("regression")
        return tuple(out)

    def strategies_for(self, task_type: str) -> frozenset[str]:
        """Return the tuning strategies valid for ``task_type``."""
        if task_type == "classification":
            return self.classification_strategies
        if task_type == "regression":
            return self.regression_strategies
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got {task_type!r}"
        )

    def supports(self, task_type: str, tuning_strategy: str) -> bool:
        """Return whether ``tuning_strategy`` is valid for ``task_type``."""
        try:
            return tuning_strategy in self.strategies_for(task_type)
        except ValueError:
            return False

    def is_experimental(self, tuning_strategy: str) -> bool:
        """Return whether support for ``tuning_strategy`` is marked experimental."""
        return tuning_strategy in self.experimental

    # ------------------------------------------------------------------ views

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, used by reports and model cards."""
        return {
            "name": self.name,
            "family": self.family,
            "aliases": list(self.aliases),
            "tasks": list(self.tasks),
            "classification_strategies": sorted(self.classification_strategies),
            "regression_strategies": sorted(self.regression_strategies),
            "finetune_modes": sorted(self.finetune_modes),
            "experimental": sorted(self.experimental),
            "preprocessor_key": self.preprocessor_key,
            "envelope": {
                "max_classes": self.envelope.max_classes,
                "max_features": self.envelope.max_features,
                "max_rows": self.envelope.max_rows,
                "max_cells": self.envelope.max_cells,
                "min_rows": self.envelope.min_rows,
                "native_nan": self.envelope.native_nan,
                "native_text": self.envelope.native_text,
                "native_categorical": self.envelope.native_categorical,
                "notes": self.envelope.notes,
                "summary": self.envelope.describe(),
            },
            "license": {
                "name": self.license.name,
                "commercial_use_ok": self.license.commercial_use_ok,
                "requires_attribution": self.license.requires_attribution,
                "url": self.license.url,
                "notes": self.license.notes,
            },
            "commercial_alternatives": list(self.commercial_alternatives),
            "paper": self.paper,
            "weights": self.weights,
            "summary": self.summary,
        }


def normalise_name(name: str) -> str:
    """Return a canonical lookup key for a model name.

    Lookup is intentionally forgiving: ``"TabPFN-v2.6"``, ``"tabpfnv26"`` and
    ``"TabPFN v26"`` all resolve to the same model, because users routinely
    copy names out of papers, READMEs and Slack messages.

    Args:
        name: A user-supplied model name.

    Returns:
        The lowercase name with ``-``, ``_``, ``.`` and whitespace removed.
    """
    if not isinstance(name, str):
        raise TypeError(f"model name must be a string, got {type(name).__name__}")
    return "".join(ch for ch in name.lower() if ch.isalnum())


def merge_alias_map(specs: Iterable[ModelSpec]) -> Mapping[str, str]:
    """Build a normalised-alias to canonical-name mapping.

    Raises:
        ValueError: If two specs claim the same alias.
    """
    mapping: dict[str, str] = {}
    for spec in specs:
        for candidate in (spec.name, *spec.aliases):
            key = normalise_name(candidate)
            existing = mapping.get(key)
            if existing is not None and existing != spec.name:
                raise ValueError(
                    f"Alias {candidate!r} is claimed by both {existing!r} and {spec.name!r}"
                )
            mapping[key] = spec.name
    return mapping


def _as_frozenset(value: Sequence[str] | frozenset[str] | None) -> frozenset[str]:
    """Coerce a strategy collection to a frozenset (helper for catalog authoring)."""
    if value is None:
        return frozenset()
    return frozenset(value)

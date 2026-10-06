"""Declarative description of a time series foundation model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .spec import LicenseSpec

__all__ = [
    "TimeSeriesModelSpec",
    "TIME_SERIES_TASKS",
    "TIME_SERIES_STRATEGIES",
    "FORECAST_TASKS",
]

TIME_SERIES_TASKS: frozenset[str] = frozenset(
    {"forecasting", "anomaly_detection", "imputation", "embedding"}
)
TIME_SERIES_STRATEGIES: frozenset[str] = frozenset({"inference", "finetune", "peft"})

FORECAST_TASKS: frozenset[str] = frozenset({"forecasting", "anomaly_detection", "imputation"})


def _license_dict(license: LicenseSpec) -> dict[str, Any]:
    """A license as JSON, including the obligations a caller has to honour."""
    return {
        "name": license.name,
        "commercial_use_ok": license.commercial_use_ok,
        "requires_attribution": license.requires_attribution,
        "url": license.url,
        "notes": license.notes,
    }


@dataclass(frozen=True)
class TimeSeriesModelSpec:
    """Everything TabTune knows about a TSFM other than its implementation.

    Attributes:
        name: Canonical model name, as accepted by ``TimeSeriesPipeline``.
        family: Coarse architecture family, used for grouping.
        adapter: The :class:`~tabtune.models.TimeSeries.base.TSFMAdapter`
            subclass that runs the model, either as a ``"package.module:Class"``
            string (imported lazily, the normal case) or as the class itself
            (convenient for tests and third-party registrations).
        aliases: Alternative spellings that resolve to ``name``.
        tasks: Task types the model implements: ``"forecasting"``,
            ``"anomaly_detection"`` and ``"imputation"`` for every forecaster,
            plus ``"embedding"`` when the adapter exposes its encoder.
        strategies: Adaptation strategies the model implements:
            ``"inference"`` (zero-shot), ``"finetune"`` (all weights) and
            ``"peft"`` (LoRA adapters).
        default_checkpoint: Checkpoint used when ``model_params`` names none.
        checkpoints: Every checkpoint TabTune has validated for this spec. A
            local checkpoint directory is also accepted.
        max_context: Longest history the model attends to. Longer histories
            are left-truncated. ``None`` means no declared limit.
        max_horizon: Longest horizon the model is documented for. Exceeding it
            is a soft violation. ``None`` means no declared limit.
        native_missing: Whether the model accepts NaN in the target history.
        supports_multivariate: Whether the model forecasts several target
            variates of one item jointly. When ``False``, a schema naming more
            than one target is rejected.
        supports_covariates: Whether the model conditions on covariates. When
            ``False``, a schema naming any covariate is rejected.
        supports_categorical_covariates: Whether non-numeric covariates are
            understood. Only meaningful when ``supports_covariates``.
        dependency_extra: For adapters backed by an optional pip package: the
            ``pip install 'tabtune[<extra>]'`` extra named in the import error
            when that package is missing. ``None`` for vendored models.
        license: License of the *weights* of the default checkpoint, and of
            every checkpoint not listed in ``checkpoint_licenses``.
        checkpoint_licenses: ``(checkpoint, LicenseSpec)`` pairs for
            checkpoints whose weights are licensed differently.
        commercial_alternatives: Models to suggest when a license check fails.
        paper: Canonical paper URL.
        summary: One-line description.
    """

    name: str
    family: str
    adapter: str | type
    aliases: tuple[str, ...] = ()
    tasks: frozenset[str] = FORECAST_TASKS
    strategies: frozenset[str] = frozenset({"inference"})
    default_checkpoint: str = ""
    checkpoints: tuple[str, ...] = ()
    max_context: int | None = None
    max_horizon: int | None = None
    native_missing: bool = False
    supports_multivariate: bool = False
    supports_covariates: bool = False
    supports_categorical_covariates: bool = False
    dependency_extra: str | None = None
    license: LicenseSpec = field(default_factory=lambda: LicenseSpec(name="unknown"))
    checkpoint_licenses: tuple[tuple[str, LicenseSpec], ...] = ()
    commercial_alternatives: tuple[str, ...] = ()
    paper: str = ""
    summary: str = ""

    def license_for(self, checkpoint: str | None) -> LicenseSpec:
        """The weight license of ``checkpoint``."""
        return dict(self.checkpoint_licenses).get(checkpoint or self.default_checkpoint, self.license)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, used by reports and model cards."""
        adapter = self.adapter if isinstance(self.adapter, str) else (
            f"{self.adapter.__module__}:{self.adapter.__qualname__}"
        )
        return {
            "name": self.name,
            "family": self.family,
            "aliases": list(self.aliases),
            "tasks": sorted(self.tasks),
            "strategies": sorted(self.strategies),
            "adapter": adapter,
            "default_checkpoint": self.default_checkpoint,
            "checkpoints": list(self.checkpoints),
            "max_context": self.max_context,
            "max_horizon": self.max_horizon,
            "native_missing": self.native_missing,
            "supports_multivariate": self.supports_multivariate,
            "supports_covariates": self.supports_covariates,
            "supports_categorical_covariates": self.supports_categorical_covariates,
            "dependency_extra": self.dependency_extra,
            "license": _license_dict(self.license),
            "checkpoint_licenses": {
                checkpoint: _license_dict(lic) for checkpoint, lic in self.checkpoint_licenses
            },
            "commercial_alternatives": list(self.commercial_alternatives),
            "paper": self.paper,
            "summary": self.summary,
        }

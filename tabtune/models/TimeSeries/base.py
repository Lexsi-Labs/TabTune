"""The boundary between ``TimeSeriesPipeline`` and a forecasting backend.

An adapter translates a validated :class:`~tabtune.TimeSeries.schema.TimeSeriesPanel`
into one backend's calling convention and returns plain numpy arrays. Data
validation, context truncation, caching, metrics and persistence live in the
pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...registry.errors import ConfigError

if TYPE_CHECKING:
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

__all__ = [
    "AdapterOutput",
    "missing_extra_error",
    "TrainingSpec",
    "TSFMAdapter",
]


def missing_extra_error(spec: Any, missing: str) -> ImportError:
    """An ImportError naming the pip extra that provides ``missing``."""
    extra = getattr(spec, "dependency_extra", None)
    name = getattr(spec, "name", "This model")
    hint = (
        f"Install it with `pip install 'tabtune[{extra}]'`."
        if extra
        else f"Install the {missing!r} package."
    )
    return ImportError(f"{name} requires {missing!r}, which is not installed. {hint}")


@dataclass(frozen=True, eq=False)
class AdapterOutput:
    """Raw forecasts returned by an adapter.

    Attributes:
        point: ``[row, horizon]`` point forecast, rows in panel order, where a
            row is one ``(item, target)`` pair.
        quantiles: ``[row, horizon, quantile]`` forecasts in the order of
            ``ForecastConfig.quantile_levels``, or ``None`` if the backend
            cannot produce them.

    The pipeline checks shapes and that every value is finite, so adapters
    need not.
    """

    point: np.ndarray
    quantiles: np.ndarray | None = None


@dataclass(frozen=True)
class TrainingSpec:
    """A fine-tuning request, built by the pipeline from ``tuning_params``.

    Attributes:
        mode: ``"full"`` updates every weight, ``"lora"`` trains low-rank
            adapters on the linear layers named by ``lora_targets``.
        prediction_length: Horizon of the training windows.
        context_length: Context of the training windows; ``None`` means the
            model's maximum.
        steps: Optimiser steps.
        learning_rate: AdamW learning rate.
        batch_size: Windows per step.
        weight_decay: AdamW weight decay.
        gradient_clip_norm: Gradient norm clip, or ``None``.
        validation: Hold out the last ``prediction_length`` steps of every
            series, evaluate on them and keep the best weights.
        patience: Validation checks without improvement before stopping, or
            ``None`` to always run every step.
        validation_every: Steps between validation checks.
        lora_r, lora_alpha, lora_dropout: LoRA hyper-parameters.
        lora_targets: Linear layer name patterns for LoRA; ``None`` uses the
            adapter's defaults.
        seed: Seed for window sampling and initialisation.
    """

    mode: str = "full"
    prediction_length: int = 16
    context_length: int | None = None
    steps: int = 250
    learning_rate: float = 1e-5
    batch_size: int = 32
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    validation: bool = False
    patience: int | None = None
    validation_every: int = 50
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] | None = None
    seed: int | None = None


class TSFMAdapter(ABC):
    """Base class for time series foundation model adapters.

    Subclasses implement :meth:`load` and :meth:`forecast`, and keep whatever
    :meth:`load` creates (model weights, a backend pipeline object) in
    ``self._model``. That convention is what lets :meth:`__getstate__` drop the
    weights when a fitted pipeline is saved; the pipeline calls :meth:`load`
    again on first use after loading.

    Two operations are optional, and the model's registry spec declares which
    ones an adapter implements:

    * :meth:`embed` for ``task_type="embedding"``;
    * :meth:`finetune` for ``tuning_strategy="finetune"`` and ``"peft"``.
      Adapters that train through TabTune's loop implement :meth:`_network`
      and :meth:`_training_loss` and inherit :meth:`finetune`. The tuned
      weights are kept in :attr:`tuned_state`, which survives ``save`` and
      ``load``, and :meth:`_restore_tuned` re-applies them after :meth:`load`.

    Attributes:
        point_forecast: Which statistic :attr:`AdapterOutput.point` holds,
            ``"mean"`` or ``"median"``; recorded in the forecast metadata.
        lora_targets: Default linear layer name patterns for LoRA.

    Args:
        spec: The registry entry this adapter serves.
        checkpoint: Checkpoint identifier (for example a Hugging Face repo id).
        device: Concrete device string, already resolved by the pipeline.
        model_params: Backend-specific keyword arguments.
        seed: Seed for backends that sample. ``None`` leaves the global RNG
            alone. :attr:`respects_seed` says whether the seed reaches the
            model at all; the pipeline warns when it would not.
    """

    point_forecast: str = "mean"
    lora_targets: tuple[str, ...] = ()
    #: Weight precisions this backend can run in; a single entry means fixed precision.
    supported_dtypes: tuple[str, ...] = ("float32",)
    #: Precision to default to on CUDA, where a backend recommends one.
    cuda_default_dtype: str | None = None
    #: Whether ``seed`` changes this backend's output.
    respects_seed: bool = False
    #: Context width of the fine-tuning windows when the caller names none.
    default_train_context: int = 512

    def __init__(
        self,
        spec: TimeSeriesModelSpec,
        *,
        checkpoint: str,
        device: str,
        model_params: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.spec = spec
        self.checkpoint = checkpoint
        self.device = device
        self.model_params = dict(model_params or {})
        self.seed = seed
        self.tuned_state: dict[str, Any] | None = None
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        """Whether :meth:`load` has run since construction or unpickling."""
        return self._model is not None

    @abstractmethod
    def load(self) -> None:
        """Load weights into ``self._model``. Import backend packages here, not at module level."""

    @abstractmethod
    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        """Forecast ``config.prediction_length`` steps past the end of every series in ``panel``.

        ``panel`` is already cut to the model's context length, and every
        series in it has at least one observed value.
        """

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Return one ``[row, dim]`` embedding per panel row."""
        raise NotImplementedError(f"{self.spec.name} does not produce embeddings")

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        """Train the loaded weights on ``panel`` and return a training report."""
        import torch

        from ._training import export_state, prepare_trainable, run_training

        if spec.seed is not None:
            torch.manual_seed(spec.seed)
        module = self._network()
        trainable, lora_meta = prepare_trainable(
            torch,
            module,
            spec.mode,
            lora_targets=spec.lora_targets or self.lora_targets,
            lora_r=spec.lora_r,
            lora_alpha=spec.lora_alpha,
            lora_dropout=spec.lora_dropout,
            device=self.device,
        )
        context = self._training_context(spec)
        horizon = spec.prediction_length
        train_values = list(panel.values)
        validation_values = None
        if spec.validation:
            validation_values = train_values
            train_values = [v[:-horizon] if len(v) > horizon + 1 else v for v in train_values]
        report = run_training(
            torch,
            module,
            trainable,
            lambda ctx, lab: self._training_loss(torch, ctx, lab),
            train_values=train_values,
            validation_values=validation_values,
            context=context,
            label=horizon,
            spec=spec,
        )
        self.tuned_state = export_state(module, lora_meta, base_modified=spec.mode == "full")
        return report

    def _network(self) -> Any:
        """The ``torch.nn.Module`` that :meth:`finetune` trains."""
        raise NotImplementedError(f"{self.spec.name} does not support fine-tuning")

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        """Loss on ``[batch, context]`` windows and their ``[batch, horizon]`` labels."""
        raise NotImplementedError(f"{self.spec.name} does not support fine-tuning")

    def _training_context(self, spec: TrainingSpec) -> int:
        """Context length of the training windows."""
        limit = self.spec.max_context or 512
        return min(spec.context_length or limit, limit)

    def _resolve_dtype(self) -> str:
        """The precision to load in, from ``model_params['dtype']``.

        ``None`` or an absent key means this backend's default. An unsupported
        value raises ``ConfigError`` for a multi-precision backend or one with
        :attr:`refuses_other_dtypes`, and otherwise warns that it is ignored.
        """
        from ..._internal.deprecation import warn_once

        name = self.spec.name
        default = (
            self.cuda_default_dtype
            if self.cuda_default_dtype and self.device.startswith("cuda")
            else self.supported_dtypes[0]
        )
        requested = self.model_params.get("dtype")
        if requested is None:
            return default
        if not isinstance(requested, str):
            raise ConfigError(f"{name} model_params['dtype'] must be a string, got {requested!r}")
        if requested in self.supported_dtypes:
            return requested
        if len(self.supported_dtypes) > 1 or self.refuses_other_dtypes:
            raise ConfigError(
                f"{name} runs in {' or '.join(self.supported_dtypes)}, "
                f"got dtype={requested!r}"
            )
        warn_once(
            f"{name} runs in {default} and ignores model_params['dtype']={requested!r}; "
            f"the value is not applied. This will raise in a future release.",
            UserWarning,
            key=f"ts-dtype-ignored:{name}:{requested}",
        )
        return default

    #: Whether an unsupported ``dtype`` raises instead of being ignored with a warning.
    refuses_other_dtypes: bool = False

    def missing_extra_error(self, exc: ImportError, default: str) -> ImportError:
        """An ImportError naming the extra to install."""
        return missing_extra_error(self.spec, exc.name or default)

    def training_horizon(self, spec: TrainingSpec, limit: int, *, reason: str) -> TrainingSpec:
        """Clamp a training horizon to ``limit``, warning when it moves."""
        from dataclasses import replace

        from ..._internal.deprecation import warn_once

        if spec.prediction_length <= limit:
            return spec
        warn_once(
            f"{self.spec.name} {reason}, so fine-tuning uses {limit}-step windows rather "
            f"than the requested {spec.prediction_length}. Longer forecasts still work at "
            f"predict time, where the model is rolled forward.",
            UserWarning,
            key=f"ts-train-horizon:{self.spec.name}:{spec.prediction_length}",
        )
        return replace(spec, prediction_length=limit)

    @contextmanager
    def _seeded_rng(self, torch: Any) -> Iterator[None]:
        """Fork the RNG of the CPU and the adapter's device, then seed it.

        Without a seed the fork is disabled and the global RNG is consumed.
        """
        device_type, _, index = self.device.partition(":")
        fork_kwargs: dict[str, Any] = {"devices": [], "enabled": self.seed is not None}
        if device_type == "cuda":
            fork_kwargs["devices"] = [int(index) if index else torch.cuda.current_device()]
            fork_kwargs["device_type"] = "cuda"
        elif device_type == "mps":
            fork_kwargs["devices"] = [0]
            fork_kwargs["device_type"] = "mps"

        with torch.random.fork_rng(**fork_kwargs):
            if self.seed is not None:
                torch.manual_seed(self.seed)
            yield

    def quantile_range(self) -> tuple[float, float] | None:
        """Lowest and highest quantile level the model predicts without
        extrapolating, or ``None`` if it serves any level in (0, 1)."""
        return None

    def _restore_tuned(self, module: Any) -> None:
        """Re-apply :attr:`tuned_state` to a freshly loaded ``module``."""
        if self.tuned_state is None:
            return
        from ._training import import_state

        import_state(module, self.tuned_state, default_targets=self.lora_targets, device=self.device)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("tuned_state", None)
        self.__dict__.update(state)

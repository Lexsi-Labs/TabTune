"""Typed, validated configuration objects for TabTune.

Historically every knob in TabTune travelled as an untyped ``dict``. Unknown
keys were silently dropped or silently forwarded into a model constructor, and
the defaults for each fine-tuning loop were duplicated in three places (the
loop body, ``TabularPipeline.get_params`` and the documentation).

These schemas make the contract explicit while staying **fully backward
compatible**: every configuration object accepts a plain dict, forwards unknown
keys unchanged, and exposes ``.to_dict()`` so existing code paths that expect a
dict keep working. The only behavioural change is that typos now produce a
warning instead of vanishing.

Example:
    >>> cfg = TuningConfig(epochs=10, learning_rate=1e-5)
    >>> cfg.epochs
    10
    >>> TuningConfig.from_dict({"epochs": 3, "lr": 1e-4}).extras  # doctest: +SKIP
    {'lr': 0.0001}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .._internal.deprecation import warn_unknown_keys
from ..registry.errors import ConfigError

__all__ = [
    "PeftConfig",
    "TuningConfig",
    "ProcessorConfig",
    "ContextSamplingConfig",
    "PipelineConfig",
    "TaskType",
    "TuningStrategy",
    "FinetuneMode",
]

TaskType = Literal["classification", "regression"]
TuningStrategy = Literal["inference", "finetune", "peft"]
FinetuneMode = Literal["meta-learning", "sft", "native", "turn_by_turn", "tbt"]

#: Aliases accepted for ``finetune_mode`` so users are not tripped by spelling.
_FINETUNE_MODE_ALIASES: dict[str, str] = {
    "meta_learning": "meta-learning",
    "metalearning": "meta-learning",
    "meta": "meta-learning",
    "episodic": "meta-learning",
    "supervised": "sft",
    "tbt": "turn_by_turn",
    "turn-by-turn": "turn_by_turn",
    "turnbyturn": "turn_by_turn",
}


class _Base(BaseModel):
    """Shared behaviour: forgiving construction, dict round-tripping."""

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @property
    def extras(self) -> dict[str, Any]:
        """Return keys TabTune does not recognise, forwarded to the model."""
        return dict(self.__pydantic_extra__ or {})

    @classmethod
    def known_fields(cls) -> tuple[str, ...]:
        """Return the field names this schema recognises."""
        return tuple(cls.model_fields)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | _Base | None,
        *,
        strict: bool = False,
        context: str | None = None,
    ):
        """Build a config from a dict, warning about unrecognised keys.

        Args:
            data: A mapping, an existing config instance, or ``None`` for
                defaults.
            strict: Raise :class:`ConfigError` on unknown keys instead of
                warning. Off by default to preserve backward compatibility with
                model-specific kwargs.
            context: Name used in warnings, e.g. ``"tuning_params"``.

        Returns:
            A validated config instance.

        Raises:
            ConfigError: On invalid values, or on unknown keys when ``strict``.
        """
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        if isinstance(data, _Base):
            data = data.to_dict()
        if not isinstance(data, Mapping):
            raise ConfigError(
                f"{cls.__name__} expects a mapping, got {type(data).__name__}"
            )

        try:
            instance = cls(**dict(data))
        except ConfigError:
            raise
        except Exception as exc:  # pydantic ValidationError and friends
            raise ConfigError(f"Invalid {context or cls.__name__}: {exc}") from exc

        unknown = sorted(instance.extras)
        if unknown:
            if strict:
                raise ConfigError(
                    f"Unrecognised {context or cls.__name__} key(s): "
                    f"{', '.join(repr(k) for k in unknown)}. "
                    f"Known keys: {', '.join(cls.known_fields())}."
                )
            warn_unknown_keys(
                unknown, context=context or cls.__name__, known=cls.known_fields()
            )
        return instance

    def to_dict(self, *, drop_none: bool = True) -> dict[str, Any]:
        """Return a plain dict, suitable for the legacy ``**params`` call sites.

        Args:
            drop_none: Omit keys whose value is ``None``, so downstream code
                that uses ``params.get(key, default)`` sees its own default
                rather than an explicit ``None``.
        """
        data = self.model_dump(exclude_none=drop_none)
        return data

    def merged(self, **overrides: Any):
        """Return a copy with ``overrides`` applied."""
        merged = {**self.to_dict(drop_none=False), **overrides}
        return type(self)(**merged)


class PeftConfig(_Base):
    """LoRA hyper-parameters.

    Attributes:
        r: LoRA rank. Lower means fewer trainable parameters.
        lora_alpha: Scaling factor applied to the low-rank update.
        lora_dropout: Dropout applied inside the LoRA branch.
        target_modules: Explicit module-name substrings to adapt. ``None``
            resolves per model from the registry, falling back to every linear
            layer.
    """

    r: int = Field(default=8, ge=1, le=1024)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] | None = None

    @field_validator("target_modules", mode="before")
    @classmethod
    def _coerce_targets(cls, value: Any) -> Any:
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, (tuple, set, frozenset)):
            return list(value)
        if isinstance(value, str):
            return [value]
        return value


class TuningConfig(_Base):
    """Parameters controlling model adaptation.

    All fields have defaults matching TabTune's historical behaviour, so
    constructing ``TuningConfig()`` reproduces the previous implicit defaults.

    Attributes:
        epochs: Number of passes over the training data.
        learning_rate: Optimiser learning rate.
        batch_size: Batch size for supervised fine-tuning.
        device: Device request; ``"auto"`` picks CUDA, then MPS, then CPU.
        finetune_mode: Fine-tuning algorithm. ``None`` selects a per-task
            default (``turn_by_turn`` for regression, ``meta-learning``
            otherwise).
        seed: Seed applied to python, numpy and torch before training. ``None``
            leaves global RNG state untouched.
        early_stopping: Enable early stopping where the underlying loop
            supports it.
        gradient_clip_norm: Max gradient norm, or ``None`` to disable clipping.
        peft_config: LoRA settings, used when ``tuning_strategy='peft'``.
    """

    epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    batch_size: int = Field(default=32, ge=1)
    device: str = "auto"
    finetune_mode: str | None = None
    seed: int | None = None

    early_stopping: bool = False
    early_stopping_patience: int = Field(default=8, ge=1)
    validation_split: float = Field(default=0.0, ge=0.0, lt=1.0)
    gradient_clip_norm: float | None = Field(default=None, gt=0.0)
    n_estimators_finetune: int | None = Field(default=None, ge=1)

    # Episodic fine-tuning knobs for the in-context models (TabFM, iLTM, LimiX,
    # EXAONE Tabular). Each step samples a fresh support/query episode out of the
    # training frame rather than a flat minibatch, so ``batch_size`` does not
    # describe the unit of work; these do.
    #
    # .. versionadded:: 0.2.0
    #    They were already the de-facto vocabulary -- the episodic tuners read
    #    exactly these keys -- but were undeclared, so passing one tripped the
    #    "unrecognised tuning_params key(s)" typo warning. The library was
    #    warning about its own parameter names.
    support_size: int | None = Field(default=None, ge=1)
    query_size: int | None = Field(default=None, ge=1)
    steps_per_epoch: int | None = Field(default=None, ge=1)
    n_episodes: int | None = Field(default=None, ge=1)

    save_checkpoint_path: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_epochs: int | None = Field(default=None, ge=1)
    show_progress: bool = True

    peft_config: PeftConfig | None = None

    @field_validator("finetune_mode", mode="before")
    @classmethod
    def _normalise_mode(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"finetune_mode must be a string, got {type(value).__name__}")
        key = value.strip().lower()
        return _FINETUNE_MODE_ALIASES.get(key, key)

    @field_validator("peft_config", mode="before")
    @classmethod
    def _coerce_peft(cls, value: Any) -> Any:
        if value is None or isinstance(value, PeftConfig):
            return value
        if isinstance(value, Mapping):
            return PeftConfig(**dict(value))
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> TuningConfig:
        if self.checkpoint_epochs is not None and self.checkpoint_dir is None:
            # Not fatal: the tuner falls back to ./checkpoints. Surface it so
            # the user is not surprised by files appearing in their CWD.
            object.__setattr__(self, "checkpoint_dir", self.checkpoint_dir)
        return self


class ProcessorConfig(_Base):
    """Preprocessing strategies applied before the model sees the data.

    .. note::
       Before v0.2.0 these fields were silently ignored for classification
       tasks: model-aware defaults overwrote them unconditionally and the
       standard preprocessing stack was only fitted for regression. They now
       take effect, with model-aware defaults used only where the user did not
       specify a value.
    """

    task_type: TaskType = "classification"
    imputation_strategy: Literal["mean", "median", "most_frequent", "iterative", "knn", "none"] | None = None
    categorical_encoding: str | None = None
    scaling_strategy: Literal["standard", "minmax", "robust", "power_transform", "none"] | None = None
    resampling_strategy: Literal[
        "smote", "random_over", "random_under", "tomek", "kmeans", "knn", "none"
    ] | None = None
    feature_selection_strategy: Literal[
        "variance", "select_k_best_anova", "select_k_best_chi2", "none"
    ] | None = None
    correlation_threshold: float | None = Field(default=None, gt=0.0, le=1.0)

    @field_validator(
        "imputation_strategy",
        "scaling_strategy",
        "resampling_strategy",
        "feature_selection_strategy",
        "categorical_encoding",
        mode="before",
    )
    @classmethod
    def _none_string(cls, value: Any) -> Any:
        """Treat the literal string ``"none"`` as ``None``, as users often do."""
        if isinstance(value, str) and value.strip().lower() in ("none", ""):
            return None
        return value


class ContextSamplingConfig(_Base):
    """Controls which labelled rows are placed in an ICL model's context.

    In-context learners consume the training set at inference time, so the
    choice of context rows matters as much as any hyper-parameter. This is the
    supported way to bound context size for large datasets.

    Attributes:
        strategy: Sampling strategy name (``uniform``, ``stratified``,
            ``balanced``, ``oversample_minority``, ``smote``,
            ``diversity_kmeans``, ``hybrid_balanced_diverse``).
        context_size: Target number of context rows.
        hybrid_ratio: Balance between the balanced and diverse halves of the
            hybrid strategy.
    """

    strategy: str | None = Field(default=None, alias="context_sampling_strategy")
    context_size: int | None = Field(default=None, ge=1)
    strat_set: int = Field(default=10, ge=1)
    hybrid_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    sampling_seed: int = 42
    allow_replacement: bool = True
    kmeans_centers: int = Field(default=2000, ge=1)
    min_pos: int = Field(default=50, ge=0)
    oversample_weight: float = Field(default=5.0, gt=0.0)

    @property
    def enabled(self) -> bool:
        """Whether context sampling will actually be applied."""
        return self.strategy is not None

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the key names ``TabularPipeline`` used before v0.2.0."""
        return {
            "context_sampling_strategy": self.strategy,
            "context_size": self.context_size,
            "strat_set": self.strat_set,
            "hybrid_ratio": self.hybrid_ratio,
            "sampling_seed": self.sampling_seed,
            "allow_replacement": self.allow_replacement,
            "kmeans_centers": self.kmeans_centers,
            "min_pos": self.min_pos,
            "oversample_weight": self.oversample_weight,
        }


class PipelineConfig(_Base):
    """A complete, serialisable description of a TabTune run.

    This is what ``tabtune fit --config run.yaml`` consumes and what model
    cards embed, so an experiment can be reproduced from a single file.

    Example:
        >>> cfg = PipelineConfig(model_name="TabICLv2", tuning_strategy="inference")
        >>> cfg.model_name
        'TabICLv2'
    """

    model_name: str
    task_type: TaskType = "classification"
    tuning_strategy: TuningStrategy = "inference"
    finetune_mode: str | None = None
    model_checkpoint_path: str | None = None

    tuning: TuningConfig = Field(default_factory=TuningConfig)
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig)
    context_sampling: ContextSamplingConfig = Field(default_factory=ContextSamplingConfig)
    model_params: dict[str, Any] = Field(default_factory=dict)

    envelope_mode: Literal["error", "warn", "ignore"] = "warn"
    license_mode: Literal["research", "commercial", "ignore"] = "research"
    cache: Literal["memory", "disk", "none"] | None = None
    random_state: int | None = None

    @field_validator("finetune_mode", mode="before")
    @classmethod
    def _normalise_mode(cls, value: Any) -> Any:
        return TuningConfig._normalise_mode(value)

    @model_validator(mode="after")
    def _canonicalise(self) -> PipelineConfig:
        """Canonicalise the model name and keep nested task types in sync.

        Registry *validation* deliberately does not happen here. Pydantic wraps
        any ``ValueError`` raised inside a validator in a ``ValidationError``,
        which would bury TabTune's actionable "here are compatible models"
        messages under a pydantic traceback. Callers invoke
        :meth:`validate_against_registry` explicitly instead, which is also
        what lets a config be constructed for a model registered later.
        """
        from ..registry import resolve_model_name
        from ..registry.errors import ModelNotFoundError

        try:
            self.__dict__["model_name"] = resolve_model_name(self.model_name)
        except ModelNotFoundError:
            # Leave the name untouched; validate_against_registry() reports it
            # with a suggestion when the caller is ready to act on it.
            pass

        if self.processor.task_type != self.task_type:
            self.processor.__dict__["task_type"] = self.task_type
        return self

    def validate_against_registry(self, *, strict_finetune_mode: bool = False) -> PipelineConfig:
        """Check this configuration against the model registry.

        Args:
            strict_finetune_mode: Raise rather than warn when the requested
                fine-tuning mode is not implemented by the model.

        Returns:
            ``self``, so this composes in a fluent chain.

        Raises:
            ModelNotFoundError: Unknown model name.
            UnsupportedTaskError: Model has no head for ``task_type``.
            UnsupportedStrategyError: Model does not implement the strategy.
        """
        from ..registry import validate_request

        validate_request(
            self.model_name,
            self.task_type,
            self.tuning_strategy,
            self.finetune_mode or self.tuning.finetune_mode,
            strict_finetune_mode=strict_finetune_mode,
        )
        return self

    def resolved_finetune_mode(self) -> str:
        """Return the effective fine-tuning mode, applying the per-task default."""
        if self.finetune_mode:
            return self.finetune_mode
        if self.tuning.finetune_mode:
            return self.tuning.finetune_mode
        return "turn_by_turn" if self.task_type == "regression" else "meta-learning"

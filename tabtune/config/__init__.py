"""Typed configuration for TabTune.

Every knob TabTune exposes is described by a pydantic model here, which gives
validation, IDE completion, YAML/JSON round-tripping and a single home for
defaults that were previously duplicated across fine-tuning loops, the
``get_params`` method and the docs.

Backward compatibility is preserved throughout: ``TabularPipeline`` still
accepts plain dicts for ``tuning_params``, ``processor_params`` and
``model_params``, and unrecognised keys are forwarded to the model rather than
dropped. The difference is that a typo now warns instead of vanishing.

Example:
    >>> from tabtune.config import PipelineConfig, TuningConfig
    >>> cfg = PipelineConfig(
    ...     model_name="TabICLv2",
    ...     tuning_strategy="finetune",
    ...     tuning=TuningConfig(epochs=10, learning_rate=1e-5, seed=0),
    ... )
    >>> cfg.resolved_finetune_mode()
    'meta-learning'
"""

from __future__ import annotations

from .loader import config_from_mapping, dump_config, load_config, save_config
from .schemas import (
    ContextSamplingConfig,
    PeftConfig,
    PipelineConfig,
    ProcessorConfig,
    TuningConfig,
)

__all__ = [
    "PipelineConfig",
    "TuningConfig",
    "PeftConfig",
    "ProcessorConfig",
    "ContextSamplingConfig",
    "load_config",
    "save_config",
    "dump_config",
    "config_from_mapping",
]

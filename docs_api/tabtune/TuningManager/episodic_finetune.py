"""Public name of the validated episodic fine-tuning driver.

The implementation is in ``tabtune._internal.episodic_finetune`` so the model
modules can import it without importing ``TuningManager``.
"""

from tabtune._internal.episodic_finetune import (  # noqa: F401
    EpisodicConfig,
    FinetuneReport,
    ParameterSnapshot,
    _lr_lambda,
    episode_sizes,
    log_loss,
    rmse,
    run_episodic_finetune,
    sample_episode,
    split_validation,
)

__all__ = [
    "EpisodicConfig",
    "FinetuneReport",
    "ParameterSnapshot",
    "episode_sizes",
    "log_loss",
    "rmse",
    "run_episodic_finetune",
    "sample_episode",
    "split_validation",
]

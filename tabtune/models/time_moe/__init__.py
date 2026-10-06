"""Time-MoE (Shi et al., ICLR 2025) vendored for TabTune.

Decoder-only mixture-of-experts forecasters from https://github.com/Time-MoE/Time-MoE
(Apache-2.0). The upstream checkpoints (``Maple728/TimeMoE-50M``,
``Maple728/TimeMoE-200M``) are published as Hugging Face "remote code" models;
vendoring the two model files lets TabTune load them with
``TimeMoeForPrediction.from_pretrained`` and **never** ``trust_remote_code``.
The code needs only torch and transformers, both core TabTune dependencies.
See ``ATTRIBUTION.md`` for provenance and the exact changes.

Nothing here is imported by ``import tabtune``: the adapter in
``tabtune.models.TimeSeries.timemoe`` loads this package on first use.
"""

from .configuration_time_moe import TimeMoeConfig
from .modeling_time_moe import TimeMoeForPrediction, TimeMoeModel

__all__ = ["TimeMoeConfig", "TimeMoeForPrediction", "TimeMoeModel"]

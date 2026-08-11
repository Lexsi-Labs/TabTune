"""Model registry: metadata, discovery and validation for the bundled TFMs.

The registry is the answer to two questions TabTune could not previously
answer without loading a multi-gigabyte checkpoint:

* *Will this model accept my data?* - capability envelopes encode the
  architectural limits of each pretrained checkpoint (TabFM's ten-class output
  head, TabPFN-3's cell budget, Mitra's practical row ceiling), so a mismatch
  raises a clear message in milliseconds instead of an out-of-memory error
  after a long download.

* *Can I actually deploy this?* - weight licenses vary sharply across the
  bundled models, from MIT through CC-BY to research-only. The registry records
  them and can fail fast when the intended use is commercial.

Example:
    >>> from tabtune.registry import list_models, get_model_spec
    >>> ", ".join(s.name for s in list_models(commercial_ok=True, task="classification"))
    'Mitra, OrionBix, OrionMSP, OrionMSPv1.5, TabICL, TabICLv2'
    >>> get_model_spec("tabpfn-v2.6").name
    'TabPFNv26'
"""

from __future__ import annotations

from .catalog import MODEL_SPECS
from .errors import (
    ConfigError,
    EnvelopeError,
    LicenseError,
    ModelNotFoundError,
    TabTuneError,
    UnsupportedStrategyError,
    UnsupportedTaskError,
)
from .registry import (
    MODEL_REGISTRY,
    check_envelope,
    check_license,
    get_model_spec,
    infer_data_shape,
    list_model_names,
    list_models,
    models_dataframe,
    register_model,
    resolve_model_name,
    validate_request,
)
from .spec import (
    CapabilityEnvelope,
    EnvelopeViolation,
    LicenseSpec,
    ModelSpec,
    normalise_name,
)

__all__ = [
    # specs
    "ModelSpec",
    "CapabilityEnvelope",
    "LicenseSpec",
    "EnvelopeViolation",
    "normalise_name",
    # registry
    "MODEL_REGISTRY",
    "MODEL_SPECS",
    "register_model",
    "get_model_spec",
    "resolve_model_name",
    "list_models",
    "list_model_names",
    "models_dataframe",
    "validate_request",
    "check_envelope",
    "check_license",
    "infer_data_shape",
    # errors
    "TabTuneError",
    "ConfigError",
    "ModelNotFoundError",
    "UnsupportedTaskError",
    "UnsupportedStrategyError",
    "EnvelopeError",
    "LicenseError",
]

# API: Model Registry

*New in 0.2.0.* Torch-free model metadata, discovery and validation.
Narrative guide: [Model Registry](../user-guide/registry.md).

```python
from tabtune.registry import (
    # specs
    ModelSpec, CapabilityEnvelope, LicenseSpec, EnvelopeViolation, normalise_name,
    # registry
    MODEL_REGISTRY, MODEL_SPECS, register_model, get_model_spec, resolve_model_name,
    list_models, list_model_names, models_dataframe,
    validate_request, check_envelope, check_license, infer_data_shape,
    # errors
    TabTuneError, ConfigError, ModelNotFoundError, UnsupportedTaskError,
    UnsupportedStrategyError, EnvelopeError, LicenseError,
)
```

---

## Functions

::: tabtune.registry.registry
    options:
      show_source: true
      members:
        - register_model
        - resolve_model_name
        - get_model_spec
        - list_model_names
        - list_models
        - models_dataframe
        - validate_request
        - check_envelope
        - check_license
        - infer_data_shape

---

## Specs

::: tabtune.registry.spec.ModelSpec
    options:
      show_source: true

::: tabtune.registry.spec.CapabilityEnvelope
    options:
      show_source: true

::: tabtune.registry.spec.LicenseSpec
    options:
      show_source: true

::: tabtune.registry.spec.EnvelopeViolation
    options:
      show_source: true

::: tabtune.registry.spec.normalise_name
    options:
      show_source: true

---

## Errors

::: tabtune.registry.errors
    options:
      show_source: true

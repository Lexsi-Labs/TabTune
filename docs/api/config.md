# API: Configuration

*New in 0.2.0.* Pydantic schemas for every TabTune knob, with YAML/JSON round-tripping.
Narrative guide: [Typed Configuration](../user-guide/configuration.md).

```python
from tabtune.config import (
    PipelineConfig, TuningConfig, PeftConfig, ProcessorConfig, ContextSamplingConfig,
    load_config, save_config, dump_config, config_from_mapping,
)
```

---

## Schemas

::: tabtune.config.schemas.PipelineConfig
    options:
      show_source: true

::: tabtune.config.schemas.TuningConfig
    options:
      show_source: true

::: tabtune.config.schemas.PeftConfig
    options:
      show_source: true

::: tabtune.config.schemas.ProcessorConfig
    options:
      show_source: true

::: tabtune.config.schemas.ContextSamplingConfig
    options:
      show_source: true

---

## Loader

::: tabtune.config.loader
    options:
      show_source: true
      members:
        - load_config
        - save_config
        - dump_config
        - config_from_mapping

# Typed Configuration

*New in TabTune 0.2.0.*

Every knob TabTune exposes is described by a **pydantic** schema. That gives validation,
IDE completion, YAML/JSON round-tripping, and a single home for defaults that were
previously duplicated across fine-tuning loops, `get_params()` and the docs.

An experiment becomes a file you can commit.

!!! success "Fully backward compatible"
    `TabularPipeline` still accepts plain dicts for `tuning_params`, `processor_params` and
    `model_params`, and unrecognised keys are still forwarded to the model rather than
    dropped. The difference is that a typo now **warns** instead of vanishing.

```python
from tabtune.config import (
    PipelineConfig, TuningConfig, PeftConfig, ProcessorConfig, ContextSamplingConfig,
    load_config, save_config, dump_config, config_from_mapping,
)
```

---

## 1. From YAML

```yaml
# experiments/tabiclv2_finetune.yaml
model_name: TabICLv2
task_type: classification
tuning_strategy: finetune
envelope_mode: error
license_mode: commercial
cache: memory
random_state: 0

tuning:
  epochs: 10
  learning_rate: 1.0e-5
  batch_size: 32
  device: auto
  seed: 0
  early_stopping: true
  early_stopping_patience: 8
  validation_split: 0.15
  gradient_clip_norm: 1.0
  show_progress: true

processor:
  scaling_strategy: standard
  imputation_strategy: none
  resampling_strategy: smote
```

```python
from tabtune.config import load_config

cfg = load_config("experiments/tabiclv2_finetune.yaml", strict=True)
cfg.validate_against_registry()     # fails before any weights load
cfg.resolved_finetune_mode()        # 'meta-learning'
```

!!! tip "Use `strict=True` in CI"
    In strict mode an unknown key is an error, so a stale config fails the job rather than
    quietly running something else. Outside CI, `strict=False` (the default) warns.

---

## 2. Loader functions

| Function | Purpose |
|---|---|
| `load_config(path, *, strict=False)` | Read YAML/JSON into a `PipelineConfig` |
| `config_from_mapping(data, *, strict=False)` | Same, from an in-memory dict |
| `dump_config(config)` | Back to a plain `dict` |
| `save_config(config, path)` | Write YAML/JSON; returns the `Path` |

---

## 3. `PipelineConfig`

| Field | Type | Default |
|---|---|---|
| `model_name` | `str` | *required* |
| `task_type` | `'classification'` / `'regression'` | `'classification'` |
| `tuning_strategy` | `'inference'` / `'finetune'` / `'peft'` | `'inference'` |
| `finetune_mode` | `str | None` | `None` (per-task default) |
| `model_checkpoint_path` | `str | None` | `None` |
| `tuning` | `TuningConfig` | defaults |
| `processor` | `ProcessorConfig` | defaults |
| `model_params` | `dict` | `{}` |
| `envelope_mode` | `'error'` / `'warn'` / `'ignore'` | `'warn'` |
| `license_mode` | `'research'` / `'commercial'` / `'ignore'` | `'research'` |
| `cache` | `'memory'` / `'disk'` / `'none'` / `None` | `None` |
| `random_state` | `int | None` | `None` |

### Methods

```python
cfg.validate_against_registry(strict_finetune_mode=False)
# Checks model / task / strategy (and optionally finetune_mode) against the registry.
# Raises ModelNotFoundError, UnsupportedTaskError or UnsupportedStrategyError.

cfg.resolved_finetune_mode()
# The mode that will actually run: 'turn_by_turn' for regression,
# 'meta-learning' for classification, unless explicitly set.

cfg.to_dict(drop_none=True)
```

---

## 4. `TuningConfig`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `epochs` | `int` | `5` | `>= 1` |
| `learning_rate` | `float` | `1e-5` | `> 0` |
| `batch_size` | `int` | `32` | `>= 1` |
| `device` | `str` | `'auto'` | |
| `finetune_mode` | `str | None` | `None` | |
| `seed` | `int | None` | `None` | |
| `early_stopping` | `bool` | `False` | TabPFNv2.6 / v3 native mode |
| `early_stopping_patience` | `int` | `8` | `>= 1` |
| `validation_split` | `float` | `0.0` | `0 <= x < 1` |
| `gradient_clip_norm` | `float | None` | `None` | `> 0` |
| `n_estimators_finetune` | `int | None` | `None` | `>= 1` |
| `support_size` | `int | None` | `None` | episodic |
| `query_size` | `int | None` | `None` | episodic |
| `steps_per_epoch` | `int | None` | `None` | episodic |
| `n_episodes` | `int | None` | `None` | episodic |
| `save_checkpoint_path` | `str | None` | `None` | |
| `checkpoint_dir` | `str | None` | `None` | |
| `checkpoint_epochs` | `int | None` | `None` | `>= 1` |
| `show_progress` | `bool` | `True` | |
| `peft_config` | `PeftConfig | None` | `None` | |

Finetune-mode aliases (`'tbt'` → `'turn_by_turn'`, etc.) are normalised on validation.

---

## 5. `PeftConfig`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `r` | `int` | `8` | `1 <= r <= 1024` |
| `lora_alpha` | `int` | `16` | `>= 1` |
| `lora_dropout` | `float` | `0.05` | `0 <= x < 1` |
| `target_modules` | `list[str] | None` | `None` | auto-detected per model |

```yaml
tuning:
  peft_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
```

---

## 6. `ProcessorConfig`

| Field | Allowed values |
|---|---|
| `imputation_strategy` | `mean`, `median`, `most_frequent`, `iterative`, `knn`, `none` |
| `categorical_encoding` | `onehot`, `ordinal`, `target`, `hashing`, `binary` |
| `scaling_strategy` | `standard`, `minmax`, `robust`, `power_transform`, `none` |
| `resampling_strategy` | `smote`, `random_over`, `random_under`, `tomek`, `kmeans`, `knn` |
| `feature_selection_strategy` | `variance`, `select_k_best_anova`, `select_k_best_chi2` |
| `correlation_threshold` | `float` in `(0, 1]` |

---

## 7. `ContextSamplingConfig`

Controls support/query construction for meta-learning models.

| Field | Alias | Default |
|---|---|---|
| `strategy` | `context_sampling_strategy` | `None` |
| `context_size` | | `None` |
| `strat_set` | | `10` |
| `hybrid_ratio` | | `0.7` |
| `sampling_seed` | | `42` |
| `allow_replacement` | | `True` |
| `kmeans_centers` | | `2000` |
| `min_pos` | | `50` |
| `oversample_weight` | | `5.0` |

See [Resampling Strategies](resampling.md).

---

## 8. Building configs in Python

```python
from tabtune.config import PipelineConfig, TuningConfig, PeftConfig, save_config

cfg = PipelineConfig(
    model_name="TabICLv2",
    tuning_strategy="peft",
    tuning=TuningConfig(
        epochs=10,
        learning_rate=1e-5,
        seed=0,
        peft_config=PeftConfig(r=16, lora_alpha=32),
    ),
    envelope_mode="error",
    license_mode="commercial",
)

save_config(cfg, "experiments/run.yaml")
```

---

## See Also

- [Model Registry](registry.md) — what `validate_against_registry()` checks
- [API: TabularPipeline](../api/pipeline.md)
